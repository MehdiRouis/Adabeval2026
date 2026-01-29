"""
Task A: Politeness/Impoliteness Detection — Fine-tuning AraBERT end-to-end.

Runs multiple configurations and picks the best one:
  A) Full fine-tune, LR=2e-5
  B) Full fine-tune, LR=1e-5 (more conservative)
  C) Differential LR (BERT body=1e-5, classifier head=5e-4)
  D) Freeze first 6 BERT layers, fine-tune top 6 + head
  E) Gradient accumulation (effective batch=64) + LR=3e-5
  F) Full fine-tune, LR=2e-5, no class weights (baseline)
"""

import os
import sys
import json
import re
import copy
import warnings
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)

try:
    from arabert.preprocess import ArabertPreprocessor
except ImportError:
    ArabertPreprocessor = None

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIG
# ============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_CSV = os.path.join(BASE_DIR, "TaskApoliteness_train.csv")
VAL_CSV = os.path.join(BASE_DIR, "TaskApoliteness_val.csv")
MODEL_NAME = "aubmindlab/bert-base-arabertv2"
OUTPUT_DIR = os.path.join(BASE_DIR, "output_task_a_finetune")
os.makedirs(OUTPUT_DIR, exist_ok=True)

NUM_LABELS = 3
MAX_LENGTH = 128
SEED = 42


@dataclass
class ExperimentConfig:
    name: str
    lr: float = 2e-5
    head_lr: Optional[float] = None       # if set, use differential LR
    batch_size: int = 16
    grad_accum_steps: int = 1
    epochs: int = 10
    warmup_ratio: float = 0.1
    patience: int = 3
    weight_decay: float = 0.01
    use_class_weights: bool = True
    freeze_layers: int = 0                 # freeze first N encoder layers


EXPERIMENTS = [
    ExperimentConfig(name="A_full_lr2e5", lr=2e-5),
    ExperimentConfig(name="B_full_lr1e5", lr=1e-5),
    ExperimentConfig(name="C_diff_lr", lr=1e-5, head_lr=5e-4),
    ExperimentConfig(name="D_freeze6", lr=2e-5, freeze_layers=6),
    ExperimentConfig(name="E_accum4_lr3e5", lr=3e-5, grad_accum_steps=4),
    ExperimentConfig(name="F_no_weights", lr=2e-5, use_class_weights=False),
]


# ============================================================================
# REPRODUCIBILITY
# ============================================================================
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# ============================================================================
# PREPROCESSING
# ============================================================================
def preprocess_text(text, arabert_prep=None):
    if pd.isna(text):
        return ""
    text = str(text)
    if arabert_prep is not None:
        try:
            text = arabert_prep.preprocess(text)
        except Exception:
            pass
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'[إأآا]', 'ا', text)
    text = re.sub(r'ة', 'ه', text)
    text = re.sub(r'ى', 'ي', text)
    text = ' '.join(text.split())
    return text


# ============================================================================
# DATASET
# ============================================================================
class PolitenessDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=MAX_LENGTH):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }


# ============================================================================
# TRAINING HELPERS
# ============================================================================
def compute_class_weights(labels, num_classes):
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    weights = len(labels) / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def freeze_encoder_layers(model, n_layers):
    """Freeze embeddings + first n_layers of the BERT encoder."""
    for param in model.bert.embeddings.parameters():
        param.requires_grad = False
    for i in range(n_layers):
        for param in model.bert.encoder.layer[i].parameters():
            param.requires_grad = False


def build_optimizer(model, cfg):
    """Build optimizer, optionally with differential LR for head vs body."""
    if cfg.head_lr is not None:
        # Separate BERT body params from classifier head params
        body_params = []
        head_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if 'classifier' in name:
                head_params.append(param)
            else:
                body_params.append(param)
        optimizer = AdamW([
            {'params': body_params, 'lr': cfg.lr},
            {'params': head_params, 'lr': cfg.head_lr},
        ], weight_decay=cfg.weight_decay)
    else:
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    return optimizer


def train_one_epoch(model, dataloader, optimizer, scheduler, criterion, device,
                    grad_accum_steps=1):
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    optimizer.zero_grad()

    for step, batch in enumerate(tqdm(dataloader, desc="  Training", leave=False)):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(outputs.logits, labels)
        loss = loss / grad_accum_steps
        loss.backward()

        total_loss += loss.item() * grad_accum_steps * len(labels)
        preds = torch.argmax(outputs.logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(dataloader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

    avg_loss = total_loss / len(dataloader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    _, _, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )
    return avg_loss, acc, f1


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for batch in tqdm(dataloader, desc="  Evaluating", leave=False):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(outputs.logits, labels)

        total_loss += loss.item() * len(labels)
        preds = torch.argmax(outputs.logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )
    return avg_loss, acc, prec, rec, f1, np.array(all_preds), np.array(all_labels)


# ============================================================================
# RUN ONE EXPERIMENT
# ============================================================================
def run_experiment(cfg, tokenizer, train_loader, val_loader,
                   class_weights, device, le):
    """Train one configuration, return best metrics + predictions."""
    set_seed(SEED)
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {cfg.name}")
    print(f"  lr={cfg.lr}, head_lr={cfg.head_lr}, batch={cfg.batch_size}, "
          f"accum={cfg.grad_accum_steps}, freeze={cfg.freeze_layers}, "
          f"class_weights={cfg.use_class_weights}")
    print(f"{'='*60}")

    # Fresh model each time
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_LABELS
    ).to(device)

    # Optionally freeze layers
    if cfg.freeze_layers > 0:
        freeze_encoder_layers(model, cfg.freeze_layers)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"  Frozen first {cfg.freeze_layers} layers: "
              f"{trainable:,}/{total:,} params trainable")

    # Loss
    if cfg.use_class_weights:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    else:
        criterion = nn.CrossEntropyLoss()

    # Optimizer & scheduler
    optimizer = build_optimizer(model, cfg)
    total_steps = (len(train_loader) // cfg.grad_accum_steps) * cfg.epochs
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # Training loop
    best_val_f1 = 0.0
    best_epoch = 0
    best_metrics = {}
    best_preds = None
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'train_f1': [], 'val_f1': []}
    save_path = os.path.join(OUTPUT_DIR, f'model_{cfg.name}')

    for epoch in range(1, cfg.epochs + 1):
        print(f"\n  Epoch {epoch}/{cfg.epochs}")

        train_loss, train_acc, train_f1 = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, device,
            cfg.grad_accum_steps
        )
        val_loss, val_acc, val_prec, val_rec, val_f1, val_preds, val_labels = evaluate(
            model, val_loader, criterion, device
        )

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_f1'].append(train_f1)
        history['val_f1'].append(val_f1)

        print(f"    Train — loss:{train_loss:.4f} acc:{train_acc:.4f} F1:{train_f1:.4f}")
        print(f"    Val   — loss:{val_loss:.4f} acc:{val_acc:.4f} F1:{val_f1:.4f} "
              f"prec:{val_prec:.4f} rec:{val_rec:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            best_metrics = {
                'accuracy': float(val_acc),
                'precision': float(val_prec),
                'recall': float(val_rec),
                'f1_score': float(val_f1),
            }
            best_preds = val_preds.copy()
            best_labels = val_labels.copy()
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            patience_counter = 0
            print(f"    >> Best so far (F1={val_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= cfg.patience:
                print(f"    >> Early stopping at epoch {epoch}")
                break

    print(f"\n  Result: best_epoch={best_epoch}, F1={best_val_f1:.4f}")
    return {
        'config': cfg,
        'best_epoch': best_epoch,
        'metrics': best_metrics,
        'preds': best_preds,
        'labels': best_labels,
        'history': history,
        'save_path': save_path,
    }


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("=" * 60)
    print("TASK A: AraBERT Fine-tuning — Multi-config Search")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # --- Load & preprocess data ---
    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV)
    print(f"\nTraining: {len(train_df)} | Validation: {len(val_df)}")
    print(f"Class distribution (train):\n{train_df['label'].value_counts().to_string()}")

    print("\nPreprocessing...")
    arabert_prep = None
    if ArabertPreprocessor is not None:
        try:
            arabert_prep = ArabertPreprocessor(model_name=MODEL_NAME)
            arabert_prep.preprocess("test")
        except Exception as e:
            print(f"ArabertPreprocessor unavailable ({e})")
            arabert_prep = None

    train_df['processed_text'] = train_df['Sentence'].apply(
        lambda t: preprocess_text(t, arabert_prep)
    )
    val_df['processed_text'] = val_df['Sentence'].apply(
        lambda t: preprocess_text(t, arabert_prep)
    )

    le = LabelEncoder()
    train_df['label_encoded'] = le.fit_transform(train_df['label'])
    val_df['label_encoded'] = le.transform(val_df['label'])
    label_mapping = {int(i): str(l) for i, l in enumerate(le.classes_)}
    print(f"Labels: {label_mapping}")

    class_weights = compute_class_weights(train_df['label_encoded'].values, NUM_LABELS)
    print(f"Class weights: {[f'{w:.3f}' for w in class_weights.tolist()]}")

    # --- Tokenizer & DataLoaders ---
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_dataset = PolitenessDataset(
        train_df['processed_text'].tolist(),
        train_df['label_encoded'].values,
        tokenizer
    )
    val_dataset = PolitenessDataset(
        val_df['processed_text'].tolist(),
        val_df['label_encoded'].values,
        tokenizer
    )
    # Use smallest batch size across experiments for shared loaders
    train_loader = DataLoader(
        train_dataset, batch_size=16, shuffle=True,
        num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=16, shuffle=False,
        num_workers=0, pin_memory=True
    )

    # --- Run all experiments ---
    all_results = []
    for cfg in EXPERIMENTS:
        result = run_experiment(
            cfg, tokenizer, train_loader, val_loader,
            class_weights, device, le
        )
        all_results.append(result)
        # Free GPU memory between experiments
        torch.cuda.empty_cache()

    # --- Compare results ---
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Config':<25} {'Epoch':>5} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7}")
    print("-" * 60)
    for r in all_results:
        m = r['metrics']
        print(f"{r['config'].name:<25} {r['best_epoch']:>5} "
              f"{m['accuracy']:>7.4f} {m['precision']:>7.4f} "
              f"{m['recall']:>7.4f} {m['f1_score']:>7.4f}")

    # --- Pick overall best ---
    best = max(all_results, key=lambda r: r['metrics']['f1_score'])
    best_cfg = best['config']
    best_metrics = best['metrics']
    print(f"\nBEST: {best_cfg.name} — F1={best_metrics['f1_score']:.4f}")

    # Copy best model as "best_model"
    import shutil
    best_model_dir = os.path.join(OUTPUT_DIR, 'best_model')
    if os.path.exists(best_model_dir):
        shutil.rmtree(best_model_dir)
    shutil.copytree(best['save_path'], best_model_dir)

    # Reload best for final report
    best_model = AutoModelForSequenceClassification.from_pretrained(
        best_model_dir
    ).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device)
    ) if best_cfg.use_class_weights else nn.CrossEntropyLoss()
    _, _, _, _, _, final_preds, final_labels = evaluate(
        best_model, val_loader, criterion, device
    )

    print("\nClassification Report (best model):")
    print(classification_report(
        final_labels, final_preds, target_names=le.classes_, zero_division=0
    ))

    # --- Visualizations ---
    # 1) Bar chart comparing all configs
    fig, ax = plt.subplots(figsize=(12, 6))
    names = [r['config'].name for r in all_results]
    f1s = [r['metrics']['f1_score'] for r in all_results]
    colors = ['#2ecc71' if n == best_cfg.name else '#3498db' for n in names]
    bars = ax.bar(names, f1s, color=colors, edgecolor='black', linewidth=0.5)
    ax.axhline(y=0.42, color='red', linestyle='--', linewidth=1.5, label='Previous (feature extraction): 0.42')
    for bar, f1 in zip(bars, f1s):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{f1:.4f}', ha='center', fontweight='bold', fontsize=10)
    ax.set_ylabel('F1 Macro')
    ax.set_title('Fine-tuning Configurations Comparison')
    ax.set_ylim(0, max(f1s) * 1.15)
    ax.legend()
    ax.tick_params(axis='x', rotation=30)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'config_comparison.png'), dpi=300, bbox_inches='tight')
    print("Saved config_comparison.png")

    # 2) Training curves for all configs
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for r in all_results:
        n = r['config'].name
        epochs_range = range(1, len(r['history']['train_loss']) + 1)
        lw = 2.5 if n == best_cfg.name else 1.0
        alpha = 1.0 if n == best_cfg.name else 0.5
        ax1.plot(epochs_range, r['history']['val_loss'], 'o-', label=n, linewidth=lw, alpha=alpha, markersize=3)
        ax2.plot(epochs_range, r['history']['val_f1'], 'o-', label=n, linewidth=lw, alpha=alpha, markersize=3)
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Val Loss'); ax1.set_title('Validation Loss')
    ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Val F1 Macro'); ax2.set_title('Validation F1')
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'training_curves.png'), dpi=300, bbox_inches='tight')
    print("Saved training_curves.png")

    # 3) Confusion matrix for best
    fig, ax = plt.subplots(figsize=(10, 8))
    cm = confusion_matrix(final_labels, final_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=le.classes_, yticklabels=le.classes_)
    ax.set_title(f'Confusion Matrix — {best_cfg.name}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    print("Saved confusion_matrix.png")

    # --- Save artifacts ---
    with open(os.path.join(OUTPUT_DIR, 'label_encoder.json'), 'w', encoding='utf-8') as f:
        json.dump(label_mapping, f, indent=2, ensure_ascii=False)

    config = {
        'best_config': best_cfg.name,
        'model_name': MODEL_NAME,
        'label_mapping': label_mapping,
        'best_metrics': best_metrics,
        'best_epoch': best['best_epoch'],
        'all_results': [
            {
                'name': r['config'].name,
                'best_epoch': r['best_epoch'],
                'metrics': r['metrics'],
                'history': r['history'],
            }
            for r in all_results
        ],
    }
    with open(os.path.join(OUTPUT_DIR, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    # --- Predictions CSV ---
    val_df['predicted_label'] = le.inverse_transform(final_preds)
    submission = val_df[['Sentence', 'predicted_label']].copy()
    submission.columns = ['Sentence', 'label']
    submission.insert(0, 'id', range(1, len(submission) + 1))
    submission.to_csv(
        os.path.join(OUTPUT_DIR, 'OSACT7_TaskA_Predictions.csv'),
        index=False, encoding='utf-8-sig'
    )
    print(f"\nSaved OSACT7_TaskA_Predictions.csv ({len(submission)} predictions)")

    print("\n" + "=" * 60)
    print("ALL DONE")
    print(f"Best config: {best_cfg.name}")
    print(f"Best F1 macro: {best_metrics['f1_score']:.4f}")
    print(f"Previous approach: ~0.42")
    print(f"Outputs: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    main()
