"""
Task A: Politeness Detection — Finding the most viable solution.

Strategy:
  1. Fine-tune MARBERT (trained on 1B Arabic tweets, better match for this data)
     with Focal Loss (handles class imbalance better than weighted CE).
  2. Extract embeddings from the fine-tuned model -> XGBoost + SMOTE (hybrid).
  3. Also re-run AraBERT v2 with Focal Loss for comparison.
  4. Ensemble the best approaches.

Outputs to output_task_a_best/
"""

import os
import json
import re
import warnings
import shutil

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
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
OUTPUT_DIR = os.path.join(BASE_DIR, "output_task_a_best")
os.makedirs(OUTPUT_DIR, exist_ok=True)

NUM_LABELS = 3
MAX_LENGTH = 128
SEED = 42

MODELS_TO_TRY = [
    "UBC-NLP/MARBERT",                   # trained on 1B Arabic tweets
    "aubmindlab/bert-base-arabertv2",     # MSA + dialectal
]


# ============================================================================
# UTILS
# ============================================================================
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


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


def compute_class_weights(labels, num_classes):
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    weights = len(labels) / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


# ============================================================================
# FOCAL LOSS — better than CE for imbalanced data
# ============================================================================
class FocalLoss(nn.Module):
    """
    Focal Loss: down-weights well-classified examples, focuses on hard ones.
    With gamma=0 this is equivalent to CrossEntropyLoss.
    """
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        if alpha is not None:
            self.alpha = alpha  # tensor of per-class weights
        else:
            self.alpha = None

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)  # probability of correct class
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


# ============================================================================
# DATASET
# ============================================================================
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=MAX_LENGTH):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx], padding='max_length', truncation=True,
            max_length=self.max_length, return_tensors='pt'
        )
        return {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }


# ============================================================================
# TRAINING
# ============================================================================
def train_one_epoch(model, loader, optimizer, scheduler, criterion, device):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []
    optimizer.zero_grad()

    for batch in tqdm(loader, desc="  Train", leave=False):
        ids = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        outputs = model(input_ids=ids, attention_mask=mask)
        loss = criterion(outputs.logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        total_loss += loss.item() * len(labels)
        all_preds.extend(torch.argmax(outputs.logits, dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    _, _, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)
    return avg_loss, f1


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for batch in tqdm(loader, desc="  Eval", leave=False):
        ids = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        outputs = model(input_ids=ids, attention_mask=mask)
        loss = criterion(outputs.logits, labels)

        total_loss += loss.item() * len(labels)
        all_preds.extend(torch.argmax(outputs.logits, dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)
    return avg_loss, acc, prec, rec, f1, np.array(all_preds), np.array(all_labels)


@torch.no_grad()
def extract_embeddings(model_base, loader, device):
    """Extract [CLS] embeddings from a fine-tuned model's base."""
    model_base.eval()
    all_emb = []
    for batch in tqdm(loader, desc="  Embeddings", leave=False):
        ids = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        outputs = model_base(input_ids=ids, attention_mask=mask)
        cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        all_emb.append(cls_emb)
    return np.concatenate(all_emb, axis=0)


# ============================================================================
# APPROACH 1: Fine-tune with Focal Loss
# ============================================================================
def finetune_approach(model_name, tokenizer, train_loader, val_loader,
                      class_weights, device, tag):
    """Fine-tune a model with focal loss. Returns best metrics + model path."""
    set_seed(SEED)
    print(f"\n{'='*60}")
    print(f"FINE-TUNE: {tag} ({model_name})")
    print(f"{'='*60}")

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=NUM_LABELS
    ).to(device)

    criterion = FocalLoss(alpha=class_weights.to(device), gamma=2.0)

    optimizer = AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
    total_steps = len(train_loader) * 15
    warmup_steps = int(total_steps * 0.1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    save_path = os.path.join(OUTPUT_DIR, f'model_{tag}')
    best_f1, best_epoch, patience = 0.0, 0, 0
    best_metrics, best_preds, best_labels = {}, None, None

    for epoch in range(1, 16):
        print(f"  Epoch {epoch}/15")
        t_loss, t_f1 = train_one_epoch(model, train_loader, optimizer, scheduler, criterion, device)
        v_loss, v_acc, v_prec, v_rec, v_f1, v_preds, v_labels = evaluate(model, val_loader, criterion, device)
        print(f"    Train loss:{t_loss:.4f} F1:{t_f1:.4f} | Val loss:{v_loss:.4f} F1:{v_f1:.4f} acc:{v_acc:.4f}")

        if v_f1 > best_f1:
            best_f1 = v_f1
            best_epoch = epoch
            best_metrics = {'accuracy': v_acc, 'precision': v_prec, 'recall': v_rec, 'f1_score': v_f1}
            best_preds = v_preds.copy()
            best_labels = v_labels.copy()
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            patience = 0
            print(f"    >> Best (F1={v_f1:.4f})")
        else:
            patience += 1
            if patience >= 4:
                print(f"    >> Early stop at epoch {epoch}")
                break

    print(f"  RESULT: epoch={best_epoch}, F1={best_f1:.4f}")
    return {
        'tag': tag, 'model_name': model_name,
        'best_epoch': best_epoch, 'metrics': best_metrics,
        'preds': best_preds, 'labels': best_labels,
        'save_path': save_path,
    }


# ============================================================================
# APPROACH 2: Fine-tuned embeddings -> XGBoost + SMOTE (hybrid)
# ============================================================================
def hybrid_approach(model_path, tokenizer, train_loader, val_loader,
                    y_train, y_val, device, tag):
    """Use fine-tuned model as feature extractor, then XGBoost + SMOTE."""
    set_seed(SEED)
    print(f"\n{'='*60}")
    print(f"HYBRID: {tag} (fine-tuned embeddings -> XGBoost + SMOTE)")
    print(f"{'='*60}")

    # Load fine-tuned model's base (without classification head)
    full_model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    # Get the base model (bert/roberta/etc)
    if hasattr(full_model, 'bert'):
        base_model = full_model.bert
    elif hasattr(full_model, 'roberta'):
        base_model = full_model.roberta
    else:
        # fallback: try to get base_model attribute
        base_model = getattr(full_model, 'base_model', full_model)

    print("  Extracting train embeddings...")
    train_emb = extract_embeddings(base_model, train_loader, device)
    print("  Extracting val embeddings...")
    val_emb = extract_embeddings(base_model, val_loader, device)
    print(f"  Shapes: train={train_emb.shape}, val={val_emb.shape}")

    # SMOTE
    print("  Applying SMOTE...")
    smote = SMOTE(random_state=SEED)
    X_bal, y_bal = smote.fit_resample(train_emb, y_train)

    # Train XGBoost
    print("  Training XGBoost...")
    xgb = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        random_state=SEED, n_jobs=-1, eval_metric='mlogloss'
    )
    xgb.fit(X_bal, y_bal)
    preds = xgb.predict(val_emb)
    proba = xgb.predict_proba(val_emb)

    acc = accuracy_score(y_val, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(y_val, preds, average='macro', zero_division=0)
    metrics = {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1_score': f1}
    print(f"  RESULT: F1={f1:.4f}, acc={acc:.4f}")

    return {
        'tag': tag, 'metrics': metrics,
        'preds': preds, 'proba': proba, 'labels': y_val,
    }


# ============================================================================
# ENSEMBLE
# ============================================================================
def ensemble_predictions(results_list, num_classes, y_val):
    """Soft vote ensemble using prediction probabilities or one-hot preds."""
    print(f"\n{'='*60}")
    print("ENSEMBLE")
    print(f"{'='*60}")

    n = len(y_val)
    vote_matrix = np.zeros((n, num_classes))

    for r in results_list:
        if 'proba' in r and r['proba'] is not None:
            vote_matrix += r['proba']
        else:
            # One-hot from hard predictions
            for i, p in enumerate(r['preds']):
                vote_matrix[i, p] += 1.0

    preds = np.argmax(vote_matrix, axis=1)
    acc = accuracy_score(y_val, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(y_val, preds, average='macro', zero_division=0)
    print(f"  Ensemble F1={f1:.4f}, acc={acc:.4f}")
    return preds, {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1_score': f1}


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("=" * 60)
    print("TASK A: Finding the Best Solution")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # --- Load data ---
    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV)
    print(f"\nTrain: {len(train_df)} | Val: {len(val_df)}")
    print(f"Distribution:\n{train_df['label'].value_counts().to_string()}")

    le = LabelEncoder()
    train_df['label_encoded'] = le.fit_transform(train_df['label'])
    val_df['label_encoded'] = le.transform(val_df['label'])
    label_map = {int(i): str(l) for i, l in enumerate(le.classes_)}
    print(f"Labels: {label_map}")

    y_train = train_df['label_encoded'].values
    y_val = val_df['label_encoded'].values
    class_weights = compute_class_weights(y_train, NUM_LABELS)
    print(f"Class weights: {class_weights.tolist()}")

    # ==========================================
    # Run approaches for each base model
    # ==========================================
    all_results = []

    for model_name in MODELS_TO_TRY:
        short_name = model_name.split('/')[-1]
        print(f"\n{'#'*60}")
        print(f"# BASE MODEL: {model_name}")
        print(f"{'#'*60}")

        # Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Preprocess — use arabert_prep only for arabertv2
        arabert_prep = None
        if 'arabert' in model_name.lower() and ArabertPreprocessor is not None:
            try:
                arabert_prep = ArabertPreprocessor(model_name=model_name)
                arabert_prep.preprocess("test")
            except Exception:
                arabert_prep = None

        train_texts = train_df['Sentence'].apply(lambda t: preprocess_text(t, arabert_prep)).tolist()
        val_texts = val_df['Sentence'].apply(lambda t: preprocess_text(t, arabert_prep)).tolist()

        train_dataset = TextDataset(train_texts, y_train, tokenizer)
        val_dataset = TextDataset(val_texts, y_val, tokenizer)
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0, pin_memory=True)

        # Approach 1: Fine-tune with Focal Loss
        ft_result = finetune_approach(
            model_name, tokenizer, train_loader, val_loader,
            class_weights, device, tag=f"ft_{short_name}"
        )
        # Get probabilities from fine-tuned model for ensemble
        ft_model = AutoModelForSequenceClassification.from_pretrained(ft_result['save_path']).to(device)
        ft_model.eval()
        all_logits = []
        with torch.no_grad():
            for batch in val_loader:
                ids = batch['input_ids'].to(device)
                mask = batch['attention_mask'].to(device)
                logits = ft_model(input_ids=ids, attention_mask=mask).logits
                all_logits.append(torch.softmax(logits, dim=1).cpu().numpy())
        ft_result['proba'] = np.concatenate(all_logits, axis=0)
        all_results.append(ft_result)
        del ft_model
        torch.cuda.empty_cache()

        # Approach 2: Hybrid (fine-tuned embeddings -> XGBoost + SMOTE)
        hyb_result = hybrid_approach(
            ft_result['save_path'], tokenizer, train_loader, val_loader,
            y_train, y_val, device, tag=f"hyb_{short_name}"
        )
        all_results.append(hyb_result)
        torch.cuda.empty_cache()

    # ==========================================
    # Summary of all approaches
    # ==========================================
    print("\n" + "=" * 60)
    print("ALL RESULTS")
    print("=" * 60)
    print(f"{'Approach':<30} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7}")
    print("-" * 60)
    for r in all_results:
        m = r['metrics']
        print(f"{r['tag']:<30} {m['accuracy']:>7.4f} {m['precision']:>7.4f} "
              f"{m['recall']:>7.4f} {m['f1_score']:>7.4f}")

    # ==========================================
    # Ensembles
    # ==========================================
    # Ensemble all
    ens_all_preds, ens_all_metrics = ensemble_predictions(all_results, NUM_LABELS, y_val)
    all_results.append({
        'tag': 'ensemble_all', 'metrics': ens_all_metrics,
        'preds': ens_all_preds, 'labels': y_val,
    })

    # Ensemble best of each type (fine-tune + hybrid)
    ft_results = [r for r in all_results if r['tag'].startswith('ft_')]
    hyb_results = [r for r in all_results if r['tag'].startswith('hyb_')]
    if ft_results and hyb_results:
        best_ft = max(ft_results, key=lambda r: r['metrics']['f1_score'])
        best_hyb = max(hyb_results, key=lambda r: r['metrics']['f1_score'])
        ens_mix_preds, ens_mix_metrics = ensemble_predictions(
            [best_ft, best_hyb], NUM_LABELS, y_val
        )
        all_results.append({
            'tag': 'ensemble_ft+hyb', 'metrics': ens_mix_metrics,
            'preds': ens_mix_preds, 'labels': y_val,
        })

    # ==========================================
    # Final summary with ensembles
    # ==========================================
    print("\n" + "=" * 60)
    print("FINAL RESULTS (including ensembles)")
    print("=" * 60)
    print(f"{'Approach':<30} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7}")
    print("-" * 60)
    for r in all_results:
        m = r['metrics']
        print(f"{r['tag']:<30} {m['accuracy']:>7.4f} {m['precision']:>7.4f} "
              f"{m['recall']:>7.4f} {m['f1_score']:>7.4f}")

    # ==========================================
    # Pick overall best
    # ==========================================
    best = max(all_results, key=lambda r: r['metrics']['f1_score'])
    print(f"\nBEST OVERALL: {best['tag']} — F1={best['metrics']['f1_score']:.4f}")

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_val, best['preds'], target_names=le.classes_, zero_division=0))

    # Prediction distribution
    pred_labels = le.inverse_transform(best['preds'])
    print("Prediction distribution:")
    print(pd.Series(pred_labels).value_counts().to_string())

    # ==========================================
    # Visualizations
    # ==========================================
    # Bar chart
    fig, ax = plt.subplots(figsize=(14, 6))
    names = [r['tag'] for r in all_results]
    f1s = [r['metrics']['f1_score'] for r in all_results]
    best_name = best['tag']
    colors = ['#2ecc71' if n == best_name else '#e74c3c' if 'ensemble' in n else '#3498db' for n in names]
    bars = ax.bar(names, f1s, color=colors, edgecolor='black', linewidth=0.5)
    ax.axhline(y=0.42, color='gray', linestyle='--', linewidth=1.5, label='Previous best (XGBoost): 0.42')
    for bar, f1 in zip(bars, f1s):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{f1:.4f}', ha='center', fontweight='bold', fontsize=9)
    ax.set_ylabel('F1 Macro')
    ax.set_title('All Approaches Comparison')
    ax.set_ylim(0, max(f1s) * 1.15)
    ax.legend()
    ax.tick_params(axis='x', rotation=30)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'comparison.png'), dpi=300, bbox_inches='tight')
    print("\nSaved comparison.png")

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(10, 8))
    cm = confusion_matrix(y_val, best['preds'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=le.classes_, yticklabels=le.classes_)
    ax.set_title(f'Confusion Matrix — {best["tag"]}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    print("Saved confusion_matrix.png")

    # ==========================================
    # Save submission + artifacts
    # ==========================================
    submission = pd.DataFrame({
        'id': range(1, len(val_df) + 1),
        'label': le.inverse_transform(best['preds'])
    })
    submission.to_csv(os.path.join(OUTPUT_DIR, 'submission_task_a.csv'), index=False, encoding='utf-8-sig')
    print(f"\nSaved submission_task_a.csv ({len(submission)} rows)")

    with open(os.path.join(OUTPUT_DIR, 'label_encoder.json'), 'w', encoding='utf-8') as f:
        json.dump(label_map, f, indent=2, ensure_ascii=False)

    config = {
        'best_approach': best['tag'],
        'best_metrics': {k: float(v) for k, v in best['metrics'].items()},
        'label_mapping': label_map,
        'all_results': [
            {'tag': r['tag'], 'metrics': {k: float(v) for k, v in r['metrics'].items()}}
            for r in all_results
        ],
    }
    with open(os.path.join(OUTPUT_DIR, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("DONE")
    print(f"Best: {best['tag']} — F1={best['metrics']['f1_score']:.4f}")
    print(f"Previous best: ~0.42")
    print(f"Outputs: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    main()
