"""
Task A: Politeness/Impoliteness Detection
Converts OSACT7_AdabEval notebook to standalone Python script.
Uses AraBERT embeddings + linguistic features + ML classifiers (LR, RF, XGBoost, SVM).
"""

import os
import sys
import pickle
import json
import re
import warnings

import pandas as pd
import numpy as np
import torch
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
from sklearn.svm import SVC
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from transformers import AutoTokenizer, AutoModel
from arabert.preprocess import ArabertPreprocessor

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIG
# ============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_CSV = os.path.join(BASE_DIR, "TaskApoliteness_train.csv")
VAL_CSV = os.path.join(BASE_DIR, "TaskApoliteness_val.csv")
MODEL_NAME = "aubmindlab/bert-base-arabertv2"
OUTPUT_DIR = os.path.join(BASE_DIR, "output_task_a")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Politeness indicators
HONORIFICS = ["حضرتك","سيادتك","فضلك","حضرتكم","معاليك","سعادتك","أستاذ","دكتور","مهندس","شيخ","مولانا"]
KINSHIP = ["أخي","أختي","عمي","خالي","خالتي","ابني","بنتي","أمي","أبوي","صديقي","صديقتي","حبيبي","حبيبتي"]
GREETINGS = ["السلام عليكم","وعليكم السلام","مرحبا","أهلا","صباح الخير","مساء الخير","تحياتي","أهلاً وسهلاً"]
PRAYERS = ["اللهم","ربنا","يا رب","ان شاء الله","بإذن الله","بارك الله","جزاك الله","الله يوفقك"]
RESPECT_TERMS = ["احترامي","كل التقدير","تقديري","احترام"]


def preprocess_text(text, arabert_prep=None):
    if pd.isna(text):
        return ""
    text = str(text)
    # Try AraBERT preprocessing, fall back to manual cleaning if Farasa fails
    if arabert_prep is not None:
        try:
            text = arabert_prep.preprocess(text)
        except Exception:
            pass
    # Remove URLs, mentions, hashtags
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+|#\w+', '', text)
    # Normalize some Arabic chars
    text = re.sub(r'[إأآا]', 'ا', text)
    text = re.sub(r'ة', 'ه', text)
    text = re.sub(r'ى', 'ي', text)
    text = ' '.join(text.split())
    return text


def extract_politeness_features(text):
    features = {}
    text = str(text).lower() if pd.notna(text) else ""
    features['has_honorifics'] = int(any(t in text for t in HONORIFICS))
    features['has_kinship'] = int(any(t in text for t in KINSHIP))
    features['has_greetings'] = int(any(t in text for t in GREETINGS))
    features['has_prayers'] = int(any(t in text for t in PRAYERS))
    features['has_respect'] = int(any(t in text for t in RESPECT_TERMS))
    features['text_length'] = len(text.split()) if text else 0
    features['char_length'] = len(text)
    return features


def get_arabert_embedding(text, tokenizer, model, device, max_length=128):
    if pd.isna(text) or text == "":
        return np.zeros(768)
    inputs = tokenizer(
        text, padding='max_length', truncation=True,
        max_length=max_length, return_tensors='pt'
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
    return embedding.flatten()


def extract_embeddings(texts, tokenizer, model, device, desc="Extracting embeddings"):
    embeddings = []
    for text in tqdm(texts, desc=desc):
        emb = get_arabert_embedding(text, tokenizer, model, device)
        embeddings.append(emb)
    return np.array(embeddings)


def main():
    # --- Load data ---
    print("=" * 60)
    print("TASK A: Politeness/Impoliteness Detection")
    print("=" * 60)

    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV)
    print(f"Training set: {len(train_df)} samples")
    print(f"Validation set: {len(val_df)} samples")
    print(f"\nClass distribution (train):\n{train_df['label'].value_counts().to_string()}")

    # --- Preprocessing ---
    print("\nPreprocessing text...")
    try:
        arabert_prep = ArabertPreprocessor(model_name=MODEL_NAME)
        # Test it works
        arabert_prep.preprocess("test")
    except Exception as e:
        print(f"ArabertPreprocessor unavailable ({e}), using manual preprocessing")
        arabert_prep = None
    train_df['processed_text'] = train_df['Sentence'].apply(lambda t: preprocess_text(t, arabert_prep))
    val_df['processed_text'] = val_df['Sentence'].apply(lambda t: preprocess_text(t, arabert_prep))

    # Politeness features
    print("Extracting politeness features...")
    train_feat = train_df['Sentence'].apply(extract_politeness_features).apply(pd.Series)
    val_feat = val_df['Sentence'].apply(extract_politeness_features).apply(pd.Series)
    train_df = pd.concat([train_df, train_feat], axis=1)
    val_df = pd.concat([val_df, val_feat], axis=1)

    # --- Label encoding ---
    le = LabelEncoder()
    train_df['label_encoded'] = le.fit_transform(train_df['label'])
    val_df['label_encoded'] = le.transform(val_df['label'])
    print(f"\nLabel mapping: {dict(enumerate(le.classes_))}")

    # --- AraBERT embeddings ---
    print("\nLoading AraBERT model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    bert_model = AutoModel.from_pretrained(MODEL_NAME)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bert_model = bert_model.to(device)
    bert_model.eval()
    print(f"Device: {device}")

    train_embeddings = extract_embeddings(
        train_df['processed_text'], tokenizer, bert_model, device, "Train embeddings"
    )
    val_embeddings = extract_embeddings(
        val_df['processed_text'], tokenizer, bert_model, device, "Val embeddings"
    )
    print(f"Train embeddings: {train_embeddings.shape}")
    print(f"Val embeddings: {val_embeddings.shape}")

    # --- Feature matrix ---
    feat_cols = ['has_honorifics', 'has_kinship', 'has_greetings',
                 'has_prayers', 'has_respect', 'text_length', 'char_length']
    X_train = np.concatenate([train_embeddings, train_df[feat_cols].values], axis=1)
    X_val = np.concatenate([val_embeddings, val_df[feat_cols].values], axis=1)
    y_train = train_df['label_encoded'].values
    y_val = val_df['label_encoded'].values
    print(f"\nFeature matrix: X_train={X_train.shape}, X_val={X_val.shape}")

    # --- SMOTE ---
    print("\nApplying SMOTE...")
    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
    unique, counts = np.unique(y_train_bal, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"  Class {u} ({le.classes_[u]}): {c}")

    # --- Train models ---
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'XGBoost': XGBClassifier(n_estimators=100, random_state=42, n_jobs=-1, eval_metric='mlogloss'),
        'SVM': SVC(kernel='rbf', probability=True, random_state=42),
    }

    results = {}
    print("\n" + "=" * 60)
    print("Training models...")
    print("=" * 60)

    for name, clf in models.items():
        print(f"\nTraining {name}...")
        clf.fit(X_train_bal, y_train_bal)
        y_pred = clf.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_val, y_pred, average='macro', zero_division=0)
        results[name] = {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1_score': f1, 'predictions': y_pred}
        print(f"  Accuracy={acc:.4f}  Precision={prec:.4f}  Recall={rec:.4f}  F1={f1:.4f}")
        print(classification_report(y_val, y_pred, target_names=le.classes_, zero_division=0))

    # --- Best model ---
    best_name = max(results, key=lambda x: results[x]['f1_score'])
    best_clf = models[best_name]
    y_pred_best = results[best_name]['predictions']

    print("\n" + "=" * 60)
    print(f"BEST MODEL: {best_name}")
    r = results[best_name]
    print(f"  Accuracy={r['accuracy']:.4f}  F1={r['f1_score']:.4f}")
    print("=" * 60)

    # --- Visualizations ---
    results_df = pd.DataFrame({
        'Model': list(results.keys()),
        'Accuracy': [r['accuracy'] for r in results.values()],
        'Precision': [r['precision'] for r in results.values()],
        'Recall': [r['recall'] for r in results.values()],
        'F1-Score': [r['f1_score'] for r in results.values()],
    })

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
    for ax, metric in zip(axes.flatten(), ['Accuracy', 'Precision', 'Recall', 'F1-Score']):
        sns.barplot(data=results_df, x='Model', y=metric, ax=ax, palette='viridis')
        ax.set_title(metric)
        ax.set_ylim(0, 1)
        for i, v in enumerate(results_df[metric]):
            ax.text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'model_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"Saved model_comparison.png")

    fig, ax = plt.subplots(figsize=(10, 8))
    cm = confusion_matrix(y_val, y_pred_best)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=le.classes_, yticklabels=le.classes_)
    ax.set_title(f'Confusion Matrix - {best_name}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    print(f"Saved confusion_matrix.png")

    # --- Save best model + artifacts ---
    with open(os.path.join(OUTPUT_DIR, 'best_model.pkl'), 'wb') as f:
        pickle.dump(best_clf, f)
    with open(os.path.join(OUTPUT_DIR, 'label_encoder.pkl'), 'wb') as f:
        pickle.dump(le, f)

    config = {
        'best_model': best_name,
        'label_mapping': {int(i): str(l) for i, l in enumerate(le.classes_)},
        'metrics': {k: float(v) for k, v in results[best_name].items() if k != 'predictions'},
    }
    with open(os.path.join(OUTPUT_DIR, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    # --- Generate predictions CSV ---
    val_df['predicted_label'] = le.inverse_transform(y_pred_best)
    submission = val_df[['Sentence', 'predicted_label']].copy()
    submission.columns = ['Sentence', 'label']
    submission.insert(0, 'id', range(1, len(submission) + 1))
    submission.to_csv(os.path.join(OUTPUT_DIR, 'OSACT7_TaskA_Predictions.csv'), index=False, encoding='utf-8-sig')
    print(f"\nSaved OSACT7_TaskA_Predictions.csv ({len(submission)} predictions)")

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED")
    print("=" * 60)
    print(f"All outputs in: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
