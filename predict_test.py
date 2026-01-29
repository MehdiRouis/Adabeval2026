"""
Prediction sur le fichier test avec le modele MARBERT entraine.
"""

import os
import re
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_CSV = os.path.join(BASE_DIR, "TaskApoliteness_test.csv")
MODEL_PATH = os.path.join(BASE_DIR, "output_task_a_best", "model_ft_MARBERT")
OUTPUT_CSV = os.path.join(BASE_DIR, "predictions_test.csv")

MAX_LENGTH = 128
BATCH_SIZE = 32
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Label mapping (from training)
LABEL_MAP = {0: "Impolite", 1: "Neutral", 2: "Polite"}


def preprocess_text(text):
    """Preprocess Arabic text."""
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'[إأآا]', 'ا', text)
    text = re.sub(r'ة', 'ه', text)
    text = re.sub(r'ى', 'ي', text)
    text = ' '.join(text.split())
    return text


def predict_batch(texts, model, tokenizer, device):
    """Predict labels for a batch of texts."""
    processed = [preprocess_text(t) for t in texts]

    encodings = tokenizer(
        processed,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors='pt'
    )

    with torch.no_grad():
        outputs = model(
            input_ids=encodings['input_ids'].to(device),
            attention_mask=encodings['attention_mask'].to(device)
        )
        probs = torch.softmax(outputs.logits, dim=1)
        preds = torch.argmax(probs, dim=1).cpu().numpy()
        confidences = probs.max(dim=1).values.cpu().numpy()

    return preds, confidences


def main():
    print("=" * 60)
    print("PREDICTION SUR LE FICHIER TEST")
    print("=" * 60)

    print(f"\nDevice: {DEVICE}")
    if DEVICE.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load model
    print(f"\nChargement du modele: {MODEL_PATH}")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model.eval()
    print("Modele charge!")

    # Load test data
    print(f"\nChargement des donnees: {TEST_CSV}")
    test_df = pd.read_csv(TEST_CSV)
    print(f"Nombre d'exemples: {len(test_df)}")

    # Predict in batches
    print("\nPrediction en cours...")
    all_preds = []
    all_confidences = []

    texts = test_df['Sentence'].tolist()

    for i in tqdm(range(0, len(texts), BATCH_SIZE)):
        batch_texts = texts[i:i + BATCH_SIZE]
        preds, confs = predict_batch(batch_texts, model, tokenizer, DEVICE)
        all_preds.extend(preds)
        all_confidences.extend(confs)

    # Create output dataframe
    test_df['Predicted_Label'] = [LABEL_MAP[p] for p in all_preds]
    test_df['Confidence'] = [f"{c:.2%}" for c in all_confidences]

    # Save predictions
    test_df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    print(f"\nPredictions sauvegardees: {OUTPUT_CSV}")

    # Distribution
    print("\nDistribution des predictions:")
    print(test_df['Predicted_Label'].value_counts().to_string())

    # Show examples
    print("\n" + "=" * 60)
    print("EXEMPLES DE PREDICTIONS")
    print("=" * 60)

    for label in ["Polite", "Impolite", "Neutral"]:
        subset = test_df[test_df['Predicted_Label'] == label].head(3)
        print(f"\n--- {label.upper()} ---")
        for _, row in subset.iterrows():
            text = row['Sentence'][:60] + "..." if len(row['Sentence']) > 60 else row['Sentence']
            print(f"  [{row['Confidence']}] {text}")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == '__main__':
    main()
