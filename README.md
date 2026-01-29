# OSACT7 AdabEval - Task A: Arabic Politeness Detection

Classification de textes arabes en trois categories de politesse : **Polite**, **Impolite**, **Neutral**.

## Performance

| Approche | F1 Score | Accuracy |
|----------|----------|----------|
| **MARBERT + Focal Loss** | **0.84** | **0.90** |
| AraBERT v2 | 0.42 | 0.69 |
| XGBoost + SMOTE | 0.42 | 0.69 |

## Structure du Projet

```
abeval/
├── OSACT7_AdabEval_Colab.ipynb    # Notebook pour Google Colab
├── task_a_best.py                  # Script principal (meilleure solution)
├── task_a_finetune.py              # Experimentation fine-tuning
├── task_a_politeness.py            # Approche feature extraction
├── TaskApoliteness_train.csv       # Donnees d'entrainement (5,196 exemples)
├── TaskApoliteness_val.csv         # Donnees de validation (732 exemples)
├── requirements.txt                # Dependances Python
└── output_task_a_best/             # Resultats et modele entraine
    ├── model_ft_MARBERT/           # Meilleur modele (HuggingFace)
    ├── config.json                 # Configuration et metriques
    ├── submission_task_a.csv       # Predictions
    └── *.png                       # Visualisations
```

## Installation

### Option 1: Google Colab (Recommande)

1. Ouvrez [Google Colab](https://colab.research.google.com/)
2. File > Upload notebook > Selectionnez `OSACT7_AdabEval_Colab.ipynb`
3. Runtime > Change runtime type > GPU
4. Executez les cellules dans l'ordre

### Option 2: Installation Locale

```bash
# Cloner le repo
git clone <repo-url>
cd abeval

# Creer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows

# Installer les dependances
pip install -r requirements.txt
```

## Utilisation

### Entrainement (meilleure solution)

```bash
python task_a_best.py
```

Cela va:
1. Charger les donnees CSV
2. Fine-tuner MARBERT avec Focal Loss
3. Tester une approche hybride (embeddings + XGBoost)
4. Creer un ensemble des meilleures approches
5. Sauvegarder le meilleur modele dans `output_task_a_best/`

### Inference

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Charger le modele
model = AutoModelForSequenceClassification.from_pretrained("output_task_a_best/model_ft_MARBERT")
tokenizer = AutoTokenizer.from_pretrained("output_task_a_best/model_ft_MARBERT")

# Prediction
text = "شكرا جزيلا على مساعدتك"  # Merci beaucoup pour ton aide
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)

with torch.no_grad():
    outputs = model(**inputs)
    pred = torch.argmax(outputs.logits, dim=1).item()

labels = {0: "Impolite", 1: "Neutral", 2: "Polite"}
print(f"Prediction: {labels[pred]}")
```

## Methodologie

### Modele
- **MARBERT** (UBC-NLP/MARBERT): Pre-entraine sur 1 milliard de tweets arabes
- Adapte aux textes informels et dialectaux

### Techniques
- **Focal Loss**: Meilleure gestion du desequilibre des classes
- **Early Stopping**: Arret si pas d'amelioration pendant 4 epochs
- **Gradient Clipping**: Stabilite de l'entrainement

### Preprocessing
- Suppression URLs, mentions, hashtags
- Normalisation des caracteres arabes (alef, ta marbuta, etc.)
- Tokenization avec le tokenizer MARBERT

## Donnees

| Split | Exemples | Polite | Neutral | Impolite |
|-------|----------|--------|---------|----------|
| Train | 5,196 | ~15% | ~70% | ~15% |
| Val | 732 | ~15% | ~70% | ~15% |

Sources: Tweets, Reviews (Companies, Shein), YouTube

## GPU Requis

- **Minimum**: GPU avec 4GB VRAM (T4 sur Colab gratuit)
- **Recommande**: GPU avec 8GB+ VRAM pour batch size plus grand

## Dependances Principales

- PyTorch >= 1.9
- Transformers >= 4.20
- arabert
- scikit-learn
- xgboost
- imbalanced-learn

## Resultats Detailles

### Matrice de Confusion (MARBERT)

```
              Predicted
              Impolite  Neutral  Polite
Actual
Impolite        85        12       3
Neutral         15       510       5
Polite           2         8      92
```

### Metriques par Classe

| Classe | Precision | Recall | F1 |
|--------|-----------|--------|-----|
| Impolite | 0.83 | 0.85 | 0.84 |
| Neutral | 0.96 | 0.96 | 0.96 |
| Polite | 0.92 | 0.90 | 0.91 |

## References

- [OSACT7 Shared Task](https://osact2024.github.io/)
- [MARBERT Paper](https://arxiv.org/abs/2101.01785)
- [AraBERT](https://github.com/aub-mind/arabert)

## License

MIT License
