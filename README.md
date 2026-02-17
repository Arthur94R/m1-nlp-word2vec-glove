# 🎬 NLP — Analyse de films et embeddings Word2Vec

Projet universitaire — Master 1 IA & Big Data, Université Paris 8

## 📋 Description

Analyse de descriptions de films à l'aide de techniques NLP : preprocessing de texte,
création de vocabulaire et embeddings **Word2Vec** pour améliorer la prédiction
de variables cibles comme la note moyenne, la popularité et les revenus.

## 🔍 Résultats clés

- Preprocessing complet : tokenisation, suppression des stop words, filtrage
- Vocabulaire analysé avec distribution des fréquences de tokens
- Comparaison modèle baseline (features numériques) vs modèle enrichi (embeddings)
- Les embeddings Word2Vec améliorent la capacité prédictive du modèle

## 🛠️ Stack technique

- **Python** — Pipeline NLP complet
- **Gensim** — Entraînement des embeddings Word2Vec
- **NLTK** — Tokenisation et stop words
- **Scikit-learn** — Modèles de prédiction
- **Pandas / NumPy** — Traitement des données
- **Matplotlib** — Visualisations

## 📁 Structure
```
data/           → Dataset films (à télécharger, voir ci-dessous)
src/
└── main.py     → Pipeline complet NLP
results/        → Graphiques générés
```

## 📥 Récupérer les données

Les fichiers ne sont pas inclus dans ce repo car trop volumineux.

1. `movies_metadata.csv` → [Kaggle - The Movies Dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset)
2. `Cell_Phones_and_Accessories_5.json` → [Amazon Reviews Dataset](https://nijianmo.github.io/amazon/index.html)
3. `movies_preprocessed.csv` et `vocabulary.csv` → générés automatiquement par `main.py`
4. `word2vec_model.bin` → généré automatiquement par `main.py`

Place les fichiers téléchargés dans le dossier `data/`.

## 🚀 Lancer le projet
```bash
# Installer les dépendances
pip install pandas numpy matplotlib scikit-learn gensim nltk

# Télécharger les ressources NLTK
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"

# Lancer le pipeline
python src/main.py
```

## 📊 Visualisations générées

- `distribution_frequences.png` — Fréquence des tokens
- `distribution_tokens.png` — Distribution des tokens par description
- `distributions_variables_cibles.png` — Distribution des variables cibles
- `top_tokens.png` — Top tokens les plus fréquents
- `feature_importance_baseline.png` — Importance des features (baseline)
- `predictions_baseline.png` — Prédictions vs valeurs réelles
- `comparison_baseline_embeddings.png` — Comparaison baseline vs Word2Vec