import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
import warnings
warnings.filterwarnings('ignore')

# Télécharger ressources NLTK
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)

# Chemins
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')
RESULTS_DIR = os.path.join(BASE_DIR, '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

print("="*60)
print("TP WORD2VEC - DATASET REVIEWS AMAZON")
print("="*60 + "\n")

# ===========================
# 1. CHARGEMENT
# ===========================
print("1. Chargement des données...")
df = pd.read_json(os.path.join(DATA_DIR, 'Cell_Phones_and_Accessories_5.json'), lines=True)

print(f"   Dataset : {len(df)} reviews")
print(f"   Colonnes : {len(df.columns)}")
print(f"\n   Colonnes disponibles :")
print(f"   {df.columns.tolist()}\n")

# Analyse variable cible
print("   Variable cible : overall (note 1-5)")
print(f"   Distribution :")
for note in sorted(df['overall'].unique()):
    count = (df['overall'] == note).sum()
    pct = count / len(df) * 100
    print(f"      {int(note)} étoiles : {count:6d} ({pct:5.1f}%)")
print()

# Analyse colonne texte
print("   Colonne texte : reviewText (avis clients)")
print(f"   Valeurs manquantes : {df['reviewText'].isnull().sum()}")
df['text_length'] = df['reviewText'].apply(len)
print(f"   Longueur moyenne : {df['text_length'].mean():.0f} caractères")
print(f"   Longueur médiane : {df['text_length'].median():.0f} caractères\n")

# ===========================
# 2. PREPROCESSING
# ===========================
print("2. Preprocessing du texte...")

stop_words = set(stopwords.words('english'))

def preprocess(text):
    if pd.isna(text):
        return []
    text = text.lower()
    text = re.sub(r"[^a-zA-Z ]", "", text)
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words and len(w) > 2]
    return tokens

df['tokens'] = df['reviewText'].apply(preprocess)
print("   Preprocessing terminé\n")

# Vocabulaire
all_tokens = [token for tokens_list in df['tokens'] for token in tokens_list]
token_counts = Counter(all_tokens)
print(f"   Vocabulaire : {len(token_counts):,} mots uniques")
print(f"   Tokens totaux : {len(all_tokens):,}\n")

# ===========================
# 3. WORD2VEC
# ===========================
print("3. Entraînement Word2Vec...")

sentences = df['tokens'].tolist()

model = Word2Vec(
    sentences=sentences,
    vector_size=100,
    window=5,
    min_count=5,
    workers=4,
    sg=1,
    epochs=10
)

print(f"   Modèle entraîné")
print(f"   Vocabulaire (min_count=5) : {len(model.wv):,} mots\n")

# ===========================
# 4. ANALYSE DES EMBEDDINGS
# ===========================
print("4. Analyse des embeddings\n")

# Mots similaires
test_words = ['phone', 'battery', 'screen', 'quality', 'case', 'price']
print("   === MOTS SIMILAIRES ===\n")
for word in test_words:
    if word in model.wv:
        similar = model.wv.most_similar(word, topn=5)
        print(f"   '{word}' est proche de :")
        for sim_word, score in similar:
            print(f"      {sim_word:15s} : {score:.3f}")
        print()

# Relations vectorielles
print("   === RELATIONS VECTORIELLES ===\n")
if all(w in model.wv for w in ['good', 'bad', 'quality']):
    result = model.wv.most_similar(
        positive=['bad', 'quality'],
        negative=['good'],
        topn=3
    )
    print("   quality - good + bad ≈")
    for word, score in result:
        print(f"      {word:15s} : {score:.3f}")
    print()

# Exemples de vecteurs
print("   === EXEMPLES DE VECTEURS (10 premières dimensions) ===\n")
example_words = ['phone', 'battery', 'quality']
for word in example_words:
    if word in model.wv:
        vector = model.wv[word]
        print(f"   '{word}' →")
        print(f"      {vector[:10]}")
        print(f"      ... (100 dimensions au total)\n")

# ===========================
# 5. SAUVEGARDE
# ===========================
model.save(os.path.join(DATA_DIR, 'word2vec_phones.bin'))
print("Modèle sauvegardé : word2vec_phones.bin\n")