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
print("TP WORD2VEC - DATASET FILMS")
print("="*60 + "\n")

# ===========================
# 1. CHARGEMENT
# ===========================
print("1. Chargement des données...")
df = pd.read_csv(os.path.join(DATA_DIR, 'movies_metadata.csv'), low_memory=False)

print(f"   Dataset brut : {len(df)} films")
print(f"   Colonnes : {len(df.columns)}")
print(f"\n   Colonnes disponibles :")
print(f"   {df.columns.tolist()}\n")

# Analyse variable cible
print("   Variable cible : vote_average")
df['vote_average'] = pd.to_numeric(df['vote_average'], errors='coerce')
print(f"   Min : {df['vote_average'].min()}")
print(f"   Max : {df['vote_average'].max()}")
print(f"   Moyenne : {df['vote_average'].mean():.2f}\n")

# Analyse colonne texte
print("   Colonne texte : overview (descriptions)")
print(f"   Descriptions manquantes : {df['overview'].isnull().sum()} ({df['overview'].isnull().sum()/len(df)*100:.1f}%)")

df = df[df['overview'].notna()].copy()
print(f"   Après nettoyage : {len(df)} films\n")

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

df['tokens'] = df['overview'].apply(preprocess)
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
test_words = ['love', 'action', 'hero', 'world', 'family']
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
if all(w in model.wv for w in ['man', 'woman', 'king']):
    result = model.wv.most_similar(
        positive=['woman', 'king'],
        negative=['man'],
        topn=3
    )
    print("   king - man + woman ≈")
    for word, score in result:
        print(f"      {word:15s} : {score:.3f}")
    print()

# Exemples de vecteurs
print("   === EXEMPLES DE VECTEURS (10 premières dimensions) ===\n")
example_words = ['love', 'hero', 'adventure']
for word in example_words:
    if word in model.wv:
        vector = model.wv[word]
        print(f"   '{word}' →")
        print(f"      {vector[:10]}")
        print(f"      ... (100 dimensions au total)\n")

# ===========================
# 5. SAUVEGARDE
# ===========================
model.save(os.path.join(DATA_DIR, 'word2vec_films.bin'))
print("Modèle sauvegardé : word2vec_films.bin\n")