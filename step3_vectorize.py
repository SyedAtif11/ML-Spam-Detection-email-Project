# step3_vectorize.py
# Assignment 4 — TF-IDF Vectorization
# -------------------------------------------------------
# TF-IDF (Term Frequency × Inverse Document Frequency)
# converts each cleaned message into a numeric vector.
#
#   TF  = how often a word appears in THIS message
#   IDF = log(N / df)  — rewards rare, distinctive words
#
# We keep the top 5 000 uni-grams + bi-grams.
# -------------------------------------------------------

import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# ── Load cleaned data ────────────────────────────────────
df = pd.read_csv("data_clean.csv")
df.dropna(subset=["clean"], inplace=True)   # drop any rows where cleaning produced NaN
df = df[df["clean"].str.strip() != ""]      # also drop rows that became empty strings
df.reset_index(drop=True, inplace=True)

# ── Build the vectorizer ─────────────────────────────────
vectorizer = TfidfVectorizer(
    max_features=5000,    # keep the 5 000 most informative tokens
    ngram_range=(1, 2),   # unigrams ("free") AND bigrams ("free win")
    min_df=2              # ignore tokens seen in fewer than 2 messages
)

X = vectorizer.fit_transform(df["clean"])   # sparse matrix (n_docs × vocab)
y = df["label"]

print("Matrix shape:", X.shape)
print("Example features:", vectorizer.get_feature_names_out()[:15])
print("\nLabel counts:")
print(y.value_counts())

# ── Save artefacts for later steps ──────────────────────
joblib.dump(vectorizer, "vectorizer.pkl")
joblib.dump((X, y),     "Xy.pkl")
print("\n[✓] Saved → vectorizer.pkl  and  Xy.pkl")
