# step2_clean.py
# Assignment 4 — Text Cleaning / Preprocessing
# -------------------------------------------------------
# Pipeline:
#   1. Lowercase everything
#   2. Remove URLs and email addresses
#   3. Remove numbers
#   4. Remove punctuation
#   5. Tokenise (split into words)
#   6. Drop stopwords  (e.g. "the", "is", "at")
#   7. Stem every token  (e.g. "running" → "run")
# -------------------------------------------------------

import re
import string
import pandas as pd
import nltk

# Download NLTK data the first time (safe to re-run)
nltk.download("stopwords", quiet=True)
nltk.download("punkt",     quiet=True)

from nltk.corpus import stopwords
from nltk.stem   import PorterStemmer

STOP    = set(stopwords.words("english"))
stemmer = PorterStemmer()


def clean_text(text: str) -> str:
    text = str(text).lower()                                         # 1. lowercase
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)             # 2. remove URLs
    text = re.sub(r"\S+@\S+", "", text)                              # 2. remove emails
    text = re.sub(r"\d+", "", text)                                  # 3. remove numbers
    text = text.translate(str.maketrans("", "", string.punctuation)) # 4. punctuation
    tokens = text.split()
    tokens = [stemmer.stem(w) for w in tokens
              if w not in STOP and len(w) > 2]                       # 5-7. filter+stem
    return " ".join(tokens)


# ── Load → clean → save ──────────────────────────────────
df = pd.read_csv("data_loaded.csv")

df["clean"] = df["text"].apply(clean_text)

print("Before cleaning:")
print(df["text"].iloc[2])
print("\nAfter cleaning:")
print(df["clean"].iloc[2])

print("\nFirst 5 rows (text vs clean):")
print(df[["label", "text", "clean"]].head(5).to_string())

df.to_csv("data_clean.csv", index=False)
print("\n[✓] Saved → data_clean.csv")
