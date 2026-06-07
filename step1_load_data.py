# step1_load_data.py
# Assignment 4 — Load and inspect the spam dataset
# -------------------------------------------------------
# The CSV has 5 columns; only the first two matter:
#   v1 = label  ("ham" or "spam")
#   v2 = message text
# -------------------------------------------------------

import pandas as pd

CSV_PATH = r"C:\Users\HP\Desktop\spam.csv"   # <-- your actual file location

# Load with latin-1 encoding (original UCI file uses it)
df = pd.read_csv(CSV_PATH, encoding="latin-1")

# Keep only the two useful columns and rename them
df = df[["v1", "v2"]].copy()
df.columns = ["label", "text"]

# Quick sanity checks
print("Shape:", df.shape)
print("\nClass distribution:")
print(df["label"].value_counts())
print("\nSample rows:")
print(df.head(5).to_string())

# Save cleaned frame for the next step
df.to_csv("data_loaded.csv", index=False)
print("\n[✓] Saved → data_loaded.csv")
