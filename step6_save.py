# step6_save.py
# Assignment 4 — Persist the Final Model + Vectorizer
# -------------------------------------------------------
# IMPORTANT: always save BOTH objects together.
# The vectorizer holds the vocabulary learned from training.
# If you save only the model, you can't reproduce the same
# feature space at prediction time.
# -------------------------------------------------------

import joblib

# Re-load so we can save the "official" final versions
model      = joblib.load("spam_model_tmp.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Save with production filenames
joblib.dump(model,      "spam_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")   # already there; re-dump for clarity

print("[✓] spam_model.pkl  — the trained Naive Bayes classifier")
print("[✓] vectorizer.pkl  — the fitted TF-IDF vectorizer")
print("\nBoth files are ready to be used by step7_email_bot.py")
