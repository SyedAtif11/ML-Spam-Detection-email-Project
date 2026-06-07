# step4_train.py
# Assignment 4 — Train the Naive Bayes Classifier
# -------------------------------------------------------
# Model: Multinomial Naive Bayes
#   — The textbook choice for bag-of-words / TF-IDF features
#   — Bayes' theorem: P(spam | words) ∝ P(words | spam) × P(spam)
#   — Very fast; works well even with a small dataset
#
# Split: 80 % train / 20 % test  (stratified to keep class ratio)
# -------------------------------------------------------

import joblib
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes     import MultinomialNB

# ── Load X, y ────────────────────────────────────────────
X, y = joblib.load("Xy.pkl")

# ── Split ────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y          # keeps ham:spam ratio identical in both halves
)

print(f"Training samples : {X_train.shape[0]}")
print(f"Test samples     : {X_test.shape[0]}")

# ── Train ─────────────────────────────────────────────────
model = MultinomialNB()
model.fit(X_train, y_train)

print("\n[✓] Model trained!")

# ── Persist split for evaluation ─────────────────────────
joblib.dump(model,                        "spam_model_tmp.pkl")
joblib.dump((X_train, X_test, y_train, y_test), "split.pkl")
print("[✓] Saved → spam_model_tmp.pkl  and  split.pkl")
