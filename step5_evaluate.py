# step5_evaluate.py
# Assignment 4 — Model Evaluation
# -------------------------------------------------------
# Metrics explained:
#
#   Accuracy  = (TP + TN) / total            — overall correct %
#   Precision = TP / (TP + FP)               — of predicted spam, how many really are?
#   Recall    = TP / (TP + FN)               — of real spam, how many did we catch?
#   F1        = 2 × (P × R) / (P + R)       — harmonic mean; best single metric
#
# Confusion matrix:
#              Predicted Ham   Predicted Spam
#   Actual Ham      TN               FP
#   Actual Spam     FN               TP
# -------------------------------------------------------

import joblib
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# ── Load ─────────────────────────────────────────────────
model = joblib.load("spam_model_tmp.pkl")
X_train, X_test, y_train, y_test = joblib.load("split.pkl")

# ── Predict ───────────────────────────────────────────────
y_pred = model.predict(X_test)

# ── Print results ─────────────────────────────────────────
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy : {acc:.4f}  ({acc*100:.2f}%)\n")

print("Confusion Matrix:")
print("              Predicted Ham   Predicted Spam")
cm = confusion_matrix(y_test, y_pred, labels=["ham", "spam"])
print(f"  Actual Ham      {cm[0][0]:<14}  {cm[0][1]}")
print(f"  Actual Spam     {cm[1][0]:<14}  {cm[1][1]}\n")

print("Full Classification Report:")
print(classification_report(y_test, y_pred))

# ── What to say in your writeup ──────────────────────────
print("─" * 50)
print("Key insight: precision on the SPAM class tells you")
print("how often a 'spam' prediction is actually spam.")
print("High precision = fewer false alarms (ham → spam).")
print("High recall    = fewer missed spams (spam → ham).")
