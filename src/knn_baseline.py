import json
import os

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from utils import load_data

X, y, class_names = load_data('data')

# Same split as train_model.py (test_size=0.3 then 0.5, random_state=42,
# stratified) so KNN and the ANN are compared on the identical held-out
# test set.
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

# Pick k on the validation set only, so the test set stays untouched until
# the final evaluation below (same leakage-avoidance rationale as the ANN).
print("Selecting k on validation set:")
best_k, best_val_acc = None, -1.0
for k in [1, 3, 5, 7, 9, 11, 15]:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    val_acc = accuracy_score(y_val, knn.predict(X_val))
    print(f"  k={k:>2}  val_acc={val_acc:.4f}")
    if val_acc > best_val_acc:
        best_k, best_val_acc = k, val_acc
print(f"Best k: {best_k} (val_acc={best_val_acc:.4f})")

# Retrain on train+val with the chosen k, evaluate once on the held-out test set.
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(np.vstack([X_train, X_val]), np.concatenate([y_train, y_val]))
y_pred = knn.predict(X_test)

print("\nClassification Report (held-out test set):")
print(classification_report(y_test, y_pred, target_names=class_names))

print("Confusion Matrix (rows = actual, columns = predicted):")
cm = confusion_matrix(y_test, y_pred)
print("        " + " ".join(f"{name:>5}" for name in class_names))
for name, row in zip(class_names, cm):
    print(f"{name:>8} " + " ".join(f"{v:>5}" for v in row))

knn_acc = accuracy_score(y_test, y_pred)
print(f"\nKNN test accuracy (k={best_k}): {knn_acc:.4f}")

if os.path.exists('model_accuracy.json'):
    with open('model_accuracy.json') as f:
        ann_acc = json.load(f).get('accuracy')
    print(f"ANN test accuracy (from train_model.py): {ann_acc:.4f}")
    print(f"Difference (ANN - KNN): {ann_acc - knn_acc:+.4f}")
