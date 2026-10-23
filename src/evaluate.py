"""
evaluate.py
-----------
Loads the trained model and evaluates it on the test set:
    - overall accuracy
    - per-class precision / recall / F1 (classification_report)
    - confusion matrix heatmap saved as models/confusion_matrix.png

Usage:
    python evaluate.py

CodeAlpha ML Internship - Task 3: Handwritten Character Recognition
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model

from data_loader import load_combined_dataset
from labels import LABEL_MAP

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "handwritten_char_cnn.keras")
CM_PATH = os.path.join(MODEL_DIR, "confusion_matrix.png")


def main():
    print("Loading test data...")
    (_, _), (x_test, y_test) = load_combined_dataset()

    print(f"Loading model from {MODEL_PATH} ...")
    model = load_model(MODEL_PATH)

    print("Running predictions...")
    y_pred_probs = model.predict(x_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    class_names = [LABEL_MAP[i] for i in range(len(LABEL_MAP))]

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))

    acc = (y_pred == y_test).mean()
    print(f"Overall accuracy: {acc:.4f}")

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(16, 14))
    sns.heatmap(cm, annot=False, cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix - Handwritten Character Recognition")
    plt.tight_layout()
    plt.savefig(CM_PATH, dpi=150)
    print(f"\nConfusion matrix saved to: {CM_PATH}")


if __name__ == "__main__":
    main()
