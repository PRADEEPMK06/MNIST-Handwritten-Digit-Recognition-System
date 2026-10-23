"""
train.py
--------
Trains the CNN on the combined MNIST + EMNIST dataset and saves the best
model to models/handwritten_char_cnn.keras, plus training curve plots.

Usage:
    python -u train.py --epochs 20 --batch-size 128

CodeAlpha ML Internship - Task 3: Handwritten Character Recognition
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from data_loader import load_combined_dataset
from labels import NUM_CLASSES
from model import build_cnn

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "handwritten_char_cnn.keras")
HISTORY_PLOT_PATH = os.path.join(MODEL_DIR, "training_history.png")


def parse_args():
    parser = argparse.ArgumentParser(description="Train handwritten character CNN")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--val-split", type=float, default=0.1)
    return parser.parse_args()


def plot_history(history):
    print("\n[PROGRESS] Plotting training history...", flush=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["accuracy"], label="train")
    axes[0].plot(history.history["val_accuracy"], label="validation")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="train")
    axes[1].plot(history.history["val_loss"], label="validation")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(HISTORY_PLOT_PATH, dpi=150)
    print(f"[PROGRESS] Training curves saved to: {HISTORY_PLOT_PATH}", flush=True)


def main():
    args = parse_args()

    print("=" * 60, flush=True)
    print("HANDWRITTEN CHARACTER CNN TRAINING STARTED", flush=True)
    print("=" * 60, flush=True)
    print(f"[INFO] Epochs: {args.epochs}", flush=True)
    print(f"[INFO] Batch size: {args.batch_size}", flush=True)

    os.makedirs(MODEL_DIR, exist_ok=True)

    print("\n[PROGRESS 1/5] Loading MNIST + EMNIST dataset...", flush=True)
    (x_train, y_train), (x_test, y_test) = load_combined_dataset()

    print("[DONE] Dataset loaded successfully!", flush=True)
    print(f"[INFO] Train samples: {len(x_train)}", flush=True)
    print(f"[INFO] Test samples: {len(x_test)}", flush=True)

    print("\n[PROGRESS 2/5] Building CNN model...", flush=True)
    model = build_cnn(input_shape=x_train.shape[1:], num_classes=NUM_CLASSES)
    model.summary()
    print("[DONE] CNN model built successfully!", flush=True)

    # Stratified split guarantees every class appears in both the training
    # and validation subsets.
    print("\n[PROGRESS 3/5] Splitting training and validation data...", flush=True)

    x_tr, x_val, y_tr, y_val = train_test_split(
        x_train, y_train, test_size=args.val_split, stratify=y_train, random_state=42
    )

    print(f"[INFO] Training subset: {len(x_tr)} samples", flush=True)
    print(f"[INFO] Validation subset: {len(x_val)} samples", flush=True)

    # Light augmentation on the training set only.
    print("\n[PROGRESS 4/5] Preparing image augmentation...", flush=True)

    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
    )
    datagen.fit(x_tr)
    train_gen = datagen.flow(x_tr, y_tr, batch_size=args.batch_size)

    print("[DONE] Image augmentation prepared!", flush=True)
    print("[PROGRESS] Calculating class weights...", flush=True)

    class_labels = np.unique(y_tr)
    weights = compute_class_weight("balanced", classes=class_labels, y=y_tr)
    class_weight = dict(zip(class_labels.tolist(), weights.tolist()))

    print("[DONE] Class weights calculated!", flush=True)

    callbacks = [
        ModelCheckpoint(MODEL_PATH, monitor="val_accuracy", save_best_only=True, verbose=1),
        EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6),
    ]

    print("\n" + "=" * 60, flush=True)
    print("[PROGRESS 5/5] CNN TRAINING STARTING...", flush=True)
    print("=" * 60, flush=True)

    history = model.fit(
        train_gen,
        validation_data=(x_val, y_val),
        epochs=args.epochs,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1,
    )

    print("\n[DONE] CNN training finished!", flush=True)

    plot_history(history)

    print("\n[PROGRESS] Evaluating model on held-out test data...", flush=True)

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)

    print("\n" + "=" * 60, flush=True)
    print("FINAL TRAINING RESULTS", flush=True)
    print("=" * 60, flush=True)
    print(f"Test accuracy: {test_acc:.4f}", flush=True)
    print(f"Test loss: {test_loss:.4f}", flush=True)
    print(f"Best model saved to: {MODEL_PATH}", flush=True)
    print(f"Training curves saved to: {HISTORY_PLOT_PATH}", flush=True)
    print("[SUCCESS] Training process completed!", flush=True)


if __name__ == "__main__":
    main()kkkkkkkkkkkkkk