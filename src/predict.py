"""
predict.py
----------
Predicts the digit/letter in a single handwritten character image.

Usage:
    python predict.py --image path/to/character.png

Notes:
    - Works best with a single character on a plain background.
    - Handles both "dark text on light background" and
      "light text on dark background" images automatically.

CodeAlpha ML Internship - Task 3: Handwritten Character Recognition
"""

import argparse
import os

import cv2
import numpy as np
from tensorflow.keras.models import load_model

from labels import LABEL_MAP

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "handwritten_char_cnn.keras")


def crop_and_center(gray, margin_frac=0.18, threshold=20):
    """Crop to the character's bounding box and pad back into a square
    with a margin, matching the MNIST/EMNIST convention of a centered,
    non-edge-to-edge character. Kept identical to the logic in app.py so
    CLI predictions and the web app behave consistently.
    """
    ys, xs = np.where(gray > threshold)
    if len(ys) == 0 or len(xs) == 0:
        return gray

    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    cropped = gray[y0:y1 + 1, x0:x1 + 1]

    h, w = cropped.shape
    side = max(h, w)
    pad = max(1, int(side * margin_frac))
    side += pad * 2

    square = np.zeros((side, side), dtype=gray.dtype)
    y_off = (side - h) // 2
    x_off = (side - w) // 2
    square[y_off:y_off + h, x_off:x_off + w] = cropped
    return square


def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    # MNIST/EMNIST convention: white character on black background.
    # If the image looks like dark text on a light background, invert it.
    if img.mean() > 127:
        img = 255 - img

    img = crop_and_center(img)
    img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    img = img[np.newaxis, ..., np.newaxis]  # shape (1, 28, 28, 1)
    return img


def parse_args():
    parser = argparse.ArgumentParser(description="Predict a handwritten character")
    parser.add_argument("--image", type=str, required=True, help="Path to the character image")
    parser.add_argument("--top-k", type=int, default=3, help="Show top-k predictions")
    return parser.parse_args()


def main():
    args = parse_args()

    model = load_model(MODEL_PATH)
    img = preprocess_image(args.image)

    probs = model.predict(img, verbose=0)[0]
    top_indices = np.argsort(probs)[::-1][: args.top_k]

    print(f"\nPredictions for '{args.image}':")
    for rank, idx in enumerate(top_indices, start=1):
        print(f"  {rank}. {LABEL_MAP[idx]}  (confidence: {probs[idx] * 100:.2f}%)")


if __name__ == "__main__":
    main()
