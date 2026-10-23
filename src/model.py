"""
model.py
--------
CNN architecture for handwritten character recognition (digits + letters).

CodeAlpha ML Internship - Task 3: Handwritten Character Recognition
"""

from tensorflow.keras import layers, models


def build_cnn(input_shape=(28, 28, 1), num_classes=36):
    """
    Builds a CNN for 28x28 grayscale character classification.

    Architecture:
        Conv(32) -> BN -> Conv(32) -> BN -> MaxPool -> Dropout
        Conv(64) -> BN -> Conv(64) -> BN -> MaxPool -> Dropout
        Flatten -> Dense(256) -> BN -> Dropout -> Dense(num_classes, softmax)
    """
    model = models.Sequential(name="HandwrittenCharCNN")

    model.add(layers.Input(shape=input_shape))

    # Block 1
    model.add(layers.Conv2D(32, (3, 3), padding="same", activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(32, (3, 3), padding="same", activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))

    # Block 2
    model.add(layers.Conv2D(64, (3, 3), padding="same", activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(64, (3, 3), padding="same", activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))

    # Classifier head
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation="softmax"))

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


if __name__ == "__main__":
    model = build_cnn()
    model.summary()
