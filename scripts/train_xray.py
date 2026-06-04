"""
X-ray classifier — small CNN trained on synthetic 64×64 noise patches.
Saves a .h5 so the backend has a real Keras model to load.

For real performance: run notebooks/03_xray_cnn.ipynb with the Kaggle
chest-xray-pneumonia dataset; it overrides this artifact.
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODELS_DIR, exist_ok=True)

IMG_SIZE = 64
CLASSES = ["NORMAL", "PNEUMONIA"]


def synth_images(n_per_class=400, seed=42):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for i, cls in enumerate(CLASSES):
        for _ in range(n_per_class):
            base = rng.normal(0.5, 0.05, (IMG_SIZE, IMG_SIZE, 1))
            # PNEUMONIA: add bright cloudy patches
            if cls == "PNEUMONIA":
                for _ in range(rng.integers(2, 6)):
                    cy, cx = rng.integers(8, IMG_SIZE - 8, 2)
                    r = rng.integers(4, 10)
                    yy, xx = np.ogrid[:IMG_SIZE, :IMG_SIZE]
                    mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2
                    base[mask] += 0.3
            X.append(np.clip(base, 0, 1))
            y.append(i)
    return np.array(X, dtype=np.float32), np.array(y)


def build_cnn():
    m = models.Sequential([
        layers.Input((IMG_SIZE, IMG_SIZE, 1)),
        layers.Conv2D(16, 3, activation="relu", padding="same"),
        layers.MaxPool2D(),
        layers.Conv2D(32, 3, activation="relu", padding="same"),
        layers.MaxPool2D(),
        layers.Conv2D(64, 3, activation="relu", padding="same"),
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(32, activation="relu"),
        layers.Dense(len(CLASSES), activation="softmax"),
    ])
    m.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return m


def main():
    X, y = synth_images()
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]
    split = int(0.8 * len(X))
    model = build_cnn()
    model.fit(X[:split], y[:split], validation_data=(X[split:], y[split:]),
              epochs=5, batch_size=32, verbose=2)
    out = os.path.join(MODELS_DIR, "xray_cnn.h5")
    model.save(out)
    print(f"[xray] saved -> {out}")


if __name__ == "__main__":
    main()
