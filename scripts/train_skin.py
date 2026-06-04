"""
Skin disease classifier — small CNN on synthetic 64×64 RGB patches.
Saves .h5 + label map.

For real performance: run notebooks/04_skin_disease_cnn.ipynb on HAM10000.
"""
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODELS_DIR, exist_ok=True)

IMG_SIZE = 64
CLASSES = ["Melanoma", "Nevus", "Basal_Cell_Carcinoma", "Benign_Keratosis", "Eczema"]
COLOR_HINTS = {
    "Melanoma":             (0.25, 0.15, 0.10),
    "Nevus":                (0.55, 0.40, 0.30),
    "Basal_Cell_Carcinoma": (0.65, 0.45, 0.45),
    "Benign_Keratosis":     (0.70, 0.55, 0.40),
    "Eczema":               (0.85, 0.55, 0.50),
}


def synth_images(n_per_class=300, seed=42):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for i, cls in enumerate(CLASSES):
        base_color = np.array(COLOR_HINTS[cls])
        for _ in range(n_per_class):
            img = np.ones((IMG_SIZE, IMG_SIZE, 3)) * base_color
            img += rng.normal(0, 0.06, img.shape)
            # add a darker irregular lesion blob
            cy, cx = rng.integers(20, 44, 2)
            r = rng.integers(8, 20)
            yy, xx = np.ogrid[:IMG_SIZE, :IMG_SIZE]
            mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2
            img[mask] *= rng.uniform(0.55, 0.85)
            X.append(np.clip(img, 0, 1))
            y.append(i)
    return np.array(X, dtype=np.float32), np.array(y)


def build_cnn():
    m = models.Sequential([
        layers.Input((IMG_SIZE, IMG_SIZE, 3)),
        layers.Conv2D(16, 3, activation="relu", padding="same"),
        layers.MaxPool2D(),
        layers.Conv2D(32, 3, activation="relu", padding="same"),
        layers.MaxPool2D(),
        layers.Conv2D(64, 3, activation="relu", padding="same"),
        layers.MaxPool2D(),
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
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
              epochs=6, batch_size=32, verbose=2)
    out = os.path.join(MODELS_DIR, "skin_cnn.h5")
    model.save(out)
    with open(os.path.join(MODELS_DIR, "skin_labels.json"), "w") as f:
        json.dump(CLASSES, f)
    print(f"[skin] saved -> {out}")


if __name__ == "__main__":
    main()
