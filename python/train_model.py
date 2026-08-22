"""
train_model.py
----------------
Trains on data/combined_dataset.csv, which is already normalized the same
way model_deploy.ino will normalize live sensor readings - so there's no
separate mean/std header that could ever drift out of sync with the device.

Validates on a held-out SESSION (not a random row split) so the reported
accuracy is a real estimate of live performance, not an inflated number
from near-duplicate frames leaking between train and test.

Produces, all in ../models/:
  - asl_model.keras
  - asl_model_int8.tflite
  - model_data.h      <- copy this straight into esp32/model_deploy/
  - labels.txt

Usage:
    python train_model.py
"""

import os

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

FEATURE_NAMES = [
    "flex_pinky", "flex_ring", "flex_middle", "flex_index", "flex_thumb",
    "touch_index", "touch_middle", "touch_ring", "touch_pinky", "touch_r", "touch_u",
    "ax", "ay", "az", "gx", "gy", "gz",
]

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "combined_dataset.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

print("\n======================================")
print(" ASL TinyML MODEL TRAINING")
print("======================================\n")

# ============================================================
# 1. LOAD
# ============================================================

if not os.path.exists(DATA_PATH):
    raise SystemExit(f"{DATA_PATH} not found - run merge_data.py first.")

df = pd.read_csv(DATA_PATH)
print("Loaded", len(df), "rows,", df["session"].nunique(), "session(s).")

# Sanity check: flag any feature with ~zero variance before it can cause
# the same silent train/deploy mismatch as before.
for col in FEATURE_NAMES:
    if df[col].std() < 1e-6:
        print(f"WARNING: '{col}' has near-zero variance (constant={df[col].iloc[0]}). "
              f"It carries no signal for the model and, if its live value ever differs "
              f"from this constant, it can distort every prediction. Check this sensor "
              f"physically before trusting the model.")

X = df[FEATURE_NAMES].values.astype(np.float32)
y_text = df["label"].values
sessions = df["session"].values

# ============================================================
# 2. ENCODE LABELS
# ============================================================

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_text)
class_names = label_encoder.classes_

with open(os.path.join(MODEL_DIR, "labels.txt"), "w") as f:
    for label in class_names:
        f.write(str(label) + "\n")

# ============================================================
# 3. SESSION-GROUPED SPLIT
# ============================================================

unique_sessions = sorted(set(sessions))
print(f"\n{len(unique_sessions)} unique session(s) found.")

if len(unique_sessions) >= 2:
    n_test = max(1, round(len(unique_sessions) * 0.2))
    test_sessions = set(unique_sessions[-n_test:])  # most recent session(s) held out
    train_mask = ~pd.Series(sessions).isin(test_sessions).values
    print(f"Held-out test session(s): {test_sessions}")
else:
    print("WARNING: only one session - falling back to a random split. "
          "Collect a second session before trusting these numbers.")
    from sklearn.model_selection import train_test_split
    idx_train, idx_test = train_test_split(
        np.arange(len(y)), test_size=0.2, stratify=y, random_state=42
    )
    train_mask = np.zeros(len(y), dtype=bool)
    train_mask[idx_train] = True

X_train, X_test = X[train_mask], X[~train_mask]
y_train, y_test = y[train_mask], y[~train_mask]
print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

# ============================================================
# 4. BUILD + TRAIN
# ============================================================
# No extra scaling here - X is already normalized to sensible ranges by
# collect_data.py (flex/touch in [0,1], accel/gyro in roughly [-1,1]).

print("\nBuilding neural network...")

num_features = X_train.shape[1]
num_classes = len(class_names)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(num_features,)),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(num_classes, activation="softmax"),
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()

print("\nTraining...")

model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    verbose=1,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)
    ],
)

# ============================================================
# 5. EVALUATE ON HELD-OUT SESSION
# ============================================================

test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print("\n======================================")
print("HELD-OUT SESSION ACCURACY:", round(test_accuracy * 100, 2), "%")
print("======================================")

predictions = model.predict(X_test, verbose=0)
predicted_classes = np.argmax(predictions, axis=1)

print("\nClassification report (held-out session):\n")
print(classification_report(
    y_test, predicted_classes,
    labels=np.unique(y_test),
    target_names=class_names[np.unique(y_test)],
))
print("Confusion matrix:\n")
print(confusion_matrix(y_test, predicted_classes))

# ============================================================
# 6. SAVE KERAS MODEL
# ============================================================

model.save(os.path.join(MODEL_DIR, "asl_model.keras"))

# ============================================================
# 7. CONVERT TO INT8 TFLITE
# ============================================================

def representative_dataset():
    for i in range(min(300, len(X_train))):
        yield [np.expand_dims(X_train[i], axis=0).astype(np.float32)]

print("\nConverting to INT8 TFLite...")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()

tflite_path = os.path.join(MODEL_DIR, "asl_model_int8.tflite")
with open(tflite_path, "wb") as f:
    f.write(tflite_model)
print("Saved", tflite_path, "-", len(tflite_model), "bytes")

# ============================================================
# 8. WRITE model_data.h DIRECTLY (folds in create_model_header.py)
# ============================================================

header_path = os.path.join(MODEL_DIR, "model_data.h")
with open(header_path, "w") as f:
    f.write("#ifndef MODEL_DATA_H\n#define MODEL_DATA_H\n\n#include <stdint.h>\n\n")
    f.write("const unsigned char asl_model_int8_tflite[] = {\n")
    for i, byte in enumerate(tflite_model):
        if i % 12 == 0:
            f.write("    ")
        f.write(f"0x{byte:02x}")
        if i < len(tflite_model) - 1:
            f.write(", ")
        if (i + 1) % 12 == 0:
            f.write("\n")
    f.write("\n};\n\n")
    f.write(f"const unsigned int asl_model_int8_tflite_len = {len(tflite_model)};\n\n")
    f.write("#endif\n")

print("Saved", header_path)

print("\n======================================")
print(" DONE")
print("======================================")
print("\nNext steps:")
print(f"1. Copy {header_path} into esp32/model_deploy/")
print("2. Make sure esp32/model_deploy/calibration.h is up to date")
print("   (run generate_calibration_header.py with your freshest calibration)")
print("3. Reflash the ESP32 and test live")
print("4. Log results in DAILY_LOG_TEMPLATE.md")