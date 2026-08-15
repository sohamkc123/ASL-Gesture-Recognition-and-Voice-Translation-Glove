import os
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# ============================================================
# SETTINGS
# ============================================================

CSV_FILE = "asl_dataset.csv"

MODEL_DIR = "asl_model"

os.makedirs(MODEL_DIR, exist_ok=True)

print("\n======================================")
print(" ASL TinyML MODEL TRAINING")
print("======================================\n")


# ============================================================
# 1. LOAD DATASET
# ============================================================

print("[1] Loading dataset...")

# Your CSV does NOT contain column headers
df = pd.read_csv(CSV_FILE, header=None)

print("Original dataset shape:", df.shape)


# ============================================================
# 2. SEPARATE LABEL AND SENSOR DATA
# ============================================================

# Column 0 = gesture label
# Columns 1 onward = sensor values

labels = df.iloc[:, 0].astype(str)

features = df.iloc[:, 1:].apply(
    pd.to_numeric,
    errors="coerce"
)


# ============================================================
# 3. REMOVE INVALID ROWS
# ============================================================

print("\n[2] Checking invalid rows...")

invalid_rows = features.isna().any(axis=1)

print("Invalid rows found:", invalid_rows.sum())

if invalid_rows.sum() > 0:
    print("Removing invalid rows...")

    df = df.loc[~invalid_rows].reset_index(drop=True)

    labels = df.iloc[:, 0].astype(str)

    features = df.iloc[:, 1:].apply(
        pd.to_numeric,
        errors="coerce"
    )


print("Clean dataset shape:", features.shape)


# ============================================================
# 4. CONVERT TO NUMPY
# ============================================================

X = features.values.astype(np.float32)
y_text = labels.values


print("\nNumber of features:", X.shape[1])
print("Number of samples:", X.shape[0])

print("\nClasses:")
print(np.unique(y_text))


# ============================================================
# 5. ENCODE LABELS
# ============================================================

label_encoder = LabelEncoder()

y = label_encoder.fit_transform(y_text)

class_names = label_encoder.classes_

print("\nClass mapping:")

for i, name in enumerate(class_names):
    print(i, "=", name)


# Save class labels
with open(
    os.path.join(MODEL_DIR, "labels.txt"),
    "w"
) as f:

    for label in class_names:
        f.write(str(label) + "\n")


# ============================================================
# 6. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples :", len(X_test))


# ============================================================
# 7. FEATURE NORMALIZATION
# ============================================================

print("\n[3] Normalizing sensor data...")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled = X_train_scaled.astype(np.float32)
X_test_scaled = X_test_scaled.astype(np.float32)


# ============================================================
# 8. SAVE NORMALIZATION PARAMETERS
# ============================================================

mean_values = scaler.mean_
std_values = scaler.scale_

with open(
    os.path.join(MODEL_DIR, "normalization.h"),
    "w"
) as f:

    f.write("#ifndef NORMALIZATION_H\n")
    f.write("#define NORMALIZATION_H\n\n")

    f.write(
        f"#define NUM_FEATURES {len(mean_values)}\n\n"
    )

    f.write(
        "const float feature_mean[NUM_FEATURES] = {\n"
    )

    for value in mean_values:
        f.write(f"    {value:.10f}f,\n")

    f.write("};\n\n")

    f.write(
        "const float feature_std[NUM_FEATURES] = {\n"
    )

    for value in std_values:
        f.write(f"    {value:.10f}f,\n")

    f.write("};\n\n")

    f.write("#endif\n")


print("Normalization parameters saved.")


# ============================================================
# 9. BUILD SMALL NEURAL NETWORK
# ============================================================

print("\n[4] Building neural network...")

num_features = X_train_scaled.shape[1]
num_classes = len(class_names)

model = tf.keras.Sequential([

    tf.keras.layers.Input(
        shape=(num_features,)
    ),

    tf.keras.layers.Dense(
        32,
        activation="relu"
    ),

    tf.keras.layers.Dense(
        16,
        activation="relu"
    ),

    tf.keras.layers.Dense(
        num_classes,
        activation="softmax"
    )
])


model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]
)


model.summary()


# ============================================================
# 10. TRAIN MODEL
# ============================================================

print("\n[5] Training model...")

history = model.fit(

    X_train_scaled,
    y_train,

    validation_split=0.20,

    epochs=100,

    batch_size=32,

    verbose=1,

    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=15,
            restore_best_weights=True
        )
    ]
)


# ============================================================
# 11. EVALUATE MODEL
# ============================================================

print("\n[6] Evaluating model...")

test_loss, test_accuracy = model.evaluate(
    X_test_scaled,
    y_test,
    verbose=0
)

print("\n======================================")
print("TEST ACCURACY:", test_accuracy * 100, "%")
print("TEST LOSS:", test_loss)
print("======================================")


# ============================================================
# 12. CLASSIFICATION REPORT
# ============================================================

predictions = model.predict(
    X_test_scaled,
    verbose=0
)

predicted_classes = np.argmax(
    predictions,
    axis=1
)

print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        predicted_classes,
        target_names=class_names
    )
)


# ============================================================
# 13. CONFUSION MATRIX
# ============================================================

print("\nConfusion Matrix:\n")

print(
    confusion_matrix(
        y_test,
        predicted_classes
    )
)


# ============================================================
# 14. SAVE KERAS MODEL
# ============================================================

keras_model_path = os.path.join(
    MODEL_DIR,
    "asl_model.keras"
)

model.save(keras_model_path)

print(
    "\nKeras model saved:",
    keras_model_path
)


# ============================================================
# 15. TFLITE REPRESENTATIVE DATASET
# ============================================================

def representative_dataset():

    for i in range(
        min(300, len(X_train_scaled))
    ):

        sample = X_train_scaled[i]

        sample = np.expand_dims(
            sample,
            axis=0
        ).astype(np.float32)

        yield [sample]


# ============================================================
# 16. CONVERT TO FULL INTEGER INT8 TFLITE
# ============================================================

print("\n[7] Converting to INT8 TensorFlow Lite...")

converter = tf.lite.TFLiteConverter.from_keras_model(
    model
)

converter.optimizations = [
    tf.lite.Optimize.DEFAULT
]

converter.representative_dataset = (
    representative_dataset
)

converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8
]

converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()


tflite_path = os.path.join(
    MODEL_DIR,
    "asl_model_int8.tflite"
)

with open(
    tflite_path,
    "wb"
) as f:

    f.write(tflite_model)


print(
    "\nINT8 TFLite model saved:",
    tflite_path
)

print(
    "Model size:",
    len(tflite_model),
    "bytes"
)


# ============================================================
# 17. CHECK TFLITE MODEL
# ============================================================

print("\n[8] Testing TFLite model...")

interpreter = tf.lite.Interpreter(
    model_path=tflite_path
)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("\nInput details:")
print(input_details)

print("\nOutput details:")
print(output_details)


# ============================================================
# 18. GET QUANTIZATION PARAMETERS
# ============================================================

input_scale, input_zero_point = (
    input_details[0]["quantization"]
)

output_scale, output_zero_point = (
    output_details[0]["quantization"]
)

print("\nInput quantization:")
print("Scale:", input_scale)
print("Zero point:", input_zero_point)

print("\nOutput quantization:")
print("Scale:", output_scale)
print("Zero point:", output_zero_point)


# ============================================================
# 19. TEST ONE SAMPLE
# ============================================================

sample = X_test_scaled[0]

sample = np.expand_dims(
    sample,
    axis=0
).astype(np.float32)


# Convert float input to INT8
sample_int8 = np.round(
    sample / input_scale
    + input_zero_point
).astype(np.int8)


interpreter.set_tensor(
    input_details[0]["index"],
    sample_int8
)

interpreter.invoke()

output = interpreter.get_tensor(
    output_details[0]["index"]
)

# Convert output back to float
output_float = (
    output.astype(np.float32)
    - output_zero_point
) * output_scale


predicted_class = np.argmax(
    output_float
)

print("\nSample prediction:")
print(
    "Actual:",
    class_names[y_test[0]]
)

print(
    "Predicted:",
    class_names[predicted_class]
)


# ============================================================
# 20. PRINT FINAL INFORMATION
# ============================================================

print("\n======================================")
print(" TRAINING COMPLETE")
print("======================================")

print("\nFiles generated:")

print(
    "1.",
    os.path.join(
        MODEL_DIR,
        "asl_model.keras"
    )
)

print(
    "2.",
    os.path.join(
        MODEL_DIR,
        "asl_model_int8.tflite"
    )
)

print(
    "3.",
    os.path.join(
        MODEL_DIR,
        "normalization.h"
    )
)

print(
    "4.",
    os.path.join(
        MODEL_DIR,
        "labels.txt"
    )
)

print("\nYour model is ready for ESP32.")