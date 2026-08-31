"""
train_random_forest.py
----------------------
Train a Random Forest ASL classifier using the same 17 features used by
collect_data.py and the NN model.

Outputs:
  - ../models_rf/random_forest.joblib
  - ../models_rf/labels.txt
  - ../models_rf/rf_model.h
  - ../esp32/rf_model_deploy/model_deploy_rf/rf_model.h

Usage:
  python train_random_forest.py
"""

import os
import argparse
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder

FEATURE_NAMES = [
    "flex_pinky", "flex_ring", "flex_middle", "flex_index", "flex_thumb",
    "touch_index", "touch_middle", "touch_ring", "touch_pinky", "touch_r", "touch_u",
    "ax", "ay", "az", "gx", "gy", "gz",
]

ROOT = os.path.join(os.path.dirname(__file__), "..")
DEFAULT_DATA_PATH = os.path.join(ROOT, "data", "ASL_Gesture_Dataset_Cleaned.csv")
MODEL_DIR = os.path.join(ROOT, "models_rf")
ESP32_RF_DIR = os.path.join(ROOT, "esp32", "rf_model_deploy", "model_deploy_rf")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(ESP32_RF_DIR, exist_ok=True)


def tree_nodes_from_sklearn(tree):
    nodes = []
    left = tree.children_left
    right = tree.children_right
    feature = tree.feature
    threshold = tree.threshold
    value = tree.value

    for i in range(tree.node_count):
        is_leaf = left[i] == -1 and right[i] == -1
        if is_leaf:
            pred = int(np.argmax(value[i][0]))
            nodes.append((-1, 0.0, -1, -1, pred))
        else:
            nodes.append((int(feature[i]), float(threshold[i]), int(left[i]), int(right[i]), -1))
    return nodes


def write_rf_header(rf, class_names, out_path):
    guard = "RF_MODEL_H"

    tree_node_counts = []
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"#ifndef {guard}\n#define {guard}\n\n")
        f.write("#include <stdint.h>\n\n")
        f.write("typedef struct {\n")
        f.write("  int16_t feature;\n")
        f.write("  float threshold;\n")
        f.write("  int16_t left;\n")
        f.write("  int16_t right;\n")
        f.write("  int16_t pred;\n")
        f.write("} RFNode;\n\n")
        f.write("typedef struct {\n")
        f.write("  const RFNode* nodes;\n")
        f.write("  uint16_t node_count;\n")
        f.write("} RFTree;\n\n")

        for ti, est in enumerate(rf.estimators_):
            nodes = tree_nodes_from_sklearn(est.tree_)
            tree_node_counts.append(len(nodes))
            f.write(f"static const RFNode RF_TREE_{ti}[] = {{\n")
            for (feat, thr, lft, rgt, pred) in nodes:
                f.write(
                    f"  {{{feat}, {thr:.7f}f, {lft}, {rgt}, {pred}}},\n"
                )
            f.write("};\n\n")

        f.write("static const RFTree RF_TREES[] = {\n")
        for ti, ncount in enumerate(tree_node_counts):
            f.write(f"  {{RF_TREE_{ti}, {ncount}}},\n")
        f.write("};\n\n")

        f.write(f"static const uint16_t RF_NUM_TREES = {len(rf.estimators_)};\n")
        f.write(f"static const uint8_t RF_NUM_CLASSES = {len(class_names)};\n")
        f.write(f"static const uint8_t RF_NUM_FEATURES = {len(FEATURE_NAMES)};\n\n")

        label_list = ", ".join([f'\"{c}\"' for c in class_names])
        f.write(f"static const char* RF_LABELS[{len(class_names)}] = {{{label_list}}};\n\n")

        f.write(f"#endif // {guard}\n")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=DEFAULT_DATA_PATH, help="Path to training CSV")
    return p.parse_args()


def main():
    args = parse_args()
    data_path = args.data
    print("\n======================================")
    print(" RANDOM FOREST TRAINING")
    print("======================================\n")

    if not os.path.exists(data_path):
        raise SystemExit(f"{data_path} not found. Run merge_data.py first.")

    df = pd.read_csv(data_path)
    print("Loaded", len(df), "rows,", df["session"].nunique(), "session(s).")

    X = df[FEATURE_NAMES].values.astype(np.float32)
    y_text = df["label"].values
    sessions = df["session"].values

    le = LabelEncoder()
    y = le.fit_transform(y_text)
    class_names = le.classes_

    with open(os.path.join(MODEL_DIR, "labels.txt"), "w", encoding="utf-8") as f:
        for c in class_names:
            f.write(str(c) + "\n")

    unique_sessions = sorted(set(sessions))
    print(f"\n{len(unique_sessions)} unique session(s) found.")

    if len(unique_sessions) >= 2:
        n_test = max(1, round(len(unique_sessions) * 0.2))
        test_sessions = set(unique_sessions[-n_test:])
        train_mask = ~pd.Series(sessions).isin(test_sessions).values
        print("Held-out test session(s):", test_sessions)
    else:
        print("WARNING: only one session. Falling back to random split.")
        from sklearn.model_selection import train_test_split
        idx_train, idx_test = train_test_split(
            np.arange(len(y)), test_size=0.2, stratify=y, random_state=42
        )
        train_mask = np.zeros(len(y), dtype=bool)
        train_mask[idx_train] = True

    X_train, X_test = X[train_mask], X[~train_mask]
    y_train, y_test = y[train_mask], y[~train_mask]

    print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

    # Flash-size-safe configuration for ESP32 deployment.
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )

    print("\nTraining random forest...")
    rf.fit(X_train, y_train)

    pred = rf.predict(X_test)
    acc = accuracy_score(y_test, pred)

    print("\n======================================")
    print("HELD-OUT SESSION ACCURACY:", round(acc * 100, 2), "%")
    print("======================================")

    print("\nClassification report (held-out session):\n")
    print(classification_report(
        y_test,
        pred,
        labels=np.unique(y_test),
        target_names=class_names[np.unique(y_test)],
        zero_division=0,
    ))

    print("Confusion matrix:\n")
    print(confusion_matrix(y_test, pred))

    model_path = os.path.join(MODEL_DIR, "random_forest.joblib")
    dump(rf, model_path)
    print("\nSaved", model_path)

    rf_header_path = os.path.join(MODEL_DIR, "rf_model.h")
    write_rf_header(rf, class_names, rf_header_path)
    print("Saved", rf_header_path)

    deploy_header_path = os.path.join(ESP32_RF_DIR, "rf_model.h")
    with open(rf_header_path, "r", encoding="utf-8") as src, \
            open(deploy_header_path, "w", encoding="utf-8") as dst:
        dst.write(src.read())
    print("Saved", deploy_header_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
