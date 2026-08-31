"""
dataset_analysis_and_split.py
-----------------------------
ASL Gesture Dataset Analysis

Input:
    ASL_Gesture_Dataset_Cleaned.csv

Outputs:
    1. Overall class distribution graph
    2. Overall class distribution pie chart
    3. Training/test class distribution graph
    4. Training/test class distribution graph by percentage
    5. train_dataset.csv
    6. test_dataset.csv

IMPORTANT:
    The dataset is split by SESSION, not by individual rows.
    This avoids very similar consecutive sensor samples from the
    same recording appearing in both training and testing sets.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = r"C:\Users\shrawan prasai\OneDrive\Desktop\ASL-Gesture-Recognition-and-Voice-Translation-Glove\data\ASL_Gesture_Dataset_Cleaned.csv"

OUTPUT_DIR = "dataset_analysis"

TRAIN_RATIO = 0.80


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD DATASET
# ============================================================

print("\n==========================================")
print(" ASL DATASET ANALYSIS")
print("==========================================\n")

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Dataset not found:\n{DATA_PATH}"
    )

df = pd.read_csv(DATA_PATH)

print(f"Dataset loaded successfully.")
print(f"Rows    : {len(df):,}")
print(f"Columns : {len(df.columns)}")

print("\nColumns:")
print(df.columns.tolist())


# ============================================================
# BASIC VALIDATION
# ============================================================

required_columns = ["label", "session"]

for col in required_columns:
    if col not in df.columns:
        raise ValueError(
            f"Required column '{col}' is missing."
        )


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

class_counts = (
    df["label"]
    .value_counts()
    .sort_index()
)

class_percentages = (
    class_counts / len(df) * 100
)


print("\n==========================================")
print(" OVERALL CLASS DISTRIBUTION")
print("==========================================\n")

for label in class_counts.index:

    print(
        f"{label}: "
        f"{class_counts[label]:,} samples "
        f"({class_percentages[label]:.2f}%)"
    )


# ============================================================
# GRAPH 1: OVERALL CLASS DISTRIBUTION
# ============================================================

plt.figure(figsize=(14, 7))

plt.bar(
    class_counts.index,
    class_counts.values
)

plt.xlabel("ASL Letter")
plt.ylabel("Number of Samples")
plt.title(
    "ASL Gesture Dataset Distribution Across Classes"
)

plt.xticks(rotation=0)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

overall_graph = os.path.join(
    OUTPUT_DIR,
    "overall_class_distribution.png"
)

plt.savefig(
    overall_graph,
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# GRAPH 2: PIE CHART
# ============================================================

plt.figure(figsize=(10, 10))

plt.pie(
    class_counts.values,
    labels=class_counts.index,
    autopct="%1.1f%%",
    startangle=90
)

plt.title(
    "Percentage Distribution of ASL Classes"
)

plt.tight_layout()

pie_graph = os.path.join(
    OUTPUT_DIR,
    "class_distribution_pie.png"
)

plt.savefig(
    pie_graph,
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# SESSION INFORMATION
# ============================================================

sessions = sorted(
    df["session"].astype(str).unique()
)

print("\n==========================================")
print(" SESSION INFORMATION")
print("==========================================\n")

print(
    f"Total sessions: {len(sessions)}"
)

for session in sessions:

    count = (
        df["session"].astype(str) == session
    ).sum()

    print(
        f"{session}: {count:,} samples"
    )


# ============================================================
# SESSION-WISE 80/20 SPLIT
# ============================================================

if len(sessions) < 2:

    raise ValueError(
        "At least 2 sessions are required "
        "for a session-wise train/test split."
    )


# Number of sessions used for training
n_train_sessions = max(
    1,
    int(round(len(sessions) * TRAIN_RATIO))
)


# Make sure at least one session remains for testing
if n_train_sessions >= len(sessions):

    n_train_sessions = len(sessions) - 1


train_sessions = sessions[
    :n_train_sessions
]

test_sessions = sessions[
    n_train_sessions:
]


train_df = df[
    df["session"].astype(str).isin(train_sessions)
].copy()


test_df = df[
    df["session"].astype(str).isin(test_sessions)
].copy()


# ============================================================
# PRINT SPLIT INFORMATION
# ============================================================

print("\n==========================================")
print(" DATASET SPLIT")
print("==========================================\n")

print(
    f"Training sessions: {len(train_sessions)}"
)

for s in train_sessions:
    print(f"  TRAIN: {s}")


print(
    f"\nTesting sessions: {len(test_sessions)}"
)

for s in test_sessions:
    print(f"  TEST : {s}")


print("\nSample counts:")

print(
    f"Training: "
    f"{len(train_df):,} samples "
    f"({len(train_df) / len(df) * 100:.2f}%)"
)

print(
    f"Testing : "
    f"{len(test_df):,} samples "
    f"({len(test_df) / len(df) * 100:.2f}%)"
)


# ============================================================
# CLASS DISTRIBUTION FOR TRAINING AND TESTING
# ============================================================

train_counts = (
    train_df["label"]
    .value_counts()
    .reindex(class_counts.index, fill_value=0)
)

test_counts = (
    test_df["label"]
    .value_counts()
    .reindex(class_counts.index, fill_value=0)
)


# ============================================================
# GRAPH 3: TRAIN VS TEST SAMPLE COUNTS
# ============================================================

x = range(len(class_counts))
width = 0.4

plt.figure(figsize=(15, 7))

plt.bar(
    [i - width / 2 for i in x],
    train_counts.values,
    width=width,
    label="Training"
)

plt.bar(
    [i + width / 2 for i in x],
    test_counts.values,
    width=width,
    label="Testing"
)

plt.xlabel("ASL Letter")
plt.ylabel("Number of Samples")

plt.title(
    "Training and Testing Distribution Across ASL Classes"
)

plt.xticks(
    list(x),
    class_counts.index
)

plt.legend()

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

split_graph = os.path.join(
    OUTPUT_DIR,
    "train_test_class_distribution.png"
)

plt.savefig(
    split_graph,
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# GRAPH 4: TRAIN VS TEST PERCENTAGE
# ============================================================

train_percent = (
    train_counts /
    len(train_df) *
    100
)

test_percent = (
    test_counts /
    len(test_df) *
    100
)


plt.figure(figsize=(15, 7))

plt.bar(
    [i - width / 2 for i in x],
    train_percent.values,
    width=width,
    label="Training"
)

plt.bar(
    [i + width / 2 for i in x],
    test_percent.values,
    width=width,
    label="Testing"
)

plt.xlabel("ASL Letter")
plt.ylabel("Percentage of Samples (%)")

plt.title(
    "Percentage Distribution of ASL Classes "
    "in Training and Testing Sets"
)

plt.xticks(
    list(x),
    class_counts.index
)

plt.legend()

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

percentage_graph = os.path.join(
    OUTPUT_DIR,
    "train_test_class_percentage.png"
)

plt.savefig(
    percentage_graph,
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# SAVE SPLIT DATASETS
# ============================================================

train_path = os.path.join(
    OUTPUT_DIR,
    "train_dataset.csv"
)

test_path = os.path.join(
    OUTPUT_DIR,
    "test_dataset.csv"
)


train_df.to_csv(
    train_path,
    index=False
)

test_df.to_csv(
    test_path,
    index=False
)


# ============================================================
# SAVE CLASS DISTRIBUTION TABLE
# ============================================================

distribution_table = pd.DataFrame({

    "Class": class_counts.index,

    "Total Samples":
        class_counts.values,

    "Overall Percentage":
        class_percentages.values,

    "Training Samples":
        train_counts.values,

    "Training Percentage":
        train_percent.values,

    "Testing Samples":
        test_counts.values,

    "Testing Percentage":
        test_percent.values
})


distribution_path = os.path.join(
    OUTPUT_DIR,
    "class_distribution.csv"
)


distribution_table.to_csv(
    distribution_path,
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n==========================================")
print(" FILES CREATED")
print("==========================================\n")

print(
    f"Overall graph      : {overall_graph}"
)

print(
    f"Pie chart           : {pie_graph}"
)

print(
    f"Train/test graph    : {split_graph}"
)

print(
    f"Percentage graph    : {percentage_graph}"
)

print(
    f"Training dataset    : {train_path}"
)

print(
    f"Testing dataset     : {test_path}"
)

print(
    f"Distribution table  : {distribution_path}"
)

print("\nAnalysis completed successfully.")