"""
merge_data.py
--------------
Combines every data/<date>/session_<timestamp>.csv into one file for
training. Run this after each collection session (or just once before
training, it'll pick up everything).

Usage:
    python merge_data.py
"""

import pandas as pd
import glob
import os
import numpy as np

FEATURE_NAMES = [
    "flex_pinky", "flex_ring", "flex_middle", "flex_index", "flex_thumb",
    "touch_index", "touch_middle", "touch_ring", "touch_pinky", "touch_r", "touch_u",
    "ax", "ay", "az", "gx", "gy", "gz",
]
FLEX_COLS = FEATURE_NAMES[0:5]
TOUCH_COLS = FEATURE_NAMES[5:11]
MPU_COLS = FEATURE_NAMES[11:17]

DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "data")
OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "combined_dataset.csv")


def clean_dataframe(df):
    start_rows = len(df)

    # Coerce features to numeric so bad tokens become NaN.
    for col in FEATURE_NAMES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Mark impossible MPU frames (all six channels exactly zero) as missing,
    # then repair them within each (session, label, rep) sequence.
    mpu_all_zero_mask = (df[MPU_COLS].abs().sum(axis=1) == 0)
    mpu_all_zero_rows = int(mpu_all_zero_mask.sum())
    df.loc[mpu_all_zero_mask, MPU_COLS] = np.nan

    # Interpolate time-wise inside each repetition to avoid deleting rows.
    for col in MPU_COLS:
        df[col] = df.groupby(["session", "label", "rep"], dropna=False)[col].transform(
            lambda s: s.interpolate(limit_direction="both")
        )

    # Fill any still-missing MPU values with label medians.
    for col in MPU_COLS:
        df[col] = df.groupby("label", dropna=False)[col].transform(
            lambda s: s.fillna(s.median()) if s.notna().any() else s
        )

    # Last fallback: global median.
    for col in MPU_COLS:
        df[col] = df[col].fillna(df[col].median())

    # Keep touch values binary only.
    touch_valid = df[TOUCH_COLS].isin([0, 1]).all(axis=1)
    dropped_touch_invalid = int((~touch_valid).sum())
    df = df[touch_valid].copy()

    # Drop rows with remaining non-finite values in any feature.
    finite_mask = np.isfinite(df[FEATURE_NAMES].to_numpy()).all(axis=1)
    dropped_non_finite = int((~finite_mask).sum())
    df = df[finite_mask].copy()

    print("Cleaning summary:")
    print(f"  MPU all-zero rows repaired       : {mpu_all_zero_rows}")
    print(f"  dropped invalid touch rows       : {dropped_touch_invalid}")
    print(f"  dropped remaining non-finite rows: {dropped_non_finite}")
    print(f"  kept rows                        : {len(df)} / {start_rows}")

    print("\nFlex zero ratios (after cleaning):")
    for col in FLEX_COLS:
        ratio = float((df[col] == 0).mean()) if len(df) else 0.0
        warn = "  <-- high; possible loose sensor" if ratio > 0.40 else ""
        print(f"  {col:12s}: {ratio:.1%}{warn}")

    return df


def main():
    files = glob.glob(os.path.join(DATA_ROOT, "*", "session_*.csv"))
    if not files:
        raise SystemExit(f"No session CSVs found under {DATA_ROOT}/<date>/. Run collect_data.py first.")

    dfs = [pd.read_csv(f) for f in files]
    full = pd.concat(dfs, ignore_index=True)

    print(f"Merged {len(files)} session file(s), {len(full)} total rows before cleaning.\n")
    full = clean_dataframe(full)
    print()
    print("Samples per letter:")
    counts = full["label"].value_counts().sort_index()
    print(counts)

    if len(counts):
        min_count = int(counts.min())
        max_count = int(counts.max())
        if min_count < 0.5 * max_count:
            print("\nWARNING: class imbalance detected.")
            print(f"  smallest class: {min_count} samples")
            print(f"  largest class : {max_count} samples")
            print("  Consider recollecting weak letters or using class weights in training.")
    print("\nSamples per session:")
    print(full["session"].value_counts())
    print("\nSamples per day:")
    print(full["day"].value_counts())

    full.to_csv(OUT_PATH, index=False)
    print(f"\nSaved {OUT_PATH}")

    if full["session"].nunique() < 2:
        print("\nWARNING: only one session so far - train_model.py will fall back to a "
              "random split, which won't tell you much about live performance. "
              "Collect at least one more session before trusting the reported accuracy.")


if __name__ == "__main__":
    main()