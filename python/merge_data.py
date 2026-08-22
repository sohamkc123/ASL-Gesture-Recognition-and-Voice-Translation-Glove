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

DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "data")
OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "combined_dataset.csv")


def main():
    files = glob.glob(os.path.join(DATA_ROOT, "*", "session_*.csv"))
    if not files:
        raise SystemExit(f"No session CSVs found under {DATA_ROOT}/<date>/. Run collect_data.py first.")

    dfs = [pd.read_csv(f) for f in files]
    full = pd.concat(dfs, ignore_index=True)

    print(f"Merged {len(files)} session file(s), {len(full)} total rows.\n")
    print("Samples per letter:")
    print(full["label"].value_counts().sort_index())
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