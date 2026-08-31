"""
repair_index_flex.py
--------------------
Repairs a broken `flex_index` column in one session CSV by copying
`flex_index` values from one or more reference session CSVs.

Why this script:
- collect_data.py stores normalized flex values (not raw ADC), so if raw logs
  are not available, you cannot re-normalize raw with another calibration.
- This script reuses normalized index-flex values from good sessions.

Usage:
    python repair_index_flex.py <broken_csv> <output_csv> <ref_csv1> [<ref_csv2> ...]

Example:
    python repair_index_flex.py \
      ../data/2026-08-23/session_2026-08-23_09-51-49.csv \
      ../data/2026-08-23/session_2026-08-23_09-51-49.csv \
      ../data/2026-08-22/session_2026-08-22_23-06-48.csv \
      ../data/2026-08-23/session_2026-08-22_23-06-48.csv
"""

import os
import sys
import pandas as pd

REQ_COLS = ["label", "rep", "flex_index"]


def load_csv(path):
    df = pd.read_csv(path)
    missing = [c for c in REQ_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")
    return df


def main():
    if len(sys.argv) < 5:
        raise SystemExit(
            "Usage: python repair_index_flex.py <broken_csv> <output_csv> <ref_csv1> [<ref_csv2> ...]"
        )

    broken_path = sys.argv[1]
    output_path = sys.argv[2]
    ref_paths = sys.argv[3:]

    broken = load_csv(broken_path).copy()
    refs = [load_csv(p) for p in ref_paths]
    ref = pd.concat(refs, ignore_index=True)

    # within-label index to align sequence deterministically
    broken["_k"] = broken.groupby("label").cumcount()
    ref["_k"] = ref.groupby("label").cumcount()

    # Build donor table
    donor = ref[["label", "_k", "flex_index"]].rename(columns={"flex_index": "flex_index_ref"})

    repaired = broken.merge(donor, on=["label", "_k"], how="left")

    # Fallback if any label has fewer donor rows than needed.
    # Use per-label median from references, then global median.
    label_median = ref.groupby("label")["flex_index"].median()
    global_median = float(ref["flex_index"].median())

    missing_mask = repaired["flex_index_ref"].isna()
    if missing_mask.any():
        repaired.loc[missing_mask, "flex_index_ref"] = repaired.loc[missing_mask, "label"].map(label_median)
        repaired["flex_index_ref"] = repaired["flex_index_ref"].fillna(global_median)

    before_std = float(repaired["flex_index"].std())
    repaired["flex_index"] = repaired["flex_index_ref"]
    after_std = float(repaired["flex_index"].std())

    repaired = repaired.drop(columns=["_k", "flex_index_ref"])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    repaired.to_csv(output_path, index=False)

    print("Repair complete")
    print(f"  broken file : {broken_path}")
    print(f"  output file : {output_path}")
    print(f"  refs used   : {len(ref_paths)}")
    print(f"  flex_index std before: {before_std:.6f}")
    print(f"  flex_index std after : {after_std:.6f}")


if __name__ == "__main__":
    main()
