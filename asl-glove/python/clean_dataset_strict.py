"""
clean_dataset_strict.py
-----------------------
Strict, label-aware dataset cleaning for ASL glove data.

Features:
1) Numeric + finite checks
2) Touch binary check (0/1)
3) Optional touch-template filtering (per label)
4) Per-label robust outlier filtering (MAD-based) on flex + MPU features
5) Optional class balancing (same sample count per label)

Usage examples:

# A) Build inferred touch template from current dataset (edit this file manually)
python clean_dataset_strict.py --make-touch-template data/touch_template_inferred.json

# B) Strict clean with touch template + balancing
python clean_dataset_strict.py \
  --touch-template data/touch_template_inferred.json \
  --touch-max-mismatches 1 \
  --balance

# C) Strict clean without touch template
python clean_dataset_strict.py --balance
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import numpy as np
import pandas as pd

FEATURE_NAMES = [
    "flex_pinky", "flex_ring", "flex_middle", "flex_index", "flex_thumb",
    "touch_index", "touch_middle", "touch_ring", "touch_pinky", "touch_r", "touch_u",
    "ax", "ay", "az", "gx", "gy", "gz",
]

FLEX_COLS = FEATURE_NAMES[0:5]
TOUCH_COLS = FEATURE_NAMES[5:11]
MPU_COLS = FEATURE_NAMES[11:17]
OUTLIER_COLS = FLEX_COLS + MPU_COLS


ROOT = os.path.join(os.path.dirname(__file__), "..")
DEFAULT_INPUT = os.path.join(ROOT, "data", "combined_dataset.csv")
DEFAULT_OUTPUT = os.path.join(ROOT, "data", "combined_dataset_final.csv")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default=DEFAULT_INPUT)
    p.add_argument("--output", default=DEFAULT_OUTPUT)
    p.add_argument("--touch-template", default=None, help="JSON mapping label->6 touch bits")
    p.add_argument("--touch-max-mismatches", type=int, default=1)
    p.add_argument("--mad-z", type=float, default=4.0)
    p.add_argument("--min-per-label", type=int, default=80)
    p.add_argument("--balance", action="store_true")
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--make-touch-template", default=None, help="Write inferred touch template and exit")
    return p.parse_args()


def ensure_numeric_finite(df: pd.DataFrame) -> pd.DataFrame:
    for col in FEATURE_NAMES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    finite_mask = np.isfinite(df[FEATURE_NAMES].to_numpy()).all(axis=1)
    return df[finite_mask].copy()


def filter_touch_binary(df: pd.DataFrame) -> pd.DataFrame:
    mask = df[TOUCH_COLS].isin([0, 1]).all(axis=1)
    return df[mask].copy()


def infer_touch_template(df: pd.DataFrame) -> Dict[str, List[int]]:
    out: Dict[str, List[int]] = {}
    for label, g in df.groupby("label"):
        # majority vote per touch bit
        bits = (g[TOUCH_COLS].mean(axis=0) >= 0.5).astype(int).tolist()
        out[str(label)] = [int(x) for x in bits]
    return dict(sorted(out.items(), key=lambda kv: kv[0]))


def load_touch_template(path: str) -> Dict[str, List[int]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    out = {}
    for k, v in data.items():
        if not isinstance(v, list) or len(v) != 6:
            raise ValueError(f"touch template for label '{k}' must be list of 6 values")
        vv = [int(x) for x in v]
        if any(x not in (0, 1) for x in vv):
            raise ValueError(f"touch template for label '{k}' must contain only 0/1")
        out[str(k)] = vv
    return out


def filter_by_touch_template(df: pd.DataFrame, template: Dict[str, List[int]], max_mismatches: int) -> pd.DataFrame:
    def row_ok(row):
        lbl = str(row["label"])
        if lbl not in template:
            return True
        expected = template[lbl]
        actual = [int(row[c]) for c in TOUCH_COLS]
        mism = sum(int(a != b) for a, b in zip(actual, expected))
        return mism <= max_mismatches

    mask = df.apply(row_ok, axis=1)
    return df[mask].copy()


def robust_label_outlier_filter(df: pd.DataFrame, mad_z: float) -> pd.DataFrame:
    keep = np.ones(len(df), dtype=bool)
    idx = df.index.to_numpy()

    for label, g in df.groupby("label"):
        g_idx = g.index.to_numpy()
        g_keep = np.ones(len(g), dtype=bool)

        for col in OUTLIER_COLS:
            x = g[col].to_numpy(dtype=float)
            med = np.median(x)
            mad = np.median(np.abs(x - med))
            scale = 1.4826 * mad

            if scale < 1e-9:
                # Nearly constant feature for this label; keep all for this feature.
                continue

            z = np.abs((x - med) / scale)
            g_keep &= (z <= mad_z)

        # map back to global keep mask
        local_pos = np.searchsorted(idx, g_idx)
        keep[local_pos] &= g_keep

    return df.iloc[np.where(keep)[0]].copy()


def balance_classes(df: pd.DataFrame, min_per_label: int, random_state: int) -> pd.DataFrame:
    counts = df["label"].value_counts().sort_index()
    target = int(counts.min())
    if target < min_per_label:
        print(f"WARNING: smallest class has only {target} rows (< {min_per_label}).")
        print("Keeping unbalanced data to avoid throwing away too much.")
        return df

    parts = []
    for lbl, g in df.groupby("label"):
        parts.append(g.sample(n=target, random_state=random_state))
    out = pd.concat(parts, ignore_index=True)
    return out.sample(frac=1.0, random_state=random_state).reset_index(drop=True)


def print_counts(title: str, df: pd.DataFrame):
    print(f"\n{title}")
    print(df["label"].value_counts().sort_index())


def main():
    args = parse_args()

    if not os.path.exists(args.input):
        raise SystemExit(f"Input file not found: {args.input}")

    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} rows from {args.input}")

    # Coerce + finite + touch binary first
    df = ensure_numeric_finite(df)
    df = filter_touch_binary(df)

    if args.make_touch_template:
        tmpl = infer_touch_template(df)
        os.makedirs(os.path.dirname(args.make_touch_template), exist_ok=True)
        with open(args.make_touch_template, "w", encoding="utf-8") as f:
            json.dump(tmpl, f, indent=2)
        print(f"Wrote inferred touch template: {args.make_touch_template}")
        return

    start = len(df)

    # Optional touch template filtering
    if args.touch_template:
        tmpl = load_touch_template(args.touch_template)
        before = len(df)
        df = filter_by_touch_template(df, tmpl, args.touch_max_mismatches)
        print(f"Touch-template filter: kept {len(df)} / {before}")

    # Label-wise robust outlier filter
    before = len(df)
    df = robust_label_outlier_filter(df, mad_z=args.mad_z)
    print(f"Robust outlier filter: kept {len(df)} / {before}")

    # Remove labels that are too tiny after filtering
    counts = df["label"].value_counts()
    valid_labels = counts[counts >= args.min_per_label].index
    before = len(df)
    df = df[df["label"].isin(valid_labels)].copy()
    print(f"Min-per-label filter (>= {args.min_per_label}): kept {len(df)} / {before}")

    if args.balance:
        before = len(df)
        df = balance_classes(df, args.min_per_label, args.random_state)
        print(f"Balance classes: kept {len(df)} / {before}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False)

    print_counts("Final samples per label:", df)
    print(f"\nSaved cleaned dataset: {args.output}")
    print(f"Overall kept: {len(df)} / {start}")


if __name__ == "__main__":
    main()
