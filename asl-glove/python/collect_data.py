"""
collect_data.py
----------------
Creates ONE CSV per session (all 26 letters go in the same file for that
session), saved to data/<date>/session_<timestamp>.csv. Run it twice today
and you'll get two files under data/2026-08-22/.

Normalization is applied HERE at collection time, and is fully
deterministic - no dataset statistics involved:
  - flex: (raw - flat) / (fist - flat), clamped to [0, 1], using THIS
    session's calibration.json
  - touch: passed through as-is (1 = not touched, 0 = touched)
  - accel/gyro: raw_int16 / 32768.0, giving roughly [-1, 1]

Because this exact transform is also what model_deploy.ino applies live,
train/deploy can never drift apart the way the old StandardScaler-in-header
approach could.

Usage:
    python collect_data.py ../calibration/calibration_2026-08-22_09-15-00.json
"""

import serial
import time
import json
import sys
import csv
import os
import math
from datetime import date

PORT = "COM10"
BAUD = 115200
REPS_PER_LETTER = 5
SAMPLES_PER_REP = 40
SETTLE_SAMPLES = 6   # discard first ~0.3s of each rep while hand settles

LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

FEATURE_NAMES = [
    "flex_pinky", "flex_ring", "flex_middle", "flex_index", "flex_thumb",
    "touch_index", "touch_middle", "touch_ring", "touch_pinky", "touch_r", "touch_u",
    "ax", "ay", "az", "gx", "gy", "gz",
]

DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "data")


def normalize_flex(raw, flat, fist):
    if fist == flat:
        return 0.5
    val = (raw - flat) / (fist - flat)
    return max(0.0, min(1.0, val))


def read_line(ser):
    line = ser.readline().decode(errors="ignore").strip()
    if not line or line.startswith("flex_pinky"):
        return None
    parts = line.split(",")
    if len(parts) != 17:
        return None

    try:
        vals = [float(x) for x in parts]
    except ValueError:
        return None

    # Reject malformed numeric frames early.
    if not all(math.isfinite(v) for v in vals):
        return None

    accel_gyro_raw = vals[11:17]

    # NOTE: normalized flex can validly be 0.0, so we do NOT reject flex==0 here.
    # Reject only impossible/corrupted MPU frames.
    if all(v == 0 for v in accel_gyro_raw):
        return None

    return vals


def process_sample(raw, calib):
    flex_raw = raw[0:5]
    touch = raw[5:11]
    accel_gyro_raw = raw[11:17]
    flex_norm = [
        normalize_flex(flex_raw[i], calib["flex_flat"][i], calib["flex_fist"][i])
        for i in range(5)
    ]
    accel_gyro_norm = [v / 32768.0 for v in accel_gyro_raw]
    return flex_norm + touch + accel_gyro_norm


def collect_letter(ser, calib, letter, rep_ids):
    rows = []
    total = len(rep_ids)
    for idx, rep in enumerate(rep_ids, start=1):
        input(f"[{letter}] rep {idx}/{total}: make the sign, hold steady, "
              f"press Enter to start recording...")
        ser.reset_input_buffer()
        collected = 0
        while collected < SAMPLES_PER_REP + SETTLE_SAMPLES:
            raw = read_line(ser)
            if raw is None:
                continue
            collected += 1
            if collected <= SETTLE_SAMPLES:
                continue
            feat = process_sample(raw, calib)
            rows.append(feat + [letter, rep])
        print(f"  captured {collected - SETTLE_SAMPLES} samples")
    return rows


def main():
    if len(sys.argv) < 2:
        print("Usage: python collect_data.py <calibration_file.json>")
        sys.exit(1)

    with open(sys.argv[1]) as f:
        calib = json.load(f)

    session_id = calib["timestamp"]
    day_dir = os.path.join(DATA_ROOT, str(date.today()))
    os.makedirs(day_dir, exist_ok=True)
    out_path = os.path.join(day_dir, f"session_{session_id}.csv")

    # Resume support: if file exists for this session id, append instead of overwrite.
    existing_progress = {letter: set() for letter in LETTERS}
    existing_max_rep = {letter: -1 for letter in LETTERS}
    file_exists = os.path.exists(out_path)
    has_data = file_exists and os.path.getsize(out_path) > 0

    if has_data:
        with open(out_path, "r", newline="") as rf:
            reader = csv.DictReader(rf)
            for row in reader:
                lbl = str(row.get("label", "")).strip().upper()
                rep_txt = str(row.get("rep", "")).strip()
                if lbl in LETTERS:
                    try:
                        rep = int(float(rep_txt))
                        if rep >= 0:
                            existing_max_rep[lbl] = max(existing_max_rep[lbl], rep)
                        if 0 <= rep < REPS_PER_LETTER:
                            existing_progress[lbl].add(rep)
                    except ValueError:
                        pass

    ser = serial.Serial(PORT, BAUD, timeout=1)
    time.sleep(2)
    ser.reset_input_buffer()

    mode = "a" if has_data else "w"
    with open(out_path, mode, newline="") as f:
        writer = csv.writer(f)
        if not has_data:
            writer.writerow(FEATURE_NAMES + ["label", "rep", "session", "day"])
            print(f"Session file (new): {out_path}")
        else:
            print(f"Session file (resume): {out_path}")
            print("Already recorded reps per letter:")
            for letter in LETTERS:
                done = sorted(existing_progress[letter])
                if done:
                    print(f"  {letter}: {len(done)}/{REPS_PER_LETTER} reps done")

        print("Enter a letter A-Z to collect, or 'done' to finish this session.")
        print("If a letter is complete, you can still retake it and add 5 new reps.\n")

        while True:
            letter = input("Letter (or 'done'): ").strip().upper()
            if letter == "DONE":
                break
            if letter not in LETTERS:
                print("Enter one letter A-Z, or 'done'.")
                continue

            done_reps = existing_progress.get(letter, set())

            if len(done_reps) >= REPS_PER_LETTER:
                ans = input(f"{letter} already complete. Add 5 new retake reps anyway? (y/n): ").strip().lower()
                if ans not in ("y", "yes"):
                    continue
                start_rep = existing_max_rep[letter] + 1
                rep_ids = list(range(start_rep, start_rep + REPS_PER_LETTER))
                print(f"Adding retake block for {letter} with rep ids {rep_ids[0]}..{rep_ids[-1]}")
            else:
                rep_ids = [r for r in range(REPS_PER_LETTER) if r not in done_reps]

            rows = collect_letter(ser, calib, letter, rep_ids=rep_ids)
            for r in rows:
                writer.writerow(r + [session_id, str(date.today())])
                rep_val = int(r[-1])
                existing_max_rep[letter] = max(existing_max_rep[letter], rep_val)
                if 0 <= rep_val < REPS_PER_LETTER:
                    existing_progress[letter].add(rep_val)
            f.flush()
            print(f"Finished {letter}\n")

    print(f"\nSession saved: {out_path}")


if __name__ == "__main__":
    main()