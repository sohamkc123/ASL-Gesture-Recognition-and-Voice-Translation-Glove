"""
calibrate.py
------------
Run this before EVERY session (both sessions today, and every day after).
Records this session's flex sensor baseline so flex normalization stays
consistent even as the raw ADC values drift with temperature/fit/mounting.

Usage:
    python calibrate.py

Produces: ../calibration/calibration_<timestamp>.json
Pass this file's path to collect_data.py for this same session.
"""

import serial
import time
import json
import os
import statistics

PORT = "COM5"   # <-- set to your ESP32 port
BAUD = 115200

CALIB_DIR = os.path.join(os.path.dirname(__file__), "..", "calibration")


def read_samples(ser, duration_s):
    samples = []
    t0 = time.time()
    while time.time() - t0 < duration_s:
        line = ser.readline().decode(errors="ignore").strip()
        if not line or line.startswith("flex_pinky"):
            continue
        parts = line.split(",")
        if len(parts) != 17:
            continue
        try:
            samples.append([float(x) for x in parts])
        except ValueError:
            continue
    return samples


def main():
    os.makedirs(CALIB_DIR, exist_ok=True)

    ser = serial.Serial(PORT, BAUD, timeout=1)
    time.sleep(2)
    ser.reset_input_buffer()

    input("Hold your hand FLAT, all fingers fully straight, then press Enter...")
    print("Recording flat pose for 3 seconds, hold still...")
    flat_samples = read_samples(ser, 3)

    input("Now make a FULL FIST, all fingers fully bent, then press Enter...")
    print("Recording fist pose for 3 seconds, hold still...")
    fist_samples = read_samples(ser, 3)

    if len(flat_samples) < 5 or len(fist_samples) < 5:
        print("Not enough samples captured - check the serial connection and retry.")
        return

    # First 5 columns are the flex sensors, in this order:
    # pinky, ring, middle, index, thumb
    flex_flat = [statistics.median(s[i] for s in flat_samples) for i in range(5)]
    flex_fist = [statistics.median(s[i] for s in fist_samples) for i in range(5)]

    calib = {
        "flex_order": ["pinky", "ring", "middle", "index", "thumb"],
        "flex_flat": flex_flat,
        "flex_fist": flex_fist,
        "timestamp": time.strftime("%Y-%m-%d_%H-%M-%S"),
    }

    fname = os.path.join(CALIB_DIR, f"calibration_{calib['timestamp']}.json")
    with open(fname, "w") as f:
        json.dump(calib, f, indent=2)

    print(f"\nSaved calibration to {fname}")
    print("flat :", [round(v, 1) for v in flex_flat])
    print("fist :", [round(v, 1) for v in flex_fist])
    print("\nUse this exact file with collect_data.py for this session.")


if __name__ == "__main__":
    main()