"""
generate_calibration_header.py
--------------------------------
Converts a calibration_<timestamp>.json (from calibrate.py) into
calibration.h for the ESP32 deploy sketch.

Run this right before your final flash of the day, using whichever
calibration file is freshest (ideally: recalibrate one more time with the
glove in the state you're about to test it in, then generate from that).

Usage:
    python generate_calibration_header.py ../calibration/calibration_2026-08-22_18-40-00.json
"""

import json
import sys
import os

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "esp32", "model_deploy")


def main():
    if len(sys.argv) < 2:
        print("Usage: python generate_calibration_header.py <calibration_file.json>")
        sys.exit(1)

    with open(sys.argv[1]) as f:
        calib = json.load(f)

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "calibration.h")

    with open(out_path, "w") as f:
        f.write("#ifndef CALIBRATION_H\n#define CALIBRATION_H\n\n")
        f.write("// Generated from " + os.path.basename(sys.argv[1]) + "\n")
        f.write("// Order: pinky, ring, middle, index, thumb (matches FEATURE_NAMES)\n\n")
        f.write("const float flex_flat[5] = {" +
                ", ".join(f"{v:.2f}f" for v in calib["flex_flat"]) + "};\n")
        f.write("const float flex_fist[5] = {" +
                ", ".join(f"{v:.2f}f" for v in calib["flex_fist"]) + "};\n\n")
        f.write("#endif\n")

    print(f"Saved {out_path}")
    print("Copy/keep this in esp32/model_deploy/ alongside model_data.h before flashing.")


if __name__ == "__main__":
    main()