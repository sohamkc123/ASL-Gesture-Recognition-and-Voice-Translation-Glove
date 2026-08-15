import csv
import glob
import os
import time

import serial

# =====================================================
# SETTINGS
# =====================================================

BAUDRATE = 115200
SAMPLES_PER_LETTER = 200
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "asl_dataset.csv")


# =====================================================
# CSV HEADER
# =====================================================

HEADER = [
    "Label",
    "Mode",
    "FlexPinky",
    "FlexRing",
    "FlexMiddle",
    "FlexIndex",
    "FlexThumb",
    "TouchIndex",
    "TouchMiddle",
    "TouchRing",
    "TouchPinky",
    "TouchR",
    "TouchU",
    "AccX",
    "AccY",
    "AccZ",
    "GyroX",
    "GyroY",
    "GyroZ",
]


def find_esp32_port(port_candidates=None):
    """Find the first likely ESP32 serial port."""
    if port_candidates is None:
        port_candidates = []
        try:
            port_candidates.extend(port.device for port in serial.tools.list_ports.comports())
        except Exception:
            pass

        for pattern in [
            "/dev/ttyUSB*",
            "/dev/ttyACM*",
            "/dev/cu.usbserial*",
            "/dev/tty.usbserial*",
            "/dev/cu.SLAB_USBtoUART",
        ]:
            port_candidates.extend(glob.glob(pattern))

    if not port_candidates:
        return None

    preferred = []
    for port in port_candidates:
        name = os.path.basename(port)
        if any(token in name for token in ["ttyACM", "ttyUSB", "usbserial", "SLAB_USBtoUART"]):
            preferred.append(port)

    if not preferred:
        preferred = list(port_candidates)

    preferred.sort(key=lambda p: (0 if "ttyACM" in os.path.basename(p) else 1, p))
    return preferred[0]


def is_valid_sensor_line(line):
    if not line or line.startswith("#"):
        return False

    if line.startswith("Mode,"):
        return False

    data = [item.strip() for item in line.split(",")]

    if len(data) != 18:
        return False

    return all(part != "" for part in data)


def connect_to_esp32(port=None):
    port = port or os.environ.get("ESP32_PORT") or find_esp32_port()

    if not port:
        raise RuntimeError(
            "No ESP32 serial port found.\n"
            "1) Plug in the ESP32\n"
            "2) Check the port in Arduino IDE -> Tools -> Port\n"
            "3) Or set ESP32_PORT=/dev/ttyUSB0 before running this script"
        )

    print(f"Connecting to ESP32 on {port}...")

    try:
        ser = serial.Serial(port, BAUDRATE, timeout=1)
    except serial.SerialException as exc:
        message = str(exc)
        if "Permission denied" in message or "Access is denied" in message:
            raise RuntimeError(
                f"Permission denied for {port}.\n"
                "Run this once in a terminal: sudo usermod -a -G dialout $USER\n"
                "Then log out and log back in, or run: newgrp dialout"
            ) from exc
        raise

    time.sleep(2)
    print("Connected!")
    print()
    return ser


def main():
    port = os.environ.get("ESP32_PORT") or find_esp32_port()

    ser = connect_to_esp32(port)

    # =====================================================
    # CREATE CSV
    # =====================================================

    file_exists = os.path.exists(OUTPUT_FILE)

    csv_file = open(OUTPUT_FILE, "a", newline="")
    writer = csv.writer(csv_file)

    if not file_exists:
        writer.writerow(HEADER)

    try:
        print("==========================================")
        print("       ASL DATASET COLLECTION")
        print("==========================================")
        print()
        print("Enter a letter A-Z to collect data.")
        print("Press Ctrl+C to quit.")
        print()

        while True:
            label = input("Enter letter: ").strip().upper()

            if len(label) != 1 or label not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                print("Please enter ONE letter from A-Z.")
                continue

            print()
            print("------------------------------------------")
            print("Collecting:", label)
            print("Samples:", SAMPLES_PER_LETTER)
            print("------------------------------------------")

            input("Place your hand in the ASL position, then press ENTER...")
            ser.reset_input_buffer()

            count = 0
            while count < SAMPLES_PER_LETTER:
                line = ser.readline().decode("utf-8", errors="ignore").strip()
                if not line:
                    continue

                if not is_valid_sensor_line(line):
                    continue

                data = line.split(",")
                row = [label] + data

                writer.writerow(row)
                csv_file.flush()
                count += 1
                print(f"\r{label}: {count}/{SAMPLES_PER_LETTER}", end="")

            print()
            print(f"Finished collecting {label}!")
            print()

    except KeyboardInterrupt:
        print()
        print("Collection stopped by user.")

    finally:
        csv_file.close()
        ser.close()

        print()
        print("CSV file saved:")
        print(OUTPUT_FILE)


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, serial.SerialException) as exc:
        print(f"\nError: {exc}\n")
        raise SystemExit(1)
