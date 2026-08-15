import serial
import csv
import os
import time

# =====================================================
# SETTINGS
# =====================================================

PORT = "COM13"       # CHANGE THIS to your ESP32 port
BAUDRATE = 115200

SAMPLES_PER_LETTER = 200

OUTPUT_FILE = "asl_dataset.csv"


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
    "GyroZ"
]


# =====================================================
# CONNECT TO ESP32
# =====================================================

print("Connecting to ESP32...")

ser = serial.Serial(
    PORT,
    BAUDRATE,
    timeout=1
)

time.sleep(2)

print("Connected!")
print()


# =====================================================
# CREATE CSV
# =====================================================

file_exists = os.path.exists(OUTPUT_FILE)

csv_file = open(
    OUTPUT_FILE,
    "a",
    newline=""
)

writer = csv.writer(csv_file)

if not file_exists:
    writer.writerow(HEADER)


# =====================================================
# MAIN DATA COLLECTION
# =====================================================

try:

    print("==========================================")
    print("       ASL DATASET COLLECTION")
    print("==========================================")
    print()
    print("Enter a letter A-Z to collect data.")
    print("Enter Q to quit.")
    print()

    while True:

        label = input("Enter letter: ").strip().upper()

        if label == "Q":
            break

        if len(label) != 1 or label not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            print("Please enter ONE letter from A-Z.")
            continue


        print()
        print("------------------------------------------")
        print("Collecting:", label)
        print("Samples:", SAMPLES_PER_LETTER)
        print("------------------------------------------")

        input("Place your hand in the ASL position, then press ENTER...")


        # -------------------------------------------------
        # Clear old serial data
        # -------------------------------------------------

        ser.reset_input_buffer()


        count = 0

        while count < SAMPLES_PER_LETTER:

            line = ser.readline().decode(
                "utf-8",
                errors="ignore"
            ).strip()


            if not line:
                continue


            # Ignore ESP32 status messages
            if line.startswith("#"):
                continue


            data = line.split(",")


            # Arduino should send:
            # Mode + 17 sensor values = 18 values

            if len(data) != 18:
                continue


            # -------------------------------------------------
            # Add LABEL to beginning
            # -------------------------------------------------

            row = [label] + data


            # -------------------------------------------------
            # Save
            # -------------------------------------------------

            writer.writerow(row)

            csv_file.flush()

            count += 1


            print(
                f"\r{label}: {count}/{SAMPLES_PER_LETTER}",
                end=""
            )


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