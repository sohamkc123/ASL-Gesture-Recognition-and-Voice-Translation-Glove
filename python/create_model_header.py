import os

# Path to your TFLite model
MODEL_FILE = r"C:\Users\shrawan prasai\OneDrive\Desktop\ASL-Gesture-Recognition-and-Voice-Translation-Glove\asl_model\asl_model_int8.tflite"

# Output header file
OUTPUT_FILE = "model_data.h"


with open(MODEL_FILE, "rb") as f:
    model_data = f.read()


with open(OUTPUT_FILE, "w") as f:

    f.write("#ifndef MODEL_DATA_H\n")
    f.write("#define MODEL_DATA_H\n\n")

    f.write("#include <stdint.h>\n\n")

    f.write("const unsigned char asl_model_int8_tflite[] = {\n")

    for i, byte in enumerate(model_data):

        if i % 12 == 0:
            f.write("    ")

        f.write(f"0x{byte:02x}")

        if i < len(model_data) - 1:
            f.write(", ")

        if (i + 1) % 12 == 0:
            f.write("\n")

    f.write("\n};\n\n")

    f.write(
        f"const unsigned int asl_model_int8_tflite_len = "
        f"{len(model_data)};\n\n"
    )

    f.write("#endif // MODEL_DATA_H\n")


print("======================================")
print(" Model conversion complete")
print("======================================")
print()
print("Input :", MODEL_FILE)
print("Output:", OUTPUT_FILE)
print("Size  :", len(model_data), "bytes")
print()
print("model_data.h created successfully!")