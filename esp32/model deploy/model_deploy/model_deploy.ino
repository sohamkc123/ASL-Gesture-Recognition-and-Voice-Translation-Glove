#include <Arduino.h>
#include "model_data.h"
#include "normalization.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"


// ============================================================
// MODEL
// ============================================================

const tflite::Model* model;

tflite::MicroInterpreter* interpreter;

TfLiteTensor* input;
TfLiteTensor* output;


// ============================================================
// TENSOR ARENA
// ============================================================

// Increase this if ESP32 gives allocation errors.

constexpr int kTensorArenaSize = 40 * 1024;

uint8_t tensor_arena[kTensorArenaSize];


// ============================================================
// SETUP
// ============================================================

void setup()
{
    Serial.begin(115200);

    delay(2000);

    Serial.println();
    Serial.println("================================");
    Serial.println(" ASL TINYML ESP32 TEST");
    Serial.println("================================");


    // --------------------------------------------------------
    // Load model
    // --------------------------------------------------------

    model = tflite::GetModel(
        asl_model_int8_tflite
    );


    if (model->version() != TFLITE_SCHEMA_VERSION)
    {
        Serial.println(
            "ERROR: Model schema mismatch!"
        );

        while (1);
    }


    Serial.println("Model loaded successfully.");


    // --------------------------------------------------------
    // Register operators
    // --------------------------------------------------------

    static tflite::MicroMutableOpResolver<10> resolver;

    resolver.AddFullyConnected();
    resolver.AddSoftmax();
    resolver.AddRelu();


    // --------------------------------------------------------
    // Create interpreter
    // --------------------------------------------------------

    static tflite::MicroInterpreter static_interpreter(

        model,
        resolver,
        tensor_arena,
        kTensorArenaSize
    );

    interpreter = &static_interpreter;


    // --------------------------------------------------------
    // Allocate tensors
    // --------------------------------------------------------

    TfLiteStatus allocate_status =
        interpreter->AllocateTensors();


    if (allocate_status != kTfLiteOk)
    {
        Serial.println(
            "ERROR: Tensor allocation failed!"
        );

        while (1);
    }


    Serial.println(
        "Tensor allocation successful."
    );


    // --------------------------------------------------------
    // Get input/output tensors
    // --------------------------------------------------------

    input =
        interpreter->input(0);

    output =
        interpreter->output(0);


    Serial.println();

    Serial.print(
        "Input tensor type: "
    );

    Serial.println(
        input->type
    );


    Serial.print(
        "Input elements: "
    );

    Serial.println(
        input->bytes
    );


    Serial.print(
        "Output tensor type: "
    );

    Serial.println(
        output->type
    );


    Serial.print(
        "Output elements: "
    );

    Serial.println(
        output->bytes
    );


    Serial.println();

    Serial.println(
        "Model is ready!"
    );
}


// ============================================================
// LOOP
// ============================================================

void loop()
{
    delay(3000);

    Serial.println(
        "Running test inference..."
    );


    // --------------------------------------------------------
    // Create dummy input
    // --------------------------------------------------------

    for (int i = 0; i < NUM_FEATURES; i++)
    {
        input->data.int8[i] = 0;
    }


    // --------------------------------------------------------
    // Run inference
    // --------------------------------------------------------

    TfLiteStatus status =
        interpreter->Invoke();


    if (status != kTfLiteOk)
    {
        Serial.println(
            "ERROR: Inference failed!"
        );

        return;
    }


    // --------------------------------------------------------
    // Find predicted class
    // --------------------------------------------------------

    int8_t best_value =
        output->data.int8[0];

    int best_index = 0;


    for (int i = 1; i < output->bytes; i++)
    {
        if (
            output->data.int8[i]
            >
            best_value
        )
        {
            best_value =
                output->data.int8[i];

            best_index = i;
        }
    }


    // --------------------------------------------------------
    // Print result
    // --------------------------------------------------------

    Serial.print(
        "Predicted class index: "
    );

    Serial.println(
        best_index
    );


    Serial.print(
        "Predicted letter: "
    );


    // Current classes
    // A = 0
    // B = 1
    // C = 2
    // ...


    Serial.println(
        char('A' + best_index)
    );


    Serial.println(
        "------------------------------"
    );
}