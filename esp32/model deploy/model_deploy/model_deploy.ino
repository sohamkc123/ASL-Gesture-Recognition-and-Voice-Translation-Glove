#include <Arduino.h>
#include "model_data.h"
#include "normalization.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "DFRobotDFPlayerMini.h"

// ============================================================
// DFPLAYER
// ============================================================

HardwareSerial dfSerial(2);
DFRobotDFPlayerMini dfPlayer;

// ESP32 pins
#define DFPLAYER_RX 16   // ESP32 receives from DFPlayer TX
#define DFPLAYER_TX 17   // ESP32 sends to DFPlayer RX


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
    Serial.println(" ASL TINYML ESP32");
    Serial.println(" DFPLAYER VOICE OUTPUT");
    Serial.println("================================");


    // --------------------------------------------------------
    // Start DFPlayer
    // --------------------------------------------------------

    dfSerial.begin(
        9600,
        SERIAL_8N1,
        DFPLAYER_RX,
        DFPLAYER_TX
    );

    delay(1000);

    Serial.println("Initializing DFPlayer...");

    if (!dfPlayer.begin(dfSerial))
    {
        Serial.println("ERROR: DFPlayer not detected!");
        Serial.println("Check RX, TX, VCC and GND.");

        while (1)
        {
            delay(1000);
        }
    }

    Serial.println("DFPlayer initialized.");


    // Volume: 0-30
    dfPlayer.volume(25);

    delay(500);


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
    // Get tensors
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
        "Input bytes: "
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
        "Output bytes: "
    );

    Serial.println(
        output->bytes
    );


    Serial.println();
    Serial.println("Model + DFPlayer ready!");

    delay(1000);
}


// ============================================================
// LOOP
// ============================================================

void loop()
{
    delay(3000);


    // --------------------------------------------------------
    // Dummy input
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
    // Convert class to letter
    // --------------------------------------------------------

    if (best_index < 0 || best_index > 25)
    {
        Serial.println("Invalid class!");
        return;
    }


    char letter =
        'A' + best_index;


    // --------------------------------------------------------
    // PLAY AUDIO
    // --------------------------------------------------------

    // A = 1
    // B = 2
    // C = 3
    // ...
    // Z = 26

    int trackNumber =
        best_index + 1;


    Serial.print("Playing: ");
    Serial.println(letter);


    dfPlayer.play(trackNumber);


    delay(2500);


    Serial.println(
        "------------------------------"
    );
}