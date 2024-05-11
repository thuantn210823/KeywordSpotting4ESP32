//#include <eloquent_tensorflow32.h>
//#include <ESP_TF.h>
#include <TFLite_Micro.h>
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_op_resolver.h"
//#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
//#include "tensorflow/lite/micro/system_setup.h"
//#include "tensorflow/lite/version.h"
#include "cls_2.h"

#define TOUCH_PIN T0
#define LED_PIN 2

namespace
{
  tflite::ErrorReporter *error_reporter = nullptr;
  const tflite::Model *model = nullptr;
  tflite::MicroInterpreter *interpreter = nullptr;
  TfLiteTensor *input = nullptr;
  TfLiteTensor *output = nullptr;
  // An area of memory to use for input, output, and intermediate arrays.
  constexpr int kTensorArenaSize = 2*1024;
  uint8_t tensor_arena[kTensorArenaSize];
  using AllOpsResolver = tflite::MicroMutableOpResolver<4>;
TfLiteStatus RegisterOps(AllOpsResolver& resolver) {
  TF_LITE_ENSURE_STATUS(resolver.AddFullyConnected());
  TF_LITE_ENSURE_STATUS(resolver.AddSoftmax());
  TF_LITE_ENSURE_STATUS(resolver.AddRelu());
  return kTfLiteOk;
}
} 


const int doRoi = 40;
int touchValue;
bool ledSTT;

void setup() {
  // put your code here, to run once:
  Serial.begin(115200);
  pinMode(LED_PIN, OUTPUT);

  // Defines an micro_error_reporter instance
  static tflite::MicroErrorReporter micro_error_reporter;
  // Creates an error_reporter pointer and assigns it to the mirco_error_reporter
  error_reporter = &micro_error_reporter;

  model = tflite::GetModel(_content_lr_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION)
  {
      error_reporter->Report(
        "Model provided is schma version %d not equal"
        "to supported version %d.",
        model->version(), TFLITE_SCHEMA_VERSION);
        return;
  }

  AllOpsResolver resolver;
  RegisterOps(resolver);
  
  tflite::MicroInterpreter static_interpreter(
    model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;


  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    error_reporter->Report("AllocateTensors() failed");
    return;
  }

  input = interpreter->input(0);
  output = interpreter->output(0);

  Serial.print("setup complete");
}


void loop() {
  // put your main code here, to run repeatedly:
  touchValue = touchRead(TOUCH_PIN);
  if (touchValue < doRoi)
  {
    ledSTT = !ledSTT;
    digitalWrite(LED_PIN, ledSTT);
    float x_val = (float)touchValue;
    //uint8_t x_val = (uint8_t)((float)touchValue/0.03886239 + 0);
    // Place your input
    input->data.f[0] = x_val;
    // Run inference, and report any error
    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
      error_reporter->Report("Invoke failed on x_val: %f\n",
                           static_cast<double>(x_val));
      return;}
    // Read the predicted y value from the model's output tensor
    uint8_t y_val = argmax(output, 3);
    Serial.println((String)touchValue + " _ class: " + (String)y_val + " prob: " + (String)(output->data.f[y_val]));
  }
}


uint8_t argmax(TfLiteTensor *p, uint8_t num_classes)
{
    float max = 0;
    uint8_t idx;
    for(uint8_t i = 0; i < num_classes; i++)
    {
        if (max < p->data.f[i])
        {
            max = p->data.f[i];
            idx = i;
        }
    }
    return idx;
}
