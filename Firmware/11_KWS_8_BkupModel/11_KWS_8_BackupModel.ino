#include <eloquent_tensorflow32.h>
/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "main_functions.h"

#include "audio_provider.h"
#include "command_responder.h"
#include "feature_provider.h"
#include "micro_model_settings.h"
#include "bkup_model2.h"
#include "recognize_commands.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Globals, used for compatibility with Arduino-style sketches.
namespace {
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* model_input = nullptr;
FeatureProvider* feature_provider = nullptr;
RecognizeCommands* recognizer = nullptr;
int32_t previous_time = 0;

// Create an area of memory to use for input, output, and intermediate arrays.
// The size of this will depend on the model you're using, and may need to be
// determined by experimentation.
constexpr int kTensorArenaSize = 50 * 1024;
uint8_t tensor_arena[kTensorArenaSize];
int8_t feature_buffer[kFeatureElementCount];
int8_t* model_input_buffer = nullptr;
}  // namespace
QueueHandle_t xQueueAudioWave;
#define QueueAudioWaveSize 32
// The name of this function is important for Arduino compatibility.
int8_t wuw_flag = 0;
int8_t score;
int8_t pre_output[kCategoryCount*3];
void setup() {
  Serial.begin(115200);
  xQueueAudioWave = xQueueCreate(QueueAudioWaveSize, sizeof(int16_t));

  // Set up logging. Google style is to avoid globals or statics because of
  // lifetime uncertainty, but since this has a trivial destructor it's okay.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  model = tflite::GetModel(bkup_model_conv);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.",
                         model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // Pull in only the operation implementations we need.
  // This relies on a complete list of all the ops needed by this graph.
  // An easier approach is to just use the AllOpsResolver, but this will
  // incur some penalty in code space for op implementations that are not
  // needed by this graph.
  //
  // tflite::AllOpsResolver resolver;
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroMutableOpResolver<10> micro_op_resolver(error_reporter);
  if (micro_op_resolver.AddConv2D() != kTfLiteOk) {
    return;
  }
  if (micro_op_resolver.AddFullyConnected() != kTfLiteOk) {
    return;
  }
  if (micro_op_resolver.AddSoftmax() != kTfLiteOk) {
    return;
  }
  if (micro_op_resolver.AddReshape() != kTfLiteOk) {
    return;
  }
  if (micro_op_resolver.AddQuantize() != kTfLiteOk) {
    return;
  }
  if (micro_op_resolver.AddShape() != kTfLiteOk) {
    return;
  }
  if (micro_op_resolver.AddMean() != kTfLiteOk) {
    return;
  }
  if (micro_op_resolver.AddMaxPool2D() != kTfLiteOk) {
    return;
  }
  // Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(
      model, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
    return;
  }

  // Get information about the memory area to use for the model's input.
  model_input = interpreter->input(0);
  if ((model_input->dims->size != 4) || (model_input->dims->data[0] != 1) ||
      (model_input->dims->data[1] != kFeatureSliceCount) ||
      (model_input->dims->data[2] != kFeatureSliceSize) ||
      (model_input->type != kTfLiteInt8)) {
    error_reporter->Report("Bad input tensor parameters in model");
    return;
  }
  model_input_buffer = model_input->data.int8;

  // Prepare to access the audio spectrograms from a microphone or other source
  // that will provide the inputs to the neural network.
  // NOLINTNEXTLINE(runtime-global-variables)
  static FeatureProvider static_feature_provider(kFeatureElementCount,
                                                 feature_buffer);
  feature_provider = &static_feature_provider;

  static RecognizeCommands static_recognizer(error_reporter);
  recognizer = &static_recognizer;

  previous_time = 0;

  // Define pre output
  for(int i = 0; i<kCategoryCount*3; i++)
  {
      pre_output[i] = 0;
  }
}

// The name of this function is important for Arduino compatibility.
void loop() {
  // Fetch the spectrogram for the current time.
  const int32_t current_time = LatestAudioTimestamp();
  int how_many_new_slices = 0;
  TfLiteStatus feature_status = feature_provider->PopulateFeatureData(
      error_reporter, previous_time, current_time, &how_many_new_slices);
  if (feature_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "Feature generation failed");
    return;
  }
  previous_time = current_time;
  // If no new audio samples have been received since last time, don't bother
  // running the network model.
  if (how_many_new_slices == 0) {
    return;
  }

  // Copy feature buffer to input tensor
  for (int i = 0; i < kFeatureElementCount; i++) {
    model_input_buffer[i] = feature_buffer[i];
  }

  // Run the model on the spectrogram input and make sure it succeeds.
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed");
    return;
  }

  // Obtain a pointer to the output tensor
  TfLiteTensor* output = interpreter->output(0);
  
  // Determine whether a command was recognized based on the output of inference
  //const char* found_command = nullptr;
  //uint8_t score = 0;
  //bool is_new_command = false;
  //TfLiteStatus process_status = recognizer->ProcessLatestResults(
  //    output, current_time, &found_command, &score, &is_new_command);
  //if (process_status != kTfLiteOk) {
  //  TF_LITE_REPORT_ERROR(error_reporter,
  //                       "RecognizeCommands::ProcessLatestResults() failed");
  //  return;
  //}

  // Do something based on the recognized command. The default implementation
  // just prints to the error console, but you should replace this with your
  // own function for a real application.
  //RespondToCommand(error_reporter, current_time, found_command, score,
  //                 is_new_command);
  int8_t score;
  uint8_t y_val = argmax(output->data.int8, pre_output, kCategoryCount, 2, &score);
  //int8_t score = output->data.int8[y_val];
  if (score >40)
  {
      if (y_val == 5) // 'on'
      {
        Serial.println("Already on!");
        wuw_flag = 1;
      }
      if (y_val == 4)
      {
        wuw_flag = 0;
        Serial.println("Off! Goodbye!");
        clear_history(pre_output, kCategoryCount*3);
        delay(1000);
      }
      if (wuw_flag == 1)
      {
          if ((y_val != 0)&&(y_val != 1))
          {
             Serial.println((String)y_val + " score: " + (String)(score) + " Command: " + kCategoryLabels[y_val]);
             clear_history(pre_output, kCategoryCount*3);
             delay(1000);
          }
      }
  }
  //delay(1000);
}

uint8_t argmax(int8_t *input_scores, int8_t *des_scores, uint8_t num_classes, uint8_t num_win, int8_t *score)
{
    int8_t max_value = 0;
    uint8_t idx;
    int8_t mean_score;
    for(int i = 0; i < num_classes; i++)
    {
        mean_score = input_scores[i];
        for(int j = 0; j<num_win; j++)
        {
            mean_score += des_scores[i + j*kCategoryCount];
        }
        mean_score /= num_win;
        if (max_value < mean_score)
        {
            max_value = mean_score;
            idx = i;
        }
        for(int j = 0; j<num_win-1; j++)
        {
            des_scores[i + j*kCategoryCount] = des_scores[i + (j+1)*kCategoryCount];
        }
        des_scores[i + (num_win-1)*kCategoryCount] = input_scores[i];
    }
    *score = max_value;
    return idx;
}
/*uint8_t argmax(int8_t *input_scores,uint8_t num_classes, int8_t *score)
{
    int8_t max_value = 0;
    uint8_t idx;
    for(int i = 0; i < num_classes; i++)
    {
        if (max_value < input_scores[i])
        {
            max_value = input_scores[i];
            idx = i;
        }
    }
    *score = max_value;
    return idx;
}*/
void clear_history(int8_t *p, int8_t len)
{
    for(int i = 0; i< len; i++)
    {
        p[i] = 0;
    }
}
