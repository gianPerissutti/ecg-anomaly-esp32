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

// Modified from the TensorFlow Lite Micro example for ECG inference.

#include "neural_network.h"

#include <cmath>
#include <cstdint>
#include <limits>

#include "model.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace {

constexpr int kTensorArenaSize = 40000;
constexpr float kMinInputMillivolts = -5.12f;
constexpr float kMaxInputMillivolts = 5.12f;
constexpr float kInvalidError = std::numeric_limits<float>::quiet_NaN();

alignas(16) std::uint8_t tensor_arena[kTensorArenaSize];
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* model_input = nullptr;
TfLiteTensor* model_output = nullptr;

float normalize(float sample)
{
    return (sample - kMinInputMillivolts) / (kMaxInputMillivolts - kMinInputMillivolts);
}

bool is_ecg_tensor(const TfLiteTensor* tensor)
{
    return tensor != nullptr && tensor->type == kTfLiteFloat32 &&
           tensor->bytes == ecg::kWindowSamples * sizeof(float) && tensor->data.f != nullptr;
}

}  // namespace

namespace ecg {

bool initialize_model()
{
    if (interpreter != nullptr) {
        return true;
    }

    const auto* model = tflite::GetModel(g_model);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        MicroPrintf("Esquema del modelo %u, esperado %u", static_cast<unsigned>(model->version()),
                    static_cast<unsigned>(TFLITE_SCHEMA_VERSION));
        return false;
    }

    static tflite::MicroMutableOpResolver<11> resolver;
    static const bool operators_registered =
        resolver.AddFullyConnected() == kTfLiteOk &&
        resolver.AddConv2D() == kTfLiteOk &&
        resolver.AddReshape() == kTfLiteOk &&
        resolver.AddTransposeConv() == kTfLiteOk &&
        resolver.AddQuantize() == kTfLiteOk &&
        resolver.AddDequantize() == kTfLiteOk &&
        resolver.AddExpandDims() == kTfLiteOk &&
        resolver.AddShape() == kTfLiteOk &&
        resolver.AddStridedSlice() == kTfLiteOk &&
        resolver.AddPack() == kTfLiteOk &&
        resolver.AddLogistic() == kTfLiteOk;
    if (!operators_registered) {
        MicroPrintf("No se pudieron registrar los operadores del modelo");
        return false;
    }

    static tflite::MicroInterpreter model_interpreter(model, resolver, tensor_arena,
                                                        sizeof(tensor_arena));
    if (model_interpreter.AllocateTensors() != kTfLiteOk) {
        MicroPrintf("No se pudieron reservar los tensores del modelo");
        return false;
    }

    auto* input = model_interpreter.input(0);
    auto* output = model_interpreter.output(0);
    if (!is_ecg_tensor(input) || !is_ecg_tensor(output)) {
        MicroPrintf("El modelo requiere entrada y salida float32 de %d muestras", kWindowSamples);
        return false;
    }

    model_input = input;
    model_output = output;
    interpreter = &model_interpreter;
    return true;
}

float reconstruction_error(const float* samples)
{
    if (interpreter == nullptr) {
        return kInvalidError;
    }

    for (int i = 0; i < kWindowSamples; ++i) {
        if (!std::isfinite(samples[i])) {
            return kInvalidError;
        }
        model_input->data.f[i] = normalize(samples[i]);
    }

    if (interpreter->Invoke() != kTfLiteOk) {
        MicroPrintf("Falló la inferencia del modelo ECG");
        return kInvalidError;
    }

    float absolute_error = 0.0f;
    for (int i = 0; i < kWindowSamples; ++i) {
        const float diff = model_output->data.f[i] - normalize(samples[i]);
        absolute_error += std::fabs(diff);
    }
    return std::isfinite(absolute_error) ? absolute_error / kWindowSamples : kInvalidError;
}

}  // namespace ecg
