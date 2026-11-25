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

// Modified from the TensorFlow Lite Micro example for the ECG model.

#ifndef ECG_MODEL_H
#define ECG_MODEL_H

// TensorFlow Lite FlatBuffer stored as an aligned array in model.cc.
extern const unsigned char g_model[];
extern const int g_model_len;

#endif  // ECG_MODEL_H
