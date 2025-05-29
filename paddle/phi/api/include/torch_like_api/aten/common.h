// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "paddle/phi/api/include/torch_like_api/c10/array_ref.h"
#include "paddle/phi/api/include/torch_like_api/c10/data_type.h"
#include "paddle/phi/api/include/torch_like_api/c10/device.h"
#include "paddle/phi/api/include/torch_like_api/c10/exception.h"
#include "paddle/phi/api/include/torch_like_api/c10/memory_format.h"
#include "paddle/phi/api/include/torch_like_api/c10/scalar_type.h"
#include "paddle/phi/api/include/torch_like_api/c10/tensor_options.h"
#include "paddle/phi/common/scalar.h"

namespace at {

// TensorOptions
using c10::TensorOptions;

// DataType
using Half = c10::Half;
using BFloat16 = c10::BFloat16;

// ScalarType
using c10::ScalarType;

#define REDEFINE_CONSTANT_IN_AT(_1, _2, name) \
  constexpr ScalarType k##name = c10::k##name;
FOREACH_PADDLE_AND_TORCH_DTYPES(REDEFINE_CONSTANT_IN_AT)
#undef REDEFINE_CONSTANT_IN_AT

// IntArrayRef
using c10::IntArrayRef;

// MemoryFormat
using c10::MemoryFormat;

// Scalar
using Scalar = paddle::experimental::Scalar;

// Device
using c10::Device;
using c10::DeviceType;
constexpr c10::DeviceType kCPU = c10::kCPU;
constexpr c10::DeviceType kCUDA = c10::kCUDA;
constexpr c10::DeviceType kCUSTOM = c10::kCUSTOM;  // Paddle only

struct Layout {};
}  // namespace at
