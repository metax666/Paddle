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

#include "paddle/common/macros.h"
#include <c10/data_type.h>

namespace c10 {
#define FOREACH_PADDLE_AND_TORCH_DTYPES(_)            \
  _(uint8_t, UINT8, Byte)                             \
  _(int8_t, INT8, Char)                               \
  _(int16_t, INT16, Short)                            \
  _(int32_t, INT32, Int)                              \
  _(int64_t, INT64, Long)                             \
  _(at::Half, FLOAT16, Half)                          \
  _(float, FLOAT32, Float)                            \
  _(double, FLOAT64, Double)                          \
  _(c10::complex<float>, COMPLEX64, ComplexFloat)     \
  _(c10::complex<double>, COMPLEX128, ComplexDouble)  \
  _(bool, BOOL, Bool)                                 \
  _(at::BFloat16, BFLOAT16, BFloat16)                 \
  _(c10::Float8_e5m2, FLOAT8_E5M2, Float8_e5m2)       \
  _(c10::Float8_e4m3fn, FLOAT8_E4M3FN, Float8_e4m3fn) \
  _(uint16_t, UINT16, UInt16)                         \
  _(uint32_t, UINT32, UInt32)

enum class PADDLE_API ScalarType : int8_t {
#define DEFINE_ST_ENUM_VAL_(_1, _2, n) n,
  FOREACH_PADDLE_AND_TORCH_DTYPES(DEFINE_ST_ENUM_VAL_)
#undef DEFINE_ENUM_ST_ENUM_VAL_
      Undefined,
  NumOptions
};

#define DEFINE_CONSTANT(_1, _2, name) \
  constexpr ScalarType k##name = ScalarType::name;

FOREACH_PADDLE_AND_TORCH_DTYPES(DEFINE_CONSTANT)
#undef DEFINE_CONSTANT

}  // namespace c10
