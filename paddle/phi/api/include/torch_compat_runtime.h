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

/**
 * torch compat runtime (ptrt)
 */

#pragma once

#include <iostream>
#include <optional>
#include <variant>
#include "paddle/common/macros.h"
#include "paddle/phi/api/include/api.h"
#include "paddle/phi/api/include/tensor.h"
#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/common/complex.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/common/float8_e4m3fn.h"
#include "paddle/phi/common/float8_e5m2.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/ddim.h"

namespace logging {
#define UNSUPPORTED_FEATURE_IN_PADDLE(feature)                            \
  std::cerr << "Unsupported feature in Paddle: " << feature << std::endl; \
  std::abort();
}  // namespace logging

namespace c10 {
template <typename T>
using complex = ::phi::dtype::complex<T>;
using Half = ::phi::dtype::float16;
using Float8_e5m2 = ::phi::dtype::float8_e5m2;
using Float8_e4m3fn = ::phi::dtype::float8_e4m3fn;
using BFloat16 = ::phi::dtype::bfloat16;
// using
}  // namespace c10

namespace at {

using Scalar = paddle::experimental::Scalar;
struct Device {};
struct Layout {};

using PaddleTensor = paddle::Tensor;

// IntArrayRef
template <typename T>
class ArrayRef {
 private:
  /// The start of the array, in an external buffer.
  const T* Data;

  /// The number of elements.
  size_t Length;

 public:
  /// Construct an empty ArrayRef.
  /* implicit */ constexpr ArrayRef() : Data(nullptr), Length(0) {}

  constexpr ArrayRef(const T& OneElt) : Data(&OneElt), Length(1) {}  // NOLINT

  /// Construct an ArrayRef from a pointer and length.
  constexpr ArrayRef(const T* data, size_t length)
      : Data(data), Length(length) {}

  /// Construct an ArrayRef from a range.
  constexpr ArrayRef(const T* begin, const T* end)
      : Data(begin), Length(end - begin) {}

  /* implicit */ constexpr ArrayRef(const std::initializer_list<T>& Vec)
      : Data(std::begin(Vec) == std::end(Vec) ? static_cast<T*>(nullptr)
                                              : std::begin(Vec)),
        Length(Vec.size()) {}

  const paddle::IntArray _PD_ToPaddleIntArray() const {
    return paddle::IntArray(Data, Length);
  }
};
using IntArrayRef = ArrayRef<int64_t>;

enum class PADDLE_API MemoryFormat : int8_t {
  Contiguous,
  Preserve,
  ChannelsLast,
  ChannelsLast3d,
  NumOptions
};

// Datatype
using Half = c10::Half;
using BFloat16 = c10::BFloat16;

// ScalarType
#define FORALL_PADDLE_AND_TORCH_DTYPES(_)             \
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
  FORALL_PADDLE_AND_TORCH_DTYPES(DEFINE_ST_ENUM_VAL_)
#undef DEFINE_ENUM_ST_ENUM_VAL_
      Undefined,
  NumOptions
};

struct PADDLE_API TensorOptions {
  TensorOptions()
      : requires_grad_(false),
        pinned_memory_(false),
        has_device_(false),
        has_dtype_(false),
        has_layout_(false),
        has_requires_grad_(false),
        has_pinned_memory_(false),
        has_memory_format_(false) {}

  ScalarType _PD_GetScalarType() const { return dtype; }
  ::phi::Place _PD_GetPlace() const { return place; }

 private:
  ScalarType dtype = ScalarType::Float;
  ::phi::Place place = ::phi::CPUPlace();
  bool requires_grad_ : 1;
  bool pinned_memory_ : 1;

  bool has_device_ : 1;
  bool has_dtype_ : 1;
  bool has_layout_ : 1;
  bool has_requires_grad_ : 1;
  bool has_pinned_memory_ : 1;
  bool has_memory_format_ : 1;
};

namespace conversion {
inline IntArrayRef _PD_PhiDDimToIntArrayRef(const phi::DDim& ddim) {
  return IntArrayRef(ddim.Get(), ddim.size());
}

inline phi::DataType _PD_AtenScalarTypeToPhiDataType(ScalarType dtype) {
  switch (dtype) {
#define DEFINE_ST_TO_DT_CASE_(_1, _dt, _st) \
  case ScalarType::_st:                     \
    return phi::DataType::_dt;
    FORALL_PADDLE_AND_TORCH_DTYPES(DEFINE_ST_TO_DT_CASE_)
#undef DEFINE_ST_TO_DT_CASE_
    default:
      UNSUPPORTED_FEATURE_IN_PADDLE("Unsupported ScalarType")
  }
}  // namespace conversion

}  // namespace conversion
class PADDLE_API Tensor {
 public:
  Tensor(const PaddleTensor& tensor) : tensor_(tensor){};  // NOLINT

  void* data_ptr() { return tensor_.data(); }
  template <typename T>
  T* data_ptr() const {
    return const_cast<T*>(tensor_.data<T>());
  }
  int64_t stride(int64_t dim) const {
    return tensor_.strides()[static_cast<int>(dim)];
  }
  IntArrayRef strides() const {
    return conversion::_PD_PhiDDimToIntArrayRef(tensor_.strides());
  }

  int64_t size(int64_t dim) const {
    return tensor_.dims()[static_cast<int>(dim)];
  }

  IntArrayRef sizes() const {
    return conversion::_PD_PhiDDimToIntArrayRef(tensor_.dims());
  }

  int64_t numel() const { return tensor_.numel(); }

  at::Tensor contiguous(
      MemoryFormat memory_format = MemoryFormat::Contiguous) const {
    if (memory_format != MemoryFormat::Contiguous) {
      UNSUPPORTED_FEATURE_IN_PADDLE("`MemoryFormat` other than Contiguous")
    }
    return tensor_.contiguous();
  }

  TensorOptions options() const {
    // TODO(SigureMo): Implement this
    return TensorOptions();
  }

  PaddleTensor _PD_GetInner() const { return tensor_; }

 private:
  PaddleTensor tensor_;
};

at::Tensor empty(
    at::IntArrayRef size,
    at::TensorOptions options = {},
    ::std::optional<at::MemoryFormat> memory_format = ::std::nullopt) {
  if (memory_format.has_value()) {
    UNSUPPORTED_FEATURE_IN_PADDLE("`MemoryFormat`")
  }
  return paddle::experimental::empty(
      size._PD_ToPaddleIntArray(),
      conversion::_PD_AtenScalarTypeToPhiDataType(options._PD_GetScalarType()),
      options._PD_GetPlace());
}
at::Tensor ones(at::IntArrayRef size, at::TensorOptions options = {}) {
  return paddle::experimental::ones(
      size._PD_ToPaddleIntArray(),
      conversion::_PD_AtenScalarTypeToPhiDataType(options._PD_GetScalarType()),
      options._PD_GetPlace());
}
at::Tensor zeros(at::IntArrayRef size, at::TensorOptions options = {}) {
  return paddle::experimental::zeros(
      size._PD_ToPaddleIntArray(),
      conversion::_PD_AtenScalarTypeToPhiDataType(options._PD_GetScalarType()),
      options._PD_GetPlace());
}

at::Tensor full(at::IntArrayRef size,
                const at::Scalar& fill_value,
                ::std::optional<at::ScalarType> dtype = {},
                ::std::optional<at::Layout> layout = {},
                ::std::optional<at::Device> device = {},
                ::std::optional<bool> pin_memory = {}) {
  if (pin_memory.has_value()) {
    UNSUPPORTED_FEATURE_IN_PADDLE("`pin_memory` option in full")
  }
  return paddle::experimental::full(
      size._PD_ToPaddleIntArray(),
      fill_value,
      dtype.has_value() ? conversion::_PD_AtenScalarTypeToPhiDataType(*dtype)
                        : phi::DataType::FLOAT32,
      phi::CPUPlace()  // TODO(SigureMo): support other places
  );
}

}  // namespace at

namespace torch {
using Tensor = at::Tensor;
using Dtype = at::ScalarType;
}  // namespace torch

void compiling_test() {
  // Example usage of the Tensor class
  at::Tensor a = at::ones({2, 3}, at::TensorOptions());
  at::Tensor b = at::full({2, 3}, 1, at::ScalarType::Float);
  double c = 10;
  at::Tensor a_contig = a.contiguous();
  at::Tensor b_contig = b.contiguous();
  at::Tensor result = at::empty(a_contig.sizes(), a_contig.options());
  const float* a_ptr = a_contig.data_ptr<float>();
  const float* b_ptr = b_contig.data_ptr<float>();
  float* result_ptr = result.data_ptr<float>();
  for (int64_t i = 0; i < a_contig.numel(); i++) {
    result_ptr[i] = a_ptr[i] * b_ptr[i] + c;
  }
  // Show result
  for (int64_t i = 0; i < a_contig.numel(); i++) {
    std::cout << "Result[" << i << "] = " << a_ptr[i] * b_ptr[i] + c
              << std::endl;
  }
}
