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

#include "paddle/phi/api/include/tensor.h"
#include "paddle/phi/api/include/torch_like_api/c10/device.h"
#include "paddle/phi/api/include/torch_like_api/c10/memory_format.h"
#include "paddle/phi/api/include/torch_like_api/c10/scalar_type.h"
#include "paddle/phi/api/include/torch_like_api/c10/tensor_options.h"
#include "paddle/phi/api/include/torch_like_api/compat/int_array_ref_conversion.h"
#include "paddle/phi/api/include/torch_like_api/compat/scalar_type_conversion.h"

namespace at {
using PaddleTensor = paddle::Tensor;

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
  c10::IntArrayRef strides() const {
    return compat::_PD_PhiDDimToIntArrayRef(tensor_.strides());
  }

  int64_t size(int64_t dim) const {
    return tensor_.dims()[static_cast<int>(dim)];
  }

  c10::IntArrayRef sizes() const {
    return compat::_PD_PhiDDimToIntArrayRef(tensor_.dims());
  }

  int64_t numel() const { return tensor_.numel(); }

  c10::ScalarType dtype() const {  // Should we use `TypeMeta` here?
    return compat::_PD_PhiDataTypeToAtenScalarType(tensor_.dtype());
  }

  c10::Device device() const { return c10::Device(tensor_.place()); }

  at::Tensor contiguous(
      c10::MemoryFormat memory_format = c10::MemoryFormat::Contiguous) const {
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

}  // namespace at
