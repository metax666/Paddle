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

#include "paddle/phi/api/include/api.h"
#include "paddle/phi/api/include/torch_like_api/aten/common.h"
#include "paddle/phi/api/include/torch_like_api/aten/tensor.h"

namespace at {

at::Tensor empty(
    at::IntArrayRef size,
    at::TensorOptions options = {},
    ::std::optional<at::MemoryFormat> memory_format = ::std::nullopt) {
  if (memory_format.has_value()) {
    UNSUPPORTED_FEATURE_IN_PADDLE("`MemoryFormat`")
  }
  return paddle::experimental::empty(
      size._PD_ToPaddleIntArray(),
      compat::_PD_AtenScalarTypeToPhiDataType(options._PD_GetScalarType()),
      options._PD_GetPlace());
}
at::Tensor ones(at::IntArrayRef size, at::TensorOptions options = {}) {
  return paddle::experimental::ones(
      size._PD_ToPaddleIntArray(),
      compat::_PD_AtenScalarTypeToPhiDataType(options._PD_GetScalarType()),
      options._PD_GetPlace());
}
at::Tensor zeros(at::IntArrayRef size, at::TensorOptions options = {}) {
  return paddle::experimental::zeros(
      size._PD_ToPaddleIntArray(),
      compat::_PD_AtenScalarTypeToPhiDataType(options._PD_GetScalarType()),
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
      dtype.has_value() ? compat::_PD_AtenScalarTypeToPhiDataType(*dtype)
                        : phi::DataType::FLOAT32,
      device.has_value() ? device.value()._PD_GetInner() : phi::CPUPlace());
}
at::Tensor abs(const at::Tensor& self) {
  return paddle::experimental::abs(self._PD_GetInner());
}

}  // namespace at
