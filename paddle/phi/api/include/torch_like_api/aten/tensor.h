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
#include "paddle/phi/api/include/tensor.h"
#include "paddle/phi/api/include/torch_like_api/aten/common.h"
#include "paddle/phi/api/include/torch_like_api/c10/device.h"
#include "paddle/phi/api/include/torch_like_api/c10/memory_format.h"
#include "paddle/phi/api/include/torch_like_api/c10/scalar_type.h"
#include "paddle/phi/api/include/torch_like_api/c10/tensor_options.h"
#include "paddle/phi/api/include/torch_like_api/compat/int_array_ref_conversion.h"
#include "paddle/phi/api/include/torch_like_api/compat/scalar_type_conversion.h"
#include "paddle/phi/common/place.h"

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
  c10::DeviceIndex get_device() const {
    return c10::Device(tensor_.place()).index();
  }

  int64_t dim() const { return tensor_.dims().size(); }
  int64_t ndimension() const { return dim(); }

  at::Tensor contiguous(
      c10::MemoryFormat memory_format = c10::MemoryFormat::Contiguous) const {
    if (memory_format != c10::MemoryFormat::Contiguous) {
      UNSUPPORTED_FEATURE_IN_PADDLE("`MemoryFormat` other than Contiguous")
    }
    return tensor_.contiguous();
  }

  bool is_contiguous(
      at::MemoryFormat memory_format = at::MemoryFormat::Contiguous) const {
    if (memory_format != c10::MemoryFormat::Contiguous) {
      UNSUPPORTED_FEATURE_IN_PADDLE("`MemoryFormat` other than Contiguous")
    }
    return tensor_.is_contiguous();
  }

  c10::ScalarType scalar_type() const {
    return compat::_PD_PhiDataTypeToAtenScalarType(tensor_.dtype());
  }

  c10::TensorOptions options() const {
    // TODO(SigureMo): Implement layout
    return c10::TensorOptions().dtype(dtype()).device(device());
  }

  const Tensor& fill_(const at::Scalar& scalar) const {
    paddle::experimental::fill_(const_cast<PaddleTensor&>(tensor_), scalar);
    return *this;
  }

  const Tensor& zero_() const {
    paddle::experimental::fill_(const_cast<PaddleTensor&>(tensor_), 0.0);
    return *this;
  }

  bool is_cpu() const { return phi::is_cpu_place(tensor_.place()); }
  bool is_cuda() const { return phi::is_gpu_place(tensor_.place()); }

  at::Tensor reshape(at::IntArrayRef shape) const {
    return Tensor(
        paddle::experimental::reshape(tensor_, shape._PD_ToPaddleIntArray()));
  }

  at::Tensor transpose(int64_t dim0, int64_t dim1) const {
    return Tensor(paddle::experimental::transpose(
        tensor_, {static_cast<int>(dim0), static_cast<int>(dim1)}));
  }

  at::Tensor& copy_(const at::Tensor& src, bool non_blocking = false) const {
    const_cast<PaddleTensor&>(tensor_).copy_(
        src._PD_GetInner(), tensor_.place(), /*blocking=*/!non_blocking);
    return const_cast<at::Tensor&>(*this);
  }

  at::Tensor view(at::IntArrayRef size) const {
    return Tensor(paddle::experimental::view_shape(tensor_, size.vec()));
  }

  at::Tensor view(at::ScalarType dtype) const {
    return Tensor(paddle::experimental::view_dtype(
        tensor_, compat::_PD_AtenScalarTypeToPhiDataType(dtype)));
  }

  PaddleTensor _PD_GetInner() const { return tensor_; }
  PaddleTensor& _PD_GetInner() { return tensor_; }

 private:
  PaddleTensor tensor_;
};

}  // namespace at
