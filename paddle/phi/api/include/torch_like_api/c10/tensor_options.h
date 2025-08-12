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

#include <c10/core/ScalarType.h>

#include "paddle/common/macros.h"
#include "paddle/phi/common/place.h"

#include <optional>

namespace c10 {
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

  /* implicit */ TensorOptions(c10::ScalarType dtype)  // NOLINT
      : TensorOptions() {
    this->set_dtype(dtype);
  }

  TensorOptions dtype(std::optional<ScalarType> dtype) const noexcept {
    TensorOptions r = *this;
    r.set_dtype(dtype);
    return r;
  }

  TensorOptions device(std::optional<Device> device) const noexcept {
    TensorOptions r = *this;
    r.set_device(device);
    return r;
  }

  template <typename T>
  TensorOptions& dtype() {
    has_dtype_ = true;
    return *this;
  }

  /// Mutably set the device of `TensorOptions`.
  void set_device(std::optional<Device> device) & noexcept {
    if (device) {
      device_ = *device;
      has_device_ = true;
    } else {
      has_device_ = false;
    }
  }

  // legacy function to support ScalarType
  void set_dtype(std::optional<ScalarType> dtype) & noexcept {
    if (dtype) {
      dtype_ = *dtype;
      has_dtype_ = true;
    } else {
      has_dtype_ = false;
    }
  }

  //   /// Mutably set the layout of `TensorOptions`.
  //   void set_layout(std::optional<Layout> layout) & noexcept {
  //     if (layout) {
  //       layout_ = *layout;
  //       has_layout_ = true;
  //     } else {
  //       has_layout_ = false;
  //     }
  //   }

  /// Mutably set the `requires_grad` property of `TensorOptions`.
  void set_requires_grad(std::optional<bool> requires_grad) & noexcept {
    if (requires_grad) {
      requires_grad_ = *requires_grad;
      has_requires_grad_ = true;
    } else {
      has_requires_grad_ = false;
    }
  }

  /// Mutably set the `pinned_memory` property of `TensorOptions`.
  void set_pinned_memory(std::optional<bool> pinned_memory) & noexcept {
    if (pinned_memory) {
      pinned_memory_ = *pinned_memory;
      has_pinned_memory_ = true;
    } else {
      has_pinned_memory_ = false;
    }
  }

  //   /// Mutably set the `memory_Format` property of `TensorOptions`.
  //   void set_memory_format(std::optional<MemoryFormat> memory_format) &
  //   noexcept {
  //     if (memory_format) {
  //       memory_format_ = *memory_format;
  //       has_memory_format_ = true;
  //     } else {
  //       has_memory_format_ = false;
  //     }
  //   }

  std::optional<c10::ScalarType> dtype_opt() const noexcept {
    return has_dtype_ ? std::make_optional(dtype_) : std::nullopt;
  }

  // std::optional<Layout> layout_opt() const noexcept {
  //   return has_layout_ ? std::make_optional(layout_) : std::nullopt;
  // }

  std::optional<Device> device_opt() const noexcept {
    return has_device_ ? std::make_optional(device_) : std::nullopt;
  }

  c10::ScalarType _PD_GetScalarType() const { return dtype_; }
  ::phi::Place _PD_GetPlace() const { return device_._PD_GetInner(); }

 private:
  Device device_ = c10::kCPU;
  c10::ScalarType dtype_ = c10::ScalarType::Float;
  bool requires_grad_ : 1;
  bool pinned_memory_ : 1;

  bool has_device_ : 1;
  bool has_dtype_ : 1;
  bool has_layout_ : 1;
  bool has_requires_grad_ : 1;
  bool has_pinned_memory_ : 1;
  bool has_memory_format_ : 1;
};

}  // namespace c10
