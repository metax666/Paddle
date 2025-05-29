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
#include "paddle/phi/api/include/torch_like_api/c10/scalar_type.h"
#include "paddle/phi/common/place.h"

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

  c10::ScalarType _PD_GetScalarType() const { return dtype; }
  ::phi::Place _PD_GetPlace() const { return place; }

 private:
  c10::ScalarType dtype = c10::ScalarType::Float;
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

}  // namespace c10
