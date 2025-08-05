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
#include <c10/device.h>
#include "paddle/phi/core/platform/cuda_device_guard.h"

namespace c10::cuda {
struct CUDAGuard {
  /// No default constructor; see Note [Omitted default constructor from RAII]
  explicit CUDAGuard() = delete;  // NOLINT

  /// Set the current CUDA device to the passed device index.
  explicit CUDAGuard(DeviceIndex device_index) : guard_(device_index) {}

  /// Sets the current CUDA device to the passed device.  Errors if the passed
  /// device is not a CUDA device.
  explicit CUDAGuard(Device device) : guard_(device._PD_GetInner()) {}

  // Copy is not allowed
  CUDAGuard(const CUDAGuard&) = delete;
  CUDAGuard& operator=(const CUDAGuard&) = delete;

  // Move is not allowed (there is no uninitialized state)
  CUDAGuard(CUDAGuard&& other) = delete;
  CUDAGuard& operator=(CUDAGuard&& other) = delete;
  ~CUDAGuard() = default;

  void set_device(Device device) { guard_.SetDevice(device._PD_GetInner()); }
  void reset_device(Device device) { set_device(device); }
  void set_index(DeviceIndex device_index) {
    guard_.SetDeviceIndex(device_index);
  }
  // Device original_device() const { NotImplemented; }
  Device current_device() const {
    return c10::Device(c10::kCUDA, phi::backends::gpu::GetCurrentDeviceId());
  }

 private:
  paddle::platform::CUDADeviceGuard guard_;
};
}  // namespace c10::cuda
