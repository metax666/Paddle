#pragma once

#include "paddle/phi/api/include/context_pool.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_info.h"
#include "paddle/phi/core/cuda_stream.h"

namespace at::cuda {

using StreamId = int64_t;

class CUDAStream {
 public:
  CUDAStream() = delete;
  explicit CUDAStream(const cudaStream_t& stream) : raw_stream_(stream) {}
  StreamId id() const { return reinterpret_cast<StreamId>(raw_stream_); }

  operator cudaStream_t() const { return raw_stream_; }

  const cudaStream_t& stream() const { return raw_stream_; }

 private:
  cudaStream_t raw_stream_;
};

inline CUDAStream getCurrentCUDAStream(c10::DeviceIndex device_index = -1) {
  if (device_index == -1) {
    device_index = phi::backends::gpu::GetCurrentDeviceId();
  }

  return CUDAStream(
      paddle::GetCurrentCUDAStream(phi::GPUPlace(device_index))->raw_stream());
}

}  // namespace at::cuda