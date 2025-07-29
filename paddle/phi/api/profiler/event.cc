#include "paddle/phi/api/profiler/event.h"

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
#include "glog/logging.h"
#endif

namespace phi {

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)

CudaEvent::CudaEvent() {
#ifdef PADDLE_WITH_HIP
  hipEventCreateWithFlags(&event_, flags_);
#else
  cudaEventCreateWithFlags(&event_, flags_);
#endif
  VLOG(4) << "CudaEvent " << event_;
}

CudaEvent::CudaEvent(unsigned int flags) : flags_(flags) {
#ifdef PADDLE_WITH_HIP
  hipEventCreateWithFlags(&event_, flags_);
#else
  cudaEventCreateWithFlags(&event_, flags_);
#endif
  VLOG(4) << "CudaEvent " << event_;
}

bool CudaEvent::Query() {
#ifdef PADDLE_WITH_HIP
  gpuError_t err = hipEventQuery(event_);
  if (err == hipSuccess) {
    return true;
  }
  if (err == hipErrorNotReady) {
    return false;
  }
#else
  gpuError_t err = cudaEventQuery(event_);
  if (err == cudaSuccess) {
    return true;
  }
  if (err == cudaErrorNotReady) {
    return false;
  }
#endif
  PADDLE_ENFORCE_GPU_SUCCESS(err);
  return false;
}

float CudaEvent::ElapsedTime(CudaEvent *end_event) {
  float milliseconds = 0;
#ifdef PADDLE_WITH_HIP
  hipEventSynchronize(end_event->GetRawCudaEvent());
  PADDLE_ENFORCE_GPU_SUCCESS(
      hipEventElapsedTime(&milliseconds, event_, end_event->GetRawCudaEvent()));
#else
  cudaEventSynchronize(end_event->GetRawCudaEvent());
  PADDLE_ENFORCE_GPU_SUCCESS(cudaEventElapsedTime(
      &milliseconds, event_, end_event->GetRawCudaEvent()));
#endif
  return milliseconds;
}

#endif

}  // namespace phi