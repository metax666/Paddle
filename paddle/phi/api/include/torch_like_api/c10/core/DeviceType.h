#pragma once

// This is directly synchronized with caffe2/proto/caffe2.proto, but
// doesn't require me to figure out how to get Protobuf headers into
// ATen/core (which would require a lot more build system hacking.)
// If you modify me, keep me synchronized with that file.

#include <ostream>

#include "paddle/phi/common/place.h"

namespace c10 {

using DeviceType = phi::AllocationType;

constexpr DeviceType kCPU = DeviceType::CPU;
constexpr DeviceType kCUDA = DeviceType::GPU;
constexpr DeviceType kCUSTOM = DeviceType::CUSTOM;
// C10_API std::ostream& operator<<(std::ostream& stream, DeviceType type);

} // namespace c10

namespace torch {
// NOLINTNEXTLINE(misc-unused-using-decls)
using c10::DeviceType;
} // namespace torch
