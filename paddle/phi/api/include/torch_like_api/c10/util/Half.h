#pragma once

#include "paddle/phi/common/float16.h"

namespace c10 {
template <typename T>
using Half = ::phi::dtype::float16;
}  // namespace c10
