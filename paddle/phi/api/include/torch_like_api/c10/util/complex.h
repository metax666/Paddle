#pragma once

#include "paddle/phi/common/complex.h"

namespace c10 {
template <typename T>
using complex = ::phi::dtype::complex<T>;
}  // namespace c10
