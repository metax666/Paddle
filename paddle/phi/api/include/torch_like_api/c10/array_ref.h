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
#include <cstdint>
#include <vector>
#include "paddle/phi/common/int_array.h"

namespace c10 {
template <typename T>
class ArrayRef {
 private:
  /// The start of the array, in an external buffer.
  const T* Data;

  /// The number of elements.
  size_t Length;

 public:
  using iterator = const T*;
  using const_iterator = const T*;
  using size_type = size_t;
  using value_type = T;

  using reverse_iterator = std::reverse_iterator<iterator>;

  /// Construct an empty ArrayRef.
  /* implicit */ constexpr ArrayRef() : Data(nullptr), Length(0) {}

  constexpr ArrayRef(const T& OneElt) : Data(&OneElt), Length(1) {}  // NOLINT

  /// Construct an ArrayRef from a pointer and length.
  constexpr ArrayRef(const T* data, size_t length)
      : Data(data), Length(length) {}

  /// Construct an ArrayRef from a range.
  constexpr ArrayRef(const T* begin, const T* end)
      : Data(begin), Length(end - begin) {}

  /* implicit */ constexpr ArrayRef(const std::initializer_list<T>& Vec)
      : Data(std::begin(Vec) == std::end(Vec) ? static_cast<T*>(nullptr)
                                              : std::begin(Vec)),
        Length(Vec.size()) {}

  constexpr iterator begin() const { return Data; }
  constexpr iterator end() const { return Data + Length; }

  constexpr bool equals(ArrayRef RHS) const {
    return Length == RHS.Length && std::equal(begin(), end(), RHS.begin());
  }

  std::vector<T> vec() const { return std::vector<T>(Data, Data + Length); }

  const paddle::experimental::IntArray _PD_ToPaddleIntArray() const {
    return paddle::experimental::IntArray(Data, Length);
  }
};

template <typename T>
bool operator==(c10::ArrayRef<T> a1, c10::ArrayRef<T> a2) {
  return a1.equals(a2);
}

template <typename T>
bool operator!=(c10::ArrayRef<T> a1, c10::ArrayRef<T> a2) {
  return !a1.equals(a2);
}

template <typename T>
bool operator==(const std::vector<T>& a1, c10::ArrayRef<T> a2) {
  return c10::ArrayRef<T>(a1).equals(a2);
}

template <typename T>
bool operator!=(const std::vector<T>& a1, c10::ArrayRef<T> a2) {
  return !c10::ArrayRef<T>(a1).equals(a2);
}

template <typename T>
bool operator==(c10::ArrayRef<T> a1, const std::vector<T>& a2) {
  return a1.equals(c10::ArrayRef<T>(a2));
}

template <typename T>
bool operator!=(c10::ArrayRef<T> a1, const std::vector<T>& a2) {
  return !a1.equals(c10::ArrayRef<T>(a2));
}
using IntArrayRef = ArrayRef<int64_t>;

}  // namespace c10
