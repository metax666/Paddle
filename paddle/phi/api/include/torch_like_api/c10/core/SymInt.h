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
#include <c10/util/accumulate.h>
#include <cstdint>

namespace c10 {
using SymInt = int64_t;

// /// Sum of a list of SymInt; accumulates into the c10::SymInt expression
// template <typename C,
//           typename std::enable_if_t<
//               std::is_same_v<typename C::value_type, c10::SymInt>,
//               int> = 0>
// inline c10::SymInt multiply_integers(const C& container) {
//   return std::accumulate(
//       container.begin(),
//       container.end(),
//       c10::SymInt(1),
//       [](const c10::SymInt& a, const c10::SymInt& b) { return a * b; });
// }

// template <typename Iter,
//           typename = std::enable_if_t<
//               std::is_same_v<typename std::iterator_traits<Iter>::value_type,
//                              c10::SymInt>>>
// inline c10::SymInt multiply_integers(Iter begin, Iter end) {
//   return std::accumulate(
//       begin,
//       end,
//       c10::SymInt(1),
//       [](const c10::SymInt& a, const c10::SymInt& b) { return a * b; });
// }

}  // namespace c10
