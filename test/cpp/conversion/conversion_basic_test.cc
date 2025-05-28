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

#include "paddle/phi/api/include/torch_compat_runtime.h"

#include "gtest/gtest.h"

TEST(conversion_basic_test, BasicCase) {
  at::Tensor a = at::ones({2, 3}, at::TensorOptions());
  at::Tensor b = at::full({2, 3}, 2, at::ScalarType::Float);
  double c = 10;
  at::Tensor a_contig = a.contiguous();
  at::Tensor b_contig = b.contiguous();
  at::Tensor result = at::empty(a_contig.sizes(), a_contig.options());
  const float* a_ptr = a_contig.data_ptr<float>();
  const float* b_ptr = b_contig.data_ptr<float>();
  float* result_ptr = result.data_ptr<float>();
  for (int64_t i = 0; i < a_contig.numel(); i++) {
    result_ptr[i] = a_ptr[i] * b_ptr[i] + c;
  }
  // Show result
  for (int64_t i = 0; i < a_contig.numel(); i++) {
    std::cout << "Result[" << i << "] = " << a_ptr[i] * b_ptr[i] + c
              << std::endl;
    ASSERT_EQ(result_ptr[i], 12);
  }
}
