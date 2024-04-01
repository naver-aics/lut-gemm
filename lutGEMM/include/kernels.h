/* LUT-GEMM
 * Copyright (c) 2024-present NAVER Cloud Corp. All rights reserved.
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma ones

#ifndef KERNELS_H
#define KERNELS_H

#include "nQWeight_fp16.h"

namespace lutGEMM{

void matmul(void* output, nQWeight_fp16 &nqW, void* input, int n, int algo=0);
void matmul(void* output, void* input, nQWeight_fp16 &nqW, int m, int algo=0);
void matmul_gptq(
    int m, int n, int k, void *scale, void *bias,
    void *A, void *B, void *C);
void matmul_gptq_faster(
    int m, int n, int k, void *scale, void *bias,
    void *A, void *B, void *C);
}

#endif

