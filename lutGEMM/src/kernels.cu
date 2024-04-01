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

#include "../include/kernels.h"
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_fp16.h>

namespace lutGEMM{

#include "../src/cuda/kernels/cublas.h"
#include "../src/cuda/kernels/mv_fp16.hpp"
#include "../src/cuda/kernels/mv_fp16_bias.hpp"
#include "../src/cuda/kernels/gptq_fp16_bias.hpp"
#include "../src/cuda/kernels/gptq_faster_fp16_bias.hpp"

void matmul(void* output, nQWeight_fp16 &nqW, void* input, int n, int algo);
void matmul(void* output, void* input, nQWeight_fp16 &nqW, int m, int algo);



/* float16 */
inline void matmul_useCublas(__half* output, nQWeight_fp16 &nqW, __half* input, int n);
inline void matmul_useCublas(__half* output, __half* input, nQWeight_fp16 &nqW, int m);
/************************** float16 ***********************/

void matmul_gptq(
    int m, int n, int k, void *scale, void *bias,
    void *A, void *B, void *C){
    cudaMemset(C, 0, sizeof(__half) * m * n);
    kernel::gptq(n, k, (__half*)scale, (__half*)bias,
                (__half*)A, (uint32_t*)B, (__half*)C);
}

void matmul_gptq_faster(
    int m, int n, int k, void *scale, void *bias,
    void *A, void *B, void *C){
    cudaMemset(C, 0, sizeof(__half) * m * n);
    kernel::gptq_faster(n, k, (__half*)scale, (__half*)bias,
                (half2*)A, (uint32_t*)B, (__half*)C);
}

void matmul(void* output, nQWeight_fp16 &nqW, void* input, int n, int algo){
    if(n==1){
        cudaMemset(output, 0, sizeof(__half) * nqW.mSize);  // 0.007ms 0.04
        if(nqW.q_bias == nullptr)  kernel::nqmv((__half*)output, nqW, (__half*)input, algo);
        else                       kernel::nqmv_bias((__half*)output, nqW, (__half*)input, algo);
    } 
    else     matmul_useCublas((__half*)output, nqW, (__half*)input, n);
}
void matmul(void* output, void* input, nQWeight_fp16 &nqW, int m, int algo){
    if(m==1){
        cudaMemset(output, 0, sizeof(__half) * nqW.mSize);
        if(nqW.q_bias == nullptr)  kernel::nqmv((__half*)output, nqW, (__half*)input, algo);
        else                       kernel::nqmv_bias((__half*)output, nqW, (__half*)input, algo);
    } 
    else     matmul_useCublas((__half*)output, (__half*)input, nqW, m);
}

inline void matmul_useCublas(__half* output, nQWeight_fp16 &nqW, __half* input, int n) {
    kernel::cublas_gemm_ex((__half*)nqW.getDequantiedWeight(true), input, output, nqW.mSize, n, nqW.kSize);
}

inline void matmul_useCublas(__half* output, __half* input, nQWeight_fp16 &nqW, int m) {
    kernel::cublas_gemm_ex(input, (__half*)nqW.getDequantiedWeight(true), output, m, nqW.mSize, nqW.kSize);
}

}



