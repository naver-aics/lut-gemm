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

#include <stdio.h>

#include "../include/nQWeight_fp16.h"

#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_fp16.h>

namespace lutGEMM{
#include "../src/cuda/tmpWeight.hpp"
#include "../src/cuda/kernels/dequant_fp16.hpp"

void dequantize_gpu(nQWeight_fp16 &nqw, void *d_fW, int algo){
    if(nqw.is_row_wise_quantize)
        kernel::dequantize((__half*)d_fW, nqw.bWeight, (__half*)nqw.alpha, nqw.mSize, nqw.kSize, nqw.nb);
    else
        kernel::dequantize_t((__half*)d_fW, nqw.bWeight, (__half*)nqw.alpha, nqw.mSize, nqw.kSize, nqw.nb, nqw.group_size);
    cudaDeviceSynchronize();
}

/* fW[M][K] */
void dequantize_cpu(nQWeight_fp16 &nqw, void *fW){
    __half* ffW = (__half*)fW;
    __half* alpha = (__half*)nqw.alpha;
    int group_size = nqw.group_size;
    unsigned int *bWeight = nqw.bWeight;
    int kSize = nqw.kSize;
    int mSize = nqw.mSize;
    int nb = nqw.nb;
    for(int k=0;k<kSize;k++){
        for(int m=0;m<mSize;m++){
            float tmp = 0.0;
            for(int b=0;b<nb;b++){
                if((bWeight[(k/32)*nb*mSize + b*mSize + m] >> (k%32)) & 1)  tmp += float(alpha[(k/group_size)*nb*mSize + b * mSize + m]);
                else                                                        tmp -= float(alpha[(k/group_size)*nb*mSize + b * mSize + m]);
            }
            if(nqw.is_row_wise_quantize) 
                ffW[m*kSize + k] = __float2half(tmp);
            else                         
                ffW[k*mSize + m] = __float2half(tmp);           
        }
    }
}
void nQWeight_fp16::parsing(unsigned int *bW, float *A, int row, int col, int num_bits, 
        bool is_row_wise_quantize, int num_alpha_groups, float* q_bias){
    this->num_groups = num_alpha_groups;
    this->group_size =  kSize/num_alpha_groups;

    __half* p_alpha;
    __half* p_q_bias;
    nb=num_bits;
    this->is_row_wise_quantize = is_row_wise_quantize;
    if(is_row_wise_quantize){
        mSize = row; 
        kSize = col; 
    }
    else{
        mSize = col; 
        kSize = row;             
    }

    if(q_bias == nullptr) p_q_bias = nullptr;
    else{
        cudaMallocManaged(&p_q_bias    ,sizeof(__half  ) * num_groups * mSize);
        for(int i=0;i<num_groups*mSize;i++) p_q_bias[i] = __float2half(q_bias[i]);
    }
    
    cudaMallocManaged(&p_alpha    ,sizeof(__half  ) * num_groups * mSize * nb);
    for(int i=0;i<num_groups*mSize*nb;i++) p_alpha[i] = __float2half(A[i]);

    cudaMallocManaged(&bWeight  ,sizeof(uint32_t) * kSize * mSize * nb / 32);
    cudaMemcpy(bWeight ,bW      ,sizeof(uint32_t) * kSize * mSize * nb / 32,    cudaMemcpyHostToDevice);
    this->alpha = (void*)p_alpha;
    this->q_bias = (void*)p_q_bias;
}

void* nQWeight_fp16::getDequantiedWeight(bool onGPU){
    __half* fW = (__half*)tmpWeight::getInstance().getWeight(mSize*kSize/2);
    cudaDeviceSynchronize();
    if(onGPU) dequantize_gpu(*this, fW);
    else      dequantize_cpu(*this, fW);
    return fW;
}

nQWeight_fp16::~nQWeight_fp16(){
    cudaFree(alpha);
    cudaFree(alpha);
    if(q_bias!= nullptr) cudaFree(q_bias);
}

}

