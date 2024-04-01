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

#ifndef N_Q_WEIGHT_FP16_H
#define N_Q_WEIGHT_FP16_H

namespace lutGEMM{

class nQWeight_fp16;

void dequantize_gpu(nQWeight_fp16 &nqw, void *d_fW, int algo=0);
void dequantize_cpu(nQWeight_fp16 &nqw, void *fW);

class nQWeight_fp16{
public:
    unsigned int* bWeight;  // Weight[kSize/32][nb][mSize]   
    void* alpha;     //  alpha[num_alpha_groups][nb][mSize]
    void* q_bias;   //q_bias[num_alpha_groups][mSize]
    int num_groups;
    int group_size;
    int mSize;
    int kSize;   
    int nb;
    bool is_row_wise_quantize;
    nQWeight_fp16() {}

    /* uint32 bW[kSize/32][nb][mSize]  alpha[num_alpha_groups][mSize][nb] */
    nQWeight_fp16(unsigned int *bW, float *A, int row, int col, int num_bits, 
        bool is_row_wise_quantize, int num_alpha_groups=1, float* q_bias=nullptr){
        parsing(bW, A, row, col, num_bits, is_row_wise_quantize, num_alpha_groups, q_bias);
    }

    void parsing(unsigned int *bW, float *A, int row, int col, int num_bits, 
        bool is_row_wise_quantize, int num_alpha_groups=1, float* q_bias=nullptr);

    ~nQWeight_fp16();
    
    void* getDequantiedWeight(bool onGPU=true);
};

}
#endif // N_Q_WEIGHT_FP16_H
