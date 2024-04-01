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

#ifndef KERNELS_DEQUANT_HPP
#define KERNELS_DEQUANT_HPP
/*************************************************
 * float    W[M][K]
 * uint32_t bW[K/32][M][NUM_BITS]
 * float    A[M][NUM_BITS]
*/

namespace kernel{
template<size_t NUM_BTIS, size_t M_TILE_SIZE, size_t K_TILE_SIZE>
__global__ void _dequantize(float* W, uint32_t *bW, float *A, size_t M, size_t K){
    int m_step = blockDim.y;

    int m_start = blockIdx.y * M_TILE_SIZE + threadIdx.y;
    int m_end = (blockIdx.y + 1) * M_TILE_SIZE;
    m_end = (m_end < M) ? m_end : M;

    int k     = blockIdx.x * K_TILE_SIZE + threadIdx.x;
    int tk = k/32;
    int t  = k%32;
    int k_end = (blockIdx.x + 1) * K_TILE_SIZE;
    k_end = (k_end < K) ? k_end : K;

    for(int m = m_start;m<m_end;m += m_step){
        if(k < k_end){
            float r = 0;
            for(int b = 0;b<NUM_BTIS;b++){
                if((bW[tk * NUM_BTIS * M + b * M + m] >> t) & 1) r += A[b * M + m];
                else                                             r -= A[b * M + m];
            } 
            W[m * K + k] = r;
        }
    }
}

template<size_t NUM_BTIS, size_t M_TILE_SIZE, size_t K_TILE_SIZE>
__global__ void _dequantize_t(float* W, uint32_t *bW, float *A, size_t M, size_t K){
    int m_step = blockDim.y;

    int m_start = blockIdx.y * M_TILE_SIZE + threadIdx.y;
    int m_end = (blockIdx.y + 1) * M_TILE_SIZE;
    m_end = (m_end < M) ? m_end : M;

    int k     = blockIdx.x * K_TILE_SIZE + threadIdx.x;
    int tk = k/32;
    int t  = k%32;
    int k_end = (blockIdx.x + 1) * K_TILE_SIZE;
    k_end = (k_end < K) ? k_end : K;

    for(int m = m_start;m<m_end;m += m_step){
        if(k < k_end){
            float r = 0;
            for(int b = 0;b<NUM_BTIS;b++){
                if((bW[tk * NUM_BTIS * M + b * M + m] >> t) & 1) r += A[b * M + m];
                else                                             r -= A[b * M + m];
            } 
            W[k * M + m] = r;
        }
    }
}

inline int div_roundup(int x , int y){return (x + y - 1)/ y;}
inline void dequantize(float* W, uint32_t *bW, float *A, size_t m, size_t k, size_t num_bits){
    const int k_tile_size = 32;
    const int m_tile_size = 32;
    const int num_thraeds = 64;
    dim3 block(k_tile_size,  num_thraeds/k_tile_size);
    dim3 grid(div_roundup(k, k_tile_size), div_roundup(m, m_tile_size)); 
    
    if     (num_bits == 1) _dequantize<1, m_tile_size, k_tile_size><<<grid, block>>>(W, bW, A, m, k);
    else if(num_bits == 2) _dequantize<2, m_tile_size, k_tile_size><<<grid, block>>>(W, bW, A, m, k);
    else if(num_bits == 3) _dequantize<3, m_tile_size, k_tile_size><<<grid, block>>>(W, bW, A, m, k);
    else if(num_bits == 4) _dequantize<4, m_tile_size, k_tile_size><<<grid, block>>>(W, bW, A, m, k);
}

inline void dequantize_t(float* W, uint32_t *bW, float *A, size_t m, size_t k, size_t num_bits){
    const int k_tile_size =   4;
    const int m_tile_size =  64;
    const int num_thraeds =  64;
    dim3 block(k_tile_size,  num_thraeds/k_tile_size);
    dim3 grid(div_roundup(k, k_tile_size), div_roundup(m, m_tile_size)); 
    
    if     (num_bits == 1) _dequantize_t<1, m_tile_size, k_tile_size><<<grid, block>>>(W, bW, A, m, k);
    else if(num_bits == 2) _dequantize_t<2, m_tile_size, k_tile_size><<<grid, block>>>(W, bW, A, m, k);
    else if(num_bits == 3) _dequantize_t<3, m_tile_size, k_tile_size><<<grid, block>>>(W, bW, A, m, k);
    else if(num_bits == 4) _dequantize_t<4, m_tile_size, k_tile_size><<<grid, block>>>(W, bW, A, m, k);
    // cudaDeviceSynchronize();
}

}
#endif