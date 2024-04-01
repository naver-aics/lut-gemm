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

#include "tests.h"

#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_fp16.h>



template <typename T, typename S>
inline cublasStatus_t cublas_gemm_ex(T *A,  T *B,  S *C,
                                    int m, int n, int k);
                                    
template<int M, int N, int K, int NUM_BITS, int A_GROUP_SIZE=K>
class int3_col_wise_matmul_fp16{
public:
    static const int num_groups = K/A_GROUP_SIZE;
    float     qW[K   ][NUM_BITS][N]; // (-1, 1) 
    uint32_t  bW[K/32][NUM_BITS][N]; // bit packed
    float     alpha[num_groups][NUM_BITS][N];
    float    q_bias[num_groups][N];

    float   weight[K][N];           // float weight
    float    input[M][K];
    float   output[M][N];

    int K_new = K * 3 / 32; // 3bit weights are packed into int32
    int   weight_int3[K * 3 / 32][N];
    float scale[N];
    float bias[N];

    int*    d_weight_int3;
    __half* d_scale;
    __half* d_bias;
    __half* d_gptq_output;


    __half* d_weight_fp16;
    __half*  d_input;

    __half* d_cu_output;
    __half* d_nq_output;

    lutGEMM::nQWeight_fp16 nqW;

    double run(bool run_cublas=true, bool run_lutgemm=false, bool run_gptq=false, int iter=16){
        alloc_cuda();
        makeRandomInput();
        makeRandomWeight();
        makeRandomWeight_int3();
        makeRandomAlpha();
        //dequantizeFrom_qW();
        copy_cpuToCuda();

        nqW.parsing((uint32_t*)bW, (float*)alpha, K, N, NUM_BITS, false, num_groups, (float*)q_bias);
        cudaDeviceSynchronize();

        //double meanError = checkErr();
        double meanError = 0;
        cudaDeviceSynchronize();

        if(run_cublas) cublas_latency(M, N, K, d_input, d_weight_fp16, d_cu_output, iter);
        if(run_lutgemm) lutgemm_latency(nqW, M, N, K, d_input, d_weight_fp16, d_cu_output, iter);
        //if(run_gptq) gptq_latency(M, N, K_new, d_scale, d_bias, d_input, d_weight_int3, d_gptq_output, iter);
        if(run_gptq) gptq_faster_latency(M, N, K_new, d_scale, d_bias, d_input, d_weight_int3, d_gptq_output, iter);

        free_cuda();
        return meanError;
    }

    void gptq_latency(int m, int n, int k, __half* scale, __half* bias, __half* A, int *B, __half *C, int iter=64){
        timer tm;

        lutGEMM::matmul_gptq(m, n, k, (void*)scale, (void*)bias,
                        (void*)A, (void*)B, (void*)C);
        cudaDeviceSynchronize();

        for(int i=0;i<iter;i++){
            tm.start();
            lutGEMM::matmul_gptq(m, n, k, (void*)scale, (void*)bias,
                        (void*)A, (void*)B, (void*)C);
            cudaDeviceSynchronize();
            tm.end();
        }
        printf("latency min : %.5fms, max : %.5fms, avg:%.5f\n", tm.min(), tm.max(), tm.mean());
    }

    void gptq_faster_latency(int m, int n, int k, __half* scale, __half* bias, __half* A, int *B, __half *C, int iter=64){
        timer tm;

        lutGEMM::matmul_gptq_faster(m, n, k, (void*)scale, (void*)bias,
                        (void*)A, (void*)B, (void*)C);
        cudaDeviceSynchronize();

        for(int i=0;i<iter;i++){
            tm.start();
            lutGEMM::matmul_gptq_faster(m, n, k, (void*)scale, (void*)bias,
                        (void*)A, (void*)B, (void*)C);
            cudaDeviceSynchronize();
            tm.end();
        }
        printf("latency min : %.5fms, max : %.5fms, avg:%.5f\n", tm.min(), tm.max(), tm.mean());
    }

    void lutgemm_latency(lutGEMM::nQWeight_fp16 &nqW, int m, int n, int k, __half* A, __half *B, __half *C, int iter=64){
        timer tm;

        lutGEMM::matmul((void*)C, (void*)A, nqW, m);
        cudaDeviceSynchronize();

        for(int i=0;i<iter;i++){
            tm.start();
            lutGEMM::matmul((void*)C, (void*)A, nqW, m);
            cudaDeviceSynchronize();
            tm.end();
        }
        printf("latency min : %.5fms, max : %.5fms, avg:%.5f\n", tm.min(), tm.max(), tm.mean());
    }

    void cublas_latency(int m, int n, int k, __half* A, __half *B, __half *C, int iter=64){
        timer tm;
        float th = 0;
        cublas_gemm_ex(A, B, C,
                            m, n, k);
        cudaDeviceSynchronize();
        for (int i = 0; i < iter; ++i) {
            tm.start();
            cublasStatus_t success;
            success = cublas_gemm_ex(A, B, C,
                                    m, n, k);
            cudaDeviceSynchronize();
            tm.end();

        }
            printf("latency min : %.5fms, max : %.5fms, avg:%.5f\n", tm.min(), tm.max(), tm.mean());

    }


    double checkErr(){
        cublas_gemm_ex(d_input, d_weight_fp16, d_cu_output, M, N, K);
        cudaMemset(d_nq_output, 0, sizeof(float) * M * N);
        lutGEMM::matmul(d_nq_output, d_input, nqW, M);
        cudaDeviceSynchronize();
        return checkOutputMeanError(d_cu_output, d_nq_output);
    }

    double checkOutputMeanError(__half *o1, __half *o2){
        double err=0;
        for(int m=0;m<M;m++){
            for(int n=0;n<N;n++){
                err += std::abs(float(o1[m*N + n]) - float(o2[m*N + n]));
                // if(n<100) printf("%f %f\n", float(o1[m*N + n]), float(o2[m*N + n]));
            }
        }
        return err/M/N;
    }

    void matmul_cpu(){
        for(int m=0;m<M;m++){
            for(int n=0;n<N;n++){
                output[m][n] = 0;
                for(int k=0;k<K;k++){
                    output[m][n] += input[m][k] * weight[k][n];
                }
            }
        }
    }

    void makeRandomInput(){
        for(int m=0;m<M;m++)
            for(int k=0;k<K;k++)
                input[m][k] = rand_fp32(); // (-1.0, 1.0) / 2^b
    }

    void makeRandomAlpha(){
        for(int g=0;g<num_groups;g++)
            for(int n=0;n<N;n++){
                q_bias[g][n] = rand_fp32()/(1<< NUM_BITS);
                for(int b=0;b<NUM_BITS;b++)
                    alpha[g][b][n] = rand_fp32()/(1<<b); // (-1.0, 1.0) / 2^b
            }
    }

    void makeRandomWeight(){
        for(int n=0;n<N;n++){
            for(int b=0;b<NUM_BITS;b++){
                for(int k=0;k<K;k+=32){  //32 단위
                    uint32_t s=0;
                    for(int t=0;t<32;t++){
                        if(rand_bool()){
                                s |= 1<<t;
                                qW[k + t][b][n] = +1;
                        } else  qW[k + t][b][n] = -1;
                    }
                    bW[k/32][b][n] = s;
                }
            }
        }
    }

    void makeRandomWeight_int3(){
        for(int n=0;n<N;n++){
            for(int k=0;k<K_new;k++){
                weight_int3[k][n] = rand();
            }
        }
    }

    void makeRandomScale(){
        for(int n=0;n<N;n++)
            scale[n] = rand_fp32();
    }

    void makeRandomBias(){
        for(int n=0;n<N;n++)
            bias[n] = rand_fp32();
    }

    void dequantizeFrom_qW(){
        for(int n=0;n<N;n++){
            for(int k=0;k<K;k++){  //32 단위
                weight[k][n] = q_bias[k/A_GROUP_SIZE][n];
                for(int b=0;b<NUM_BITS;b++){
                    weight[k][n] += alpha[k/A_GROUP_SIZE][b][n] * qW[k][b][n]; 
                }
            }
        }        
    }    

    void alloc_cuda(){
        cudaMallocManaged(&d_input    , sizeof(float) * M * K);   
        cudaMallocManaged(&d_weight_fp16, sizeof(float) * K * N);   

        cudaMallocManaged(&d_cu_output, sizeof(float) * M * N);       
        cudaMallocManaged(&d_nq_output, sizeof(float) * M * N);

        cudaMallocManaged(&d_weight_int3, sizeof(int) * K_new * N);   
        cudaMallocManaged(&d_scale, sizeof(float) * N);   
        cudaMallocManaged(&d_bias, sizeof(float) * N);   
        cudaMallocManaged(&d_gptq_output, sizeof(float) * M * N);

    }
    
    void free_cuda(){
        cudaFree(d_input);
        cudaFree(d_weight_fp16);
        cudaFree(d_cu_output);
        cudaFree(d_nq_output);

        cudaFree(d_weight_int3);
        cudaFree(d_scale);
        cudaFree(d_bias);
        cudaFree(d_gptq_output);
    }
    void copy_cpuToCuda(){
        fhCpy(d_input , (float*)input  ,M * K);
        fhCpy(d_weight_fp16, (float*)weight ,K * N);

        cudaMemcpy(d_weight_int3, (int*)weight_int3,
            K_new * N, cudaMemcpyHostToDevice);
        fhCpy(d_scale, (float*)scale , N);
        fhCpy(d_bias, (float*)bias , N);

        cudaDeviceSynchronize();
    }

    void hfCpy(float* a, __half* b, int size){
       for(int i=0;i<size;i++) a[i] = float(b[i]);
    }
    void fhCpy(__half* a, float* b, int size){
       for(int i=0;i<size;i++) a[i] = __float2half(b[i]);
    }

};

const int H = 7168;
TEST(int3_col_wise_matmul_fp16, layer_175b){
    double total_error = 0;
    int e_cnt = 0;

    { auto t = std::make_shared<int3_col_wise_matmul_fp16<1, H*4, H, 3, 128>>(); total_error += t->run(true, true, true); e_cnt++; }
    printf("----------------------------------------------------------------\n");
    printf("Warm up done.\n");
    printf("----------------------------------------------------------------\n");
    printf("M = 1, N = %d, K = %d\n", 4*H, H);
    printf("cuBLAS [FP16, FP16, FP16]\t");
    { auto t = std::make_shared<int3_col_wise_matmul_fp16<1, H*4, H, 3, 128>>(); total_error += t->run(true, false, false); e_cnt++; }
    printf("OPTQ [INT3, FP16, FP16]\t\t");
    { auto t = std::make_shared<int3_col_wise_matmul_fp16<1, H*4, H, 4, 128>>(); total_error += t->run(false, false, true); e_cnt++; }  

    printf("LUT-GEMM [INT8, FP16, FP16]\t");
    { auto t = std::make_shared<int3_col_wise_matmul_fp16<1, H*4, H, 8, 128>>(); total_error += t->run(false, true, false); e_cnt++; } 
    
    printf("LUT-GEMM [INT4, FP16, FP16]\t");
    { auto t = std::make_shared<int3_col_wise_matmul_fp16<1, H*4, H, 4, 128>>(); total_error += t->run(false, true, false); e_cnt++; }  
    printf("LUT-GEMM [INT3, FP16, FP16]\t");
    { auto t = std::make_shared<int3_col_wise_matmul_fp16<1, H*4, H, 3, 128>>(); total_error += t->run(false, true, false); e_cnt++; }
}




template <typename T, typename S>
inline cublasStatus_t cublas_gemm_ex(T *A,  T *B,  S *C,
                                    int m, int n, int k) {
    static S alpha = 1;
    static S beta  = 0;
    static cublasHandle_t handle = nullptr;
    if(handle == nullptr) cublasCreate(&handle);
    
    cudaDataType_t AType, BType, CType;
    cublasComputeType_t  ComputeType;
    if (std::is_same<T, float>::value) {
        AType = BType = CType = CUDA_R_32F;
        ComputeType = CUBLAS_COMPUTE_32F_FAST_TF32;
    } else if (std::is_same<T, __half>::value) {
        AType = BType = CType = CUDA_R_16F;
        ComputeType = CUBLAS_COMPUTE_16F;
    } else if (std::is_same<T, int8_t>::value) {
        AType = BType = CUDA_R_8I;
        CType = CUDA_R_32I;
        ComputeType = CUBLAS_COMPUTE_32I;
    } else {
        return CUBLAS_STATUS_NOT_SUPPORTED;
    }
    return cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                          n, m, k, 
                          &alpha,
                          B, BType, n,
                          A, AType, k,
                          &beta,
                          C, CType, n,
                          ComputeType,
                          CUBLAS_GEMM_DFALT);
}