#include "tests.h"

#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_fp16.h>


template <typename T, typename S>
inline cublasStatus_t cublas_sgemm_ex(T *A,  T *B,  S *C,
                                    int m, int n, int k) {
    static float alpha = 1;
    static float beta  = 0;
    static cublasHandle_t handle = nullptr;
    if(handle == nullptr) cublasCreate(&handle);
    
    cudaDataType_t AType, BType, CType;
    AType = BType = CUDA_R_8I;
    CType = CUDA_R_32I;

    return cublasSgemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                          n, m, k, 
                          &alpha,
                          B, BType, n,
                          A, AType, k,
                          &beta,
                          C, CType, n);
}


void floatToInt8(int8_t *out, float *inp, int size, float scale=1.0){
    for(int i=0;i<size;i++){
        out[i] = inp[i]/scale*127;
    }
}

int8_t float2int8(float f, float scale) {
    int8_t i = int8_t(f * scale);
    if (i < -127) i = -127;
    if (i > 127) i = 127;
    return i;
}

template <typename T, typename S>
void allocate_memory(int m, int n, int k, T **A, T **B, S **C) {
    cudaMallocManaged(A, m * k * sizeof(T));
    cudaMallocManaged(B, k * n * sizeof(T));
    cudaMallocManaged(C, m * n * sizeof(S));
}

template <typename T, typename S>
void free_memory(T *A, T *B, S *C) {
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
}

template <typename T, typename S>
inline cublasStatus_t cublas_gemm_ex(T *A,  T *B,  S *C,
                                    int m, int n, int k) {
    static S alpha = 1;
    static S beta  = 0;
    static cublasHandle_t handle = nullptr;
    if(handle == nullptr) cublasCreate(&handle);
    cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
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
        printf("Not supported data type.");
        return CUBLAS_STATUS_NOT_SUPPORTED;
    }
    return cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                          n, m, k, 
                          &alpha,
                          B, AType, n,
                          A, BType, k,
                          &beta,
                          C, CType, n,
                          ComputeType,
                          CUBLAS_GEMM_ALGO0_TENSOR_OP);
}

template <typename T, typename S>
void test_gemm (int m, int n, int k, 
               T *A, T *B, S *C,
               int iteration)  {

    cublas_gemm_ex(A, B, C,
                                 m, n, k);
    cudaDeviceSynchronize();

    timer tm;

    for (int i = 0; i < iteration; ++i) {
        tm.start();
        cublasStatus_t success;
        success = cublas_gemm_ex(A, B, C,
                                 m, n, k);
        cudaDeviceSynchronize();
        // printf("%d\n", success);

        if (success == CUBLAS_STATUS_SUCCESS){
            tm.end();
        }
    }
    if (tm.arr.size() > 0){
        printf("iter : %lu, latency : (%lf, %lf) mean %lfms\n", tm.arr.size() , tm.min(), tm.max(), tm.mean());
    }
}

template <typename T, typename S>
void test_sgemm (int m, int n, int k, 
               T *A, T *B, S *C,
               int iteration)  {

    cublas_gemm_ex(A, B, C,
                   m, n, k);
    cudaDeviceSynchronize();

     timer tm;

    for (int i = 0; i < iteration; ++i) {
        tm.start();
        cublasStatus_t success;
        success = cublas_gemm_ex(A, B, C,
                                 m, n, k);
        cudaDeviceSynchronize();
        // printf("%d\n", success);

        if (success == CUBLAS_STATUS_SUCCESS){
            tm.end();
        }
    }
    if (tm.arr.size() > 0){
        printf("iter : %lu, latency : (%lf, %lf) mean %lfms\n", tm.arr.size() , tm.min(), tm.max(), tm.mean());
    }
}
void gMatmul(double* C, double *A, double *B, int M, int N, int K, int tran_A, int trans_B){
    memset(C, 0, M*N*sizeof(double));
    for(int i=0;i<N;i++){
        for(int j=0;j<M;j++){
            for(int k=0;k<K;k++){
                C[i + j*N] += A[K*j + k] * B[k*N + i];
            }
        }
    }
    
}

float _abs(float d){return d<0?-d:d;}

void test_cuBlas(int m, int n, int k, bool cmp_check, int iter = 128){
    printf("=================== cuBlas: =================== \n");
    printf("shape: (%d, %d) x (%d, %d)\n", m, k, k, n);
    int iteration = iter;
    
    double *A = new double[m*k];
    double *B = new double[k*n];
    double *C = new double[m*n];

    float *fA, *fB, *fC;
    __half *hA, *hB, *hC;
    int8_t *iA, *iB; int32_t *iC;
    float f_alpha = 1, f_beta = 0;
    __half h_alpha = __float2half_rn(1.0), h_beta = __float2half_rn(0.0);
    int32_t i_alpha = 1, i_beta = 0;
    allocate_memory(m, n, k, &fA, &fB, &fC);
    allocate_memory(m, n, k, &hA, &hB, &hC);
    allocate_memory(m, n, k, &iA, &iB, &iC);
    
    
    for (int i = 0; i < m * k; ++i) {
        fA[i] = A[i] = rand_fp32();
        hA[i] = __float2half_rn(fA[i]);
    } 
    floatToInt8(iA, fA, m*k);
    for (int i = 0; i < k * n; ++i) {
        fB[i] = B[i] = rand_fp32();
        hB[i] = __float2half_rn(fB[i]);
        
    }
    floatToInt8(iB, fB, k*n);

    cublasHandle_t handle;
    cublasCreate(&handle);
    
    printf(">>>>>>>>>>>>>>>>> test fp32 >>>>>>>>>>>>>>>>>\n");
    test_gemm(m, n, k, fA, fB, fC, iteration);
    printf(">>>>>>>>>>>>>>>>> test fp16 >>>>>>>>>>>>>>>>>\n");
    test_gemm(m, n, k, hA, hB, hC, iteration);
    printf(">>>>>>>>>>>>>>>>> test int8 >>>>>>>>>>>>>>>>>\n");
    test_sgemm(m, n, k, iA, iB, iC, iteration);
    if(cmp_check){
        printf(">>>>>>>>>>>>>>>>> compare result >>>>>>>>>>>>>>>>>\n");
        gMatmul(C, A, B, m, n, k, 0, 0);

        printf("gfp32: ");

        double ferr = 0.0;
        for (int i = 0; i < m; i++)
            for(int j = 0; j < n;j++)
                ferr += _abs(C[i * n + j] - fC[i * n + j]); 

        double herr = 0.0;
        for (int i = 0; i < m; i++)
            for(int j = 0; j < n;j++)
                herr += _abs(C[i * n + j] - float(hC[i * n + j])); 

        double ierr = 0.0;
        for (int i = 0; i < m; i++)
            for(int j = 0; j < n;j++)
                ierr += _abs(C[i * n + j] - float(iC[i * n + j])/127/127); 


        printf("fp32 mean error : %lf\n", ferr/m/n);
        printf("fp16 mean error : %lf\n", herr/m/n);
        printf("int8 mean error : %lf\n", ierr/m/n);
    }
    free_memory(iA, iB, iC);
    free_memory(fA, fB, fC);
    free_memory(hA, hB, hC);

    
    delete[] A;
    delete[] B;
    delete[] C;
}

