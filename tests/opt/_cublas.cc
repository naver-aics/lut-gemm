#include "gtest/gtest.h"
#include "_cublas.h"

TEST(tests, cublas_att){
    test_cuBlas(32, 1, 32, true, 0);
    EXPECT_TRUE(true);
}
const int hidden_size = 8192;
const int num_gpu = 1;
TEST(tests, cublas_ffn){
    test_cuBlas(251*128,   1024,    1024,   true, 128);

    EXPECT_TRUE(true);
}
