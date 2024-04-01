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

#ifndef TMP_WEIGHT_HPP
#define TMP_WEIGHT_HPP


class tmpWeight{
public:
    static tmpWeight& getInstance(){
        static tmpWeight ins;
        return ins;
    }

    float* getWeight(int Size){
        if(Size > size){
            mem_free();
            size = Size;
            cudaMallocManaged(&mem, sizeof(float) * size);
        }
        return mem;
    }

private:
    void mem_free(){
        if(mem != nullptr)
            cudaFree(mem);
    }
    float *mem = nullptr;
    int size = 0;

    tmpWeight(const tmpWeight&) = delete;
    tmpWeight& operator=(const tmpWeight&) = delete;
    tmpWeight(/* args */){ }
    ~tmpWeight(){
        mem_free();
    }
};


#endif // TMP_WEIGHT_HPP