cmake_minimum_required(VERSION 3.18)
project(nQmatmul CXX C)
enable_language(CUDA)

find_package(CUDA 10.1 REQUIRED)


if(NOT DEFINED ${CMAKE_CUDA_ARCHITECTURES})
    set(CMAKE_CUDA_ARCHITECTURES 80)
endif()

add_subdirectory(thirdparty/googletest)
add_subdirectory(lutGEMM)
add_subdirectory(tests)

add_library(${PROJECT_NAME}
    INTERFACE
)

target_link_libraries(${PROJECT_NAME} 
    INTERFACE
        gtest
        lutGEMM
)
