cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(tests LANGUAGES CXX CUDA)
enable_language(CUDA)

# set(CMAKE_CXX_STANDARD 14)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)
# set(CMAKE_CXX_EXTENSIONS OFF)
# add_library(my_lib STATIC opt/_cublas.cu)
# set_property(TARGET my_lib PROPERTY POSITION_INDEPENDENT_CODE  ON)
# set_property(TARGET my_lib PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
# target_link_libraries(my_lib PUBLIC -lcublas -lcudart -lcurand -lgomp)
# target_include_directories(my_lib
#     PUBLIC
#         include
#         gtest
#         lutGEMM

# )
add_executable(tests 
    src/custom_random.cpp
    main.cc

    opt/fp16/int3_col_wise_matmul_fp16.cu

)
set_target_properties(tests PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(tests PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_target_properties(tests PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(tests -lcublasLt -lcublas -lcurand -lcudart -lcuda -lgomp lutgemm)

# set(CMAKE_CXX_FLAGS_DEBUG "-O3 -fopenmp")



set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

target_include_directories(${PROJECT_NAME}
    PUBLIC
        include
        gtest
        lutGEMM

)

target_link_libraries(${PROJECT_NAME} 
    # PRIVATE 
    gtest
    lutGEMM
)