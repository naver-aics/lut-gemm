

# project(lutGEMM)
project(lutGEMM LANGUAGES CXX CUDA)
enable_language(CUDA)

add_library(${PROJECT_NAME} INTERFACE)


add_library(lutgemm STATIC src/nQWeight_fp16.cu src/kernels.cu)
set_property(TARGET lutgemm PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET lutgemm PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(lutgemm PUBLIC -lcublas -lcudart -lcurand -lgomp)


if(NOT DEFINED ${CMAKE_CUDA_ARCHITECTURES})
    set(CMAKE_CUDA_ARCHITECTURES 80)
endif()

target_include_directories(${PROJECT_NAME}
    INTERFACE
        include
)
