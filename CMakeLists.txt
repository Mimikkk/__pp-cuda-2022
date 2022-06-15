cmake_minimum_required(VERSION 3.22)

project(pp_cuda_reduction CUDA)

set(CMAKE_CUDA_STANDARD 23)

set(SRC_DIR ${CMAKE_SOURCE_DIR}/src)
set(BIN_DIR ${CMAKE_SOURCE_DIR}/bin)

set(EXECUTABLE_OUTPUT_PATH ${BIN_DIR})
file(GLOB_RECURSE SOURCES ${SRC_DIR}/*.hpp ${SRC_DIR}/*.cpp ${SRC_DIR}/*.cu)

add_executable(pp_cuda_reduction src/main.cu ${SOURCES})

set_target_properties(pp_cuda_reduction PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
