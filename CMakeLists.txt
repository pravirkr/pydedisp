cmake_minimum_required(VERSION 3.0.0)

include(CheckLanguage)
check_language(CUDA)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED True)

project(dedisp VERSION 1.0.1 LANGUAGES CXX CUDA)

# CUDA NVCC FLAGS
set(CMAKE_CUDA_FLAGS -c -Xcompiler "-fPIC -Wall" -O3 -arch sm_60)

add_subdirectory("src")
add_subdirectory("example")
