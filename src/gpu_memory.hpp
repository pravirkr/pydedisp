/*
 *  Copyright 2012 Ben Barsdell
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *      distributed under the License is distributed on an "AS IS" BASIS,
 *      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

/*
  This file just contains crappy wrappers for CUDA memory functions
*/

#pragma once

#include <cstddef>

using gpu_size_t = std::size_t;

/**
 * @brief Allocate memory on the device
 *
 * @tparam T
 * @param addr   Pointer to allocate memory.
 * @param count  Number of memory blocks to allocate.
 * @return true  If allocation is successful.
 * @return false If device memory allocation fails.
 */
template <typename T>
bool malloc_device(T*& addr, gpu_size_t count) {
    cudaError_t error = cudaMalloc((void**)&addr, count * sizeof(T));
    if (error != cudaSuccess) {
        return false;
    }
    return true;
}

/**
 * @brief Free device memory
 *
 * @tparam T
 * @param addr Device pointer
 */
template <typename T>
void free_device(T*& addr) {
    cudaFree(addr);
    addr = 0;
}

/**
 * @brief Copy host memory to device memory.
 *
 * @tparam T
 * @param dst    Device pointer.
 * @param src    Host pointer.
 * @param count  Number of memory blocks to copy.
 * @return true  If copy is successful.
 * @return false If data copy fails.
 */
template <typename T>
bool copy_host_to_device(T* dst, const T* src, gpu_size_t count) {
    cudaMemcpy(dst, src, count * sizeof(T), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        return false;
    }
    return true;
}

/**
 * @brief Copy device memory to host memory.
 *
 * @tparam T
 * @param dst    Host pointer.
 * @param src    Device pointer.
 * @param count  Number of memory blocks to copy.
 * @return true  If copy is successful.
 * @return false If data copy fails.
 */
template <typename T>
bool copy_device_to_host(T* dst, const T* src, gpu_size_t count) {
    cudaMemcpy(dst, src, count * sizeof(T), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        return false;
    }
    return true;
}

template <typename T>
bool copy_host_to_symbol(const T* symbol, const T* src, gpu_size_t count) {
    cudaMemcpyToSymbol(symbol, src, count * sizeof(T), 0,
                       cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        return false;
    }
    return true;
}

template <typename T>
bool copy_device_to_symbol(const T* symbol, const T* src, gpu_size_t count) {
    cudaMemcpyToSymbol(symbol, src, count * sizeof(T), 0,
                       cudaMemcpyDeviceToDevice);
    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        return false;
    }
    return true;
}

// Note: Strides must be given in units of bytes
template <typename T, typename U>
bool copy_host_to_device_2d(T* dst,
                            gpu_size_t dst_stride,
                            const U* src,
                            gpu_size_t src_stride,
                            gpu_size_t width_bytes,
                            gpu_size_t height) {
    cudaMemcpy2D(dst, dst_stride, src, src_stride, width_bytes, height,
                 cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        return false;
    }
    return true;
}

template <typename T, typename U>
bool copy_device_to_host_2d(T* dst,
                            gpu_size_t dst_stride,
                            const U* src,
                            gpu_size_t src_stride,
                            gpu_size_t width_bytes,
                            gpu_size_t height) {
    cudaMemcpy2D(dst, dst_stride, src, src_stride, width_bytes, height,
                 cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        return false;
    }
    return true;
}

template <typename T, typename U>
bool copy_device_to_device_2d(T* dst,
                              gpu_size_t dst_stride,
                              const U* src,
                              gpu_size_t src_stride,
                              gpu_size_t width_bytes,
                              gpu_size_t height) {
    cudaMemcpy2D(dst, dst_stride, src, src_stride, width_bytes, height,
                 cudaMemcpyDeviceToDevice);
    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        return false;
    }
    return true;
}
