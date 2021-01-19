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
 * This file contains a CUDA implementation of the array transpose operation.
 *
 * Parts of this file are based on the transpose implementation in the
 * NVIDIA CUDA SDK.
 * https://docs.nvidia.com/cuda/samples/6_Advanced/transpose/doc/MatrixTranspose.pdf
 *
 * https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/
 */

#pragma once

#include <cstddef>
#include <algorithm>

#include "utils.hpp"

using gpu_size_t = std::size_t;

template <typename T>
class Transpose {
public:
    Transpose() {}

    void transpose(const T* in,
                   gpu_size_t width,
                   gpu_size_t height,
                   gpu_size_t in_stride,
                   gpu_size_t out_stride,
                   T* out,
                   cudaStream_t stream = 0);

    void transpose(const T* in,
                   gpu_size_t width,
                   gpu_size_t height,
                   T* out,
                   cudaStream_t stream = 0) {
        transpose(in, width, height, width, height, out, stream);
    }

private:
    // cuda specs
    static const gpu_size_t TILE_DIM           = 32;
    static const gpu_size_t BLOCK_ROWS         = 8;
    static const gpu_size_t MAX_GRID_DIMENSION = 65535;
};

/**
 * @brief  CUDA kernel for a strided transpose matrix operation on a given
 * strided square matrix.
 *
 * @tparam TILE_DIM   CUDA Shared memory tile size for kernel.
 * @tparam BLOCK_ROWS CUDA blocksize used for kernel launch.
 * @tparam T          Template parameter type of the data.
 * @param in          Input non-transposed strided matrix of type T.
 * @param out         Output transposed strided matrix of type T.
 * @param width
 * @param height
 * @param in_stride
 * @param out_stride
 * @return
 */
template <int TILE_DIM, int BLOCK_ROWS, typename T>
__global__ void transpose_kernel(const T* __restrict__ in,
                                 T* __restrict__ out,
                                 gpu_size_t width,
                                 gpu_size_t height,
                                 gpu_size_t in_stride,
                                 gpu_size_t out_stride) {
    __shared__ T tile[TILE_DIM][TILE_DIM + 1];

    gpu_size_t index_in_x = blockIdx.x * TILE_DIM + threadIdx.x;
    gpu_size_t index_in_y = blockIdx.y * TILE_DIM + threadIdx.y;
    gpu_size_t index_in   = index_in_x + (index_in_y)*in_stride;

#pragma unroll
    for (gpu_size_t i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
        // Avoid excess threads
        if (index_in_x < width && index_in_y + i < height) {
            tile[threadIdx.y + i][threadIdx.x]
                = LDG_LOAD(in[index_in + i * in_stride]);
        }
    }

    __syncthreads();

    gpu_size_t index_out_x = blockIdx.y * TILE_DIM + threadIdx.x;
    gpu_size_t index_out_y = blockIdx.x * TILE_DIM + threadIdx.y;
    gpu_size_t index_out   = index_out_x + (index_out_y)*out_stride;

#pragma unroll
    for (gpu_size_t i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
        // Avoid excess threads
        if (index_out_y + i < width && index_in_x < height) {
            out[index_out + i * out_stride]
                = tile[threadIdx.x][threadIdx.y + i];
        }
    }
}

template <typename T>
void Transpose<T>::transpose(const T* in,
                             gpu_size_t width,
                             gpu_size_t height,
                             gpu_size_t in_stride,
                             gpu_size_t out_stride,
                             T* out,
                             cudaStream_t stream) {
    // Parameter checks
    // TODO: Implement some sort of error returning!
    if (0 == width || 0 == height)
        return;
    if (0 == in)
        return;  // throw std::runtime_error("Transpose: in is NULL");
    if (0 == out)
        return;  // throw std::runtime_error("Transpose: out is NULL");
    if (width > in_stride)
        return;  // throw std::runtime_error("Transpose: width exceeds
                 // in_stride");
    if (height > out_stride)
        return;  // throw std::runtime_error("Transpose: height exceeds
                 // out_stride");

    // Specify thread decomposition (uses up-rounded divisions)
    dim3 tot_block_count(div_up(width, TILE_DIM), div_up(height, TILE_DIM));

    // Partition the grid into chunks that the GPU can accept at once
    for (gpu_size_t block_y_offset = 0; block_y_offset < tot_block_count.y;
         block_y_offset += MAX_GRID_DIMENSION) {

        dim3 block_count;
        // Handle the possibly incomplete final grid
        block_count.y
            = std::min(MAX_GRID_DIMENSION, tot_block_count.y - block_y_offset);

        for (gpu_size_t block_x_offset = 0; block_x_offset < tot_block_count.x;
             block_x_offset += MAX_GRID_DIMENSION) {

            // Handle the possibly incomplete final grid
            block_count.x = std::min(MAX_GRID_DIMENSION,
                                     tot_block_count.x - block_x_offset);

            // Compute the chunked parameters
            gpu_size_t x_offset   = block_x_offset * TILE_DIM;
            gpu_size_t y_offset   = block_y_offset * TILE_DIM;
            gpu_size_t in_offset  = x_offset + y_offset * in_stride;
            gpu_size_t out_offset = y_offset + x_offset * out_stride;
            gpu_size_t w
                = std::min(MAX_GRID_DIMENSION * TILE_DIM, width - x_offset);
            gpu_size_t h
                = std::min(MAX_GRID_DIMENSION * TILE_DIM, height - y_offset);

            dim3 block(TILE_DIM, BLOCK_ROWS);
            dim3 grid(block_count.x, block_count.y);
            // Run the CUDA kernel
            transpose_kernel<TILE_DIM, BLOCK_ROWS><<<grid, block, 0, stream>>>(
                in + in_offset, out + out_offset, w, h, in_stride, out_stride);
        }
    }
}
