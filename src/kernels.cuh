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
  This file contains the important stuff like the CUDA kernel and physical
    equations.
*/

#pragma once

#include <cstddef>
#include <iostream>
#include <cstdio>

#include <thrust/transform.h>  // For scrunch_x2
#include <thrust/iterator/counting_iterator.h>

using gpu_size_t = std::size_t;

// Kernel tuning parameters
#define DEDISP_SAMPS_PER_THREAD 2  // 4 is better for Fermi?

__constant__ dedisp_float c_delay_table[DEDISP_MAX_NCHANS];
__constant__ dedisp_bool c_killmask[DEDISP_MAX_NCHANS];

template <int NBITS, typename T = unsigned int>
struct max_value {
    static const T value = (((unsigned)1 << (NBITS - 1)) - 1) * 2 + 1;
};

template <int NBITS, typename T>
inline __host__ __device__ T extract_subword(T value, int idx) {
    enum { MASK = max_value<NBITS, T>::value };
    return (value >> (idx * NBITS)) & MASK;
}

// Summation type metafunction
template <int IN_NBITS>
struct SumType {
    typedef dedisp_word type;
};
// Note: For 32-bit input, we must accumulate using a larger data type
template <>
struct SumType<32> {
    typedef unsigned long long type;
};

class Dedisperse {
    // dedisp_specs
    // Kernel tuning parameters
    static const gpu_size_t BLOCK_SIZE         = 256;
    static const gpu_size_t BLOCK_SAMPS        = 8;
    static const gpu_size_t SAMPS_PER_THREAD   = 2;  // 4 is better for Fermi?
    static const gpu_size_t MAX_GRID_DIMENSION = 65535;

    static const gpu_size_t BLOCK_DIM_X = BLOCK_SAMPS;
    static const gpu_size_t BLOCK_DIM_Y = BLOCK_SIZE / BLOCK_SAMPS;

public:
    Dedisperse() {}

    void dedisperse(const dedisp_word* d_in,
                    dedisp_size in_stride,
                    dedisp_size nsamps,
                    dedisp_size in_nbits,
                    dedisp_size nchans,
                    dedisp_size chan_stride,
                    const dedisp_float* d_dm_list,
                    dedisp_size dm_count,
                    dedisp_size dm_stride,
                    dedisp_float* d_out,
                    dedisp_size out_stride,
                    cudaStream_t stream = 0);
};

// Note: This assumes consecutive input words are consecutive times,
//         but that consecutive subwords are consecutive channels.
//       E.g., Words bracketed: (t0c0,t0c1,t0c2,t0c3), (t1c0,t1c1,t1c2,t1c3),...
// Note: out_stride should be in units of samples
template <int IN_NBITS, int SAMPS_PER_THREAD>
__global__ void dedisperse_kernel(const dedisp_word* d_in,
                                  dedisp_size nsamps,
                                  dedisp_size nsamps_reduced,
                                  dedisp_size stride,
                                  dedisp_size dm_count,
                                  dedisp_size dm_stride,
                                  dedisp_size nchans,
                                  dedisp_size chan_stride,
                                  dedisp_float* d_out,
                                  dedisp_size out_stride,
                                  const dedisp_float* d_dm_list) {
    // Compute compile-time constants
    enum {
        BITS_PER_BYTE  = 8,
        CHANS_PER_WORD = sizeof(dedisp_word) * BITS_PER_BYTE / IN_NBITS
    };

    // Compute the thread decomposition
    dedisp_size samp_idx = blockIdx.x * blockDim.x + threadIdx.x;
    dedisp_size dm_idx   = blockIdx.y * blockDim.y + threadIdx.y;
    // For Grid-Stride Loops
    dedisp_size nsamp_threads = gridDim.x * blockDim.x;
    dedisp_size ndm_threads   = gridDim.y * blockDim.y;

    // Iterate over grids of DMs
    for (; dm_idx < dm_count; dm_idx += ndm_threads) {

        // Look up the dispersion measure
        dedisp_float dm = d_dm_list[dm_idx * dm_stride];

        // Loop over samples
        for (; samp_idx < nsamps_reduced; samp_idx += nsamp_threads) {
            // all elements 0 in C++
			typedef typename SumType<IN_NBITS>::type sum_type;
			sum_type sum[SAMPS_PER_THREAD] = {};

            // Loop over channel words
            for (dedisp_size chan_word = 0; chan_word < nchans;
                 chan_word += CHANS_PER_WORD) {
                // Pre-compute the memory offset
                dedisp_size offset = samp_idx * SAMPS_PER_THREAD
                                     + chan_word / CHANS_PER_WORD * stride;

                // Loop over channel subwords
                for (dedisp_size chan_sub = 0; chan_sub < CHANS_PER_WORD;
                     ++chan_sub) {
                    dedisp_size chan_idx = (chan_word + chan_sub) * chan_stride;

                    // Look up the fractional delay
                    dedisp_float frac_delay = c_delay_table[chan_idx];
                    // Compute the integer delay
                    dedisp_size delay = __float2uint_rn(dm * frac_delay);

// Note: Unrolled to ensure the sum[] array is stored in regs
#pragma unroll
                    for (dedisp_size s = 0; s < SAMPS_PER_THREAD; ++s) {
                        // Grab the word containing the sample from global
                        // mem
                        dedisp_word sample = d_in[offset + s + delay];
						//printf("sample = %u \n", sample);
                        // Extract the desired subword and accumulate
                        sum[s] += c_killmask[chan_idx]
                                  * extract_subword<IN_NBITS>(sample, chan_sub);
						//printf("sum[s] 1 = %f \n", sum[s]);
                    }
                }
            }

            // Write sums to global mem
            dedisp_size out_idx
                = (samp_idx * SAMPS_PER_THREAD + dm_idx * out_stride);
#pragma unroll
            for (dedisp_size s = 0; s < SAMPS_PER_THREAD; ++s) {
                if (samp_idx * SAMPS_PER_THREAD + s < nsamps) {
                    d_out[out_idx + s] = (dedisp_float)sum[s];
					//printf("sum[s] 2 = %f \n", sum[s]);
                }
            }

        }  // End of sample loop

    }  // End of DM loop
}

void Dedisperse::dedisperse(const dedisp_word* d_in,
                            dedisp_size in_stride,
                            dedisp_size nsamps,
                            dedisp_size in_nbits,
                            dedisp_size nchans,
                            dedisp_size chan_stride,
                            const dedisp_float* d_dm_list,
                            dedisp_size dm_count,
                            dedisp_size dm_stride,
                            dedisp_float* d_out,
                            dedisp_size out_stride,
                            cudaStream_t stream) {
	//printf("sample = %u \n", d_in[0]);
    // Define thread decomposition
    dim3 grid, block;
    // Note: Block dimensions x and y represent time samples and DMs
    //       respectively.
    block.x = BLOCK_DIM_X;
    block.y = BLOCK_DIM_Y;
    // Note: Grid dimension x represents time samples. Dimension y represents
    //       DMs and batch jobs flattened together.
    // Constrain the grid size to the maximum allowed
    grid.x = div_up(nsamps, (dedisp_size)BLOCK_DIM_X * SAMPS_PER_THREAD);
    grid.y = std::min(div_up(dm_count, (dedisp_size)BLOCK_DIM_Y),
                      MAX_GRID_DIMENSION);

    // Divide and round up
    dedisp_size nsamps_reduced = div_up(nsamps, SAMPS_PER_THREAD);

    // Execute the kernel
#define DEDISP_CALL_KERNEL(NBITS)                                             \
    dedisperse_kernel<NBITS, SAMPS_PER_THREAD><<<grid, block, 0, stream>>>(   \
        d_in, nsamps, nsamps_reduced, in_stride, dm_count, dm_stride, nchans, \
        chan_stride, d_out, out_stride, d_dm_list)

    // Note: Here we dispatch dynamically on nbits for supported values
    switch (in_nbits) {
        case 1:
            DEDISP_CALL_KERNEL(1);
            break;
        case 2:
            DEDISP_CALL_KERNEL(2);
            break;
        case 4:
            DEDISP_CALL_KERNEL(4);
            break;
        case 8:
            DEDISP_CALL_KERNEL(8);
            break;
        case 16:
            DEDISP_CALL_KERNEL(16);
            break;
        case 32:
            DEDISP_CALL_KERNEL(32);
            break;
        default: /* should never be reached */
            break;
    }

#undef DEDISP_CALL_KERNEL
    // Execute the kernel
    /*
    void* args[] = {&d_in,     &nsamps,     &nsamps_reduced, &in_stride,
                    &dm_count, &dm_stride,  &nchans,         &chan_stride,
                    &d_out,    &out_stride, &d_dm_list};

    cudaLaunchKernel((void*)dedisperse_kernel<1, SAMPS_PER_THREAD>, grid, block,
                     &args[0], 0, stream);
    */
}

template <typename WordType>
struct scrunch_x2_functor
    : public thrust::unary_function<unsigned int, WordType> {
    const WordType* in;
    int nbits;
    WordType mask;
    unsigned int in_nsamps;
    unsigned int out_nsamps;
    scrunch_x2_functor(const WordType* in_, int nbits_, unsigned int in_nsamps_)
        : in(in_), nbits(nbits_), mask((1 << nbits) - 1), in_nsamps(in_nsamps_),
          out_nsamps(in_nsamps_ / 2) {}
    inline __host__ __device__ WordType operator()(unsigned int out_i) const {
        unsigned int c     = out_i / out_nsamps;
        unsigned int out_t = out_i % out_nsamps;
        unsigned int in_t0 = out_t * 2;
        unsigned int in_t1 = out_t * 2 + 1;
        unsigned int in_i0 = c * in_nsamps + in_t0;
        unsigned int in_i1 = c * in_nsamps + in_t1;

        dedisp_word in0 = in[in_i0];
        dedisp_word in1 = in[in_i1];
        dedisp_word out = 0;
        for (int k = 0; k < sizeof(WordType) * 8; k += nbits) {
            dedisp_word s0  = (in0 >> k) & mask;
            dedisp_word s1  = (in1 >> k) & mask;
            dedisp_word avg = ((unsigned long long)s0 + s1) / 2;
            out |= avg << k;
        }
        return out;
    }
};

// Reduces the time resolution by 2x
dedisp_error scrunch_x2(const dedisp_word* d_in,
                        dedisp_size nsamps,
                        dedisp_size nchan_words,
                        dedisp_size nbits,
                        dedisp_word* d_out) {
    thrust::device_ptr<dedisp_word> d_out_begin(d_out);

    dedisp_size out_nsamps = nsamps / 2;
    dedisp_size out_count  = out_nsamps * nchan_words;

    using thrust::make_counting_iterator;

    thrust::transform(make_counting_iterator<unsigned int>(0),
                      make_counting_iterator<unsigned int>(out_count),
                      d_out_begin,
                      scrunch_x2_functor<dedisp_word>(d_in, nbits, nsamps));

    return DEDISP_NO_ERROR;
}

template <typename WordType>
struct unpack_functor : public thrust::unary_function<unsigned int, WordType> {
    const WordType* in;
    int nsamps;
    int in_nbits;
    int out_nbits;
    unpack_functor(const WordType* in_,
                   int nsamps_,
                   int in_nbits_,
                   int out_nbits_)
        : in(in_), nsamps(nsamps_), in_nbits(in_nbits_), out_nbits(out_nbits_) {
    }
    inline __host__ __device__ WordType operator()(unsigned int i) const {
        int out_chans_per_word = sizeof(WordType) * 8 / out_nbits;
        int in_chans_per_word  = sizeof(WordType) * 8 / in_nbits;
        // int expansion = out_nbits / in_nbits;
        int norm          = ((1l << out_nbits) - 1) / ((1l << in_nbits) - 1);
        WordType in_mask  = (1 << in_nbits) - 1;
        WordType out_mask = (1 << out_nbits) - 1;

        /*
          cw\k 0123 0123
          0    0123|0123
          1    4567|4567

          cw\k 0 1
          0    0 1 | 0 1
          1    2 3 | 2 3
          2    4 5 | 4 5
          3    6 7 | 6 7


         */

        unsigned int t = i % nsamps;
        // Find the channel word indices
        unsigned int out_cw = i / nsamps;
        // unsigned int in_cw  = out_cw / expansion;
        // unsigned int in_i   = in_cw * nsamps + t;
        // WordType word = in[in_i];

        WordType result = 0;
        for (int k = 0; k < sizeof(WordType) * 8; k += out_nbits) {

            int c         = out_cw * out_chans_per_word + k / out_nbits;
            int in_cw     = c / in_chans_per_word;
            int in_k      = c % in_chans_per_word * in_nbits;
            int in_i      = in_cw * nsamps + t;
            WordType word = in[in_i];

            WordType val = (word >> in_k) & in_mask;
            result |= ((val * norm) & out_mask) << k;
        }
        return result;
    }
};

dedisp_error unpack(const dedisp_word* d_transposed,
                    dedisp_size nsamps,
                    dedisp_size nchan_words,
                    dedisp_word* d_unpacked,
                    dedisp_size in_nbits,
                    dedisp_size out_nbits) {
    thrust::device_ptr<dedisp_word> d_unpacked_begin(d_unpacked);

    dedisp_size expansion = out_nbits / in_nbits;
    dedisp_size in_count  = nsamps * nchan_words;
    dedisp_size out_count = in_count * expansion;

    using thrust::make_counting_iterator;

    thrust::transform(
        make_counting_iterator<unsigned int>(0),
        make_counting_iterator<unsigned int>(out_count), d_unpacked_begin,
        unpack_functor<dedisp_word>(d_transposed, nsamps, in_nbits, out_nbits));

    return DEDISP_NO_ERROR;
}
