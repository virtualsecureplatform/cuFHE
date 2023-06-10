/**
 * Copyright 2018 Wei Dai <wdai3141@gmail.com>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#pragma once

#include "ntt_ffp.cuh"
#include "ntt_single_thread.cuh"
#include "ntt_shifting.cuh"
#include <include/details/utils_gpu.cuh>

namespace cufhe {

__constant__ FFP con_1024_twd[1024];
__constant__ FFP con_1024_twd_inv[1024];
__constant__ FFP con_1024_twd_sqrt[1024];
__constant__ FFP con_1024_twd_sqrt_inv[1024];

constexpr uint numdataperthreadbit = 3;
constexpr uint dimoverthreadbit = 10 - numdataperthreadbit;

constexpr uint numdataperthread = 1<<numdataperthreadbit;

__device__ inline
void NTT1024Core(FFP* const r,
                 FFP* const s,
                 const uint32_t& t1d,
                 const uint3& t3d) {
  FFP *ptr = nullptr;
  #pragma unroll
  for (int i = 0; i < numdataperthread; i ++)
    r[i] *= con_1024_twd_sqrt[(i << dimoverthreadbit) | t1d]; // mult twiddle sqrt
  NTT8(r);
  NTT8x2Lsh(r, t3d.z); // if (t1d >= 64) NTT8x2<1>(r);
  ptr = &s[(t3d.y << 7) | (t3d.z << 6) | (t3d.x << 2)];
  #pragma unroll
  for (int i = 0; i < numdataperthread; i ++)
    ptr[(i >> 2 << 5) | (i & 0x3)] = r[i];
  __syncthreads();

  ptr = &s[(t3d.z << 9) | (t3d.y << 3) | t3d.x];
  #pragma unroll
  for (int i = 0; i < numdataperthread; i ++)
    r[i] = ptr[i << 6];
  #pragma unroll
  for (int i = 0; i < numdataperthread/2; i ++)
    NTT2(r + 2*i);
  #pragma unroll
  for (int i = 0; i < numdataperthread; i ++)
    ptr[i << 6] = r[i];
  __syncthreads();

  ptr = &s[t1d];
  #pragma unroll
  for (int i = 0; i < numdataperthread; i ++)
    r[i] = ptr[i << 7] * con_1024_twd[(i << dimoverthreadbit) | t1d]; // mult twiddle
  NTT8(r);
  #pragma unroll
  for (int i = 0; i < numdataperthread; i ++)
    ptr[i << 7] = r[i];
  __syncthreads();

  ptr = &s[(t1d >> 2 << 5) | (t3d.x & 0x3)];
  #pragma unroll
  for (int i = 0; i < numdataperthread; i ++)
    r[i] = ptr[i << 2];
  NTT8x8Lsh(r, t1d >> 4); // less divergence if put here!
  NTT8(r);
}

__device__ inline
void NTTInv1024Core(FFP* const r,
                    FFP* const s,
                    const uint32_t& t1d,
                    const uint3& t3d) {

  FFP *ptr = nullptr;
  NTTInv8(r);
  NTTInv8x2Lsh(r, t3d.z); // if (t1d >= 64) NTT8x2<1>(r);
  ptr = &s[(t3d.y << 7) | (t3d.z << 6) | (t3d.x << 2)];
  #pragma unroll
  for (int i = 0; i < numdataperthread; i ++)
    ptr[(i >> 2 << 5) | (i & 0x3)] = r[i];
  __syncthreads();

  ptr = &s[(t3d.z << 9) | (t3d.y << 3) | t3d.x];
  #pragma unroll
  for (int i = 0; i < numdataperthread; i ++)
    r[i] = ptr[i << 6];
  #pragma unroll
  for (int i = 0; i < numdataperthread/2; i ++)
    NTT2(r+2*i);
  #pragma unroll
  for (int i = 0; i < numdataperthread; i ++)
    ptr[i << 6] = r[i];
  __syncthreads();

  ptr = &s[t1d];
  #pragma unroll
  for (int i = 0; i < numdataperthread; i ++)
    r[i] = ptr[i << 7] * con_1024_twd_inv[(i << 7) | t1d]; // mult twiddle
  NTTInv8(r);
  #pragma unroll
  for (int i = 0; i < numdataperthread; i ++)
    ptr[i << 7] = r[i];
  __syncthreads();

  ptr = &s[(t1d >> 2 << 5) | (t3d.x & 0x3)];
  #pragma unroll
  for (int i = 0; i < numdataperthread; i ++)
    r[i] = ptr[i << 2];
  NTTInv8x8Lsh(r, t1d >> 4); // less divergence if put here!
  NTTInv8(r);
  #pragma unroll
  for (int i = 0; i < numdataperthread; i ++)
    r[i] *= con_1024_twd_sqrt_inv[(i << 7) | t1d]; // mult twiddle sqrt
}

template <typename T>
__device__
void NTT1024(FFP* const out,
             const T* const in,
             FFP* const temp_shared,
             const uint32_t leading_thread) {
  const uint32_t t1d = ThisThreadRankInBlock() - leading_thread;
  uint3 t3d;
  Index3DFrom1D<8, 8, 2>(t3d, t1d);
  register FFP r[numdataperthread];
  #pragma unroll
  for (int i = 0; i < numdataperthread; i ++)
    r[i] = FFP((T)in[(i << dimoverthreadbit) | t1d]);
  __syncthreads();
  NTT1024Core(r, temp_shared, t1d, t3d);
  __syncthreads();
  #pragma unroll
  for (int i = 0; i < numdataperthread; i ++)
    out[(i << dimoverthreadbit) | t1d] = r[i];
}

template <typename T>
__device__
void NTTInv1024(T* const out,
                const FFP* const in,
                FFP* const temp_shared,
                const uint32_t leading_thread) {
  const uint32_t t1d = ThisThreadRankInBlock() - leading_thread;
  uint3 t3d;
  Index3DFrom1D<8, 8, 2>(t3d, t1d);
  register FFP r[numdataperthread];
  #pragma unroll
  for (int i = 0; i < numdataperthread; i ++)
    r[i] = in[(i << dimoverthreadbit) | t1d];
  __syncthreads();
  NTTInv1024Core(r, temp_shared, t1d, t3d);
  __syncthreads();
  // mod 2^32 specifically
  constexpr uint64_t med = FFP::kModulus() / 2;
  #pragma unroll
  for (int i = 0; i < numdataperthread; i ++)
    out[(i << dimoverthreadbit) | t1d] = (T)r[i].val() - (r[i].val() > med);
}

template <typename T>
__device__
void NTTInv1024Add(T* const out,
                   const FFP* const in,
                   FFP* const temp_shared,
                   const uint32_t leading_thread) {
  const uint32_t t1d = ThisThreadRankInBlock() - leading_thread;
  uint3 t3d;
  Index3DFrom1D<8, 8, 2>(t3d, t1d);
  register FFP r[numdataperthread];
  #pragma unroll
  for (int i = 0; i < numdataperthread; i ++)
    r[i] = in[(i << dimoverthreadbit) | t1d];
  __syncthreads();
  NTTInv1024Core(r, temp_shared, t1d, t3d);
  __syncthreads();
  // mod 2^32 specifically
  constexpr uint64_t med = FFP::kModulus() / 2;
  #pragma unroll
  for (int i = 0; i < numdataperthread; i ++)
    out[(i << dimoverthreadbit) | t1d] += T(r[i].val() - (r[i].val() >= med));
}

} // namespace cufhe
