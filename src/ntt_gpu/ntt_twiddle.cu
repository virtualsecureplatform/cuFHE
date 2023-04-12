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

#include <include/ntt_gpu/ntt_twiddle.cuh>
#include <include/details/error_gpu.cuh>
#include <include/details/assert.h>
#include <include/details/allocator_gpu.cuh>
#include <params.hpp>

namespace cufhe {

constexpr int numgentwdthreadbit = 3;
constexpr int numgentwdthread = 1<<numgentwdthreadbit;
constexpr int remnumgentwdthreadbit = TFHEpp::lvl1param::nbit % numgentwdthreadbit;
constexpr int remnumgentwdthread = 1<<remnumgentwdthreadbit;
__global__
void __GenTwd__(FFP* twd, FFP* twd_inv) {
  constexpr uint32_t n = TFHEpp::lvl1param::n;
  FFP w = FFP::Root(n);
  const uint32_t cid = (threadIdx.z << (2*numgentwdthreadbit)) + (threadIdx.y << numgentwdthreadbit) + threadIdx.x;
  for (int i = 0; i < numgentwdthread; i ++) {
    const uint32_t e = (threadIdx.z * numgentwdthread + threadIdx.y / 4 * 4 + (threadIdx.x % 4))
      * (i * numgentwdthread + (threadIdx.y % 4) * 2 + threadIdx.x / 4);
    const uint32_t idx = (i * n / numgentwdthread) + cid;
    twd[idx] = FFP::Pow(w, e);
    twd_inv[idx] = FFP::Pow(w, (n - e) % n);
  }
}

__global__
void __GenTwdSqrt__(FFP* twd_sqrt, FFP* twd_sqrt_inv) {
  constexpr uint32_t n = TFHEpp::lvl1param::n;
  uint32_t idx = (uint32_t)blockIdx.x * blockDim.x + threadIdx.x;
  FFP w = FFP::Root(2 * n);
  FFP n_inv = FFP::InvPow2(TFHEpp::lvl1param::nbit);
  twd_sqrt[idx] = FFP::Pow(w, idx);
  twd_sqrt_inv[idx] = FFP::Pow(w, (2 * n - idx) % (2 * n)) * n_inv;
}

constexpr int numgentwdsqrtthread = 1<<6;
template <>
void CuTwiddle<1024>::Create() {
  assert(this->twd_ == nullptr);
  size_t nbytes = sizeof(FFP) * TFHEpp::lvl1param::n * 4;
  this->twd_ = (FFP*)AllocatorGPU::New(nbytes).first;
  this->twd_inv_ = this->twd_ + TFHEpp::lvl1param::n;
  this->twd_sqrt_ = this->twd_inv_ + TFHEpp::lvl1param::n;
  this->twd_sqrt_inv_ = this->twd_sqrt_ + TFHEpp::lvl1param::n;
  __GenTwd__<<<1, dim3(numgentwdthread, numgentwdthread, remnumgentwdthread)>>>(this->twd_, this->twd_inv_);
  __GenTwdSqrt__<<<TFHEpp::lvl1param::n/numgentwdsqrtthread, numgentwdsqrtthread>>>(this->twd_sqrt_, this->twd_sqrt_inv_);
  cudaDeviceSynchronize();
  CuCheckError();
}

template <>
void CuTwiddle<1024>::Destroy() {
  assert(this->twd_ != nullptr);
  CuSafeCall(cudaFree(this->twd_));
  this->twd_ = nullptr;
  this->twd_inv_ = nullptr;
  this->twd_sqrt_ = nullptr;
  this->twd_sqrt_inv_ = nullptr;
}

} // namespace cufhe
