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

/**
 * Unified NTT interface for cuFHE
 *
 * Switches between two implementations based on USE_SMALL_NTT_MODULUS:
 *
 * 1. Large modulus (default): ~2^60 modulus, exact integer arithmetic
 *    - NTT domain uses 64-bit FFP type
 *    - No modulus switching needed
 *    - BK storage: 8 bytes per NTT element
 *
 * 2. Small modulus (USE_SMALL_NTT_MODULUS): ~2^29 modulus (RAINTT-style)
 *    - NTT domain uses 32-bit uint32_t type
 *    - Torus discretization switching: 2^32 <-> NTT modulus P
 *    - BK storage: 4 bytes per NTT element (50% memory reduction)
 *
 * Set USE_SMALL_NTT_MODULUS via CMake: -DUSE_SMALL_NTT_MODULUS=ON
 */

#pragma once

#include <include/details/error_gpu.cuh>
#include <params.hpp>

// Always include GPU-NTT for FFP type (CMUX operations need it)
#include "ntt_gpuntt.cuh"

#ifdef USE_SMALL_NTT_MODULUS
// Also include small modulus implementation
#include "ntt_small_modulus.cuh"
#endif

namespace cufhe {

#ifdef USE_SMALL_NTT_MODULUS

// ============================================================================
// Small Modulus NTT Implementation (RAINTT-style)
// P = 655360001 (~2^29) with Torus discretization switching
// ============================================================================

// NTT value type: 32-bit for small modulus
using NTTValue = uint32_t;

// Shared memory size per gate: (k+2) * N * sizeof(uint32_t)
template<class P = TFHEpp::lvl1param>
constexpr uint32_t MEM4HOMGATE = (P::k + 2) * P::n * sizeof(uint32_t);

// Number of threads for NTT (N/2 = 512 for N=1024)
template<class P = TFHEpp::lvl1param>
constexpr uint32_t NUM_THREAD4HOMGATE = P::n >> 1;

// Modulus switching is enabled for small modulus
constexpr bool NTT_USE_MODSWITCH = true;

#ifdef __CUDACC__

// Convert Torus32 to NTT domain (with modswitch)
__device__ __forceinline__ NTTValue torusToNTT(uint32_t torus_val) {
    return torus32_to_ntt_mod(torus_val);
}

// Convert NTT domain to Torus32 (with modswitch)
__device__ __forceinline__ uint32_t nttToTorus(int32_t ntt_val) {
    return ntt_mod_to_torus32(ntt_val);
}

// Modular multiplication in NTT domain
__device__ __forceinline__ NTTValue nttMult(NTTValue a, NTTValue b) {
    return small_mod_mult(a, b);
}

// Modular addition in NTT domain
__device__ __forceinline__ NTTValue nttAdd(NTTValue a, NTTValue b) {
    return small_mod_add(a, b);
}

// Convert signed integer to NTT domain value
__device__ __forceinline__ NTTValue intToNTT(int32_t val) {
    return (val < 0) ? (small_ntt::P + val) : static_cast<uint32_t>(val);
}

// Centered reduction: NTT value to signed
__device__ __forceinline__ int32_t nttToSigned(NTTValue val) {
    constexpr uint32_t half_mod = small_ntt::P / 2;
    return (val > half_mod) ? static_cast<int32_t>(val - small_ntt::P) : static_cast<int32_t>(val);
}

#endif  // __CUDACC__

// Get NTT modulus
constexpr uint32_t getNTTModulus() { return small_ntt::P; }
constexpr uint32_t getNTTHalfModulus() { return small_ntt::P / 2; }

#else  // !USE_SMALL_NTT_MODULUS

// ============================================================================
// Large Modulus NTT Implementation (default)
// ~2^60 modulus with exact integer arithmetic
// ============================================================================

// NTT value type: 64-bit FFP for large modulus
using NTTValue = FFP;

// Shared memory size per gate: (k+2) * N * sizeof(FFP)
template<class P = TFHEpp::lvl1param>
constexpr uint32_t MEM4HOMGATE = (P::k + 2) * P::n * sizeof(FFP);

// Number of threads for NTT (N/2 = 512 for N=1024)
template<class P = TFHEpp::lvl1param>
constexpr uint32_t NUM_THREAD4HOMGATE = P::n >> NTT_THREAD_UNITBIT;

// No modulus switching for large modulus
constexpr bool NTT_USE_MODSWITCH = false;

#ifdef __CUDACC__

// Convert Torus32 to NTT domain (direct cast, no modswitch)
__device__ __forceinline__ NTTValue torusToNTT(uint32_t torus_val) {
    return FFP(torus_val);
}

// Convert NTT domain to Torus32 (direct cast after centered reduction)
__device__ __forceinline__ uint32_t nttToTorus(int64_t ntt_val) {
    return static_cast<uint32_t>(ntt_val);
}

// Modular multiplication in NTT domain
__device__ __forceinline__ NTTValue nttMult(NTTValue a, NTTValue b) {
    return a * b;
}

// Modular addition in NTT domain
__device__ __forceinline__ NTTValue nttAdd(NTTValue a, NTTValue b) {
    return a + b;
}

// Convert signed integer to NTT domain value
__device__ __forceinline__ NTTValue intToNTT(int32_t val) {
    return FFP(val);
}

// Centered reduction: NTT value to signed
__device__ __forceinline__ int64_t nttToSigned(NTTValue val) {
    constexpr uint64_t half_mod = GPUNTT_DEFAULT_MODULUS / 2;
    return (val.val() > half_mod)
        ? static_cast<int64_t>(val.val()) - static_cast<int64_t>(GPUNTT_DEFAULT_MODULUS)
        : static_cast<int64_t>(val.val());
}

#endif  // __CUDACC__

// Get NTT modulus
constexpr uint64_t getNTTModulus() { return GPUNTT_DEFAULT_MODULUS; }
constexpr uint64_t getNTTHalfModulus() { return GPUNTT_DEFAULT_MODULUS / 2; }

#endif  // USE_SMALL_NTT_MODULUS

}  // namespace cufhe
