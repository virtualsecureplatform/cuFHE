/**
 * Optimized Small NTT implementation for cuFHE
 * Based on HEonGPU's approach for maximum performance with N=1024
 *
 * Key optimizations:
 * - Uses N/2 threads (512 for N=1024), each handling 2 elements
 * - Optimized sync pattern: 5 syncs for forward NTT, 11 for inverse
 * - Barrett reduction with 60-bit prime (same as HEonGPU)
 */

#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <vector>
#include "gpuntt/common/modular_arith.cuh"
#include <params.hpp>
#include <include/details/utils_gpu.cuh>

namespace cufhe {

// Re-export GPU-NTT types for convenience
using NTTData = Data64;
using NTTModulus = Modulus<Data64>;
using NTTRoot = Root<Data64>;
using NTTNinverse = Ninverse<Data64>;

// Device-compatible modulus struct (same layout as NTTModulus)
struct DeviceModulus {
    Data64 value;
    Data64 bit;
    Data64 mu;
};

// HEonGPU's TFHE-optimized prime: 1152921504606877697 (~2^60)
constexpr Data64 GPUNTT_DEFAULT_MODULUS = 1152921504606877697ULL;

// Pre-computed Barrett reduction parameters
constexpr Data64 GPUNTT_MU = 9223372036854530040ULL;
constexpr Data64 GPUNTT_BIT = 61;

// Thread configuration for NTT
// N/2 threads, each handles 2 elements (e.g., 512 threads for N=1024)
// This matches HEonGPU's sequential NTT approach for maximum performance
constexpr uint32_t NTT_THREAD_UNITBIT = 1;

/**
 * @class FFP
 * @brief Finite Field element wrapper using GPU-NTT's prime
 */
class FFP {
private:
    uint64_t val_;

public:
    __host__ __device__ inline FFP() {}
    __host__ __device__ inline FFP(uint8_t a) : val_(a) {}
    __host__ __device__ inline FFP(uint16_t a) : val_(a) {}
    __host__ __device__ inline FFP(uint32_t a) : val_(a) {}
    __host__ __device__ inline FFP(uint64_t a) : val_(a) {}

    __host__ __device__ inline FFP(int8_t a) {
        val_ = (a < 0) ? (GPUNTT_DEFAULT_MODULUS - static_cast<uint64_t>(-a)) : static_cast<uint64_t>(a);
    }
    __host__ __device__ inline FFP(int16_t a) {
        val_ = (a < 0) ? (GPUNTT_DEFAULT_MODULUS - static_cast<uint64_t>(-a)) : static_cast<uint64_t>(a);
    }
    __host__ __device__ inline FFP(int32_t a) {
        val_ = (a < 0) ? (GPUNTT_DEFAULT_MODULUS - static_cast<uint64_t>(-static_cast<int64_t>(a))) : static_cast<uint64_t>(a);
    }
    __host__ __device__ inline FFP(int64_t a) {
        val_ = (a < 0) ? (GPUNTT_DEFAULT_MODULUS - static_cast<uint64_t>(-a)) : static_cast<uint64_t>(a);
    }

    __host__ __device__ inline ~FFP() {}

    __host__ __device__ inline uint64_t& val() { return val_; }
    __host__ __device__ inline const uint64_t& val() const { return val_; }
    __host__ __device__ inline static constexpr uint64_t kModulus() { return GPUNTT_DEFAULT_MODULUS; }

    __host__ __device__ inline FFP& operator=(uint8_t a) { val_ = a; return *this; }
    __host__ __device__ inline FFP& operator=(uint16_t a) { val_ = a; return *this; }
    __host__ __device__ inline FFP& operator=(uint32_t a) { val_ = a; return *this; }
    __host__ __device__ inline FFP& operator=(uint64_t a) { val_ = a; return *this; }
    __host__ __device__ inline FFP& operator=(int8_t a) {
        val_ = (a < 0) ? (GPUNTT_DEFAULT_MODULUS - static_cast<uint64_t>(-a)) : static_cast<uint64_t>(a);
        return *this;
    }
    __host__ __device__ inline FFP& operator=(int16_t a) {
        val_ = (a < 0) ? (GPUNTT_DEFAULT_MODULUS - static_cast<uint64_t>(-a)) : static_cast<uint64_t>(a);
        return *this;
    }
    __host__ __device__ inline FFP& operator=(int32_t a) {
        val_ = (a < 0) ? (GPUNTT_DEFAULT_MODULUS - static_cast<uint64_t>(-static_cast<int64_t>(a))) : static_cast<uint64_t>(a);
        return *this;
    }
    __host__ __device__ inline FFP& operator=(int64_t a) {
        val_ = (a < 0) ? (GPUNTT_DEFAULT_MODULUS - static_cast<uint64_t>(-a)) : static_cast<uint64_t>(a);
        return *this;
    }
    __host__ __device__ inline FFP& operator=(const FFP& a) { val_ = a.val_; return *this; }

    __host__ __device__ inline explicit operator uint64_t() const { return val_; }
    __host__ __device__ inline explicit operator uint32_t() const { return static_cast<uint32_t>(val_); }
    __host__ __device__ inline bool operator==(const FFP& other) const { return val_ == other.val_; }
    __host__ __device__ inline bool operator!=(const FFP& other) const { return val_ != other.val_; }
};

// Forward declaration
template <uint32_t length>
class CuNTTHandlerGPUNTT;

// Host-side storage for NTT parameters per GPU
struct NTTParams {
    Data64* forward_root;
    Data64* inverse_root;
    DeviceModulus modulus;
    Data64 n_inverse;
};

extern std::vector<NTTParams> g_ntt_params;

//=============================================================================
// Device-side implementation
//=============================================================================

#ifdef __CUDACC__

// Fixed 128-bit type for Barrett reduction
struct uint128_fixed {
    Data64 lo, hi;

    __device__ __forceinline__ uint128_fixed() : lo(0), hi(0) {}
    __device__ __forceinline__ uint128_fixed(Data64 x) : lo(x), hi(0) {}

    __device__ __forceinline__ uint128_fixed operator>>(uint32_t shift) const {
        uint128_fixed result;
        if (shift == 0) {
            result.lo = lo; result.hi = hi;
        } else if (shift < 64) {
            result.lo = (lo >> shift) | (hi << (64 - shift));
            result.hi = hi >> shift;
        } else if (shift == 64) {
            result.lo = hi; result.hi = 0;
        } else if (shift < 128) {
            result.lo = hi >> (shift - 64); result.hi = 0;
        } else {
            result.lo = 0; result.hi = 0;
        }
        return result;
    }

    __device__ __forceinline__ uint128_fixed operator-(const uint128_fixed& other) const {
        uint128_fixed result;
        asm("sub.cc.u64 %0, %2, %4; subc.u64 %1, %3, %5;"
            : "=l"(result.lo), "=l"(result.hi)
            : "l"(lo), "l"(hi), "l"(other.lo), "l"(other.hi));
        return result;
    }
};

__device__ __forceinline__ uint128_fixed mult128_fixed(Data64 a, Data64 b) {
    uint128_fixed result;
    asm("mul.lo.u64 %0, %2, %3; mul.hi.u64 %1, %2, %3;"
        : "=l"(result.lo), "=l"(result.hi) : "l"(a), "l"(b));
    return result;
}

// Barrett multiplication
__device__ __forceinline__ Data64 barrett_mult(Data64 a, Data64 b) {
    constexpr Data64 p = GPUNTT_DEFAULT_MODULUS;
    constexpr Data64 mu = GPUNTT_MU;
    constexpr uint32_t bit = GPUNTT_BIT;

    uint128_fixed z = mult128_fixed(a, b);
    uint128_fixed w = z >> (bit - 2);
    w = mult128_fixed(w.lo, mu);
    w = w >> (bit + 3);
    w = mult128_fixed(w.lo, p);
    z = z - w;
    return (z.lo >= p) ? (z.lo - p) : z.lo;
}

__device__ __forceinline__ Data64 mod_add(Data64 a, Data64 b) {
    constexpr Data64 p = GPUNTT_DEFAULT_MODULUS;
    Data64 sum = a + b;
    return (sum >= p) ? (sum - p) : sum;
}

__device__ __forceinline__ Data64 mod_sub(Data64 a, Data64 b) {
    constexpr Data64 p = GPUNTT_DEFAULT_MODULUS;
    Data64 diff = a + p - b;
    return (diff >= p) ? (diff - p) : diff;
}

// Cooley-Tukey butterfly for forward NTT: U' = U + V*root, V' = U - V*root
__device__ __forceinline__ void CooleyTukeyUnit(Data64& U, Data64& V, Data64 root) {
    Data64 u_ = U;
    Data64 v_ = barrett_mult(V, root);
    U = mod_add(u_, v_);
    V = mod_sub(u_, v_);
}

// Gentleman-Sande butterfly for inverse NTT: U' = U + V, V' = (U - V) * root
__device__ __forceinline__ void GentlemanSandeUnit(Data64& U, Data64& V, Data64 root) {
    Data64 u_ = U;
    Data64 v_ = V;
    U = mod_add(u_, v_);
    V = barrett_mult(mod_sub(u_, v_), root);
}

// FFP operators
__device__ inline FFP operator+(const FFP& a, const FFP& b) {
    FFP r; r.val() = mod_add(a.val(), b.val()); return r;
}
__device__ inline FFP operator-(const FFP& a, const FFP& b) {
    FFP r; r.val() = mod_sub(a.val(), b.val()); return r;
}
__device__ inline FFP operator*(const FFP& a, const FFP& b) {
    FFP r; r.val() = barrett_mult(a.val(), b.val()); return r;
}
__device__ inline FFP& operator+=(FFP& a, const FFP& b) { a = a + b; return a; }
__device__ inline FFP& operator-=(FFP& a, const FFP& b) { a = a - b; return a; }
__device__ inline FFP& operator*=(FFP& a, const FFP& b) { a = a * b; return a; }

/**
 * Optimized Small Forward NTT for N=1024
 * Uses 512 threads (N/2), each handles 2 elements
 * Matches HEonGPU's approach for maximum performance
 * Sync pattern: 4 syncs for first stages + 1 final sync = 5 total
 */
__device__ __forceinline__ void SmallForwardNTT_1024(
    Data64* sh,
    const Data64* root_table,
    int tid)
{
    constexpr int N_power = 10;

    int t_2 = N_power - 1;
    int t_ = 9;
    int m = 1;
    int t = 1 << t_;

    int in_shared_address = ((tid >> t_) << t_) + tid;
    int current_root_index;

    // First 4 stages need syncthreads (threads access distant memory)
    #pragma unroll
    for (int lp = 0; lp < 4; lp++) {
        current_root_index = m + (tid >> t_2);
        CooleyTukeyUnit(sh[in_shared_address], sh[in_shared_address + t],
                        root_table[current_root_index]);

        t = t >> 1;
        t_2 -= 1;
        t_ -= 1;
        m <<= 1;
        in_shared_address = ((tid >> t_) << t_) + tid;
        __syncthreads();
    }

    // Last 6 stages - warp-local, no sync needed between stages
    #pragma unroll
    for (int lp = 0; lp < 6; lp++) {
        current_root_index = m + (tid >> t_2);
        CooleyTukeyUnit(sh[in_shared_address], sh[in_shared_address + t],
                        root_table[current_root_index]);

        t = t >> 1;
        t_2 -= 1;
        t_ -= 1;
        m <<= 1;
        in_shared_address = ((tid >> t_) << t_) + tid;
    }
    __syncthreads();
}

/**
 * Optimized Small Inverse NTT for N=1024
 * Uses 512 threads (N/2), each handles 2 elements
 * Matches HEonGPU's approach for maximum performance
 * Sync pattern: 10 syncs for stages + 1 for n_inverse = 11 total
 */
__device__ __forceinline__ void SmallInverseNTT_1024(
    Data64* sh,
    const Data64* root_table,
    Data64 n_inverse,
    int tid)
{
    constexpr int N_power = 10;

    int t_2 = 0;
    int t_ = 0;
    int m = 1 << (N_power - 1);
    int t = 1;

    int in_shared_address = ((tid >> t_) << t_) + tid;
    int current_root_index;

    // All 10 stages of inverse NTT - each needs sync
    #pragma unroll
    for (int lp = 0; lp < N_power; lp++) {
        current_root_index = m + (tid >> t_2);
        GentlemanSandeUnit(sh[in_shared_address], sh[in_shared_address + t],
                           root_table[current_root_index]);

        t = t << 1;
        t_2 += 1;
        t_ += 1;
        m >>= 1;
        in_shared_address = ((tid >> t_) << t_) + tid;
        __syncthreads();
    }

    // Multiply by n^{-1} - each thread handles 2 elements
    sh[tid] = barrett_mult(sh[tid], n_inverse);
    sh[tid + 512] = barrett_mult(sh[tid + 512], n_inverse);
    __syncthreads();
}

/**
 * GPU-NTT Handler optimized for cuFHE
 */
template <uint32_t length = TFHEpp::lvl1param::n>
class CuNTTHandlerGPUNTT {
public:
    static constexpr uint32_t kLength = length;
    static constexpr uint32_t kLogLength = []() constexpr {
        uint32_t n = length, log = 0;
        while (n > 1) { n >>= 1; ++log; }
        return log;
    }();

    Data64* forward_root_;
    Data64* inverse_root_;
    DeviceModulus modulus_;
    Data64 n_inverse_;

    __host__ __device__ CuNTTHandlerGPUNTT() : forward_root_(nullptr), inverse_root_(nullptr), n_inverse_(0) {}
    __host__ __device__ ~CuNTTHandlerGPUNTT() {}

    __host__ static void Create();
    __host__ static void CreateConstant();
    __host__ static void Destroy();
    __host__ void SetDevicePointers(int device_id);

    // Forward NTT (512 threads per NTT, each handles 2 elements)
    template <typename T>
    __device__ inline void NTT(
        FFP* const out,
        const T* const in,
        FFP* const sh_temp,
        uint32_t leading_thread = 0) const
    {
        const int tid = threadIdx.x - leading_thread;
        constexpr int N = length;
        constexpr int NUM_THREADS = N >> NTT_THREAD_UNITBIT;  // 512 for N=1024

        // Load to shared memory - each thread handles 2 elements
        if (tid < NUM_THREADS) {
            if constexpr (std::is_same_v<T, FFP>) {
                sh_temp[tid].val() = in[tid].val();
                sh_temp[tid + NUM_THREADS].val() = in[tid + NUM_THREADS].val();
            } else {
                sh_temp[tid] = FFP(in[tid]);
                sh_temp[tid + NUM_THREADS] = FFP(in[tid + NUM_THREADS]);
            }
        }
        __syncthreads();

        // Forward NTT in-place (5 syncs inside: 4 + 1)
        if (tid < NUM_THREADS) {
            SmallForwardNTT_1024(reinterpret_cast<Data64*>(sh_temp), forward_root_, tid);
        } else {
            // Non-participating threads sync 5 times
            for (int i = 0; i < 5; i++) __syncthreads();
        }

        // Copy to output
        if (tid < NUM_THREADS) {
            out[tid] = sh_temp[tid];
            out[tid + NUM_THREADS] = sh_temp[tid + NUM_THREADS];
        }
        __syncthreads();
    }

    // Inverse NTT (512 threads per NTT, each handles 2 elements)
    template <typename T>
    __device__ inline void NTTInv(
        T* const out,
        const FFP* const in,
        FFP* const sh_temp,
        uint32_t leading_thread = 0) const
    {
        const int tid = threadIdx.x - leading_thread;
        constexpr int N = length;
        constexpr int NUM_THREADS = N >> NTT_THREAD_UNITBIT;  // 512 for N=1024
        constexpr Data64 half_mod = GPUNTT_DEFAULT_MODULUS / 2;

        // Load to shared memory - each thread handles 2 elements
        if (tid < NUM_THREADS) {
            sh_temp[tid] = in[tid];
            sh_temp[tid + NUM_THREADS] = in[tid + NUM_THREADS];
        }
        __syncthreads();

        // Inverse NTT in-place (11 syncs: 10 stages + 1 n_inverse)
        if (tid < NUM_THREADS) {
            SmallInverseNTT_1024(reinterpret_cast<Data64*>(sh_temp), inverse_root_, n_inverse_, tid);
        } else {
            for (int i = 0; i < 11; i++) __syncthreads();
        }

        // Convert back with centered reduction
        if (tid < NUM_THREADS) {
            Data64 val0 = sh_temp[tid].val();
            Data64 val1 = sh_temp[tid + NUM_THREADS].val();
            out[tid] = (val0 > half_mod)
                ? static_cast<T>(static_cast<int64_t>(val0) - static_cast<int64_t>(GPUNTT_DEFAULT_MODULUS))
                : static_cast<T>(val0);
            out[tid + NUM_THREADS] = (val1 > half_mod)
                ? static_cast<T>(static_cast<int64_t>(val1) - static_cast<int64_t>(GPUNTT_DEFAULT_MODULUS))
                : static_cast<T>(val1);
        }
        __syncthreads();
    }

    // Inverse NTT with addition (512 threads per NTT)
    template <typename T>
    __device__ inline void NTTInvAdd(
        T* const out,
        const FFP* const in,
        FFP* const sh_temp,
        uint32_t leading_thread = 0) const
    {
        const int tid = threadIdx.x - leading_thread;
        constexpr int N = length;
        constexpr int NUM_THREADS = N >> NTT_THREAD_UNITBIT;  // 512 for N=1024
        constexpr Data64 half_mod = GPUNTT_DEFAULT_MODULUS / 2;

        // Load to shared memory - each thread handles 2 elements
        if (tid < NUM_THREADS) {
            sh_temp[tid] = in[tid];
            sh_temp[tid + NUM_THREADS] = in[tid + NUM_THREADS];
        }
        __syncthreads();

        // Inverse NTT in-place (11 syncs: 10 stages + 1 n_inverse)
        if (tid < NUM_THREADS) {
            SmallInverseNTT_1024(reinterpret_cast<Data64*>(sh_temp), inverse_root_, n_inverse_, tid);
        } else {
            for (int i = 0; i < 11; i++) __syncthreads();
        }

        // Convert and ADD to output
        if (tid < NUM_THREADS) {
            Data64 val0 = sh_temp[tid].val();
            Data64 val1 = sh_temp[tid + NUM_THREADS].val();
            T conv0 = (val0 > half_mod)
                ? static_cast<T>(static_cast<int64_t>(val0) - static_cast<int64_t>(GPUNTT_DEFAULT_MODULUS))
                : static_cast<T>(val0);
            T conv1 = (val1 > half_mod)
                ? static_cast<T>(static_cast<int64_t>(val1) - static_cast<int64_t>(GPUNTT_DEFAULT_MODULUS))
                : static_cast<T>(val1);
            out[tid] += conv0;
            out[tid + NUM_THREADS] += conv1;
        }
        __syncthreads();
    }
};

#endif // __CUDACC__

// Type alias
template <uint32_t length = TFHEpp::lvl1param::n>
using CuNTTHandler = CuNTTHandlerGPUNTT<length>;

} // namespace cufhe
