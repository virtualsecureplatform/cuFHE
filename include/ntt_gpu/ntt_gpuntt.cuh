/**
 * GPU-NTT based NTT implementation for cuFHE
 * Replaces the original FFP-based NTT with GPU-NTT library
 *
 * This implementation uses Barrett reduction and supports configurable
 * polynomial degrees (not just 1024).
 */

#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <vector>
#include "gpuntt/common/modular_arith.cuh"
#include "gpuntt/ntt_merge/ntt.cuh"
#include <params.hpp>
// Include thread helper functions from existing utils
#include <include/details/utils_gpu.cuh>

namespace cufhe {

// Re-export GPU-NTT types for convenience
using NTTData = Data64;
using NTTModulus = Modulus<Data64>;
using NTTRoot = Root<Data64>;
using NTTNinverse = Ninverse<Data64>;

// Device-compatible modulus struct (same layout as NTTModulus)
// POD type for CUDA device variable compatibility
struct DeviceModulus {
    Data64 value;
    Data64 bit;
    Data64 mu;
};

// HEonGPU's TFHE-optimized prime parameters for 64-bit
// Prime: 1152921504606877697 (~2^60, NTT-friendly, used by HEonGPU for TFHE)
constexpr Data64 GPUNTT_DEFAULT_MODULUS = 1152921504606877697ULL;

// Implementation dependent parameter for thread count
// We use N/8 threads per NTT (128 threads for N=1024)
// Each thread handles 4 butterflies per stage (since N/2 total butterflies, N/8 threads)
constexpr uint32_t NTT_THREAD_UNITBIT = 3;

/**
 * @class FFP
 * @brief Wraps a uint64_t integer as an element in FF(P) using GPU-NTT's prime.
 *        Provides operator overloads for modular arithmetic using Barrett reduction.
 *
 * This class replaces the original FFP class that used P = 2^64-2^32+1.
 * Now uses GPU-NTT's prime P = 1152921504606877697 for compatibility.
 */
class FFP {
private:
    uint64_t val_;

public:
    // Default constructor - empty for shared memory compatibility
    __host__ __device__ inline FFP() {}

    // Constructors from various integer types
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

    // Destructor
    __host__ __device__ inline ~FFP() {}

    // Value access
    __host__ __device__ inline uint64_t& val() { return val_; }
    __host__ __device__ inline const uint64_t& val() const { return val_; }

    // Static modulus accessor
    __host__ __device__ inline static constexpr uint64_t kModulus() { return GPUNTT_DEFAULT_MODULUS; }

    // Assignment operators
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

    // Explicit conversion operators
    __host__ __device__ inline explicit operator uint64_t() const { return val_; }
    __host__ __device__ inline explicit operator uint32_t() const { return static_cast<uint32_t>(val_); }
    __host__ __device__ inline explicit operator uint16_t() const { return static_cast<uint16_t>(val_); }
    __host__ __device__ inline explicit operator uint8_t() const { return static_cast<uint8_t>(val_); }

#ifdef __CUDACC__
    // Modular addition: result = (a + b) mod P
    __device__ inline void Add(const FFP& a, const FFP& b, const DeviceModulus& mod) {
        const NTTModulus& m = reinterpret_cast<const NTTModulus&>(mod);
        val_ = OPERATOR_GPU<Data64>::add(a.val_, b.val_, m);
    }

    // Modular subtraction: result = (a - b) mod P
    __device__ inline void Sub(const FFP& a, const FFP& b, const DeviceModulus& mod) {
        const NTTModulus& m = reinterpret_cast<const NTTModulus&>(mod);
        val_ = OPERATOR_GPU<Data64>::sub(a.val_, b.val_, m);
    }

    // Modular multiplication: result = (a * b) mod P
    __device__ inline void Mul(const FFP& a, const FFP& b, const DeviceModulus& mod) {
        const NTTModulus& m = reinterpret_cast<const NTTModulus&>(mod);
        val_ = OPERATOR_GPU<Data64>::mult(a.val_, b.val_, m);
    }
#endif // __CUDACC__

    // Comparison operators
    __host__ __device__ inline bool operator==(const FFP& other) const { return val_ == other.val_; }
    __host__ __device__ inline bool operator!=(const FFP& other) const { return val_ != other.val_; }
};

// Forward declaration
template <uint32_t length>
class CuNTTHandlerGPUNTT;

// Global NTT parameters stored in constant memory for each GPU
// These are set during initialization and accessed by all kernels
struct NTTParams {
    Data64* forward_root;
    Data64* inverse_root;
    DeviceModulus modulus;
    Data64 n_inverse;
};

// Host-side storage for NTT parameters per GPU
extern std::vector<NTTParams> g_ntt_params;

// Get NTT parameters for current device (call from device code)
#ifdef __CUDACC__
__device__ inline const NTTParams& GetNTTParams();
#endif

/**
 * GPU-NTT based NTT Handler class
 * Provides same interface as original CuNTTHandler but uses GPU-NTT internally
 *
 * This class stores device pointers to NTT tables and can be passed to device kernels.
 * All setup is done via static functions before kernel launch.
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

    // Member variables - device pointers to NTT tables
    Data64* forward_root_;
    Data64* inverse_root_;
    DeviceModulus modulus_;
    Data64 n_inverse_;

    __host__ __device__ CuNTTHandlerGPUNTT() : forward_root_(nullptr), inverse_root_(nullptr), n_inverse_(0) {}
    __host__ __device__ ~CuNTTHandlerGPUNTT() {}

    // Static host functions for initialization
    __host__ static void Create();
    __host__ static void CreateConstant();
    __host__ static void Destroy();

    // Set device pointers (called after CreateConstant)
    __host__ void SetDevicePointers(int device_id);

    // Forward NTT: converts from coefficient domain to NTT domain
    // T is the input type (e.g., int32_t for coefficients, or FFP for in-place)
    template <typename T>
    __device__ inline void NTT(
        FFP* const out,
        const T* const in,
        FFP* const sh_temp,
        uint32_t leading_thread = 0) const;

    // Inverse NTT: converts from NTT domain to coefficient domain
    // T is the output type (e.g., int32_t for coefficients)
    template <typename T>
    __device__ inline void NTTInv(
        T* const out,
        const FFP* const in,
        FFP* const sh_temp,
        uint32_t leading_thread = 0) const;

    // Inverse NTT with addition to output
    template <typename T>
    __device__ inline void NTTInvAdd(
        T* const out,
        const FFP* const in,
        FFP* const sh_temp,
        uint32_t leading_thread = 0) const;

private:
    // Small forward NTT kernel (in-place, shared memory, works on FFP array)
    __device__ inline void SmallForwardNTT(FFP* sh, uint32_t tid) const;

    // Small inverse NTT kernel (in-place, shared memory, works on FFP array)
    __device__ inline void SmallInverseNTT(FFP* sh, uint32_t tid) const;
};

//=============================================================================
// Implementation of device functions (must be in header for inlining)
// Only compiled by nvcc (not regular C++ compilers)
//=============================================================================

#ifdef __CUDACC__

// Pre-computed Barrett reduction parameter mu for GPUNTT_DEFAULT_MODULUS
// bit = floor(log2(p) + 1) = 61, p = 1152921504606877697
// mu = floor(2^(2*bit+1) / p) = 2^123 // 1152921504606877697 = 9223372036854530040
constexpr Data64 GPUNTT_MU = 9223372036854530040ULL;
constexpr Data64 GPUNTT_BIT = 61;

// Fixed 128-bit type that properly handles shift by 64 bits
// GPU-NTT's uint128_t has undefined behavior when shifting by exactly 64
struct uint128_fixed {
    Data64 lo;  // LSB
    Data64 hi;  // MSB

    __device__ __forceinline__ uint128_fixed() : lo(0), hi(0) {}
    __device__ __forceinline__ uint128_fixed(Data64 x) : lo(x), hi(0) {}

    __device__ __forceinline__ uint128_fixed operator>>(uint32_t shift) const {
        uint128_fixed result;
        if (shift == 0) {
            result.lo = lo;
            result.hi = hi;
        } else if (shift < 64) {
            result.lo = (lo >> shift) | (hi << (64 - shift));
            result.hi = hi >> shift;
        } else if (shift == 64) {
            result.lo = hi;
            result.hi = 0;
        } else if (shift < 128) {
            result.lo = hi >> (shift - 64);
            result.hi = 0;
        } else {
            result.lo = 0;
            result.hi = 0;
        }
        return result;
    }

    __device__ __forceinline__ uint128_fixed operator-(const uint128_fixed& other) const {
        uint128_fixed result;
        asm("{\n\t"
            "sub.cc.u64  %0, %2, %4;\n\t"
            "subc.u64    %1, %3, %5;\n\t"
            "}"
            : "=l"(result.lo), "=l"(result.hi)
            : "l"(lo), "l"(hi), "l"(other.lo), "l"(other.hi));
        return result;
    }
};

// 128-bit multiply: returns full 128-bit product of two 64-bit values
__device__ __forceinline__ uint128_fixed mult128_fixed(Data64 a, Data64 b) {
    uint128_fixed result;
    asm("{\n\t"
        "mul.lo.u64  %0, %2, %3;\n\t"
        "mul.hi.u64  %1, %2, %3;\n\t"
        "}"
        : "=l"(result.lo), "=l"(result.hi)
        : "l"(a), "l"(b));
    return result;
}

// Barrett multiplication with correct handling of shift by 64 bits
__device__ __forceinline__ Data64 barrett_mult_fixed(Data64 a, Data64 b) {
    constexpr Data64 p = GPUNTT_DEFAULT_MODULUS;
    constexpr Data64 mu = GPUNTT_MU;
    constexpr uint32_t bit = GPUNTT_BIT;

    uint128_fixed z = mult128_fixed(a, b);
    uint128_fixed w = z >> (bit - 2);       // shift by 59
    w = mult128_fixed(w.lo, mu);
    w = w >> (bit + 3);                     // shift by 64 - now handled correctly!
    w = mult128_fixed(w.lo, p);
    z = z - w;

    // Final correction (result might be >= p but < 2p)
    return (z.lo >= p) ? (z.lo - p) : z.lo;
}

// Standalone modular add/sub for butterfly operations
__device__ __forceinline__ Data64 mod_add_fixed(Data64 a, Data64 b) {
    constexpr Data64 p = GPUNTT_DEFAULT_MODULUS;
    Data64 sum = a + b;
    return (sum >= p) ? (sum - p) : sum;
}

__device__ __forceinline__ Data64 mod_sub_fixed(Data64 a, Data64 b) {
    constexpr Data64 p = GPUNTT_DEFAULT_MODULUS;
    Data64 diff = a + p - b;
    return (diff >= p) ? (diff - p) : diff;
}

// Fixed Cooley-Tukey butterfly unit for forward NTT
// U' = U + V*root
// V' = U - V*root
__device__ __forceinline__ void CooleyTukeyUnit_fixed(Data64& U, Data64& V, Data64 root) {
    Data64 u_ = U;
    Data64 v_ = barrett_mult_fixed(V, root);
    U = mod_add_fixed(u_, v_);
    V = mod_sub_fixed(u_, v_);
}

// Fixed Gentleman-Sande butterfly unit for inverse NTT
// U' = U + V
// V' = (U - V) * root
__device__ __forceinline__ void GentlemanSandeUnit_fixed(Data64& U, Data64& V, Data64 root) {
    Data64 u_ = U;
    Data64 v_ = V;
    U = mod_add_fixed(u_, v_);
    v_ = mod_sub_fixed(u_, v_);
    V = barrett_mult_fixed(v_, root);
}

// FFP arithmetic operators using pre-computed modulus parameters
__device__ inline FFP operator+(const FFP& a, const FFP& b) {
    constexpr Data64 p = GPUNTT_DEFAULT_MODULUS;
    Data64 sum = a.val() + b.val();
    FFP r;
    r.val() = (sum >= p) ? (sum - p) : sum;
    return r;
}

__device__ inline FFP operator-(const FFP& a, const FFP& b) {
    constexpr Data64 p = GPUNTT_DEFAULT_MODULUS;
    Data64 diff = a.val() + p - b.val();
    FFP r;
    r.val() = (diff >= p) ? (diff - p) : diff;
    return r;
}

__device__ inline FFP operator*(const FFP& a, const FFP& b) {
    FFP r;
    r.val() = barrett_mult_fixed(a.val(), b.val());
    return r;
}

__device__ inline FFP& operator+=(FFP& a, const FFP& b) { a = a + b; return a; }
__device__ inline FFP& operator-=(FFP& a, const FFP& b) { a = a - b; return a; }
__device__ inline FFP& operator*=(FFP& a, const FFP& b) { a = a * b; return a; }

template <uint32_t length>
__device__ inline void CuNTTHandlerGPUNTT<length>::SmallForwardNTT(
    FFP* sh, uint32_t tid) const
{
    constexpr uint32_t LOG_N = kLogLength;
    constexpr uint32_t N = length;
    constexpr uint32_t NUM_THREADS = N >> NTT_THREAD_UNITBIT;  // N/8 threads
    constexpr uint32_t BUTTERFLIES_PER_THREAD = (N / 2) / NUM_THREADS;  // 4 for N=1024

    int t_2 = LOG_N - 1;
    int t_ = LOG_N - 1;
    int m = 1;
    int t = 1 << t_;

    // Forward NTT butterfly stages
    #pragma unroll
    for (int lp = 0; lp < LOG_N; lp++) {
        // Each thread handles BUTTERFLIES_PER_THREAD butterflies
        #pragma unroll
        for (int b = 0; b < BUTTERFLIES_PER_THREAD; b++) {
            int virtual_tid = tid + b * NUM_THREADS;
            int in_shared_address = ((virtual_tid >> t_) << t_) + virtual_tid;
            int current_root_index = m + (virtual_tid >> t_2);

            // Use fixed butterfly that handles shift-by-64 correctly
            CooleyTukeyUnit_fixed(
                sh[in_shared_address].val(),
                sh[in_shared_address + t].val(),
                forward_root_[current_root_index]);
        }

        t = t >> 1;
        t_2 -= 1;
        t_ -= 1;
        m <<= 1;
        __syncthreads();
    }
}

template <uint32_t length>
__device__ inline void CuNTTHandlerGPUNTT<length>::SmallInverseNTT(
    FFP* sh, uint32_t tid) const
{
    constexpr uint32_t N = length;
    constexpr uint32_t LOG_N = kLogLength;
    constexpr uint32_t NUM_THREADS = N >> NTT_THREAD_UNITBIT;  // N/8 threads
    constexpr uint32_t BUTTERFLIES_PER_THREAD = (N / 2) / NUM_THREADS;  // 4 for N=1024
    constexpr uint32_t ELEMENTS_PER_THREAD = N / NUM_THREADS;  // 8 for N=1024

    int t_2 = 0;
    int t_ = 0;
    int m = 1 << (LOG_N - 1);
    int t = 1;

    // Inverse NTT butterfly stages
    #pragma unroll
    for (int lp = 0; lp < LOG_N; lp++) {
        // Each thread handles BUTTERFLIES_PER_THREAD butterflies
        #pragma unroll
        for (int b = 0; b < BUTTERFLIES_PER_THREAD; b++) {
            int virtual_tid = tid + b * NUM_THREADS;
            int in_shared_address = ((virtual_tid >> t_) << t_) + virtual_tid;
            int current_root_index = m + (virtual_tid >> t_2);

            // Use fixed butterfly that handles shift-by-64 correctly
            GentlemanSandeUnit_fixed(
                sh[in_shared_address].val(),
                sh[in_shared_address + t].val(),
                inverse_root_[current_root_index]);
        }

        t = t << 1;
        t_2 += 1;
        t_ += 1;
        m >>= 1;
        __syncthreads();
    }

    // Multiply by n^{-1} - each thread handles ELEMENTS_PER_THREAD elements
    // Use fixed Barrett multiplication
    #pragma unroll
    for (int e = 0; e < ELEMENTS_PER_THREAD; e++) {
        int idx = tid + e * NUM_THREADS;
        sh[idx].val() = barrett_mult_fixed(sh[idx].val(), n_inverse_);
    }
    __syncthreads();
}

template <uint32_t length>
template <typename T>
__device__ inline void CuNTTHandlerGPUNTT<length>::NTT(
    FFP* const out,
    const T* const in,
    FFP* const sh_temp,
    uint32_t leading_thread) const
{
    const uint32_t tid = threadIdx.x - leading_thread;
    constexpr uint32_t N = length;
    constexpr uint32_t NUM_THREADS = N >> NTT_THREAD_UNITBIT;  // N/8 threads
    constexpr uint32_t ELEMENTS_PER_THREAD = N / NUM_THREADS;  // 8 elements per thread

    // Load input to shared memory with type conversion
    // Each thread loads ELEMENTS_PER_THREAD elements
    if (tid < NUM_THREADS) {
        #pragma unroll
        for (int e = 0; e < ELEMENTS_PER_THREAD; e++) {
            int idx = tid + e * NUM_THREADS;
            if constexpr (std::is_same_v<T, FFP>) {
                sh_temp[idx] = in[idx];
            } else if constexpr (std::is_signed_v<T>) {
                sh_temp[idx] = FFP(in[idx]);
            } else {
                sh_temp[idx] = FFP(in[idx]);
            }
        }
    }
    __syncthreads();

    // Perform forward NTT in-place
    if (tid < NUM_THREADS) {
        SmallForwardNTT(sh_temp, tid);
    } else {
        // Sync with NTT threads
        for (int i = 0; i < kLogLength; i++) {
            __syncthreads();
        }
    }

    // Copy to output - each thread handles ELEMENTS_PER_THREAD elements
    if (tid < NUM_THREADS) {
        #pragma unroll
        for (int e = 0; e < ELEMENTS_PER_THREAD; e++) {
            int idx = tid + e * NUM_THREADS;
            out[idx] = sh_temp[idx];
        }
    }
    __syncthreads();
}

template <uint32_t length>
template <typename T>
__device__ inline void CuNTTHandlerGPUNTT<length>::NTTInv(
    T* const out,
    const FFP* const in,
    FFP* const sh_temp,
    uint32_t leading_thread) const
{
    const uint32_t tid = threadIdx.x - leading_thread;
    constexpr uint32_t N = length;
    constexpr uint32_t NUM_THREADS = N >> NTT_THREAD_UNITBIT;  // N/8 threads
    constexpr uint32_t ELEMENTS_PER_THREAD = N / NUM_THREADS;  // 8 elements per thread
    constexpr Data64 half_mod = GPUNTT_DEFAULT_MODULUS / 2;

    // Load input to shared memory - each thread loads ELEMENTS_PER_THREAD elements
    if (tid < NUM_THREADS) {
        #pragma unroll
        for (int e = 0; e < ELEMENTS_PER_THREAD; e++) {
            int idx = tid + e * NUM_THREADS;
            sh_temp[idx] = in[idx];
        }
    }
    __syncthreads();

    // Perform inverse NTT in-place
    if (tid < NUM_THREADS) {
        SmallInverseNTT(sh_temp, tid);
    } else {
        // Sync with NTT threads (LOG_N stages + 1 for n_inverse multiply)
        for (int i = 0; i < kLogLength + 1; i++) {
            __syncthreads();
        }
    }

    // Convert back with centered reduction
    // Note: Must do subtraction in 64-bit before casting to avoid overflow
    // Each thread handles ELEMENTS_PER_THREAD elements
    if (tid < NUM_THREADS) {
        #pragma unroll
        for (int e = 0; e < ELEMENTS_PER_THREAD; e++) {
            int idx = tid + e * NUM_THREADS;
            Data64 val = sh_temp[idx].val();
            if (val > half_mod) {
                out[idx] = static_cast<T>(static_cast<int64_t>(val) - static_cast<int64_t>(GPUNTT_DEFAULT_MODULUS));
            } else {
                out[idx] = static_cast<T>(val);
            }
        }
    }
    __syncthreads();
}

template <uint32_t length>
template <typename T>
__device__ inline void CuNTTHandlerGPUNTT<length>::NTTInvAdd(
    T* const out,
    const FFP* const in,
    FFP* const sh_temp,
    uint32_t leading_thread) const
{
    const uint32_t tid = threadIdx.x - leading_thread;
    constexpr uint32_t N = length;
    constexpr uint32_t NUM_THREADS = N >> NTT_THREAD_UNITBIT;  // N/8 threads
    constexpr uint32_t ELEMENTS_PER_THREAD = N / NUM_THREADS;  // 8 elements per thread
    constexpr Data64 half_mod = GPUNTT_DEFAULT_MODULUS / 2;

    // Load input to shared memory - each thread loads ELEMENTS_PER_THREAD elements
    if (tid < NUM_THREADS) {
        #pragma unroll
        for (int e = 0; e < ELEMENTS_PER_THREAD; e++) {
            int idx = tid + e * NUM_THREADS;
            sh_temp[idx] = in[idx];
        }
    }
    __syncthreads();

    // Perform inverse NTT in-place
    if (tid < NUM_THREADS) {
        SmallInverseNTT(sh_temp, tid);
    } else {
        // Sync with NTT threads (LOG_N stages + 1 for n_inverse multiply)
        for (int i = 0; i < kLogLength + 1; i++) {
            __syncthreads();
        }
    }

    // Convert and ADD to output
    // Note: Must do subtraction in 64-bit before casting to avoid overflow
    // Each thread handles ELEMENTS_PER_THREAD elements
    if (tid < NUM_THREADS) {
        #pragma unroll
        for (int e = 0; e < ELEMENTS_PER_THREAD; e++) {
            int idx = tid + e * NUM_THREADS;
            Data64 val = sh_temp[idx].val();
            T conv = (val > half_mod) ? static_cast<T>(static_cast<int64_t>(val) - static_cast<int64_t>(GPUNTT_DEFAULT_MODULUS))
                                      : static_cast<T>(val);
            out[idx] += conv;
        }
    }
    __syncthreads();
}

#endif // __CUDACC__

// Type alias for backward compatibility
template <uint32_t length = TFHEpp::lvl1param::n>
using CuNTTHandler = CuNTTHandlerGPUNTT<length>;

} // namespace cufhe
