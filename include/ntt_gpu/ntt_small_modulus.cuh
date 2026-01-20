/**
 * Small Modulus NTT implementation for cuFHE
 *
 * This implements the RAINTT-style approach where:
 * - Torus discretization switches from 2^32 to NTT modulus before multiplication
 * - After INTT, results are converted back to 2^32 discretization
 *
 * This is more efficient than the 64-bit modulus approach for certain use cases
 * as it allows working with smaller integers.
 *
 * Current modulus: P = 625 * 2^20 + 1 = 655360001 (~29.3 bits)
 * - Compatible with TFHEpp RAINTT implementation
 * - Has 2048th primitive root of unity (required for N=1024 NTT)
 *
 * Note: A larger modulus P = 1433 * 2^21 + 1 = 3005218817 (~31.5 bits) could
 * provide better precision while still satisfying 2*P^2 < 2^64 for safe
 * Montgomery reduction. This would require updating the primitive root
 * calculation (prime factors change from {2,5} to {2,1433}).
 */

#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <vector>
#include <params.hpp>
#include <include/details/utils_gpu.cuh>

namespace cufhe {

//=============================================================================
// Small Modulus NTT Constants (RAINTT-style)
//=============================================================================

namespace small_ntt {

// NTT parameters - optimized for GPU
// P = K * 2^20 + 1 = 655360001 (~29.3 bits) - original RAINTT modulus
// Larger modulus (31.5 bits) available: K=1433, SHIFTAMOUNT=21 -> P=3005218817
constexpr uint32_t K = 625;
constexpr uint32_t SHIFTAMOUNT = 20;
constexpr uint32_t WORDBITS = 32;     // Using 32-bit arithmetic on GPU

// NTT modulus: P = K * 2^shiftamount + 1 = 655360001
constexpr uint32_t P = (K << SHIFTAMOUNT) + 1;  // 655360001

// Verify safety constraint at compile time: 2*P^2 < 2^64
// P must be less than sqrt(2^63) ≈ 3.03 billion for safe 64-bit Montgomery reduction
static_assert(P < 3037000500ULL, "Modulus too large for safe Montgomery reduction");

// Barrett reduction parameters for 32-bit modulus
constexpr uint32_t MODULUS_BITS = 32;
constexpr uint64_t BARRETT_MU = (1ULL << 60) / P;  // Pre-computed for 64-bit intermediate

// Montgomery constant: R = 2^32 mod P = 1289748479
constexpr uint64_t R_TEMP = (1ULL << 32) % P;
constexpr uint32_t R = static_cast<uint32_t>(R_TEMP);

// R^2 mod P for Montgomery domain conversion = 295825756
constexpr uint64_t R2_TEMP = (static_cast<uint64_t>(R) * R) % P;
constexpr uint32_t R2 = static_cast<uint32_t>(R2_TEMP);

// N inverse for N=1024 (in Montgomery form)
constexpr uint32_t N_1024 = 1024;

// Modulus switching constants
// For Torus32 -> NTT mod: res = round((a * P) / 2^32)
// For NTT mod -> Torus32: res = round((a * 2^32) / P)

// Pre-computed: 2^63 / P for inverse modswitch
constexpr uint64_t INV_MODSWITCH_MUL = (1ULL << 63) / P;

}  // namespace small_ntt

//=============================================================================
// 32-bit Small Modulus Finite Field Element
//=============================================================================

class FFP32 {
private:
    uint32_t val_;

public:
    __host__ __device__ inline FFP32() : val_(0) {}
    __host__ __device__ inline FFP32(uint32_t a) : val_(a % small_ntt::P) {}
    __host__ __device__ inline FFP32(int32_t a) {
        if (a < 0) {
            val_ = small_ntt::P - (static_cast<uint32_t>(-a) % small_ntt::P);
            if (val_ == small_ntt::P) val_ = 0;
        } else {
            val_ = static_cast<uint32_t>(a) % small_ntt::P;
        }
    }

    __host__ __device__ inline uint32_t& val() { return val_; }
    __host__ __device__ inline const uint32_t& val() const { return val_; }
    __host__ __device__ inline static constexpr uint32_t kModulus() { return small_ntt::P; }

    __host__ __device__ inline explicit operator uint32_t() const { return val_; }
};

//=============================================================================
// Device-side Small Modulus Operations
//=============================================================================

#ifdef __CUDACC__

// Modular addition: (a + b) mod P
__device__ __forceinline__ uint32_t small_mod_add(uint32_t a, uint32_t b) {
    uint32_t sum = a + b;
    return (sum >= small_ntt::P) ? (sum - small_ntt::P) : sum;
}

// Modular subtraction: (a - b) mod P
__device__ __forceinline__ uint32_t small_mod_sub(uint32_t a, uint32_t b) {
    uint32_t diff = a + small_ntt::P - b;
    return (diff >= small_ntt::P) ? (diff - small_ntt::P) : diff;
}

// Barrett multiplication: (a * b) mod P
// Uses 64-bit intermediate, suitable for 32-bit modulus
__device__ __forceinline__ uint32_t small_barrett_mult(uint32_t a, uint32_t b) {
    constexpr uint32_t p = small_ntt::P;

    uint64_t z = static_cast<uint64_t>(a) * b;

    // Barrett reduction: q = floor(z * mu / 2^60)
    // We use the formula: result = z - q * P
    uint64_t q = __umul64hi(z, small_ntt::BARRETT_MU);  // Approximate q
    q = (z * small_ntt::BARRETT_MU) >> 60;

    uint32_t result = static_cast<uint32_t>(z - q * p);

    // One conditional subtraction is enough
    return (result >= p) ? (result - p) : result;
}

// Alternative: Simple modular multiplication with 64-bit intermediate
__device__ __forceinline__ uint32_t small_mod_mult(uint32_t a, uint32_t b) {
    uint64_t prod = static_cast<uint64_t>(a) * b;
    return static_cast<uint32_t>(prod % small_ntt::P);
}

/**
 * Modulus switch: Torus32 (2^32 discretization) -> NTT modulus P
 *
 * Formula: res = round((a * P) / 2^32) = (a * P + 2^31) >> 32
 *
 * This switches the discretization from Torus (mod 2^32) to NTT domain (mod P)
 */
__device__ __forceinline__ uint32_t torus32_to_ntt_mod(uint32_t torus_val) {
    constexpr uint64_t P = small_ntt::P;

    // res = (a * P + 2^31) >> 32  (rounding)
    uint64_t temp = static_cast<uint64_t>(torus_val) * P + (1ULL << 31);
    return static_cast<uint32_t>(temp >> 32);
}

/**
 * Modulus switch: NTT modulus P -> Torus32 (2^32 discretization)
 *
 * Formula: res = round((a * 2^32) / P)
 *
 * We use: res = (a * (2^63 / P) + 2^30) >> 31
 *
 * This switches back from NTT domain (mod P) to Torus (mod 2^32)
 */
__device__ __forceinline__ uint32_t ntt_mod_to_torus32(int32_t ntt_val) {
    constexpr uint32_t P = small_ntt::P;

    // Handle signed value: convert to [0, P) range
    uint32_t a = (ntt_val < 0) ? static_cast<uint32_t>(ntt_val + static_cast<int32_t>(P))
                               : static_cast<uint32_t>(ntt_val);

    // res = round((a * 2^32) / P) = (a * (2^63 / P) + 2^30) >> 31
    uint64_t temp = static_cast<uint64_t>(a) * small_ntt::INV_MODSWITCH_MUL;
    temp = (temp + (1ULL << 30)) >> 31;
    return static_cast<uint32_t>(temp);
}

// Cooley-Tukey butterfly for forward NTT
__device__ __forceinline__ void SmallCooleyTukeyUnit(uint32_t& U, uint32_t& V, uint32_t root) {
    uint32_t u = U;
    uint32_t v = small_mod_mult(V, root);
    U = small_mod_add(u, v);
    V = small_mod_sub(u, v);
}

// Gentleman-Sande butterfly for inverse NTT
__device__ __forceinline__ void SmallGentlemanSandeUnit(uint32_t& U, uint32_t& V, uint32_t root) {
    uint32_t u = U;
    uint32_t v = V;
    U = small_mod_add(u, v);
    V = small_mod_mult(small_mod_sub(u, v), root);
}

/**
 * Small modulus Forward NTT for N=1024
 * Uses 512 threads, each handles 2 elements
 */
__device__ __forceinline__ void SmallForwardNTT32_1024(
    uint32_t* sh,
    const uint32_t* root_table,
    int tid)
{
    constexpr int N_power = 10;

    int t_2 = N_power - 1;
    int t_ = 9;
    int m = 1;
    int t = 1 << t_;

    int in_shared_address = ((tid >> t_) << t_) + tid;
    int current_root_index;

    // First 4 stages need syncthreads
    #pragma unroll
    for (int lp = 0; lp < 4; lp++) {
        current_root_index = m + (tid >> t_2);
        SmallCooleyTukeyUnit(sh[in_shared_address], sh[in_shared_address + t],
                             root_table[current_root_index]);

        t = t >> 1;
        t_2 -= 1;
        t_ -= 1;
        m <<= 1;
        in_shared_address = ((tid >> t_) << t_) + tid;
        __syncthreads();
    }

    // Last 6 stages - warp-local
    #pragma unroll
    for (int lp = 0; lp < 6; lp++) {
        current_root_index = m + (tid >> t_2);
        SmallCooleyTukeyUnit(sh[in_shared_address], sh[in_shared_address + t],
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
 * Small modulus Inverse NTT for N=1024
 * Uses 512 threads, each handles 2 elements
 */
__device__ __forceinline__ void SmallInverseNTT32_1024(
    uint32_t* sh,
    const uint32_t* root_table,
    uint32_t n_inverse,
    int tid)
{
    constexpr int N_power = 10;

    int t_2 = 0;
    int t_ = 0;
    int m = 1 << (N_power - 1);
    int t = 1;

    int in_shared_address = ((tid >> t_) << t_) + tid;
    int current_root_index;

    // First 6 stages - warp-local
    #pragma unroll
    for (int lp = 0; lp < 6; lp++) {
        current_root_index = m + (tid >> t_2);
        SmallGentlemanSandeUnit(sh[in_shared_address], sh[in_shared_address + t],
                                root_table[current_root_index]);

        t = t << 1;
        t_2 += 1;
        t_ += 1;
        m >>= 1;
        in_shared_address = ((tid >> t_) << t_) + tid;
    }
    __syncthreads();

    // Last 4 stages - need sync
    #pragma unroll
    for (int lp = 0; lp < 4; lp++) {
        current_root_index = m + (tid >> t_2);
        SmallGentlemanSandeUnit(sh[in_shared_address], sh[in_shared_address + t],
                                root_table[current_root_index]);

        t = t << 1;
        t_2 += 1;
        t_ += 1;
        m >>= 1;
        in_shared_address = ((tid >> t_) << t_) + tid;
        __syncthreads();
    }

    // Multiply by n^{-1}
    sh[tid] = small_mod_mult(sh[tid], n_inverse);
    sh[tid + 512] = small_mod_mult(sh[tid + 512], n_inverse);
    __syncthreads();
}

#endif // __CUDACC__

//=============================================================================
// Small Modulus NTT Handler
//=============================================================================

// Host-side storage for small NTT parameters
struct SmallNTTParams {
    uint32_t* forward_root;
    uint32_t* inverse_root;
    uint32_t n_inverse;
    bool initialized;
};

extern std::vector<SmallNTTParams> g_small_ntt_params;

/**
 * Small Modulus NTT Handler for cuFHE
 * Uses P = 655360001 (~2^29.3) instead of ~2^60
 *
 * Key difference from large modulus approach:
 * - Before NTT: Convert Torus32 to NTT modulus via modulus switching
 * - After INTT: Convert NTT modulus back to Torus32 via inverse modulus switching
 */
template <uint32_t length = TFHEpp::lvl1param::n>
class CuSmallNTTHandler {
public:
    static constexpr uint32_t kLength = length;
    static constexpr uint32_t kLogLength = []() constexpr {
        uint32_t n = length, log = 0;
        while (n > 1) { n >>= 1; ++log; }
        return log;
    }();

    uint32_t* forward_root_;
    uint32_t* inverse_root_;
    uint32_t n_inverse_;

    __host__ __device__ CuSmallNTTHandler() : forward_root_(nullptr), inverse_root_(nullptr), n_inverse_(0) {}
    __host__ __device__ ~CuSmallNTTHandler() {}

    __host__ static void Create();
    __host__ static void CreateConstant() {} // No-op for small modulus (tables already set in Create())
    __host__ static void Destroy();
    __host__ void SetDevicePointers(int device_id);

#ifdef __CUDACC__
    /**
     * Forward NTT with modulus switching (Torus32 -> NTT domain)
     *
     * This performs:
     * 1. Modulus switch: Convert input from 2^32 to P discretization
     * 2. Forward NTT in modulus P
     */
    __device__ inline void NTTWithModSwitch(
        uint32_t* const out,
        const uint32_t* const in,
        uint32_t* const sh_temp,
        uint32_t leading_thread = 0) const
    {
        const int tid = threadIdx.x - leading_thread;
        constexpr int N = length;
        constexpr int NUM_THREADS = N >> 1;  // 512 for N=1024

        // Load and modulus switch: Torus32 -> NTT modulus
        if (tid < NUM_THREADS) {
            sh_temp[tid] = torus32_to_ntt_mod(in[tid]);
            sh_temp[tid + NUM_THREADS] = torus32_to_ntt_mod(in[tid + NUM_THREADS]);
        }
        __syncthreads();

        // Forward NTT
        if (tid < NUM_THREADS) {
            if constexpr (N == 1024) {
                SmallForwardNTT32_1024(sh_temp, forward_root_, tid);
            }
        } else {
            for (int i = 0; i < 5; i++) __syncthreads();
        }

        // Copy to output
        if (tid < NUM_THREADS) {
            out[tid] = sh_temp[tid];
            out[tid + NUM_THREADS] = sh_temp[tid + NUM_THREADS];
        }
        __syncthreads();
    }

    /**
     * Forward NTT without modulus switching (for integer polynomials)
     *
     * Used for decomposed polynomials that are already integers
     */
    __device__ inline void NTT(
        uint32_t* const out,
        const int32_t* const in,
        uint32_t* const sh_temp,
        uint32_t leading_thread = 0) const
    {
        const int tid = threadIdx.x - leading_thread;
        constexpr int N = length;
        constexpr int NUM_THREADS = N >> 1;

        // Load integer values and reduce to [0, P)
        if (tid < NUM_THREADS) {
            int32_t v0 = in[tid];
            int32_t v1 = in[tid + NUM_THREADS];
            sh_temp[tid] = (v0 < 0) ? (small_ntt::P + v0) : static_cast<uint32_t>(v0);
            sh_temp[tid + NUM_THREADS] = (v1 < 0) ? (small_ntt::P + v1) : static_cast<uint32_t>(v1);
        }
        __syncthreads();

        // Forward NTT
        if (tid < NUM_THREADS) {
            if constexpr (N == 1024) {
                SmallForwardNTT32_1024(sh_temp, forward_root_, tid);
            }
        } else {
            for (int i = 0; i < 5; i++) __syncthreads();
        }

        // Copy to output
        if (tid < NUM_THREADS) {
            out[tid] = sh_temp[tid];
            out[tid + NUM_THREADS] = sh_temp[tid + NUM_THREADS];
        }
        __syncthreads();
    }

    /**
     * Inverse NTT with modulus switching (NTT domain -> Torus32)
     *
     * This performs:
     * 1. Inverse NTT in modulus P
     * 2. Modulus switch: Convert from P to 2^32 discretization
     */
    __device__ inline void NTTInvWithModSwitch(
        uint32_t* const out,
        const uint32_t* const in,
        uint32_t* const sh_temp,
        uint32_t leading_thread = 0) const
    {
        const int tid = threadIdx.x - leading_thread;
        constexpr int N = length;
        constexpr int NUM_THREADS = N >> 1;
        constexpr uint32_t half_mod = small_ntt::P / 2;

        // Load to shared
        if (tid < NUM_THREADS) {
            sh_temp[tid] = in[tid];
            sh_temp[tid + NUM_THREADS] = in[tid + NUM_THREADS];
        }
        __syncthreads();

        // Inverse NTT
        if (tid < NUM_THREADS) {
            if constexpr (N == 1024) {
                SmallInverseNTT32_1024(sh_temp, inverse_root_, n_inverse_, tid);
            }
        } else {
            for (int i = 0; i < 6; i++) __syncthreads();
        }

        // Convert to signed and apply inverse modulus switch
        if (tid < NUM_THREADS) {
            uint32_t val0 = sh_temp[tid];
            uint32_t val1 = sh_temp[tid + NUM_THREADS];

            // Centered reduction: convert [0, P) to [-P/2, P/2)
            int32_t signed0 = (val0 > half_mod) ? static_cast<int32_t>(val0 - small_ntt::P) : static_cast<int32_t>(val0);
            int32_t signed1 = (val1 > half_mod) ? static_cast<int32_t>(val1 - small_ntt::P) : static_cast<int32_t>(val1);

            // Modulus switch back to Torus32
            out[tid] = ntt_mod_to_torus32(signed0);
            out[tid + NUM_THREADS] = ntt_mod_to_torus32(signed1);
        }
        __syncthreads();
    }

    /**
     * Inverse NTT with modulus switching and addition
     */
    __device__ inline void NTTInvAddWithModSwitch(
        uint32_t* const out,
        const uint32_t* const in,
        uint32_t* const sh_temp,
        uint32_t leading_thread = 0) const
    {
        const int tid = threadIdx.x - leading_thread;
        constexpr int N = length;
        constexpr int NUM_THREADS = N >> 1;
        constexpr uint32_t half_mod = small_ntt::P / 2;

        // Load to shared
        if (tid < NUM_THREADS) {
            sh_temp[tid] = in[tid];
            sh_temp[tid + NUM_THREADS] = in[tid + NUM_THREADS];
        }
        __syncthreads();

        // Inverse NTT
        if (tid < NUM_THREADS) {
            if constexpr (N == 1024) {
                SmallInverseNTT32_1024(sh_temp, inverse_root_, n_inverse_, tid);
            }
        } else {
            for (int i = 0; i < 6; i++) __syncthreads();
        }

        // Convert and ADD to output
        if (tid < NUM_THREADS) {
            uint32_t val0 = sh_temp[tid];
            uint32_t val1 = sh_temp[tid + NUM_THREADS];

            int32_t signed0 = (val0 > half_mod) ? static_cast<int32_t>(val0 - small_ntt::P) : static_cast<int32_t>(val0);
            int32_t signed1 = (val1 > half_mod) ? static_cast<int32_t>(val1 - small_ntt::P) : static_cast<int32_t>(val1);

            out[tid] += ntt_mod_to_torus32(signed0);
            out[tid + NUM_THREADS] += ntt_mod_to_torus32(signed1);
        }
        __syncthreads();
    }
#endif // __CUDACC__
};

// Alias to provide unified interface with large modulus handler
template <uint32_t length = TFHEpp::lvl1param::n>
using CuNTTHandler = CuSmallNTTHandler<length>;

}  // namespace cufhe
