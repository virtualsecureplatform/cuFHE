#pragma once


#include <include/cufhe_gpu.cuh>
#include <include/details/error_gpu.cuh>
#include <include/details/utils_gpu.cuh>
#include <include/ntt_gpu/ntt_gpuntt.cuh>

namespace cufhe{

template <class P>
__device__ inline uint32_t modSwitchFromTorus(const typename P::domainP::T phase)
{
    constexpr uint32_t Mbit = P::targetP::nbit + 1;
    static_assert(32 >= Mbit, "Undefined modSwitchFromTorus!");
    return (phase >> (std::numeric_limits<typename P::domainP::T>::digits - 1 - P::targetP::nbit));
}

template <class P>
__device__ constexpr typename P::T offsetgen()
{
    typename P::T offset = 0;
    for (int i = 1; i <= P::l; i++)
        offset +=
            P::Bg / 2 *
            (1ULL << (std::numeric_limits<typename P::T>::digits - i * P::Bgbit));
    return offset;
}

template <class P>
__device__ inline void RotatedTestVector(typename P::T* tlwe,
                                         const int32_t bar,
                                         const typename P::T μ)
{
    // volatile is needed to make register usage of Mux to 128.
    // Reference
    // https://devtalk.nvidia.com/default/topic/466758/cuda-programming-and-performance/tricks-to-fight-register-pressure-or-how-i-got-down-from-29-to-15-registers-/
    volatile const uint32_t tid = ThisThreadRankInBlock();
    volatile const uint32_t bdim = ThisBlockSize();
#pragma unroll
    for (int i = tid; i < P::n; i += bdim) {
        #pragma unroll
        for(int k = 0; k < P::k; k++) tlwe[i + k*P::n] = 0;  // part a
        if (bar == 2 * P::n)
            tlwe[i + P::k*P::n] = μ;
        else {
            tlwe[i + P::k*P::n] = ((i < (bar & (P::n - 1))) ^ (bar >> P::nbit))
                                 ? -μ
                                 : μ;  // part b
        }
    }
    __syncthreads();
}

template<class P>
__device__ inline void PolynomialMulByXaiMinusOneAndDecompositionTRLWE(
    FFP* const dectrlwe, const typename P::T* const trlwe,
    const uint32_t a_bar)
{
    const uint32_t tid = ThisThreadRankInBlock();
    const uint32_t bdim = ThisBlockSize();
    constexpr uint32_t decomp_mask = (1 << P::Bgbit) - 1;
    constexpr int32_t decomp_half = 1 << (P::Bgbit - 1);
    constexpr uint32_t decomp_offset = offsetgen<P>();
    constexpr typename P::T roundoffset =
        1ULL << (std::numeric_limits<typename P::T>::digits -
                 P::l * P::Bgbit - 1);
#pragma unroll
    for (int i = tid; i < P::n; i += bdim) {
#pragma unroll
        for (int j = 0; j < P::k+1; j++) {
            // PolynomialMulByXaiMinus
            typename P::T temp =
                trlwe[j * P::n + ((i - a_bar) & (P::n - 1))];
            temp = ((i < (a_bar & (P::n - 1)) ^
                     (a_bar >> P::nbit)))
                       ? -temp
                       : temp;
            temp -= trlwe[j * P::n + i];
            // decomp temp
            temp += decomp_offset + roundoffset;
#pragma unroll
            for (int digit = 0; digit < P::l; digit += 1) {
                // Extract digit value and subtract half to center around 0
                // Result is in range [-decomp_half, decomp_half-1] (e.g., [-128, 127])
                // CRITICAL: Cast to signed int32_t to properly handle negative values
                // when constructing FFP. Without this, negative values like -128 become
                // 0xFFFFFF80 (uint32_t wrap) instead of P - 128 (correct modular value).
                int32_t digit_val = static_cast<int32_t>(
                    ((temp >>
                      (std::numeric_limits<typename P::T>::digits -
                       (digit + 1) * P::Bgbit)) &
                     decomp_mask) -
                    decomp_half);
                dectrlwe[j * P::l * P::n +
                         digit * P::n + i] = FFP(digit_val);
            }
        }
    }
    __syncthreads();  // must
}

/**
 * Sequential NTT Accumulate function (HEonGPU-style)
 * Uses 512 threads, processes NTTs one at a time for maximum efficiency
 *
 * Shared memory layout:
 * - sh_acc_ntt[0..N-1]: Working buffer for NTT operations (8KB)
 * - sh_acc_ntt[N..3N-1]: Accumulated products in NTT domain (16KB for k+1=2 outputs)
 */
template<class P>
__device__ inline void Accumulate(typename P::targetP::T* const trlwe, FFP* const sh_acc_ntt,
                                  const uint32_t a_bar,
                                  const FFP* const tgsw_ntt,
                                  const CuNTTHandler<> ntt)
{
    const uint32_t tid = ThisThreadRankInBlock();

    constexpr uint32_t N = P::targetP::n;
    constexpr uint32_t NUM_THREADS = N >> NTT_THREAD_UNITBIT;  // 512

    // Aliases for clarity
    FFP* const sh_work = &sh_acc_ntt[0];              // Working buffer for NTT
    FFP* const sh_accum = &sh_acc_ntt[N];             // Accumulated results (k+1 polynomials)

    // Initialize accumulated results to zero
    if (tid < NUM_THREADS) {
        for (int k_idx = 0; k_idx <= P::targetP::k; k_idx++) {
            sh_accum[k_idx * N + tid] = FFP(static_cast<Data64>(0));
            sh_accum[k_idx * N + tid + NUM_THREADS] = FFP(static_cast<Data64>(0));
        }
    }
    __syncthreads();

    // Decomposition constants
    constexpr uint32_t decomp_mask = (1 << P::targetP::Bgbit) - 1;
    constexpr int32_t decomp_half = 1 << (P::targetP::Bgbit - 1);
    constexpr uint32_t decomp_offset = offsetgen<typename P::targetP>();
    constexpr typename P::targetP::T roundoffset =
        1ULL << (std::numeric_limits<typename P::targetP::T>::digits -
                 P::targetP::l * P::targetP::Bgbit - 1);

    // Process each TRLWE component (k+1 components) and each decomposition level (l levels)
    for (int j = 0; j <= P::targetP::k; j++) {
        for (int digit = 0; digit < P::targetP::l; digit++) {
            // Step 1: Compute decomposed polynomial for component j, digit
            // MulByXaiMinusOne and decomposition - each thread handles 2 elements
            if (tid < NUM_THREADS) {
                #pragma unroll
                for (int e = 0; e < 2; e++) {
                    int i = tid + e * NUM_THREADS;

                    // PolynomialMulByXaiMinus
                    typename P::targetP::T temp =
                        trlwe[j * N + ((i - a_bar) & (N - 1))];
                    temp = ((i < (a_bar & (N - 1)) ^
                             (a_bar >> P::targetP::nbit)))
                               ? -temp
                               : temp;
                    temp -= trlwe[j * N + i];

                    // Decomposition for this digit
                    temp += decomp_offset + roundoffset;
                    int32_t digit_val = static_cast<int32_t>(
                        ((temp >>
                          (std::numeric_limits<typename P::targetP::T>::digits -
                           (digit + 1) * P::targetP::Bgbit)) &
                         decomp_mask) -
                        decomp_half);
                    sh_work[i] = FFP(digit_val);
                }
            }
            __syncthreads();

            // Step 2: Forward NTT on decomposed polynomial - dispatch based on N
            if (tid < NUM_THREADS) {
                if constexpr (N == 1024) {
                    SmallForwardNTT_1024(reinterpret_cast<Data64*>(sh_work), ntt.forward_root_, tid);
                } else if constexpr (N == 512) {
                    SmallForwardNTT_512(reinterpret_cast<Data64*>(sh_work), ntt.forward_root_, tid);
                }
            } else {
                // Sync count: 5 for N=1024, 9 for N=512 (after DEBUG syncs added)
                constexpr int sync_count = (N == 1024) ? 5 : 9;
                for (int s = 0; s < sync_count; s++) __syncthreads();
            }

            // Step 3: Multiply with BK and accumulate into sh_accum
            // tgsw_ntt layout: [(k+1)*l output components] x [N]
            // For digit d from component j: access tgsw_ntt[((k+1)*digit_linear + output_k) * N + i]
            // where digit_linear = j * l + digit
            int digit_linear = j * P::targetP::l + digit;
            if (tid < NUM_THREADS) {
                #pragma unroll
                for (int e = 0; e < 2; e++) {
                    int i = tid + e * NUM_THREADS;
                    FFP ntt_val = sh_work[i];

                    // Accumulate into each output component
                    #pragma unroll
                    for (int out_k = 0; out_k <= P::targetP::k; out_k++) {
                        FFP bk_val = tgsw_ntt[(((P::targetP::k + 1) * digit_linear + out_k) << P::targetP::nbit) + i];
                        sh_accum[out_k * N + i] += ntt_val * bk_val;
                    }
                }
            }
            __syncthreads();
        }
    }

    // Step 4: Inverse NTT on accumulated results and add to trlwe
    // Process (k+1) inverse NTTs sequentially
    for (int k_idx = 0; k_idx <= P::targetP::k; k_idx++) {
        // Copy accumulated data to work buffer
        if (tid < NUM_THREADS) {
            sh_work[tid] = sh_accum[k_idx * N + tid];
            sh_work[tid + NUM_THREADS] = sh_accum[k_idx * N + tid + NUM_THREADS];
        }
        __syncthreads();

        // Inverse NTT - dispatch based on N
        constexpr Data64 half_mod = GPUNTT_DEFAULT_MODULUS / 2;
        if (tid < NUM_THREADS) {
            if constexpr (N == 1024) {
                SmallInverseNTT_1024(reinterpret_cast<Data64*>(sh_work), ntt.inverse_root_, ntt.n_inverse_, tid);
            } else if constexpr (N == 512) {
                SmallInverseNTT_512(reinterpret_cast<Data64*>(sh_work), ntt.inverse_root_, ntt.n_inverse_, tid);
            }
        } else {
            // Sync count: 6 for N=1024, 10 for N=512 (after DEBUG syncs added)
            constexpr int sync_count = (N == 1024) ? 6 : 10;
            for (int s = 0; s < sync_count; s++) __syncthreads();
        }

        // Convert and add to trlwe
        if (tid < NUM_THREADS) {
            #pragma unroll
            for (int e = 0; e < 2; e++) {
                int i = tid + e * NUM_THREADS;
                Data64 val = sh_work[i].val();
                typename P::targetP::T conv = (val > half_mod)
                    ? static_cast<typename P::targetP::T>(static_cast<int64_t>(val) - static_cast<int64_t>(GPUNTT_DEFAULT_MODULUS))
                    : static_cast<typename P::targetP::T>(val);
                trlwe[k_idx * N + i] += conv;
            }
        }
        __syncthreads();
    }
}

template<class P>
__device__ inline void __BlindRotate__(typename P::targetP::T* const out,
                                   const typename P::domainP::T* const in,
                                   const typename P::targetP::T mu,
                                   const FFP* const bk,
                                   CuNTTHandler<> ntt)
{
    extern __shared__ FFP sh[];
    FFP* sh_acc_ntt = &sh[0];

    // test vector: acc.a = 0; acc.b = vec(mu) * x ^ (in.b()/2048)
    {
        const uint32_t bar =
            2 * P::targetP::n -
            modSwitchFromTorus<P>(in[P::domainP::k*P::domainP::n]);
        RotatedTestVector<typename P::targetP>(out, bar, mu);
    }

    // accumulate
    for (int i = 0; i < P::domainP::k*P::domainP::n; i++) {  // lvl0param::n iterations
        constexpr typename P::domainP::T roundoffset =
                1ULL << (std::numeric_limits<typename P::domainP::T>::digits - 2 -
                        P::targetP::nbit);
        const uint32_t bar = modSwitchFromTorus<P>(in[i]+roundoffset);
        Accumulate<P>(out, sh_acc_ntt, bar,
                   bk + (i << P::targetP::nbit) * (P::targetP::k+1) * (P::targetP::k+1) * P::targetP::l, ntt);
    }
}

template <class P, int casign, int cbsign, std::make_signed_t<typename P::domainP::T> offset>
__device__ inline void __BlindRotatePreAdd__(typename P::targetP::T* const out,
                                   const typename P::domainP::T* const in0,
                                   const typename P::domainP::T* const in1,
                                   const FFP* const bk,
                                   CuNTTHandler<> ntt)
{
    extern __shared__ FFP sh[];
    FFP* sh_acc_ntt = &sh[0];

    // test vector: acc.a = 0; acc.b = vec(mu) * x ^ (in.b()/2048)
    {
        const uint32_t bar =
            2 * P::targetP::n -
            modSwitchFromTorus<P>(offset + casign * in0[P::domainP::k*P::domainP::n] +
                                          cbsign * in1[P::domainP::k*P::domainP::n]);
        RotatedTestVector<typename P::targetP>(out, bar, P::targetP::μ);
    }

    // accumulate
    for (int i = 0; i < P::domainP::k*P::domainP::n; i++) {  // lvl0param::n iterations
        constexpr typename P::domainP::T roundoffset =
                1ULL << (std::numeric_limits<typename P::domainP::T>::digits - 2 -
                        P::targetP::nbit);
        const uint32_t bar = modSwitchFromTorus<P>(0 + casign * in0[i] +
                                                           cbsign * in1[i] + roundoffset);
        Accumulate<P>(out, sh_acc_ntt, bar,
                   bk + (i << P::targetP::nbit) * (P::targetP::k+1) * (P::targetP::k+1) * P::targetP::l, ntt);
    }
}
}