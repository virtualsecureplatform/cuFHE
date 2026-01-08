#pragma once


#include <include/cufhe_gpu.cuh>
#include <include/details/error_gpu.cuh>
#include <include/details/utils_gpu.cuh>
#include <include/ntt_gpu/ntt.cuh>

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

template<class P>
__device__ inline void Accumulate(typename P::targetP::T* const trlwe, FFP* const sh_acc_ntt,
                                  const uint32_t a_bar,
                                  const FFP* const tgsw_ntt,
                                  const CuNTTHandler<> ntt)
{
    const uint32_t tid = ThisThreadRankInBlock();
    const uint32_t bdim = ThisBlockSize();

    PolynomialMulByXaiMinusOneAndDecompositionTRLWE<typename P::targetP>(sh_acc_ntt, trlwe, a_bar);

    // (k+1)l NTTs
    // Input/output/buffer use the same shared memory location.
    if (tid < (P::targetP::k+1) * P::targetP::l * (P::targetP::n >> NTT_THREAD_UNITBIT)) {
        FFP* tar = &sh_acc_ntt[tid >> (P::targetP::nbit - NTT_THREAD_UNITBIT)
                                          << P::targetP::nbit];
        ntt.NTT<FFP>(tar, tar, tar,
                     tid >> (P::targetP::nbit - NTT_THREAD_UNITBIT)
                                << (P::targetP::nbit - NTT_THREAD_UNITBIT));
    }
    else {
        // Optimized NTT: 12 syncs (1 load + 10 NTT stages + 1 store)
        __syncthreads();
        __syncthreads();
        __syncthreads();
        __syncthreads();
        __syncthreads();
        __syncthreads();
        __syncthreads();
        __syncthreads();
        __syncthreads();
        __syncthreads();
        __syncthreads();
        __syncthreads();
    }
    __syncthreads();

// Multiply with bootstrapping key in global memory.
#pragma unroll
    for (int i = tid; i < P::targetP::n; i += bdim) {
        sh_acc_ntt[(P::targetP::k + 1) * P::targetP::l * P::targetP::n + i] =
            sh_acc_ntt[0 * P::targetP::n + i] *
            tgsw_ntt[(((P::targetP::k + 1) * 0 + 1) << P::targetP::nbit) + i];
        sh_acc_ntt[i] = sh_acc_ntt[0 * P::targetP::n + i] *
                        tgsw_ntt[(((P::targetP::k + 1) * 0 + 0) << P::targetP::nbit) + i];
#pragma unroll
        for (int digit = 1; digit < (P::targetP::k + 1) * P::targetP::l; digit += 1) {
            sh_acc_ntt[i] += sh_acc_ntt[digit * P::targetP::n + i] *
                             tgsw_ntt[(((P::targetP::k + 1) * digit + 0) << P::targetP::nbit) + i];
            sh_acc_ntt[(P::targetP::k + 1) * P::targetP::l * P::targetP::n + i] +=
                sh_acc_ntt[digit * P::targetP::n + i] *
                tgsw_ntt[(((P::targetP::k + 1) * digit + 1) << P::targetP::nbit) + i];
        }
    }
    __syncthreads();

    // k+1 NTTInvs and add acc
    if (tid < (P::targetP::k + 1) * (P::targetP::n >> NTT_THREAD_UNITBIT)) {
        FFP* src = &sh_acc_ntt[(tid >> (P::targetP::nbit - NTT_THREAD_UNITBIT)) *
                               (P::targetP::k + 1) * P::targetP::l * P::targetP::n];
        ntt.NTTInvAdd<typename P::targetP::T>(
            &trlwe[tid >> (P::targetP::nbit - NTT_THREAD_UNITBIT)
                              << P::targetP::nbit],
            src, src,
            tid >> (P::targetP::nbit - NTT_THREAD_UNITBIT)
                       << (P::targetP::nbit - NTT_THREAD_UNITBIT));
    }
    else {
        // Optimized INTT: 13 syncs (1 load + 11 INTT stages + 1 convert)
        __syncthreads();
        __syncthreads();
        __syncthreads();
        __syncthreads();
        __syncthreads();
        __syncthreads();
        __syncthreads();
        __syncthreads();
        __syncthreads();
        __syncthreads();
        __syncthreads();
        __syncthreads();
        __syncthreads();
    }
    __syncthreads();  // must
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