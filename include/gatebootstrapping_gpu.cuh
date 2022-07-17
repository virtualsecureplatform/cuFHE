#pragma once


#include <include/cufhe.h>

#include <include/cufhe_gpu.cuh>
#include <include/details/error_gpu.cuh>
#include <include/ntt_gpu/ntt.cuh>

namespace cufhe{

template <class P>
__device__ inline typename P::T modSwitchFromTorus(const uint32_t phase)
{
    constexpr uint32_t Mbit = P::nbit + 1;
    static_assert(32 >= Mbit, "Undefined modSwitchFromTorus!");
    return (phase + (1U << (31 - Mbit))) >> (32 - Mbit);
}

template <class P>
__device__ constexpr typename P::T offsetgen()
{
    typename P::T offset = 0;
    for (int i = 1; i <= P::l; i++)
        offset +=
            P::Bg / 2 *
            (1ULL << (numeric_limits<typename P::T>::digits - i * P::Bgbit));
    return offset;
}

template <class P>
__device__ inline void RotatedTestVector(TFHEpp::lvl1param::T* tlwe,
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

__device__ inline void PolynomialMulByXaiMinusOneAndDecompositionTRLWE(
    FFP* dectrlwe, const TFHEpp::lvl1param::T* const trlwe,
    const uint32_t a_bar)
{
    const uint32_t tid = ThisThreadRankInBlock();
    const uint32_t bdim = ThisBlockSize();
    constexpr uint32_t decomp_mask = (1 << lvl1param::Bgbit) - 1;
    constexpr int32_t decomp_half = 1 << (lvl1param::Bgbit - 1);
    constexpr uint32_t decomp_offset = offsetgen<lvl1param>();
    constexpr typename lvl1param::T roundoffset =
        1ULL << (std::numeric_limits<typename lvl1param::T>::digits -
                 lvl1param::l * lvl1param::Bgbit - 1);
#pragma unroll
    for (int i = tid; i < lvl1param::n; i += bdim) {
#pragma unroll
        for (int j = 0; j < lvl1param::k+1; j++) {
            // PolynomialMulByXaiMinus
            lvl1param::T temp =
                trlwe[j * lvl1param::n + ((i - a_bar) & (lvl1param::n - 1))];
            temp = ((i < (a_bar & (lvl1param::n - 1)) ^
                     (a_bar >> lvl1param::nbit)))
                       ? -temp
                       : temp;
            temp -= trlwe[j * lvl1param::n + i];
            // decomp temp
            temp += decomp_offset + roundoffset;
#pragma unroll
            for (int digit = 0; digit < lvl1param::l; digit += 1)
                dectrlwe[j * lvl1param::l * lvl1param::n +
                         digit * lvl1param::n + i] =
                    FFP(lvl1param::T(
                        ((temp >>
                          (std::numeric_limits<typename lvl1param::T>::digits -
                           (digit + 1) * lvl1param::Bgbit)) &
                         decomp_mask) -
                        decomp_half));
        }
    }
    __syncthreads();  // must
}

__device__ inline void Accumulate(TFHEpp::lvl1param::T* trlwe, FFP* sh_acc_ntt,
                                  const uint32_t a_bar,
                                  const FFP* const tgsw_ntt,
                                  const CuNTTHandler<> ntt)
{
    const uint32_t tid = ThisThreadRankInBlock();
    const uint32_t bdim = ThisBlockSize();

    PolynomialMulByXaiMinusOneAndDecompositionTRLWE(sh_acc_ntt, trlwe, a_bar);

    // (k+1)l NTTs
    // Input/output/buffer use the same shared memory location.
    if (tid < (lvl1param::k+1) * lvl1param::l * (lvl1param::n >> NTT_THRED_UNITBIT)) {
        FFP* tar = &sh_acc_ntt[tid >> (lvl1param::nbit - NTT_THRED_UNITBIT)
                                          << lvl1param::nbit];
        ntt.NTT<FFP>(tar, tar, tar,
                     tid >> (lvl1param::nbit - NTT_THRED_UNITBIT)
                                << (lvl1param::nbit - NTT_THRED_UNITBIT));
    }
    else {  // must meet 5 sync made by NTTInv
        __syncthreads();
        __syncthreads();
        __syncthreads();
        __syncthreads();
        __syncthreads();
    }
    __syncthreads();

// Multiply with bootstrapping key in global memory.
#pragma unroll
    for (int i = tid; i < lvl1param::n; i += bdim) {
        sh_acc_ntt[(lvl1param::k + 1) * lvl1param::l * lvl1param::n + i] =
            sh_acc_ntt[0 * lvl1param::n + i] *
            tgsw_ntt[(((lvl1param::k + 1) * 0 + 1) << lvl1param::nbit) + i];
        sh_acc_ntt[i] = sh_acc_ntt[0 * lvl1param::n + i] *
                        tgsw_ntt[(((lvl1param::k + 1) * 0 + 0) << lvl1param::nbit) + i];
#pragma unroll
        for (int digit = 1; digit < (lvl1param::k + 1) * lvl1param::l; digit += 1) {
            sh_acc_ntt[i] += sh_acc_ntt[digit * lvl1param::n + i] *
                             tgsw_ntt[(((lvl1param::k + 1) * digit + 0) << lvl1param::nbit) + i];
            sh_acc_ntt[(lvl1param::k + 1) * lvl1param::l * lvl1param::n + i] +=
                sh_acc_ntt[digit * lvl1param::n + i] *
                tgsw_ntt[(((lvl1param::k + 1) * digit + 1) << lvl1param::nbit) + i];
        }
    }
    __syncthreads();

    // k+1 NTTInvs and add acc
    if (tid < (lvl1param::k + 1) * (lvl1param::n >> NTT_THRED_UNITBIT)) {
        FFP* src = &sh_acc_ntt[(tid >> (lvl1param::nbit - NTT_THRED_UNITBIT)) *
                               (lvl1param::k + 1) * lvl1param::l * lvl1param::n];
        ntt.NTTInvAdd<typename lvl1param::T>(
            &trlwe[tid >> (lvl1param::nbit - NTT_THRED_UNITBIT)
                              << lvl1param::nbit],
            src, src,
            tid >> (lvl1param::nbit - NTT_THRED_UNITBIT)
                       << (lvl1param::nbit - NTT_THRED_UNITBIT));
    }
    else {  // must meet 5 sync made by NTTInv
        __syncthreads();
        __syncthreads();
        __syncthreads();
        __syncthreads();
        __syncthreads();
    }
    __syncthreads();  // must
}

__device__ inline void __BlindRotate__(TFHEpp::lvl1param::T* out,
                                   TFHEpp::lvl0param::T* in,
                                   const TFHEpp::lvl1param::T mu,
                                   const FFP* const bk,
                                   CuNTTHandler<> ntt)
{
    extern __shared__ FFP sh[];
    FFP* sh_acc_ntt = &sh[0];

    // test vector: acc.a = 0; acc.b = vec(mu) * x ^ (in.b()/2048)
    {
        const uint32_t bar =
            2 * lvl1param::n -
            modSwitchFromTorus<lvl1param>(in[lvl0param::k*lvl0param::n]);
        RotatedTestVector<lvl1param>(out, bar, mu);
    }

    // accumulate
    for (int i = 0; i < lvl0param::n; i++) {  // lvl0param::n iterations
        const uint32_t bar = modSwitchFromTorus<lvl1param>(in[i]);
        Accumulate(out, sh_acc_ntt, bar,
                   bk + (i << lvl1param::nbit) * (lvl1param::k+1) * (lvl1param::k+1) * lvl1param::l, ntt);
    }
}

template <int casign, int cbsign, typename lvl0param::T offset>
__device__ inline void __BlindRotatePreAdd__(TFHEpp::lvl1param::T* out,
                                   TFHEpp::lvl0param::T* in0,
                                   TFHEpp::lvl0param::T* in1,
                                   const FFP* const bk,
                                   CuNTTHandler<> ntt)
{
    extern __shared__ FFP sh[];
    FFP* sh_acc_ntt = &sh[0];

    // test vector: acc.a = 0; acc.b = vec(mu) * x ^ (in.b()/2048)
    {
        const uint32_t bar =
            2 * lvl1param::n -
            modSwitchFromTorus<lvl1param>(offset + casign * in0[lvl0param::k*lvl0param::n] +
                                          cbsign * in1[lvl0param::n]);
        RotatedTestVector<lvl1param>(out, bar, lvl1param::μ);
    }

    // accumulate
    for (int i = 0; i < lvl0param::n; i++) {  // lvl0param::n iterations
        const uint32_t bar = modSwitchFromTorus<lvl1param>(0 + casign * in0[i] +
                                                           cbsign * in1[i]);
        Accumulate(out, sh_acc_ntt, bar,
                   bk + (i << lvl1param::nbit) * (lvl1param::k+1) * (lvl1param::k+1) * lvl1param::l, ntt);
    }
}
}