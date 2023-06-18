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

#include <include/cufhe.h>

#include <include/bootstrap_gpu.cuh>
#include <include/gatebootstrapping_gpu.cuh>
#include <include/keyswitch_gpu.cuh>
#include <include/cufhe_gpu.cuh>
#include <include/details/error_gpu.cuh>
#include <include/ntt_gpu/ntt.cuh>
#include <limits>
#include <vector>

namespace cufhe {
template<class P = TFHEpp::lvl1param>
constexpr uint MEM4HOMGATE =
    ((P::k+1) * P::l + 1 + P::k) * P::n * sizeof(FFP);

using namespace std;
using namespace TFHEpp;

vector<FFP*> bk_ntts;
vector<CuNTTHandler<>*> ntt_handlers;

__global__ void __TRGSW2NTT__(FFP* const bk_ntt, const TFHEpp::lvl1param::T* const bk,
                              CuNTTHandler<> ntt)
{
    __shared__ FFP sh_temp[lvl1param::n];
    const int index = blockIdx.z * ((lvl1param::k+1) * lvl1param::l * (lvl1param::k+1) * lvl1param::n) +
                      blockIdx.y * (lvl1param::k+1) * lvl1param::n + blockIdx.x * lvl1param::n;
    ntt.NTT<lvl1param::T>(&bk_ntt[index], &bk[index], sh_temp, 0);
}

void TRGSW2NTT(cuFHETRGSWNTTlvl1& trgswntt,
               const TFHEpp::TRGSW<TFHEpp::lvl1param>& trgsw, Stream& st)
{
    cudaSetDevice(st.device_id());
    TFHEpp::lvl1param::T* d_trgsw;
    cudaMalloc((void**)&d_trgsw, sizeof(trgsw));
    cudaMemcpyAsync(d_trgsw, trgsw.data(), sizeof(trgsw),
                    cudaMemcpyHostToDevice, st.st());

    dim3 grid(lvl1param::k+1, (lvl1param::k+1) * lvl1param::l, 1);
    dim3 block(lvl1param::n >> NTT_THRED_UNITBIT);
    __TRGSW2NTT__<<<grid, block, 0, st.st()>>>(
        trgswntt.trgswdevices[st.device_id()], d_trgsw,
        *ntt_handlers[st.device_id()]);
    CuCheckError();
    cudaMemcpyAsync(
        trgswntt.trgswhost.data(), trgswntt.trgswdevices[st.device_id()],
        sizeof(trgswntt.trgswhost), cudaMemcpyDeviceToHost, st.st());
    cudaFree(d_trgsw);
}

void InitializeNTThandlers(const int gpuNum)
{
    for (int i = 0; i < gpuNum; i++) {
        cudaSetDevice(i);

        ntt_handlers.push_back(new CuNTTHandler<>());
        ntt_handlers[i]->Create();
        ntt_handlers[i]->CreateConstant();
        cudaDeviceSynchronize();
        CuCheckError();
    }
}

void BootstrappingKeyToNTT(const BootstrappingKey<lvl01param>& bk,
                           const int gpuNum)
{
    bk_ntts.resize(gpuNum);
    for (int i = 0; i < gpuNum; i++) {
        cudaSetDevice(i);

        cudaMalloc((void**)&bk_ntts[i], sizeof(FFP) * lvl0param::n * 2 *
                                            lvl1param::l * 2 * lvl1param::n);

        TFHEpp::lvl1param::T* d_bk;
        cudaMalloc((void**)&d_bk, sizeof(bk));
        cudaMemcpy(d_bk, bk.data(), sizeof(bk), cudaMemcpyHostToDevice);

        cudaDeviceSynchronize();
        CuCheckError();

        dim3 grid(lvl1param::k+1, (lvl1param::k+1) * lvl1param::l, lvl0param::n);
        dim3 block(lvl1param::n >> NTT_THRED_UNITBIT);
        __TRGSW2NTT__<<<grid, block>>>(bk_ntts[i], d_bk, *ntt_handlers[i]);
        cudaDeviceSynchronize();
        CuCheckError();

        cudaFree(d_bk);
    }
}

void DeleteBootstrappingKeyNTT(const int gpuNum)
{
    for (int i = 0; i < bk_ntts.size(); i++) {
        cudaSetDevice(i);
        cudaFree(bk_ntts[i]);

        ntt_handlers[i]->Destroy();
        delete ntt_handlers[i];
    }
    ntt_handlers.clear();
}

__device__ inline void TRLWESubAndDecomposition(
    FFP* const dectrlwe, const TFHEpp::lvl1param::T* const trlwe1,
    const TFHEpp::lvl1param::T* const trlwe0)
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
        for (int j = 0; j < (lvl1param::k+1); j++) {
            // decomp temp
            lvl1param::T temp = trlwe1[j * lvl1param::n + i] -
                                trlwe0[j * lvl1param::n + i] + decomp_offset +
                                roundoffset;
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

__global__ __launch_bounds__(NUM_THREAD4HOMGATE) void __CMUXNTT__(
    TFHEpp::lvl1param::T* out, const FFP* const tgsw_ntt,
    const TFHEpp::lvl1param::T* const trlwe1,
    const TFHEpp::lvl1param::T* const trlwe0, const CuNTTHandler<> ntt)
{
    const uint32_t tid = ThisThreadRankInBlock();
    const uint32_t bdim = ThisBlockSize();

    extern __shared__ FFP sh[];
    // To hold the data after Decomposition and NTT
    FFP* sh_acc_ntt = &sh[0];
    // To hold sum
    FFP* sh_res_ntt = &sh[(lvl1param::k+1) * lvl1param::l * lvl1param::n];
    TFHEpp::lvl1param::T* outtemp = (TFHEpp::lvl1param::T*)&sh[0];

    TRLWESubAndDecomposition(sh_acc_ntt, trlwe1, trlwe0);

    // (k+1)*l NTTs
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
        #pragma unroll
        for(int k = 0; k < lvl1param::k+1; k++)
        sh_res_ntt[i + k*lvl1param::n] = sh_acc_ntt[0 * lvl1param::n + i] *
                        tgsw_ntt[(((lvl1param::k+1) * 0 + k) << lvl1param::nbit) + i];
#pragma unroll
        for (int digit = 1; digit < 2 * lvl1param::l; digit += 1) {
            #pragma unroll
            for(int k = 0; k < lvl1param::k+1; k++)
            sh_res_ntt[i + k*lvl1param::n] = sh_acc_ntt[digit * lvl1param::n + i] *
                            tgsw_ntt[(((lvl1param::k+1) * digit + k) << lvl1param::nbit) + i];
        }
    }
    __syncthreads();

#pragma unroll
    for (int i = tid; i < (lvl1param::k+1) * lvl1param::n; i += bdim) outtemp[i] = trlwe0[i];

    // k+1 NTTInvs and add acc
    if (tid < (lvl1param::k+1) * (lvl1param::n >> NTT_THRED_UNITBIT)) {
        FFP* src = &sh_res_ntt[tid >> (lvl1param::nbit - NTT_THRED_UNITBIT)
                                          << lvl1param::nbit];
        ntt.NTTInvAdd<typename lvl1param::T>(
            &outtemp[tid >> (lvl1param::nbit - NTT_THRED_UNITBIT)
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
    for (int i = 0; i < (lvl1param::k+1) * lvl1param::n; i++) out[i] = outtemp[i];
    __syncthreads();
}

template <class bkP, class iksP>
__global__ __launch_bounds__(NUM_THREAD4HOMGATE) void __Bootstrap__(
    typename iksP::domainP::T* const out, const typename iksP::domainP::T* const in,
    const typename bkP::targetP::T mu, const FFP* const bk,
    const typename iksP::targetP::T* const ksk, const CuNTTHandler<> ntt)
{
    __shared__ typename bkP::targetP::T tlwe[(bkP::targetP::k+1)*bkP::targetP::n]; 

    __BlindRotate__<bkP>(tlwe,in,mu,bk,ntt);
    KeySwitch<iksP>(out, tlwe, ksk);
    __threadfence();
}

// template <class iksP, class bkP>
// __global__ __launch_bounds__(NUM_THREAD4HOMGATE) void __Bootstrap__(
//     typename iksP::domainP::T* const out, const typename iksP::domainP::T* const in,
//     const typename bkP::targetP::T mu, const FFP* const bk,
//     const typename iksP::targetP::T* const ksk, const CuNTTHandler<> ntt)
// {
//     __shared__ typename bkP::targetP::T tlwe[iksP::targetP::k*iksP::targetP::n+1]; 

//     KeySwitch<iksP>(tlwe, in, ksk);
//     __threadfence();
//     __BlindRotate__<bkP>(out,tlwe,mu,bk,ntt);
//     __threadfence();
// }

template<class P>
__global__ __launch_bounds__(NUM_THREAD4HOMGATE) void __BlindRotateGlobal__(
    TFHEpp::lvl1param::T* const out, const TFHEpp::lvl0param::T* const in,
    const TFHEpp::lvl1param::T mu, const FFP* const bk, const CuNTTHandler<> ntt)
{
    __BlindRotate__<P>(out, in, mu, bk, ntt);
}

__global__ __launch_bounds__(NUM_THREAD4HOMGATE) void __SEIandBootstrap2TRLWE__(
    TFHEpp::lvl1param::T* const out, const TFHEpp::lvl1param::T* const in,
    const TFHEpp::lvl1param::T mu, const FFP* const bk, const TFHEpp::lvl0param::T* const ksk,
    const CuNTTHandler<> ntt)
{
    //  Assert(bk.k() == 1);
    //  Assert(bk.l() == 2);
    //  Assert(bk.n() == lvl1param::n);
    extern __shared__ FFP sh[];
    FFP* sh_acc_ntt = &sh[0];
    // Use Last section to hold tlwe. This may to make these data in serial
    TFHEpp::lvl1param::T* tlwe =
        (TFHEpp::lvl1param::T*)&sh[((lvl1param::k+1) * lvl1param::l + 2) * lvl1param::n];

    lvl0param::T* tlwelvl0 =
        (lvl0param::T*)&sh[((lvl1param::k+1) * lvl1param::l + 2 + lvl1param::k) * lvl1param::n];

    KeySwitch<lvl10param>(tlwelvl0, in, ksk);
    __syncthreads();

    // test vector
    // acc.a = 0; acc.b = vec(mu) * x ^ (in.b()/2048)
    register uint32_t bar = 2 * lvl1param::n - modSwitchFromTorus<lvl1param>(
                                                   tlwelvl0[lvl0param::n]);
    RotatedTestVector<lvl1param>(tlwe, bar, mu);

    // accumulate
    for (int i = 0; i < lvl0param::n; i++) {  // n iterations
        bar = modSwitchFromTorus<lvl1param>(tlwelvl0[i]);
        Accumulate<lvl01param>(tlwe, sh_acc_ntt, bar,
                   bk + (i << lvl1param::nbit) * (lvl1param::k+1) * (lvl1param::k+1) * lvl1param::l, ntt);
    }
    __syncthreads();
    for (int i = 0; i < (lvl1param::k+1) * lvl1param::n; i++) {
        out[i] = tlwe[i];
    }
    __threadfence();
}

template<class P, uint index>
__device__ inline void __SampleExtractIndex__(typename P::T* const res, const typename P::T* const in){
    const uint32_t tid = ThisThreadRankInBlock();
    const uint32_t bdim = ThisBlockSize();
    constexpr uint nmask = P::n-1; 
    for (uint i = tid; i <= P::targetP::k*P::targetP::n; i += bdim) {
        if (i == P::targetP::k*P::targetP::n){
            res[P::k*P::n] = in[P::k*P::n+index];
        }else {
            const uint k = i >> P::nbit; 
            const uint n = i & nmask;
            if (n  <= index) res[index] = in[k*P::n + index - n];
            else res[index] = -in[k*P::n + P::n + index-n];
        }
    }
}

template <class iksP, class brP, typename brP::targetP::T μ, int casign, int cbsign, typename brP::domainP::T offset>
__device__ inline void __HomGate__(typename brP::targetP::T* const out,
                                   const typename iksP::domainP::T* const in0,
                                   const typename iksP::domainP::T* const in1, const FFP* const bk,
                                   const typename iksP::targetP::T* const ksk,
                                   const CuNTTHandler<> ntt)
{
    __shared__ typename iksP::targetP::T tlwe[iksP::targetP::k*iksP::targetP::n+1]; 
    __shared__ typename brP::targetP::T trlwe[(brP::targetP::k+1)*brP::targetP::n]; 

    IdentityKeySwitchPreAdd<iksP>(tlwe, in0, in1, ksk);
    __BlindRotate__<brP, casign,cbsign,offset>(trlwe, tlwe,bk,ntt);
    __SampleExtractIndex__<brP::targetP,0>(out,trlwe);
    __threadfence();
}

template <class brP, typename brP::targetP::T μ, class iksP, int casign, int cbsign, typename brP::domainP::T offset>
__device__ inline void __HomGate__(typename iksP::targetP::T* const out,
                                   const typename brP::domainP::T* const in0,
                                   const typename brP::domainP::T* const in1, const FFP* const bk,
                                   const typename iksP::targetP::T* const ksk,
                                   const CuNTTHandler<> ntt)
{
    __shared__ typename brP::targetP::T tlwe[(brP::targetP::k+1)*brP::targetP::n]; 

    __BlindRotatePreAdd__<brP, casign,cbsign,offset>(tlwe,in0,in1,bk,ntt);
    KeySwitch<iksP>(out, tlwe, ksk);
    __threadfence();
}

template<class brP = TFHEpp::lvl01param, typename brP::targetP::T μ = TFHEpp::lvl1param::μ, class iksP = TFHEpp::lvl10param>
__global__ __launch_bounds__(NUM_THREAD4HOMGATE) void __NandBootstrap__(
    typename iksP::targetP::T* const out, const typename brP::domainP::T* const in0,
    const typename brP::domainP::T* const in1, FFP* bk, const typename iksP::targetP::T* const ksk,
    const CuNTTHandler<> ntt)
{
    __HomGate__<brP, μ, iksP, -1, -1, lvl0param::μ>(out, in0, in1, bk, ksk, ntt);
}

template<class brP = TFHEpp::lvl01param, typename brP::targetP::T μ = TFHEpp::lvl1param::μ, class iksP = TFHEpp::lvl10param>
__global__ __launch_bounds__(NUM_THREAD4HOMGATE) void __NorBootstrap__(
    typename iksP::targetP::T* const out, const typename brP::domainP::T* const in0,
    const typename brP::domainP::T* const in1, FFP* bk, const typename iksP::targetP::T* const ksk,
    const CuNTTHandler<> ntt)
{
    __HomGate__<brP, μ, iksP, -1, -1, -lvl0param::μ>(out, in0, in1, bk, ksk, ntt);
}

template<class brP = TFHEpp::lvl01param, typename brP::targetP::T μ = TFHEpp::lvl1param::μ, class iksP = TFHEpp::lvl10param>
__global__ __launch_bounds__(NUM_THREAD4HOMGATE) void __XnorBootstrap__(
    typename iksP::targetP::T* const out, const typename brP::domainP::T* const in0,
    const typename brP::domainP::T* const in1, FFP* bk, const typename iksP::targetP::T* const ksk,
    const CuNTTHandler<> ntt)
{
    __HomGate__<brP, μ, iksP, -2, -2, -2 * lvl0param::μ>(out, in0, in1, bk, ksk, ntt);
}

template<class brP = TFHEpp::lvl01param, typename brP::targetP::T μ = TFHEpp::lvl1param::μ, class iksP = TFHEpp::lvl10param>
__global__ __launch_bounds__(NUM_THREAD4HOMGATE) void __AndBootstrap__(
    typename iksP::targetP::T* const out, const typename brP::domainP::T* const in0,
    const typename brP::domainP::T* const in1, FFP* bk, const typename iksP::targetP::T* const ksk,
    const CuNTTHandler<> ntt)
{
    __HomGate__<brP, μ, iksP, 1, 1, -lvl0param::μ>(out, in0, in1, bk, ksk, ntt);
}

template<class brP = TFHEpp::lvl01param, typename brP::targetP::T μ = TFHEpp::lvl1param::μ, class iksP = TFHEpp::lvl10param>
__global__ __launch_bounds__(NUM_THREAD4HOMGATE) void __OrBootstrap__(
    typename iksP::targetP::T* const out, const typename brP::domainP::T* const in0,
    const typename brP::domainP::T* const in1, FFP* bk, const typename iksP::targetP::T* const ksk,
    const CuNTTHandler<> ntt)
{
    __HomGate__<brP, μ, iksP, 1, 1, lvl0param::μ>(out, in0, in1, bk, ksk, ntt);
}

template<class brP = TFHEpp::lvl01param, typename brP::targetP::T μ = TFHEpp::lvl1param::μ, class iksP = TFHEpp::lvl10param>
__global__ __launch_bounds__(NUM_THREAD4HOMGATE) void __XorBootstrap__(
    typename iksP::targetP::T* const out, const typename brP::domainP::T* const in0,
    const typename brP::domainP::T* const in1, FFP* bk, const typename iksP::targetP::T* const ksk,
    const CuNTTHandler<> ntt)
{
    __HomGate__<brP, μ, iksP, 2, 2, 2 * lvl0param::μ>(out, in0, in1, bk, ksk, ntt);
}

template<class brP = TFHEpp::lvl01param, typename brP::targetP::T μ = TFHEpp::lvl1param::μ, class iksP = TFHEpp::lvl10param>
__global__ __launch_bounds__(NUM_THREAD4HOMGATE) void __AndNYBootstrap__(
    typename iksP::targetP::T* const out, const typename brP::domainP::T* const in0,
    const typename brP::domainP::T* const in1, FFP* bk, const typename iksP::targetP::T* const ksk,
    const CuNTTHandler<> ntt)
{
    __HomGate__<brP, μ, iksP, -1, 1, -lvl0param::μ>(out, in0, in1, bk, ksk, ntt);
}

template<class brP = TFHEpp::lvl01param, typename brP::targetP::T μ = TFHEpp::lvl1param::μ, class iksP = TFHEpp::lvl10param>
__global__ __launch_bounds__(NUM_THREAD4HOMGATE) void __AndYNBootstrap__(
    typename iksP::targetP::T* const out, const typename brP::domainP::T* const in0,
    const typename brP::domainP::T* const in1, FFP* bk, const typename iksP::targetP::T* const ksk,
    const CuNTTHandler<> ntt)
{
    __HomGate__<brP, μ, iksP, 1, -1, -lvl0param::μ>(out, in0, in1, bk, ksk, ntt);
}

template<class brP = TFHEpp::lvl01param, typename brP::targetP::T μ = TFHEpp::lvl1param::μ, class iksP = TFHEpp::lvl10param>
__global__ __launch_bounds__(NUM_THREAD4HOMGATE) void __OrNYBootstrap__(
    typename iksP::targetP::T* const out, const typename brP::domainP::T* const in0,
    const typename brP::domainP::T* const in1, FFP* bk, const typename iksP::targetP::T* const ksk,
    const CuNTTHandler<> ntt)
{
    __HomGate__<brP, μ, iksP, -1, 1, lvl0param::μ>(out, in0, in1, bk, ksk, ntt);
}

template<class brP = TFHEpp::lvl01param, typename brP::targetP::T μ = TFHEpp::lvl1param::μ, class iksP = TFHEpp::lvl10param>
__global__ __launch_bounds__(NUM_THREAD4HOMGATE) void __OrYNBootstrap__(
    typename iksP::targetP::T* const out, const typename brP::domainP::T* const in0,
    const typename brP::domainP::T* const in1, FFP* bk, const typename iksP::targetP::T* const ksk,
    const CuNTTHandler<> ntt)
{
    __HomGate__<brP, μ, iksP, 1, -1, lvl0param::μ>(out, in0, in1, bk, ksk, ntt);
}

__global__ __launch_bounds__(NUM_THREAD4HOMGATE) void __CopyBootstrap__(
    TFHEpp::lvl0param::T* const out, const TFHEpp::lvl0param::T* const in)
{
    const uint32_t tid = ThisThreadRankInBlock();
    out[tid] = in[tid];
    __syncthreads();
    __threadfence();
}

__global__ __launch_bounds__(NUM_THREAD4HOMGATE) void __NotBootstrap__(
    TFHEpp::lvl0param::T* const out, const TFHEpp::lvl0param::T* const in)
{
    const uint32_t tid = ThisThreadRankInBlock();
    out[tid] = -in[tid];
    __syncthreads();
    __threadfence();
}

// Mux(inc,in1,in0) = inc?in1:in0 = inc&in1 + (!inc)&in0
__global__ __launch_bounds__(NUM_THREAD4HOMGATE) void __MuxBootstrap__(
    TFHEpp::lvl0param::T* const out, const TFHEpp::lvl0param::T* const inc,
    const TFHEpp::lvl0param::T* const in1, const TFHEpp::lvl0param::T* const in0, const FFP* const bk,
    const TFHEpp::lvl0param::T* const ksk,  const CuNTTHandler<> ntt)
{
    // To use over 48 KiB shared Memory, the dynamic allocation is required.
    extern __shared__ FFP sh[];
    FFP* sh_acc_ntt = &sh[0];
    // Use Last section to hold tlwe. This may to make these data in serial
    TFHEpp::lvl1param::T* tlwe1 =
        (TFHEpp::lvl1param::T*)&sh[((lvl1param::k+1) * lvl1param::l + 1) * lvl1param::n];
    TFHEpp::lvl1param::T* tlwe0 =
        (TFHEpp::lvl1param::T*)&sh[((lvl1param::k+1) * lvl1param::l + 2) * lvl1param::n];
    // test vector: acc.a = 0; acc.b = vec(mu) * x ^ (in.b()/2048)
    register uint32_t bar =
        2 * lvl1param::n -
        modSwitchFromTorus<lvl1param>(-lvl0param::μ + inc[lvl0param::n] +
                                      in1[lvl0param::n]);
    RotatedTestVector<lvl1param>(tlwe1, bar, lvl1param::μ);

    // accumulate
    for (int i = 0; i < lvl0param::n; i++) {  // lvl1param::n iterations
        bar = modSwitchFromTorus<lvl1param>(0 + inc[i] + in1[i]);
        Accumulate<lvl01param>(tlwe1, sh_acc_ntt, bar,
                   bk + (i << lvl1param::nbit) * (lvl1param::k+1) * (lvl1param::k+1) * lvl1param::l, ntt);
    }

    bar = 2 * lvl1param::n -
          modSwitchFromTorus<lvl1param>(-lvl0param::μ - inc[lvl0param::n] +
                                        in0[lvl0param::n]);

    RotatedTestVector<lvl1param>(tlwe0, bar, lvl1param::μ);

    for (int i = 0; i < lvl0param::n; i++) {  // lvl1param::n iterations
        bar = modSwitchFromTorus<lvl1param>(0 - inc[i] + in0[i]);
        Accumulate<lvl01param>(tlwe0, sh_acc_ntt, bar,
                   bk + (i << lvl1param::nbit) * (lvl1param::k+1) * (lvl1param::k+1) * lvl1param::l, ntt);
    }

    volatile const uint32_t tid = ThisThreadRankInBlock();
    volatile const uint32_t bdim = ThisBlockSize();
#pragma unroll
    for (int i = tid; i <= lvl1param::n; i += bdim) {
        tlwe1[i] += tlwe0[i];
        if (i == lvl1param::n) {
            tlwe1[lvl1param::n] += lvl1param::μ;
        }
    }

    __syncthreads();

    KeySwitch<lvl10param>(out, tlwe1, ksk);
    __threadfence();
}

// NMux(inc,in1,in0) = !(inc?in1:in0) = !(inc&in1 + (!inc)&in0)
__global__ __launch_bounds__(NUM_THREAD4HOMGATE) void __NMuxBootstrap__(
    TFHEpp::lvl0param::T* const out, const TFHEpp::lvl0param::T* const inc,
    const TFHEpp::lvl0param::T* const in1, const TFHEpp::lvl0param::T* const in0, const FFP* const bk,
    const TFHEpp::lvl0param::T* const ksk,  const CuNTTHandler<> ntt)
{
    // To use over 48 KiB shared Memory, the dynamic allocation is required.
    extern __shared__ FFP sh[];
    FFP* sh_acc_ntt = &sh[0];
    // Use Last section to hold tlwe. This may to make these data in serial
    TFHEpp::lvl1param::T* tlwe1 =
        (TFHEpp::lvl1param::T*)&sh[(2 * lvl1param::l + 1) * lvl1param::n];
    TFHEpp::lvl1param::T* tlwe0 =
        (TFHEpp::lvl1param::T*)&sh[(2 * lvl1param::l + 2) * lvl1param::n];

    // test vector: acc.a = 0; acc.b = vec(mu) * x ^ (in.b()/2048)
    register uint32_t bar =
        2 * lvl1param::n -
        modSwitchFromTorus<lvl1param>(-lvl0param::μ + inc[lvl0param::n] +
                                      in1[lvl0param::n]);
    RotatedTestVector<lvl1param>(tlwe1, bar, lvl1param::μ);

    // accumulate
    for (int i = 0; i < lvl0param::n; i++) {  // lvl1param::n iterations
        bar = modSwitchFromTorus<lvl1param>(0 + inc[i] + in1[i]);
        Accumulate<lvl01param>(tlwe1, sh_acc_ntt, bar,
                   bk + (i << lvl1param::nbit) * (lvl1param::k+1) * (lvl1param::k+1) * lvl1param::l, ntt);
    }

    bar = 2 * lvl1param::n -
          modSwitchFromTorus<lvl1param>(-lvl0param::μ - inc[lvl0param::n] +
                                        in0[lvl0param::n]);

    RotatedTestVector<lvl1param>(tlwe0, bar, lvl1param::μ);

    for (int i = 0; i < lvl0param::n; i++) {  // lvl1param::n iterations
        bar = modSwitchFromTorus<lvl1param>(0 - inc[i] + in0[i]);
        Accumulate<lvl01param>(tlwe0, sh_acc_ntt, bar,
                   bk + (i << lvl1param::nbit) * (lvl1param::k+1) * (lvl1param::k+1) * lvl1param::l, ntt);
    }

    volatile const uint32_t tid = ThisThreadRankInBlock();
    volatile const uint32_t bdim = ThisBlockSize();
#pragma unroll
    for (int i = tid; i <= lvl1param::n; i += bdim) {
        tlwe1[i] = -tlwe1[i] - tlwe0[i];
        if (i == lvl1param::n) {
            tlwe1[lvl1param::n] -= lvl1param::μ;
        }
    }

    __syncthreads();

    KeySwitch<lvl10param>(out, tlwe1, ksk);
    __threadfence();
}

void Bootstrap(TFHEpp::lvl0param::T* const out, const TFHEpp::lvl0param::T* const in,
               const lvl1param::T mu, const cudaStream_t st, const int gpuNum)
{
    __Bootstrap__<lvl01param,lvl10param><<<1, NUM_THREAD4HOMGATE, 0, st>>>(
        out, in, mu, bk_ntts[gpuNum], ksk_devs[gpuNum], *ntt_handlers[gpuNum]);
    CuCheckError();
}

void CMUXNTTkernel(TFHEpp::lvl1param::T* const res, const FFP* const cs,
                   TFHEpp::lvl1param::T* const c1,
                   TFHEpp::lvl1param::T* const c0, cudaStream_t st,
                   const int gpuNum)
{
    cudaFuncSetAttribute(__CMUXNTT__,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         (2 * lvl1param::l + 2) * lvl1param::n * sizeof(FFP));
    __CMUXNTT__<<<1, NUM_THREAD4HOMGATE,
                  ((lvl1param::k+1) * lvl1param::l + 2) * lvl1param::n * sizeof(FFP), st>>>(
        res, cs, c1, c0, *ntt_handlers[gpuNum]);
    CuCheckError();
}

void BootstrapTLWE2TRLWE(TFHEpp::lvl1param::T* const out, const TFHEpp::lvl0param::T* const in,
                         const lvl1param::T mu, const cudaStream_t st, const int gpuNum)
{
    cudaFuncSetAttribute(__BlindRotate__<TFHEpp::lvl01param>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         MEM4HOMGATE<TFHEpp::lvl1param>);
    __BlindRotateGlobal__<TFHEpp::lvl01param><<<1, NUM_THREAD4HOMGATE, MEM4HOMGATE<TFHEpp::lvl1param>, st>>>(
        out, in, mu, bk_ntts[gpuNum], *ntt_handlers[gpuNum]);
    CuCheckError();
}

void SEIandBootstrap2TRLWE(TFHEpp::lvl1param::T* const out, const TFHEpp::lvl1param::T* const in,
                           const lvl1param::T mu, const cudaStream_t st, const int gpuNum)
{
    cudaFuncSetAttribute(
        __SEIandBootstrap2TRLWE__, cudaFuncAttributeMaxDynamicSharedMemorySize,
        (((lvl1param::k+1) * lvl1param::l + 3) * lvl1param::n + (lvl0param::n + 1) / 2 + 1) *
            sizeof(FFP));
    __SEIandBootstrap2TRLWE__<<<1, lvl1param::l * lvl1param::n>>
                                  NTT_THRED_UNITBIT,
                              ((2 * lvl1param::l + 3) * lvl1param::n +
                               (lvl0param::n + 1) / 2 + 1) *
                                  sizeof(FFP),
                              st>>>
        (out, in, mu, bk_ntts[gpuNum], ksk_devs[gpuNum], *ntt_handlers[gpuNum]);
    CuCheckError();
}

// template<class brP, class iksP>
void NandBootstrap(TFHEpp::lvl0param::T* const out, const TFHEpp::lvl0param::T* const in0,
                   const TFHEpp::lvl0param::T* const in1, const cudaStream_t st, const int gpuNum)
{
    using brP = TFHEpp::lvl01param;
    using iksP = TFHEpp::lvl10param;
    cudaFuncSetAttribute(__NandBootstrap__<brP, brP::targetP::μ, iksP>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         MEM4HOMGATE<TFHEpp::lvl1param>);
    __NandBootstrap__<brP, brP::targetP::μ, iksP><<<1, NUM_THREAD4HOMGATE, MEM4HOMGATE<TFHEpp::lvl1param>, st>>>(
        out, in0, in1, bk_ntts[gpuNum], ksk_devs[gpuNum],
        *ntt_handlers[gpuNum]);
    CuCheckError();
}

void OrBootstrap(TFHEpp::lvl0param::T* const out, const TFHEpp::lvl0param::T* const in0,
                 const TFHEpp::lvl0param::T* const in1, const cudaStream_t st, const int gpuNum)
{
    using brP = TFHEpp::lvl01param;
    using iksP = TFHEpp::lvl10param;

    cudaFuncSetAttribute(__OrBootstrap__<brP, brP::targetP::μ, iksP>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         MEM4HOMGATE<TFHEpp::lvl1param>);
    __OrBootstrap__<brP, brP::targetP::μ, iksP><<<1, NUM_THREAD4HOMGATE, MEM4HOMGATE<TFHEpp::lvl1param>, st>>>(
        out, in0, in1, bk_ntts[gpuNum], ksk_devs[gpuNum],
        *ntt_handlers[gpuNum]);
    CuCheckError();
}

void OrYNBootstrap(TFHEpp::lvl0param::T* const out, const TFHEpp::lvl0param::T* const in0,
                   const TFHEpp::lvl0param::T* const in1, const cudaStream_t st, const int gpuNum)
{
    using brP = TFHEpp::lvl01param;
    using iksP = TFHEpp::lvl10param;
    cudaFuncSetAttribute(__OrYNBootstrap__<brP, brP::targetP::μ, iksP>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         MEM4HOMGATE<TFHEpp::lvl1param>);
    __OrYNBootstrap__<brP, brP::targetP::μ, iksP><<<1, NUM_THREAD4HOMGATE, MEM4HOMGATE<TFHEpp::lvl1param>, st>>>(
        out, in0, in1, bk_ntts[gpuNum], ksk_devs[gpuNum],
        *ntt_handlers[gpuNum]);
    CuCheckError();
}

void OrNYBootstrap(TFHEpp::lvl0param::T* const out, const TFHEpp::lvl0param::T* const in0,
                   const TFHEpp::lvl0param::T* const in1, const cudaStream_t st, const int gpuNum)
{
    using brP = TFHEpp::lvl01param;
    using iksP = TFHEpp::lvl10param;
    cudaFuncSetAttribute(__OrNYBootstrap__<brP, brP::targetP::μ, iksP>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         MEM4HOMGATE<TFHEpp::lvl1param>);
    __OrNYBootstrap__<brP, brP::targetP::μ, iksP><<<1, NUM_THREAD4HOMGATE, MEM4HOMGATE<TFHEpp::lvl1param>, st>>>(
        out, in0, in1, bk_ntts[gpuNum], ksk_devs[gpuNum],
        *ntt_handlers[gpuNum]);
    CuCheckError();
}

void AndBootstrap(TFHEpp::lvl0param::T* const out, const TFHEpp::lvl0param::T* const in0,
                  const TFHEpp::lvl0param::T* const in1, const cudaStream_t st, const int gpuNum)
{
    using brP = TFHEpp::lvl01param;
    using iksP = TFHEpp::lvl10param;
    cudaFuncSetAttribute(__AndBootstrap__<brP, brP::targetP::μ, iksP>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         MEM4HOMGATE<TFHEpp::lvl1param>);
    __AndBootstrap__<brP, brP::targetP::μ, iksP><<<1, NUM_THREAD4HOMGATE, MEM4HOMGATE<TFHEpp::lvl1param>, st>>>(
        out, in0, in1, bk_ntts[gpuNum], ksk_devs[gpuNum],
        *ntt_handlers[gpuNum]);
    CuCheckError();
}

void AndYNBootstrap(TFHEpp::lvl0param::T* const out, const TFHEpp::lvl0param::T* const in0,
                   const TFHEpp::lvl0param::T* const in1, const cudaStream_t st, const int gpuNum)
{
    using brP = TFHEpp::lvl01param;
    using iksP = TFHEpp::lvl10param;
    cudaFuncSetAttribute(__AndYNBootstrap__<brP, brP::targetP::μ, iksP>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         MEM4HOMGATE<TFHEpp::lvl1param>);
    __AndYNBootstrap__<brP, brP::targetP::μ, iksP><<<1, NUM_THREAD4HOMGATE, MEM4HOMGATE<TFHEpp::lvl1param>, st>>>(
        out, in0, in1, bk_ntts[gpuNum], ksk_devs[gpuNum],
        *ntt_handlers[gpuNum]);
    CuCheckError();
}

void AndNYBootstrap(TFHEpp::lvl0param::T* const out, const TFHEpp::lvl0param::T* const in0,
                   const TFHEpp::lvl0param::T* const in1, const cudaStream_t st, const int gpuNum)
{
    using brP = TFHEpp::lvl01param;
    using iksP = TFHEpp::lvl10param;
    cudaFuncSetAttribute(__AndNYBootstrap__<brP, brP::targetP::μ, iksP>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         MEM4HOMGATE<TFHEpp::lvl1param>);
    __AndNYBootstrap__<brP, brP::targetP::μ, iksP><<<1, NUM_THREAD4HOMGATE, MEM4HOMGATE<TFHEpp::lvl1param>, st>>>(
        out, in0, in1, bk_ntts[gpuNum], ksk_devs[gpuNum],
        *ntt_handlers[gpuNum]);
    CuCheckError();
}

void NorBootstrap(TFHEpp::lvl0param::T* const out, const TFHEpp::lvl0param::T* const in0,
                  const TFHEpp::lvl0param::T* const in1, const cudaStream_t st, const int gpuNum)
{
    using brP = TFHEpp::lvl01param;
    using iksP = TFHEpp::lvl10param;
    cudaFuncSetAttribute(__NorBootstrap__<brP, brP::targetP::μ, iksP>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         MEM4HOMGATE<TFHEpp::lvl1param>);
    __NorBootstrap__<brP, brP::targetP::μ, iksP><<<1, NUM_THREAD4HOMGATE, MEM4HOMGATE<TFHEpp::lvl1param>, st>>>(
        out, in0, in1, bk_ntts[gpuNum], ksk_devs[gpuNum],
        *ntt_handlers[gpuNum]);
    CuCheckError();
}

void XorBootstrap(TFHEpp::lvl0param::T* const out, const TFHEpp::lvl0param::T* const in0,
                  const TFHEpp::lvl0param::T* const in1, const cudaStream_t st, const int gpuNum)
{
    using brP = TFHEpp::lvl01param;
    using iksP = TFHEpp::lvl10param;
    cudaFuncSetAttribute(__XorBootstrap__<brP, brP::targetP::μ, iksP>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         MEM4HOMGATE<TFHEpp::lvl1param>);
    __XorBootstrap__<brP, brP::targetP::μ, iksP><<<1, NUM_THREAD4HOMGATE, MEM4HOMGATE<TFHEpp::lvl1param>, st>>>(
        out, in0, in1, bk_ntts[gpuNum], ksk_devs[gpuNum],
        *ntt_handlers[gpuNum]);
    CuCheckError();
}

void XnorBootstrap(TFHEpp::lvl0param::T* const out, const TFHEpp::lvl0param::T* const in0,
                   const TFHEpp::lvl0param::T* const in1, const cudaStream_t st, const int gpuNum)
{
    using brP = TFHEpp::lvl01param;
    using iksP = TFHEpp::lvl10param;
    cudaFuncSetAttribute(__XnorBootstrap__<brP, brP::targetP::μ, iksP>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         MEM4HOMGATE<TFHEpp::lvl1param>);
    __XnorBootstrap__<brP, brP::targetP::μ, iksP><<<1, NUM_THREAD4HOMGATE, MEM4HOMGATE<TFHEpp::lvl1param>, st>>>(
        out, in0, in1, bk_ntts[gpuNum], ksk_devs[gpuNum],
        *ntt_handlers[gpuNum]);
    CuCheckError();
}

void CopyBootstrap(TFHEpp::lvl0param::T* const out, const TFHEpp::lvl0param::T* const in,
                   const cudaStream_t st, const int gpuNum)
{
    __CopyBootstrap__<<<1, lvl0param::n + 1, 0, st>>>(out, in);
    CuCheckError();
}

void NotBootstrap(TFHEpp::lvl0param::T* const out, const TFHEpp::lvl0param::T* const in,
                  const cudaStream_t st, const int gpuNum)
{
    __NotBootstrap__<<<1, lvl0param::n + 1, 0, st>>>(out, in);
    CuCheckError();
}

void MuxBootstrap(TFHEpp::lvl0param::T* const out, const TFHEpp::lvl0param::T* const inc,
                  const TFHEpp::lvl0param::T* const in1, const TFHEpp::lvl0param::T* const in0,
                  const cudaStream_t st, const int gpuNum)
{
    cudaFuncSetAttribute(__MuxBootstrap__,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         ((lvl1param::k+1) * lvl1param::l + 3) * lvl1param::n * sizeof(FFP));
    __MuxBootstrap__<<<1, NUM_THREAD4HOMGATE,
                       ((lvl1param::k+1) * lvl1param::l + 3) * lvl1param::n * sizeof(FFP),
                       st>>>(out, inc, in1, in0, bk_ntts[gpuNum],
                             ksk_devs[gpuNum], *ntt_handlers[gpuNum]);
    CuCheckError();
}

void NMuxBootstrap(TFHEpp::lvl0param::T* const out, const TFHEpp::lvl0param::T* const inc,
                  const TFHEpp::lvl0param::T* const in1, const TFHEpp::lvl0param::T* const in0,
                  const cudaStream_t st, const int gpuNum)
{
    cudaFuncSetAttribute(__NMuxBootstrap__,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         ((lvl1param::k+1) * lvl1param::l + 3) * lvl1param::n * sizeof(FFP));
    __NMuxBootstrap__<<<1, NUM_THREAD4HOMGATE,
                        ((lvl1param::k+1) * lvl1param::l + 3) * lvl1param::n * sizeof(FFP),
                        st>>>(out, inc, in1, in0, bk_ntts[gpuNum],
                              ksk_devs[gpuNum], *ntt_handlers[gpuNum]);
    CuCheckError();
}
}  // namespace cufhe
