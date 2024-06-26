#pragma once


#include <include/cufhe_gpu.cuh>
#include <include/details/error_gpu.cuh>
#include <include/details/utils_gpu.cuh>

namespace cufhe{

extern std::vector<TFHEpp::lvl0param::T*> ksk_devs;

template <class P>
__device__ inline void KeySwitch(typename P::targetP::T* const lwe,
                                 const typename P::domainP::T* const tlwe,
                                 const typename P::targetP::T* const ksk)
{
    constexpr typename P::domainP::T decomp_mask = (1U << P::basebit) - 1;
    constexpr typename P::domainP::T decomp_offset =
        1U << (std::numeric_limits<typename P::domainP::T>::digits - 1 -
               P::t * P::basebit);
    const uint32_t tid = ThisThreadRankInBlock();
    const uint32_t bdim = ThisBlockSize();
    for (int i = tid; i <= P::targetP::k*P::targetP::n; i += bdim) {
        typename P::targetP::T res = 0;
        if (i == P::targetP::k*P::targetP::n){
            constexpr uint domain_digit =
                std::numeric_limits<typename P::domainP::T>::digits;
            constexpr uint target_digit =
                std::numeric_limits<typename P::targetP::T>::digits;
            if constexpr (domain_digit == target_digit)
                res = tlwe[P::domainP::k * P::domainP::n];
            else if constexpr (domain_digit > target_digit)
                res = (tlwe[P::domainP::k * P::domainP::n] + (1ULL << (domain_digit - target_digit - 1))) >> (domain_digit - target_digit);
            else if constexpr (domain_digit < target_digit)
                res = static_cast<typename P::targetP::T>(tlwe[P::domainP::k * P::domainP::n]) << (target_digit - domain_digit);
        }
        for (int j = 0; j < P::domainP::k*P::domainP::n; j++) {
            typename P::domainP::T tmp;
            if (j == 0)
                tmp = tlwe[0];
            else
                tmp = -tlwe[P::domainP::k*P::domainP::n - j];
            tmp += decomp_offset;
            for (int k = 0; k < P::t; k++) {
                typename P::domainP::T val =
                    (tmp >>
                     (std::numeric_limits<typename P::domainP::T>::digits -
                      (k + 1) * P::basebit)) &
                    decomp_mask;
                if (val != 0) {
                    constexpr int numbase = (1 << P::basebit) - 1;
                    res -= ksk[j * (P::t * numbase *
                                    (P::targetP::k*P::targetP::n + 1)) +
                               k * (numbase * (P::targetP::k*P::targetP::n + 1)) +
                               (val - 1) * (P::targetP::k*P::targetP::n + 1) + i];
                }
            }
        }
        lwe[i] = res;
    }
}

template <class P, int casign, int cbsign, typename P::domainP::T offset>
__device__ inline void IdentityKeySwitchPreAdd(typename P::targetP::T* const lwe,
                                 const typename P::domainP::T* const ina,
                                 const typename P::domainP::T* const inb,
                                 const typename P::targetP::T* const ksk)
{
    constexpr typename P::domainP::T decomp_mask = (1U << P::basebit) - 1;
    constexpr typename P::domainP::T decomp_offset =
        1U << (std::numeric_limits<typename P::domainP::T>::digits - 1 -
               P::t * P::basebit);
    const uint32_t tid = ThisThreadRankInBlock();
    const uint32_t bdim = ThisBlockSize();
    for (int i = tid; i <= P::targetP::k*P::targetP::n; i += bdim) {
        typename P::targetP::T res = 0;
        if (i == P::targetP::k*P::targetP::n){
            constexpr uint domain_digit =
                std::numeric_limits<typename P::domainP::T>::digits;
            constexpr uint target_digit =
                std::numeric_limits<typename P::targetP::T>::digits;
            const typename P::domainP::T added = casign*ina[P::domainP::k*P::domainP::n]+ cbsign*inb[P::domainP::k*P::domainP::n] + offset;
            if constexpr (domain_digit == target_digit)
                res = added;
            else if constexpr (domain_digit > target_digit)
                res = (added + (1ULL << (domain_digit - target_digit - 1))) >> (domain_digit - target_digit);
            else if constexpr (domain_digit < target_digit)
                res = static_cast<typename P::targetP::T>(added) << (target_digit - domain_digit);
        }
        for (int j = 0; j < P::domainP::k*P::domainP::n; j++) {
            typename P::domainP::T tmp;
            tmp = casign*ina[j]+ cbsign*inb[j] + 0 + decomp_offset;
            for (int k = 0; k < P::t; k++) {
                typename P::domainP::T val =
                    (tmp >>
                     (std::numeric_limits<typename P::domainP::T>::digits -
                      (k + 1) * P::basebit)) &
                    decomp_mask;
                if (val != 0) {
                    constexpr int numbase = (1 << P::basebit) - 1;
                    res -= ksk[j * (P::t * numbase *
                                    (P::targetP::k*P::targetP::n + 1)) +
                               k * (numbase * (P::targetP::k*P::targetP::n + 1)) +
                               (val - 1) * (P::targetP::k*P::targetP::n + 1) + i];
                }
            }
        }
        lwe[i] = res;
    }
}

void KeySwitchingKeyToDevice(const TFHEpp::KeySwitchingKey<TFHEpp::lvl10param>& ksk,
                             const int gpuNum);

void DeleteKeySwitchingKey(const int gpuNum);

void SEIandKS(TFHEpp::lvl0param::T* const out, const TFHEpp::lvl1param::T* const in,
             const cudaStream_t& st, const int gpuNum);
}