#include <include/keyswitch_gpu.cuh>
namespace cufhe{

vector<lvl0param::T*> ksk_devs;

void KeySwitchingKeyToDevice(const KeySwitchingKey<lvl10param>& ksk,
                             const int gpuNum)
{
    ksk_devs.resize(gpuNum);
    for (int i = 0; i < gpuNum; i++) {
        cudaSetDevice(i);
        cudaMalloc((void**)&ksk_devs[i], sizeof(ksk));
        CuSafeCall(cudaMemcpy(ksk_devs[i], ksk.data(), sizeof(ksk),
                              cudaMemcpyHostToDevice));
    }
}

void DeleteKeySwitchingKey(const int gpuNum)
{
    for (int i = 0; i < ksk_devs.size(); i++) {
        cudaSetDevice(i);
        cudaFree(ksk_devs[i]);
    }
}

template<class P>
__global__ void __SEIandKS__(typename P::targetP::T* const out, const typename P::domainP::T* const in,
                            const typename P::targetP::T* const ksk)
{
    KeySwitch<lvl10param>(out, in, ksk);
    __threadfence();
}

void SEIandKS(TFHEpp::lvl0param::T* const out, const TFHEpp::lvl1param::T* const in,
             const cudaStream_t& st, const int gpuNum)
{
    __SEIandKS__<TFHEpp::lvl10param><<<1, lvl0param::n + 1, 0, st>>>(out, in,
                                                ksk_devs[gpuNum]);
    CuCheckError();
}
}