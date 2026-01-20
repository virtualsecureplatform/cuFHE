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
 * @file cufhe.h
 * @brief This is the user API of the cuFHE library.
 *        It hides most of the contents in the developer API and
 *        only provides essential data structures and functions.
 */

#pragma once
#include <cuda.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>

// Use optimized HEonGPU-style NTT implementation
#include "ntt_gpu/ntt_gpuntt.cuh"
#include "bootstrap_gpu.cuh"

#include <array>
#include <cloudkey.hpp>

namespace cufhe {

extern int _gpuNum;

extern int streamCount;

/**
 * Call before running gates on server.
 * 1. Generate necessary NTT data.
 * 2. Convert BootstrappingKey to NTT form.
 * 3. Copy KeySwitchingKey to GPU memory.
 */
void SetGPUNum(int gpuNum);

// Initialize NTThandlers only.
void Initialize();

void Initialize(const TFHEpp::EvalKey& ek);

/** Remove everything created in Initialize(). */
void CleanUp();

/**
 * \brief Synchronize device.
 * \details This makes it easy to wrap in python.
 */
inline void Synchronize()
{
    for (int i = 0; i < _gpuNum; i++) {
        cudaSetDevice(i);
        cudaDeviceSynchronize();
    }
};

template <typename dT, typename hT>
void ctxtInitialize(hT &host, std::vector<dT *> &devices)
{
    cudaHostRegister(host.data(), sizeof(host), cudaHostRegisterDefault);
    devices.resize(_gpuNum);
    for (int i = 0; i < _gpuNum; i++) {
        cudaSetDevice(i);
        cudaMalloc((void **)&devices[i], sizeof(host));
    }
}

template <typename dT, typename hT>
void ctxtDelete(hT &host, std::vector<dT *> &devices)
{
    cudaHostUnregister(host.data());
    for (int i = 0; i < _gpuNum; i++) {
        cudaSetDevice(i);
        cudaFree(devices[i]);
    }
}

/*****************************
 * Essential Data Structures *
 *****************************/

/** Ciphertext. */
template<class P>
struct Ctxt {
    Ctxt()
    {
        ctxtInitialize<typename P::T, TFHEpp::TLWE<P>>(
            tlwehost, tlwedevices);
    }

    ~Ctxt()
    {
        ctxtDelete<typename P::T, TFHEpp::TLWE<P>>(
            tlwehost, tlwedevices);
    }
    Ctxt(const Ctxt& that) = delete;
    Ctxt& operator=(const Ctxt& that) = delete;

    alignas(64) TFHEpp::TLWE<P> tlwehost;

    std::vector<typename P::T*> tlwedevices;
};

/** TRLWE holder */
struct cuFHETRLWElvl1 {
    TFHEpp::TRLWE<TFHEpp::lvl1param> trlwehost;
    std::vector<TFHEpp::lvl1param::T*> trlwedevices;
    cuFHETRLWElvl1();
    ~cuFHETRLWElvl1();

   private:
    // Don't allow users to copy this struct.
    cuFHETRLWElvl1(const cuFHETRLWElvl1&);
    cuFHETRLWElvl1& operator=(const cuFHETRLWElvl1&);
};

struct cuFHETRGSWNTTlvl1 {
    TFHEpp::TRGSWNTT<TFHEpp::lvl1param> trgswhost;
    std::vector<FFP*> trgswdevices;
    cuFHETRGSWNTTlvl1();
    ~cuFHETRGSWNTTlvl1();

   private:
    // Don't allow users to copy this struct.
    cuFHETRGSWNTTlvl1(const cuFHETRGSWNTTlvl1&);
    cuFHETRGSWNTTlvl1& operator=(const cuFHETRGSWNTTlvl1&);
};

/**
 * \class Stream
 * \brief This is created for easier wrapping in python.
 */
class Stream {
   public:
    inline Stream()
    {
        st_ = 0;
        _device_id = streamCount % _gpuNum;
        streamCount++;
    }
    inline Stream(int device_id)
    {
        _device_id = device_id;
        st_ = 0;
        streamCount++;
    }

    inline ~Stream()
    {
        // Destroy();
    }

    inline void Create()
    {
        cudaSetDevice(_device_id);
        cudaStreamCreateWithFlags(&this->st_, cudaStreamNonBlocking);
    }

    inline void Destroy()
    {
        cudaSetDevice(_device_id);
        cudaStreamDestroy(this->st_);
    }
    inline cudaStream_t st() { return st_; };
    inline int device_id() { return _device_id; }

   private:
    cudaStream_t st_;
    int _device_id;
};  // class Stream

bool StreamQuery(Stream st);

template<class P>
inline void CtxtCopyH2D(Ctxt<P>& c, Stream st)
{
    cudaSetDevice(st.device_id());
    cudaMemcpyAsync(c.tlwedevices[st.device_id()], c.tlwehost.data(),
                    sizeof(c.tlwehost), cudaMemcpyHostToDevice, st.st());
}

template<class P>
inline void CtxtCopyD2H(Ctxt<P>& c, Stream st)
{
    cudaSetDevice(st.device_id());
    cudaMemcpyAsync(c.tlwehost.data(), c.tlwedevices[st.device_id()],
                    sizeof(c.tlwehost), cudaMemcpyDeviceToHost, st.st());
}

void TRGSW2NTT(cuFHETRGSWNTTlvl1& trgswntt,
               const TFHEpp::TRGSW<TFHEpp::lvl1param>& trgsw, Stream& st);
void GateBootstrappingTLWE2TRLWElvl01NTT(cuFHETRLWElvl1& out, Ctxt<TFHEpp::lvl0param>& in,
                                         Stream st);
void Refresh(cuFHETRLWElvl1& out, cuFHETRLWElvl1& in, Stream st);
void SampleExtractAndKeySwitch(Ctxt<TFHEpp::lvl0param>& out, const cuFHETRLWElvl1& in, Stream st);
void CMUXNTT(cuFHETRLWElvl1& res, cuFHETRGSWNTTlvl1& cs, cuFHETRLWElvl1& c1,
             cuFHETRLWElvl1& c0, Stream st);

template<class P>
void Not(Ctxt<P>& out, Ctxt<P>& in, Stream st)
{
    cudaSetDevice(st.device_id());
    CtxtCopyH2D<P>(in, st);
    NotBootstrap<P>(out.tlwedevices[st.device_id()],
                 in.tlwedevices[st.device_id()], st.st(), st.device_id());
    CtxtCopyD2H<P>(out, st);
}

template<class P>
void gNot(Ctxt<P>& out, Ctxt<P>& in, Stream st)
{
    cudaSetDevice(st.device_id());
    NotBootstrap<P>(out.tlwedevices[st.device_id()],
                 in.tlwedevices[st.device_id()], st.st(), st.device_id());
}

template<class P>
void Copy(Ctxt<P>& out, Ctxt<P>& in, Stream st)
{
    cudaSetDevice(st.device_id());
    CtxtCopyH2D<P>(in, st);
    CopyBootstrap<P>(out.tlwedevices[st.device_id()],
                  in.tlwedevices[st.device_id()], st.st(), st.device_id());
    CtxtCopyD2H<P>(out, st);
}

template<class P>
void gCopy(Ctxt<P>& out, Ctxt<P>& in, Stream st)
{
    cudaSetDevice(st.device_id());
    CopyBootstrap<P>(out.tlwedevices[st.device_id()],
                  in.tlwedevices[st.device_id()], st.st(), st.device_id());
}

template<class P>
void CopyOnHost(Ctxt<P>& out, Ctxt<P>& in) { out.tlwehost = in.tlwehost; }

template<class P>
void And(Ctxt<P>& out, Ctxt<P>& in0, Ctxt<P>& in1, Stream st);
template<class P>
void AndYN(Ctxt<P>& out, Ctxt<P>& in0, Ctxt<P>& in1, Stream st);
template<class P>
void AndNY(Ctxt<P>& out, Ctxt<P>& in0, Ctxt<P>& in1, Stream st);
template<class P>
void Or(Ctxt<P>& out, Ctxt<P>& in0, Ctxt<P>& in1, Stream st);
template<class P>
void OrYN(Ctxt<P>& out, Ctxt<P>& in0, Ctxt<P>& in1, Stream st);
template<class P>
void OrNY(Ctxt<P>& out, Ctxt<P>& in0, Ctxt<P>& in1, Stream st);
template<class P>
void Nand(Ctxt<P>& out, Ctxt<P>& in0, Ctxt<P>& in1, Stream st);
template<class P>
void Nor(Ctxt<P>& out, Ctxt<P>& in0, Ctxt<P>& in1, Stream st);
template<class P>
void Xor(Ctxt<P>& out, Ctxt<P>& in0, Ctxt<P>& in1, Stream st);
template<class P>
void Xnor(Ctxt<P>& out, Ctxt<P>& in0, Ctxt<P>& in1, Stream st);
template<class P>
void Mux(Ctxt<P>& out, Ctxt<P>& inc, Ctxt<P>& in1, Ctxt<P>& in0, Stream st);
template<class P>
void NMux(Ctxt<P>& out, Ctxt<P>& inc, Ctxt<P>& in1, Ctxt<P>& in0, Stream st);

void gSampleExtractAndKeySwitch(Ctxt<TFHEpp::lvl0param>& out, const cuFHETRLWElvl1& in, Stream st);
void gGateBootstrappingTLWE2TRLWElvl01NTT(cuFHETRLWElvl1& out, Ctxt<TFHEpp::lvl0param>& in,
                                          Stream st);
void gRefresh(cuFHETRLWElvl1& out, cuFHETRLWElvl1& in, Stream st);
template<class P>
void gNand(Ctxt<P>& out, Ctxt<P>& in0, Ctxt<P>& in1, Stream st);
template<class P>
void gOr(Ctxt<P>& out, Ctxt<P>& in0, Ctxt<P>& in1, Stream st);
template<class P>
void gOrYN(Ctxt<P>& out, Ctxt<P>& in0, Ctxt<P>& in1, Stream st);
template<class P>
void gOrNY(Ctxt<P>& out, Ctxt<P>& in0, Ctxt<P>& in1, Stream st);
template<class P>
void gAnd(Ctxt<P>& out, Ctxt<P>& in0, Ctxt<P>& in1, Stream st);
template<class P>
void gAndYN(Ctxt<P>& out, Ctxt<P>& in0, Ctxt<P>& in1, Stream st);
template<class P>
void gAndNY(Ctxt<P>& out, Ctxt<P>& in0, Ctxt<P>& in1, Stream st);
template<class P>
void gNor(Ctxt<P>& out, Ctxt<P>& in0, Ctxt<P>& in1, Stream st);
template<class P>
void gXor(Ctxt<P>& out, Ctxt<P>& in0, Ctxt<P>& in1, Stream st);
template<class P>
void gXnor(Ctxt<P>& out, Ctxt<P>& in0, Ctxt<P>& in1, Stream st);
template<class P>
void gNot(Ctxt<P>& out, Ctxt<P>& in, Stream st);
template<class P>
void gMux(Ctxt<P>& out, Ctxt<P>& inc, Ctxt<P>& in1, Ctxt<P>& in0, Stream st);
template<class P>
void gNMux(Ctxt<P>& out, Ctxt<P>& inc, Ctxt<P>& in1, Ctxt<P>& in0, Stream st);
template<class P>
void gCopy(Ctxt<P>& out, Ctxt<P>& in, Stream st);

}  // namespace cufhe
