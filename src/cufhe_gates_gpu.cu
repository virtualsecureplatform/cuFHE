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

#include <unistd.h>

#include <array>
#include <cloudkey.hpp>
#include <include/bootstrap_gpu.cuh>
#include <include/keyswitch_gpu.cuh>
#include <include/cufhe_gpu.cuh>
#include <params.hpp>

namespace cufhe {

int _gpuNum = 1;

int streamCount = 0;

void SetGPUNum(int gpuNum) { _gpuNum = gpuNum; }

void Initialize() { InitializeNTThandlers(_gpuNum); }

void Initialize(const TFHEpp::EvalKey& ek)
{
    InitializeNTThandlers(_gpuNum);
    BootstrappingKeyToNTT<TFHEpp::lvl01param>(*ek.bklvl01, _gpuNum);
    KeySwitchingKeyToDevice(*ek.iksklvl10, _gpuNum);
}

void CleanUp()
{
    DeleteBootstrappingKeyNTT(_gpuNum);
    DeleteKeySwitchingKey(_gpuNum);
}

bool StreamQuery(Stream st)
{
    cudaSetDevice(st.device_id());
    cudaError_t res = cudaStreamQuery(st.st());
    if (res == cudaSuccess) {
        return true;
    }
    else {
        return false;
    }
}

void CMUXNTT(cuFHETRLWElvl1& res, cuFHETRGSWNTTlvl1& cs, cuFHETRLWElvl1& c1,
             cuFHETRLWElvl1& c0, Stream st)
{
    cudaSetDevice(st.device_id());
    cudaMemcpyAsync(cs.trgswdevices[st.device_id()], cs.trgswhost.data(),
                    sizeof(cs.trgswhost), cudaMemcpyHostToDevice, st.st());
    cudaMemcpyAsync(c1.trlwedevices[st.device_id()], c1.trlwehost.data(),
                    sizeof(c1.trlwehost), cudaMemcpyHostToDevice, st.st());
    cudaMemcpyAsync(c0.trlwedevices[st.device_id()], c0.trlwehost.data(),
                    sizeof(c0.trlwehost), cudaMemcpyHostToDevice, st.st());
    CMUXNTTkernel(res.trlwedevices[st.device_id()],
                  cs.trgswdevices[st.device_id()],
                  c1.trlwedevices[st.device_id()],
                  c0.trlwedevices[st.device_id()], st.st(), st.device_id());
    cudaMemcpyAsync(res.trlwehost.data(), res.trlwedevices[st.device_id()],
                    sizeof(res.trlwehost), cudaMemcpyDeviceToHost, st.st());
}

void GateBootstrappingTLWE2TRLWElvl01NTT(cuFHETRLWElvl1& out, Ctxt<TFHEpp::lvl0param>& in,
                                         Stream st)
{
    cudaSetDevice(st.device_id());
    CtxtCopyH2D<TFHEpp::lvl0param>(in, st);
    BootstrapTLWE2TRLWE(out.trlwedevices[st.device_id()],
                        in.tlwedevices[st.device_id()], TFHEpp::lvl1param::μ,
                        st.st(), st.device_id());
    cudaMemcpyAsync(out.trlwehost.data(), out.trlwedevices[st.device_id()],
                    sizeof(out.trlwehost), cudaMemcpyDeviceToHost, st.st());
}

void gGateBootstrappingTLWE2TRLWElvl01NTT(cuFHETRLWElvl1& out, Ctxt<TFHEpp::lvl0param>& in,
                                          Stream st)
{
    cudaSetDevice(st.device_id());
    BootstrapTLWE2TRLWE(out.trlwedevices[st.device_id()],
                        in.tlwedevices[st.device_id()], TFHEpp::lvl1param::μ,
                        st.st(), st.device_id());
}

void Refresh(cuFHETRLWElvl1& out, cuFHETRLWElvl1& in, Stream st)
{
    cudaSetDevice(st.device_id());
    cudaMemcpyAsync(in.trlwedevices[st.device_id()], in.trlwehost.data(),
                    sizeof(in.trlwehost), cudaMemcpyHostToDevice, st.st());
    SEIandBootstrap2TRLWE(out.trlwedevices[st.device_id()],
                          in.trlwedevices[st.device_id()], TFHEpp::lvl1param::μ,
                          st.st(), st.device_id());
    cudaMemcpyAsync(out.trlwehost.data(), out.trlwedevices[st.device_id()],
                    sizeof(out.trlwehost), cudaMemcpyDeviceToHost, st.st());
}

void gRefresh(cuFHETRLWElvl1& out, cuFHETRLWElvl1& in, Stream st)
{
    cudaSetDevice(st.device_id());
    BootstrapTLWE2TRLWE(out.trlwedevices[st.device_id()],
                        in.trlwedevices[st.device_id()], TFHEpp::lvl1param::μ,
                        st.st(), st.device_id());
}

void SampleExtractAndKeySwitch(Ctxt<TFHEpp::lvl0param>& out, const cuFHETRLWElvl1& in, Stream st)
{
    cudaSetDevice(st.device_id());
    cudaMemcpyAsync(in.trlwedevices[st.device_id()], in.trlwehost.data(),
                    sizeof(in.trlwehost), cudaMemcpyHostToDevice, st.st());
    SEIandKS(out.tlwedevices[st.device_id()], in.trlwedevices[st.device_id()],
            st.st(), st.device_id());
    CtxtCopyD2H<TFHEpp::lvl0param>(out, st);
}

void gSampleExtractAndKeySwitch(Ctxt<TFHEpp::lvl0param>& out, const cuFHETRLWElvl1& in, Stream st)
{
    cudaSetDevice(st.device_id());
    cudaMemcpyAsync(in.trlwedevices[st.device_id()], in.trlwehost.data(),
                    sizeof(in.trlwehost), cudaMemcpyHostToDevice, st.st());
    SEIandKS(out.tlwedevices[st.device_id()], in.trlwedevices[st.device_id()],
            st.st(), st.device_id());
}

template<>
void Nand<TFHEpp::lvl0param>(Ctxt<TFHEpp::lvl0param>& out, Ctxt<TFHEpp::lvl0param>& in0, Ctxt<TFHEpp::lvl0param>& in1, Stream st)
{
    cudaSetDevice(st.device_id());
    CtxtCopyH2D<TFHEpp::lvl0param>(in0, st);
    CtxtCopyH2D<TFHEpp::lvl0param>(in1, st);
    NandBootstrap<TFHEpp::lvl01param, TFHEpp::lvl1param::μ, TFHEpp::lvl10param>(out.tlwedevices[st.device_id()],
                  in0.tlwedevices[st.device_id()],
                  in1.tlwedevices[st.device_id()], st.st(), st.device_id());
    CtxtCopyD2H<TFHEpp::lvl0param>(out, st);
}

template<>
void gNand<TFHEpp::lvl0param>(Ctxt<TFHEpp::lvl0param>& out, Ctxt<TFHEpp::lvl0param>& in0, Ctxt<TFHEpp::lvl0param>& in1, Stream st)
{
    cudaSetDevice(st.device_id());
    NandBootstrap<TFHEpp::lvl01param, TFHEpp::lvl1param::μ, TFHEpp::lvl10param>(out.tlwedevices[st.device_id()],
                  in0.tlwedevices[st.device_id()],
                  in1.tlwedevices[st.device_id()], st.st(), st.device_id());
}

template<>
void Or<TFHEpp::lvl0param>(Ctxt<TFHEpp::lvl0param>& out, Ctxt<TFHEpp::lvl0param>& in0, Ctxt<TFHEpp::lvl0param>& in1, Stream st)
{
    cudaSetDevice(st.device_id());
    CtxtCopyH2D<TFHEpp::lvl0param>(in0, st);
    CtxtCopyH2D<TFHEpp::lvl0param>(in1, st);
    OrBootstrap<TFHEpp::lvl01param, TFHEpp::lvl1param::μ, TFHEpp::lvl10param>(out.tlwedevices[st.device_id()],
                in0.tlwedevices[st.device_id()],
                in1.tlwedevices[st.device_id()], st.st(), st.device_id());
    CtxtCopyD2H<TFHEpp::lvl0param>(out, st);
}

template<>
void gOr<TFHEpp::lvl0param>(Ctxt<TFHEpp::lvl0param>& out, Ctxt<TFHEpp::lvl0param>& in0, Ctxt<TFHEpp::lvl0param>& in1, Stream st)
{
    cudaSetDevice(st.device_id());
    OrBootstrap<TFHEpp::lvl01param, TFHEpp::lvl1param::μ, TFHEpp::lvl10param>(out.tlwedevices[st.device_id()],
                in0.tlwedevices[st.device_id()],
                in1.tlwedevices[st.device_id()], st.st(), st.device_id());
}

template<>
void OrYN<TFHEpp::lvl0param>(Ctxt<TFHEpp::lvl0param>& out, Ctxt<TFHEpp::lvl0param>& in0, Ctxt<TFHEpp::lvl0param>& in1, Stream st)
{
    cudaSetDevice(st.device_id());
    CtxtCopyH2D<TFHEpp::lvl0param>(in0, st);
    CtxtCopyH2D<TFHEpp::lvl0param>(in1, st);
    OrYNBootstrap<TFHEpp::lvl01param, TFHEpp::lvl1param::μ, TFHEpp::lvl10param>(out.tlwedevices[st.device_id()],
                  in0.tlwedevices[st.device_id()],
                  in1.tlwedevices[st.device_id()], st.st(), st.device_id());
    CtxtCopyD2H<TFHEpp::lvl0param>(out, st);
}

template<>
void gOrYN<TFHEpp::lvl0param>(Ctxt<TFHEpp::lvl0param>& out, Ctxt<TFHEpp::lvl0param>& in0, Ctxt<TFHEpp::lvl0param>& in1, Stream st)
{
    cudaSetDevice(st.device_id());
    OrYNBootstrap<TFHEpp::lvl01param, TFHEpp::lvl1param::μ, TFHEpp::lvl10param>(out.tlwedevices[st.device_id()],
                  in0.tlwedevices[st.device_id()],
                  in1.tlwedevices[st.device_id()], st.st(), st.device_id());
}

template<>
void OrNY<TFHEpp::lvl0param>(Ctxt<TFHEpp::lvl0param>& out, Ctxt<TFHEpp::lvl0param>& in0, Ctxt<TFHEpp::lvl0param>& in1, Stream st)
{
    cudaSetDevice(st.device_id());
    CtxtCopyH2D<TFHEpp::lvl0param>(in0, st);
    CtxtCopyH2D<TFHEpp::lvl0param>(in1, st);
    OrNYBootstrap<TFHEpp::lvl01param, TFHEpp::lvl1param::μ, TFHEpp::lvl10param>(out.tlwedevices[st.device_id()],
                  in0.tlwedevices[st.device_id()],
                  in1.tlwedevices[st.device_id()], st.st(), st.device_id());
    CtxtCopyD2H<TFHEpp::lvl0param>(out, st);
}

template<>
void gOrNY<TFHEpp::lvl0param>(Ctxt<TFHEpp::lvl0param>& out, Ctxt<TFHEpp::lvl0param>& in0, Ctxt<TFHEpp::lvl0param>& in1, Stream st)
{
    cudaSetDevice(st.device_id());
    OrNYBootstrap<TFHEpp::lvl01param, TFHEpp::lvl1param::μ, TFHEpp::lvl10param>(out.tlwedevices[st.device_id()],
                  in0.tlwedevices[st.device_id()],
                  in1.tlwedevices[st.device_id()], st.st(), st.device_id());
}

template<>
void And<TFHEpp::lvl0param>(Ctxt<TFHEpp::lvl0param>& out, Ctxt<TFHEpp::lvl0param>& in0, Ctxt<TFHEpp::lvl0param>& in1, Stream st)
{
    cudaSetDevice(st.device_id());
    CtxtCopyH2D<TFHEpp::lvl0param>(in0, st);
    CtxtCopyH2D<TFHEpp::lvl0param>(in1, st);
    AndBootstrap<TFHEpp::lvl01param, TFHEpp::lvl1param::μ, TFHEpp::lvl10param>(out.tlwedevices[st.device_id()],
                 in0.tlwedevices[st.device_id()],
                 in1.tlwedevices[st.device_id()], st.st(), st.device_id());
    CtxtCopyD2H<TFHEpp::lvl0param>(out, st);
}

template<>
void gAnd<TFHEpp::lvl0param>(Ctxt<TFHEpp::lvl0param>& out, Ctxt<TFHEpp::lvl0param>& in0, Ctxt<TFHEpp::lvl0param>& in1, Stream st)
{
    cudaSetDevice(st.device_id());
    AndBootstrap<TFHEpp::lvl01param, TFHEpp::lvl1param::μ, TFHEpp::lvl10param>(out.tlwedevices[st.device_id()],
                 in0.tlwedevices[st.device_id()],
                 in1.tlwedevices[st.device_id()], st.st(), st.device_id());
}

template<>
void AndYN<TFHEpp::lvl0param>(Ctxt<TFHEpp::lvl0param>& out, Ctxt<TFHEpp::lvl0param>& in0, Ctxt<TFHEpp::lvl0param>& in1, Stream st)
{
    cudaSetDevice(st.device_id());
    CtxtCopyH2D<TFHEpp::lvl0param>(in0, st);
    CtxtCopyH2D<TFHEpp::lvl0param>(in1, st);
    AndYNBootstrap<TFHEpp::lvl01param, TFHEpp::lvl1param::μ, TFHEpp::lvl10param>(out.tlwedevices[st.device_id()],
                   in0.tlwedevices[st.device_id()],
                   in1.tlwedevices[st.device_id()], st.st(), st.device_id());
    CtxtCopyD2H<TFHEpp::lvl0param>(out, st);
}

template<>
void gAndYN<TFHEpp::lvl0param>(Ctxt<TFHEpp::lvl0param>& out, Ctxt<TFHEpp::lvl0param>& in0, Ctxt<TFHEpp::lvl0param>& in1, Stream st)
{
    cudaSetDevice(st.device_id());
    AndYNBootstrap<TFHEpp::lvl01param, TFHEpp::lvl1param::μ, TFHEpp::lvl10param>(out.tlwedevices[st.device_id()],
                   in0.tlwedevices[st.device_id()],
                   in1.tlwedevices[st.device_id()], st.st(), st.device_id());
}

template<>
void AndNY<TFHEpp::lvl0param>(Ctxt<TFHEpp::lvl0param>& out, Ctxt<TFHEpp::lvl0param>& in0, Ctxt<TFHEpp::lvl0param>& in1, Stream st)
{
    cudaSetDevice(st.device_id());
    CtxtCopyH2D<TFHEpp::lvl0param>(in0, st);
    CtxtCopyH2D<TFHEpp::lvl0param>(in1, st);
    AndNYBootstrap<TFHEpp::lvl01param, TFHEpp::lvl1param::μ, TFHEpp::lvl10param>(out.tlwedevices[st.device_id()],
                   in0.tlwedevices[st.device_id()],
                   in1.tlwedevices[st.device_id()], st.st(), st.device_id());
    CtxtCopyD2H<TFHEpp::lvl0param>(out, st);
}

template<>
void gAndNY<TFHEpp::lvl0param>(Ctxt<TFHEpp::lvl0param>& out, Ctxt<TFHEpp::lvl0param>& in0, Ctxt<TFHEpp::lvl0param>& in1, Stream st)
{
    cudaSetDevice(st.device_id());
    AndNYBootstrap<TFHEpp::lvl01param, TFHEpp::lvl1param::μ, TFHEpp::lvl10param>(out.tlwedevices[st.device_id()],
                   in0.tlwedevices[st.device_id()],
                   in1.tlwedevices[st.device_id()], st.st(), st.device_id());
}

template<>
void Nor<TFHEpp::lvl0param>(Ctxt<TFHEpp::lvl0param>& out, Ctxt<TFHEpp::lvl0param>& in0, Ctxt<TFHEpp::lvl0param>& in1, Stream st)
{
    cudaSetDevice(st.device_id());
    CtxtCopyH2D<TFHEpp::lvl0param>(in0, st);
    CtxtCopyH2D<TFHEpp::lvl0param>(in1, st);
    NorBootstrap<TFHEpp::lvl01param, TFHEpp::lvl1param::μ, TFHEpp::lvl10param>(out.tlwedevices[st.device_id()],
                 in0.tlwedevices[st.device_id()],
                 in1.tlwedevices[st.device_id()], st.st(), st.device_id());
    CtxtCopyD2H<TFHEpp::lvl0param>(out, st);
}

template<>
void gNor<TFHEpp::lvl0param>(Ctxt<TFHEpp::lvl0param>& out, Ctxt<TFHEpp::lvl0param>& in0, Ctxt<TFHEpp::lvl0param>& in1, Stream st)
{
    cudaSetDevice(st.device_id());
    NorBootstrap<TFHEpp::lvl01param, TFHEpp::lvl1param::μ, TFHEpp::lvl10param>(out.tlwedevices[st.device_id()],
                 in0.tlwedevices[st.device_id()],
                 in1.tlwedevices[st.device_id()], st.st(), st.device_id());
}

template<>
void Xor<TFHEpp::lvl0param>(Ctxt<TFHEpp::lvl0param>& out, Ctxt<TFHEpp::lvl0param>& in0, Ctxt<TFHEpp::lvl0param>& in1, Stream st)
{
    cudaSetDevice(st.device_id());
    CtxtCopyH2D<TFHEpp::lvl0param>(in0, st);
    CtxtCopyH2D<TFHEpp::lvl0param>(in1, st);
    XorBootstrap<TFHEpp::lvl01param, TFHEpp::lvl1param::μ, TFHEpp::lvl10param>(out.tlwedevices[st.device_id()],
                 in0.tlwedevices[st.device_id()],
                 in1.tlwedevices[st.device_id()], st.st(), st.device_id());
    CtxtCopyD2H<TFHEpp::lvl0param>(out, st);
}

template<>
void gXor<TFHEpp::lvl0param>(Ctxt<TFHEpp::lvl0param>& out, Ctxt<TFHEpp::lvl0param>& in0, Ctxt<TFHEpp::lvl0param>& in1, Stream st)
{
    cudaSetDevice(st.device_id());
    XorBootstrap<TFHEpp::lvl01param, TFHEpp::lvl1param::μ, TFHEpp::lvl10param>(out.tlwedevices[st.device_id()],
                 in0.tlwedevices[st.device_id()],
                 in1.tlwedevices[st.device_id()], st.st(), st.device_id());
}

template<>
void Xnor<TFHEpp::lvl0param>(Ctxt<TFHEpp::lvl0param>& out, Ctxt<TFHEpp::lvl0param>& in0, Ctxt<TFHEpp::lvl0param>& in1, Stream st)
{
    cudaSetDevice(st.device_id());
    CtxtCopyH2D<TFHEpp::lvl0param>(in0, st);
    CtxtCopyH2D<TFHEpp::lvl0param>(in1, st);
    XnorBootstrap<TFHEpp::lvl01param, TFHEpp::lvl1param::μ, TFHEpp::lvl10param>(out.tlwedevices[st.device_id()],
                  in0.tlwedevices[st.device_id()],
                  in1.tlwedevices[st.device_id()], st.st(), st.device_id());
    CtxtCopyD2H<TFHEpp::lvl0param>(out, st);
}

template<>
void gXnor<TFHEpp::lvl0param>(Ctxt<TFHEpp::lvl0param>& out, Ctxt<TFHEpp::lvl0param>& in0, Ctxt<TFHEpp::lvl0param>& in1, Stream st)
{
    cudaSetDevice(st.device_id());
    XnorBootstrap<TFHEpp::lvl01param, TFHEpp::lvl1param::μ, TFHEpp::lvl10param>(out.tlwedevices[st.device_id()],
                  in0.tlwedevices[st.device_id()],
                  in1.tlwedevices[st.device_id()], st.st(), st.device_id());
}

// Mux(inc,in1,in0) = inc?in1:in0 = inc&in1 + (!inc)&in0
template<>
void Mux<TFHEpp::lvl0param>(Ctxt<TFHEpp::lvl0param>& out, Ctxt<TFHEpp::lvl0param>& inc, Ctxt<TFHEpp::lvl0param>& in1, Ctxt<TFHEpp::lvl0param>& in0, Stream st)
{
    cudaSetDevice(st.device_id());
    CtxtCopyH2D<TFHEpp::lvl0param>(inc, st);
    CtxtCopyH2D<TFHEpp::lvl0param>(in1, st);
    CtxtCopyH2D<TFHEpp::lvl0param>(in0, st);
    MuxBootstrap<TFHEpp::lvl01param, TFHEpp::lvl1param::μ, TFHEpp::lvl10param>(out.tlwedevices[st.device_id()],
                 inc.tlwedevices[st.device_id()],
                 in1.tlwedevices[st.device_id()],
                 in0.tlwedevices[st.device_id()], st.st(), st.device_id());
    CtxtCopyD2H<TFHEpp::lvl0param>(out, st);
}

template<>
void gMux<TFHEpp::lvl0param>(Ctxt<TFHEpp::lvl0param>& out, Ctxt<TFHEpp::lvl0param>& inc, Ctxt<TFHEpp::lvl0param>& in1, Ctxt<TFHEpp::lvl0param>& in0, Stream st)
{
    cudaSetDevice(st.device_id());
    MuxBootstrap<TFHEpp::lvl01param, TFHEpp::lvl1param::μ, TFHEpp::lvl10param>(out.tlwedevices[st.device_id()],
                 inc.tlwedevices[st.device_id()],
                 in1.tlwedevices[st.device_id()],
                 in0.tlwedevices[st.device_id()], st.st(), st.device_id());
}

template<>
void NMux<TFHEpp::lvl0param>(Ctxt<TFHEpp::lvl0param>& out, Ctxt<TFHEpp::lvl0param>& inc, Ctxt<TFHEpp::lvl0param>& in1, Ctxt<TFHEpp::lvl0param>& in0, Stream st)
{
    cudaSetDevice(st.device_id());
    CtxtCopyH2D<TFHEpp::lvl0param>(inc, st);
    CtxtCopyH2D<TFHEpp::lvl0param>(in1, st);
    CtxtCopyH2D<TFHEpp::lvl0param>(in0, st);
    NMuxBootstrap<TFHEpp::lvl01param, TFHEpp::lvl1param::μ, TFHEpp::lvl10param>(out.tlwedevices[st.device_id()],
                  inc.tlwedevices[st.device_id()],
                  in1.tlwedevices[st.device_id()],
                  in0.tlwedevices[st.device_id()], st.st(), st.device_id());
    CtxtCopyD2H<TFHEpp::lvl0param>(out, st);
}

template<>
void gNMux<TFHEpp::lvl0param>(Ctxt<TFHEpp::lvl0param>& out, Ctxt<TFHEpp::lvl0param>& inc, Ctxt<TFHEpp::lvl0param>& in1, Ctxt<TFHEpp::lvl0param>& in0, Stream st)
{
    cudaSetDevice(st.device_id());
    NMuxBootstrap<TFHEpp::lvl01param, TFHEpp::lvl1param::μ, TFHEpp::lvl10param>(out.tlwedevices[st.device_id()],
                  inc.tlwedevices[st.device_id()],
                  in1.tlwedevices[st.device_id()],
                  in0.tlwedevices[st.device_id()], st.st(), st.device_id());
}

// lvl1 ver.
template<>
void Nand<TFHEpp::lvl1param>(Ctxt<TFHEpp::lvl1param>& out, Ctxt<TFHEpp::lvl1param>& in0, Ctxt<TFHEpp::lvl1param>& in1, Stream st)
{
    cudaSetDevice(st.device_id());
    CtxtCopyH2D<TFHEpp::lvl1param>(in0, st);
    CtxtCopyH2D<TFHEpp::lvl1param>(in1, st);
    NandBootstrap<TFHEpp::lvl10param, TFHEpp::lvl01param, TFHEpp::lvl1param::μ>(out.tlwedevices[st.device_id()],
                  in0.tlwedevices[st.device_id()],
                  in1.tlwedevices[st.device_id()], st.st(), st.device_id());
    CtxtCopyD2H<TFHEpp::lvl1param>(out, st);
}

template<>
void gNand<TFHEpp::lvl1param>(Ctxt<TFHEpp::lvl1param>& out, Ctxt<TFHEpp::lvl1param>& in0, Ctxt<TFHEpp::lvl1param>& in1, Stream st)
{
    cudaSetDevice(st.device_id());
    NandBootstrap<TFHEpp::lvl10param, TFHEpp::lvl01param, TFHEpp::lvl1param::μ>(out.tlwedevices[st.device_id()],
                  in0.tlwedevices[st.device_id()],
                  in1.tlwedevices[st.device_id()], st.st(), st.device_id());
}

template<>
void Or<TFHEpp::lvl1param>(Ctxt<TFHEpp::lvl1param>& out, Ctxt<TFHEpp::lvl1param>& in0, Ctxt<TFHEpp::lvl1param>& in1, Stream st)
{
    cudaSetDevice(st.device_id());
    CtxtCopyH2D<TFHEpp::lvl1param>(in0, st);
    CtxtCopyH2D<TFHEpp::lvl1param>(in1, st);
    OrBootstrap<TFHEpp::lvl10param, TFHEpp::lvl01param, TFHEpp::lvl1param::μ>(out.tlwedevices[st.device_id()],
                in0.tlwedevices[st.device_id()],
                in1.tlwedevices[st.device_id()], st.st(), st.device_id());
    CtxtCopyD2H<TFHEpp::lvl1param>(out, st);
}

template<>
void gOr<TFHEpp::lvl1param>(Ctxt<TFHEpp::lvl1param>& out, Ctxt<TFHEpp::lvl1param>& in0, Ctxt<TFHEpp::lvl1param>& in1, Stream st)
{
    cudaSetDevice(st.device_id());
    OrBootstrap<TFHEpp::lvl10param, TFHEpp::lvl01param, TFHEpp::lvl1param::μ>(out.tlwedevices[st.device_id()],
                in0.tlwedevices[st.device_id()],
                in1.tlwedevices[st.device_id()], st.st(), st.device_id());
}

template<>
void OrYN<TFHEpp::lvl1param>(Ctxt<TFHEpp::lvl1param>& out, Ctxt<TFHEpp::lvl1param>& in0, Ctxt<TFHEpp::lvl1param>& in1, Stream st)
{
    cudaSetDevice(st.device_id());
    CtxtCopyH2D<TFHEpp::lvl1param>(in0, st);
    CtxtCopyH2D<TFHEpp::lvl1param>(in1, st);
    OrYNBootstrap<TFHEpp::lvl10param, TFHEpp::lvl01param, TFHEpp::lvl1param::μ>(out.tlwedevices[st.device_id()],
                  in0.tlwedevices[st.device_id()],
                  in1.tlwedevices[st.device_id()], st.st(), st.device_id());
    CtxtCopyD2H<TFHEpp::lvl1param>(out, st);
}

template<>
void gOrYN<TFHEpp::lvl1param>(Ctxt<TFHEpp::lvl1param>& out, Ctxt<TFHEpp::lvl1param>& in0, Ctxt<TFHEpp::lvl1param>& in1, Stream st)
{
    cudaSetDevice(st.device_id());
    OrYNBootstrap<TFHEpp::lvl10param, TFHEpp::lvl01param, TFHEpp::lvl1param::μ>(out.tlwedevices[st.device_id()],
                  in0.tlwedevices[st.device_id()],
                  in1.tlwedevices[st.device_id()], st.st(), st.device_id());
}

template<>
void OrNY<TFHEpp::lvl1param>(Ctxt<TFHEpp::lvl1param>& out, Ctxt<TFHEpp::lvl1param>& in0, Ctxt<TFHEpp::lvl1param>& in1, Stream st)
{
    cudaSetDevice(st.device_id());
    CtxtCopyH2D<TFHEpp::lvl1param>(in0, st);
    CtxtCopyH2D<TFHEpp::lvl1param>(in1, st);
    OrNYBootstrap<TFHEpp::lvl10param, TFHEpp::lvl01param, TFHEpp::lvl1param::μ>(out.tlwedevices[st.device_id()],
                  in0.tlwedevices[st.device_id()],
                  in1.tlwedevices[st.device_id()], st.st(), st.device_id());
    CtxtCopyD2H<TFHEpp::lvl1param>(out, st);
}

template<>
void gOrNY<TFHEpp::lvl1param>(Ctxt<TFHEpp::lvl1param>& out, Ctxt<TFHEpp::lvl1param>& in0, Ctxt<TFHEpp::lvl1param>& in1, Stream st)
{
    cudaSetDevice(st.device_id());
    OrNYBootstrap<TFHEpp::lvl10param, TFHEpp::lvl01param, TFHEpp::lvl1param::μ>(out.tlwedevices[st.device_id()],
                  in0.tlwedevices[st.device_id()],
                  in1.tlwedevices[st.device_id()], st.st(), st.device_id());
}

template<>
void And<TFHEpp::lvl1param>(Ctxt<TFHEpp::lvl1param>& out, Ctxt<TFHEpp::lvl1param>& in0, Ctxt<TFHEpp::lvl1param>& in1, Stream st)
{
    cudaSetDevice(st.device_id());
    CtxtCopyH2D<TFHEpp::lvl1param>(in0, st);
    CtxtCopyH2D<TFHEpp::lvl1param>(in1, st);
    AndBootstrap<TFHEpp::lvl10param, TFHEpp::lvl01param, TFHEpp::lvl1param::μ>(out.tlwedevices[st.device_id()],
                 in0.tlwedevices[st.device_id()],
                 in1.tlwedevices[st.device_id()], st.st(), st.device_id());
    CtxtCopyD2H<TFHEpp::lvl1param>(out, st);
}

template<>
void gAnd<TFHEpp::lvl1param>(Ctxt<TFHEpp::lvl1param>& out, Ctxt<TFHEpp::lvl1param>& in0, Ctxt<TFHEpp::lvl1param>& in1, Stream st)
{
    cudaSetDevice(st.device_id());
    AndBootstrap<TFHEpp::lvl10param, TFHEpp::lvl01param, TFHEpp::lvl1param::μ>(out.tlwedevices[st.device_id()],
                 in0.tlwedevices[st.device_id()],
                 in1.tlwedevices[st.device_id()], st.st(), st.device_id());
}

template<>
void AndYN<TFHEpp::lvl1param>(Ctxt<TFHEpp::lvl1param>& out, Ctxt<TFHEpp::lvl1param>& in0, Ctxt<TFHEpp::lvl1param>& in1, Stream st)
{
    cudaSetDevice(st.device_id());
    CtxtCopyH2D<TFHEpp::lvl1param>(in0, st);
    CtxtCopyH2D<TFHEpp::lvl1param>(in1, st);
    AndYNBootstrap<TFHEpp::lvl10param, TFHEpp::lvl01param, TFHEpp::lvl1param::μ>(out.tlwedevices[st.device_id()],
                   in0.tlwedevices[st.device_id()],
                   in1.tlwedevices[st.device_id()], st.st(), st.device_id());
    CtxtCopyD2H<TFHEpp::lvl1param>(out, st);
}

template<>
void gAndYN<TFHEpp::lvl1param>(Ctxt<TFHEpp::lvl1param>& out, Ctxt<TFHEpp::lvl1param>& in0, Ctxt<TFHEpp::lvl1param>& in1, Stream st)
{
    cudaSetDevice(st.device_id());
    AndYNBootstrap<TFHEpp::lvl10param, TFHEpp::lvl01param, TFHEpp::lvl1param::μ>(out.tlwedevices[st.device_id()],
                   in0.tlwedevices[st.device_id()],
                   in1.tlwedevices[st.device_id()], st.st(), st.device_id());
}

template<>
void AndNY<TFHEpp::lvl1param>(Ctxt<TFHEpp::lvl1param>& out, Ctxt<TFHEpp::lvl1param>& in0, Ctxt<TFHEpp::lvl1param>& in1, Stream st)
{
    cudaSetDevice(st.device_id());
    CtxtCopyH2D<TFHEpp::lvl1param>(in0, st);
    CtxtCopyH2D<TFHEpp::lvl1param>(in1, st);
    AndNYBootstrap<TFHEpp::lvl10param, TFHEpp::lvl01param, TFHEpp::lvl1param::μ>(out.tlwedevices[st.device_id()],
                   in0.tlwedevices[st.device_id()],
                   in1.tlwedevices[st.device_id()], st.st(), st.device_id());
    CtxtCopyD2H<TFHEpp::lvl1param>(out, st);
}

template<>
void gAndNY<TFHEpp::lvl1param>(Ctxt<TFHEpp::lvl1param>& out, Ctxt<TFHEpp::lvl1param>& in0, Ctxt<TFHEpp::lvl1param>& in1, Stream st)
{
    cudaSetDevice(st.device_id());
    AndNYBootstrap<TFHEpp::lvl10param, TFHEpp::lvl01param, TFHEpp::lvl1param::μ>(out.tlwedevices[st.device_id()],
                   in0.tlwedevices[st.device_id()],
                   in1.tlwedevices[st.device_id()], st.st(), st.device_id());
}

template<>
void Nor<TFHEpp::lvl1param>(Ctxt<TFHEpp::lvl1param>& out, Ctxt<TFHEpp::lvl1param>& in0, Ctxt<TFHEpp::lvl1param>& in1, Stream st)
{
    cudaSetDevice(st.device_id());
    CtxtCopyH2D<TFHEpp::lvl1param>(in0, st);
    CtxtCopyH2D<TFHEpp::lvl1param>(in1, st);
    NorBootstrap<TFHEpp::lvl10param, TFHEpp::lvl01param, TFHEpp::lvl1param::μ>(out.tlwedevices[st.device_id()],
                 in0.tlwedevices[st.device_id()],
                 in1.tlwedevices[st.device_id()], st.st(), st.device_id());
    CtxtCopyD2H<TFHEpp::lvl1param>(out, st);
}

template<>
void gNor<TFHEpp::lvl1param>(Ctxt<TFHEpp::lvl1param>& out, Ctxt<TFHEpp::lvl1param>& in0, Ctxt<TFHEpp::lvl1param>& in1, Stream st)
{
    cudaSetDevice(st.device_id());
    NorBootstrap<TFHEpp::lvl10param, TFHEpp::lvl01param, TFHEpp::lvl1param::μ>(out.tlwedevices[st.device_id()],
                 in0.tlwedevices[st.device_id()],
                 in1.tlwedevices[st.device_id()], st.st(), st.device_id());
}

template<>
void Xor<TFHEpp::lvl1param>(Ctxt<TFHEpp::lvl1param>& out, Ctxt<TFHEpp::lvl1param>& in0, Ctxt<TFHEpp::lvl1param>& in1, Stream st)
{
    cudaSetDevice(st.device_id());
    CtxtCopyH2D<TFHEpp::lvl1param>(in0, st);
    CtxtCopyH2D<TFHEpp::lvl1param>(in1, st);
    XorBootstrap<TFHEpp::lvl10param, TFHEpp::lvl01param, TFHEpp::lvl1param::μ>(out.tlwedevices[st.device_id()],
                 in0.tlwedevices[st.device_id()],
                 in1.tlwedevices[st.device_id()], st.st(), st.device_id());
    CtxtCopyD2H<TFHEpp::lvl1param>(out, st);
}

template<>
void gXor<TFHEpp::lvl1param>(Ctxt<TFHEpp::lvl1param>& out, Ctxt<TFHEpp::lvl1param>& in0, Ctxt<TFHEpp::lvl1param>& in1, Stream st)
{
    cudaSetDevice(st.device_id());
    XorBootstrap<TFHEpp::lvl10param, TFHEpp::lvl01param, TFHEpp::lvl1param::μ>(out.tlwedevices[st.device_id()],
                 in0.tlwedevices[st.device_id()],
                 in1.tlwedevices[st.device_id()], st.st(), st.device_id());
}

template<>
void Xnor<TFHEpp::lvl1param>(Ctxt<TFHEpp::lvl1param>& out, Ctxt<TFHEpp::lvl1param>& in0, Ctxt<TFHEpp::lvl1param>& in1, Stream st)
{
    cudaSetDevice(st.device_id());
    CtxtCopyH2D<TFHEpp::lvl1param>(in0, st);
    CtxtCopyH2D<TFHEpp::lvl1param>(in1, st);
    XnorBootstrap<TFHEpp::lvl10param, TFHEpp::lvl01param, TFHEpp::lvl1param::μ>(out.tlwedevices[st.device_id()],
                  in0.tlwedevices[st.device_id()],
                  in1.tlwedevices[st.device_id()], st.st(), st.device_id());
    CtxtCopyD2H<TFHEpp::lvl1param>(out, st);
}

template<>
void gXnor<TFHEpp::lvl1param>(Ctxt<TFHEpp::lvl1param>& out, Ctxt<TFHEpp::lvl1param>& in0, Ctxt<TFHEpp::lvl1param>& in1, Stream st)
{
    cudaSetDevice(st.device_id());
    XnorBootstrap<TFHEpp::lvl10param, TFHEpp::lvl01param, TFHEpp::lvl1param::μ>(out.tlwedevices[st.device_id()],
                  in0.tlwedevices[st.device_id()],
                  in1.tlwedevices[st.device_id()], st.st(), st.device_id());
}

// Mux(inc,in1,in0) = inc?in1:in0 = inc&in1 + (!inc)&in0
template<>
void Mux<TFHEpp::lvl1param>(Ctxt<TFHEpp::lvl1param>& out, Ctxt<TFHEpp::lvl1param>& inc, Ctxt<TFHEpp::lvl1param>& in1, Ctxt<TFHEpp::lvl1param>& in0, Stream st)
{
    cudaSetDevice(st.device_id());
    CtxtCopyH2D<TFHEpp::lvl1param>(inc, st);
    CtxtCopyH2D<TFHEpp::lvl1param>(in1, st);
    CtxtCopyH2D<TFHEpp::lvl1param>(in0, st);
    MuxBootstrap<TFHEpp::lvl10param, TFHEpp::lvl01param, TFHEpp::lvl1param::μ>(out.tlwedevices[st.device_id()],
                 inc.tlwedevices[st.device_id()],
                 in1.tlwedevices[st.device_id()],
                 in0.tlwedevices[st.device_id()], st.st(), st.device_id());
    CtxtCopyD2H<TFHEpp::lvl1param>(out, st);
}

template<>
void gMux<TFHEpp::lvl1param>(Ctxt<TFHEpp::lvl1param>& out, Ctxt<TFHEpp::lvl1param>& inc, Ctxt<TFHEpp::lvl1param>& in1, Ctxt<TFHEpp::lvl1param>& in0, Stream st)
{
    cudaSetDevice(st.device_id());
    MuxBootstrap<TFHEpp::lvl10param, TFHEpp::lvl01param, TFHEpp::lvl1param::μ>(out.tlwedevices[st.device_id()],
                 inc.tlwedevices[st.device_id()],
                 in1.tlwedevices[st.device_id()],
                 in0.tlwedevices[st.device_id()], st.st(), st.device_id());
}

template<>
void NMux(Ctxt<TFHEpp::lvl1param>& out, Ctxt<TFHEpp::lvl1param>& inc, Ctxt<TFHEpp::lvl1param>& in1, Ctxt<TFHEpp::lvl1param>& in0, Stream st)
{
    cudaSetDevice(st.device_id());
    CtxtCopyH2D<TFHEpp::lvl1param>(inc, st);
    CtxtCopyH2D<TFHEpp::lvl1param>(in1, st);
    CtxtCopyH2D<TFHEpp::lvl1param>(in0, st);
    NMuxBootstrap<TFHEpp::lvl10param, TFHEpp::lvl01param, TFHEpp::lvl1param::μ>(out.tlwedevices[st.device_id()],
                  inc.tlwedevices[st.device_id()],
                  in1.tlwedevices[st.device_id()],
                  in0.tlwedevices[st.device_id()], st.st(), st.device_id());
    CtxtCopyD2H<TFHEpp::lvl1param>(out, st);
}

template<>
void gNMux(Ctxt<TFHEpp::lvl1param>& out, Ctxt<TFHEpp::lvl1param>& inc, Ctxt<TFHEpp::lvl1param>& in1, Ctxt<TFHEpp::lvl1param>& in0, Stream st)
{
    cudaSetDevice(st.device_id());
    NMuxBootstrap<TFHEpp::lvl10param, TFHEpp::lvl01param, TFHEpp::lvl1param::μ>(out.tlwedevices[st.device_id()],
                  inc.tlwedevices[st.device_id()],
                  in1.tlwedevices[st.device_id()],
                  in0.tlwedevices[st.device_id()], st.st(), st.device_id());
}

}  // namespace cufhe
