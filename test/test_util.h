#include <include/cufhe_gpu.cuh>
#include <iostream>
#include <random>
#include <vector>
using namespace std;

namespace cufhe{
template <class P, class Func, class Check>
void Test(string type, Func func, Check check, vector<uint8_t>& pt,
          vector<Ctxt<P>>& ct, Stream* st, int kNumTests, int kNumSMs,
          TFHEpp::SecretKey& sk)
{
    cout << "------ Test " << type << " Gate ------" << endl;
    cout << "Number of streams:\t" << kNumSMs << endl;
    cout << "Number of tests:\t" << kNumTests << endl;
    cout << "Number of tests per stream:\t" << kNumTests/kNumSMs << endl;
    bool correct = true;
    int cnt_failures = 0;

    random_device seed_gen;
    default_random_engine engine(seed_gen());
    uniform_int_distribution<uint32_t> binary(0, 1);
    for (int i = 0; i < 4 * kNumTests; i++) {
        pt[i] = binary(engine) > 0;
        ct[i].tlwehost = TFHEpp::tlweSymEncrypt<P>(
            pt[i] ? P::μ : -P::μ,
            P::α, sk.key.get<P>());
    }

    float et;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);


    for (int i = 0; i < kNumTests; i++) {
        if constexpr (std::is_invocable_v<Func, Ctxt<P>&>) {
            func(ct[i]);
            check(pt[i]);
        }
        else if constexpr (std::is_invocable_v<Func, Ctxt<P>&, Ctxt<P>&, Stream>) {
            func(ct[i], ct[i + kNumTests], st[i % kNumSMs]);
            check(pt[i], pt[i + kNumTests]);
        }
        else if constexpr (std::is_invocable_v<Func, Ctxt<P>&, Ctxt<P>&, Ctxt<P>&,
                                               Stream>) {
            func(ct[i], ct[i + kNumTests], ct[i + kNumTests * 2],
                 st[i % kNumSMs]);
            check(pt[i], pt[i + kNumTests], pt[i + kNumTests * 2]);
        }
        else if constexpr (std::is_invocable_v<Func, Ctxt<P>&, Ctxt<P>&, Ctxt<P>&, Ctxt<P>&,
                                               Stream>) {
            func(ct[i], ct[i + kNumTests], ct[i + kNumTests * 2],
                 ct[i + kNumTests * 3], st[i % kNumSMs]);
            check(pt[i], pt[i + kNumTests], pt[i + kNumTests * 2],
                  pt[i + kNumTests * 3]);
        }
        else {
            std::cout << "Invalid Function" << std::endl;
        }
    }
    Synchronize();

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&et, start, stop);
    cout << "Total:" << et << "ms" << endl;
    cout << et / kNumTests << " ms / gate" << endl;
    cout << et / kNumSMs << " ms / stream" << endl;
    cout <<"Latency: "<< et * kNumSMs / kNumTests << " ms" << endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    for (int i = 0; i < kNumTests; i++) {
        uint8_t res;
        res = TFHEpp::tlweSymDecrypt<P>(ct[i].tlwehost,
                                                        sk.key.get<P>());
        if (res != pt[i]) {
            correct = false;
            cnt_failures += 1;
            // std::cout << type << " Fail at iteration: " << i << std::endl;
        }
    }
    if (correct)
        cout << "PASS" << endl;
    else
        cout << "FAIL:\t" << cnt_failures << "/" << kNumTests << endl;
}
}

