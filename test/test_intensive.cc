#include <include/cufhe_gpu.cuh>

const size_t N = 20, M = 1000;

template <class Launcher, class Verifier>
void runAndVerify(const char* name, Launcher&& launcher, Verifier&& verifier)
{
    cufhe::Stream st[M];
    for (size_t i = 0; i < M; i++) st[i].Create();

    int workingIndex[M] = {};
    for (size_t i = 0; i < M; i++) launcher(0, i, st[i]);
    while (true) {
        bool cont = false;
        for (size_t i = 0; i < M; i++) {
            if (workingIndex[i] == N) continue;
            cont = true;

            if (cufhe::StreamQuery(st[i])) {
                int j = ++workingIndex[i];
                if (j == N) continue;
                launcher(j, i, st[i]);
            }
        }
        if (!cont) break;
    }
    cudaDeviceSynchronize();
    size_t errcount = 0;
    for (size_t i = 0; i < N; i++)
        for (size_t j = 0; j < M; j++)
            if (!verifier(i, j)) errcount++;
    if (errcount != 0)
        std::cerr << "TEST FAILED! " << name << " " << errcount << "/(" << N
                  << " * " << M << ")\n";
    assert(errcount == 0);

    for (size_t i = 0; i < M; i++) st[i].Destroy();
}

void testMux(TFHEpp::SecretKey& sk)
{
    cufhe::Ctxt<TFHEpp::lvl0param> ca, cb, cc, cres[N][M];
    bool pa, pb, pc;
    pa = true;
    pb = false;
    pc = true;
    bool expected = pa ? pb : pc;
    TFHEpp::tlweSymEncrypt<TFHEpp::lvl0param>(ca.tlwehost,
        pa ? TFHEpp::lvl0param::μ : -TFHEpp::lvl0param::μ,
        sk.key.get<TFHEpp::lvl0param>());
    TFHEpp::tlweSymEncrypt<TFHEpp::lvl0param>(cb.tlwehost,
        pb ? TFHEpp::lvl0param::μ : -TFHEpp::lvl0param::μ,
        sk.key.get<TFHEpp::lvl0param>());
    TFHEpp::tlweSymEncrypt<TFHEpp::lvl0param>(cc.tlwehost,
        pc ? TFHEpp::lvl0param::μ : -TFHEpp::lvl0param::μ,
        sk.key.get<TFHEpp::lvl0param>());

    runAndVerify(
        "mux",
        [&](size_t i, size_t j, cufhe::Stream st) {
            cufhe::Mux(cres[i][j], ca, cb, cc, st);
        },
        [&](size_t i, size_t j) {
            bool decres;
            decres = TFHEpp::tlweSymDecrypt<TFHEpp::lvl0param>(
                cres[i][j].tlwehost, sk.key.get<TFHEpp::lvl0param>());
            return expected == decres;
        });
}

void testNand(TFHEpp::SecretKey& sk)
{
    cufhe::Ctxt<TFHEpp::lvl0param> ca, cb, cres[N][M];
    bool pa, pb;
    pa = true;
    pb = false;
    bool expected = !(pa && pb);
    TFHEpp::tlweSymEncrypt<TFHEpp::lvl0param>(ca.tlwehost,
        pa ? TFHEpp::lvl0param::μ : -TFHEpp::lvl0param::μ,
        sk.key.get<TFHEpp::lvl0param>());
    TFHEpp::tlweSymEncrypt<TFHEpp::lvl0param>(cb.tlwehost,
        pb ? TFHEpp::lvl0param::μ : -TFHEpp::lvl0param::μ,
        sk.key.get<TFHEpp::lvl0param>());

    runAndVerify(
        "nand",
        [&](size_t i, size_t j, cufhe::Stream st) {
            cufhe::Nand(cres[i][j], ca, cb, st);
        },
        [&](size_t i, size_t j) {
            bool decres;
            decres = TFHEpp::tlweSymDecrypt<TFHEpp::lvl0param>(
                cres[i][j].tlwehost, sk.key.get<TFHEpp::lvl0param>());
            return expected == decres;
        });
}

int main()
{
    TFHEpp::SecretKey* sk = new TFHEpp::SecretKey();
    TFHEpp::EvalKey ek(*sk);
    ek.emplacebk<TFHEpp::lvl01param>(*sk);
    ek.emplaceiksk<TFHEpp::lvl10param>(*sk);
    cufhe::Initialize(ek);

    testNand(*sk);
    testMux(*sk);

    cufhe::CleanUp();
}
