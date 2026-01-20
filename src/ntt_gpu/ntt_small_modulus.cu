/**
 * Small Modulus NTT implementation for cuFHE
 * Host-side initialization functions
 *
 * Uses RAINTT-style modulus: P = (K << shiftamount) + 1 = 655360001
 */

#include <include/ntt_gpu/ntt_small_modulus.cuh>
#include <include/details/error_gpu.cuh>
#include <cmath>
#include <vector>
#include <stdexcept>

namespace cufhe {

// Host-side storage for small NTT parameters per GPU
std::vector<SmallNTTParams> g_small_ntt_params;

namespace {

// Bit reversal helper
int bitreverse(int index, int n_power) {
    int res = 0;
    for (int i = 0; i < n_power; i++) {
        res <<= 1;
        res = (index & 1) | res;
        index >>= 1;
    }
    return res;
}

// Modular exponentiation for 32-bit values
uint32_t mod_exp(uint64_t base, uint64_t exp, uint32_t mod) {
    uint64_t result = 1;
    base = base % mod;
    while (exp > 0) {
        if (exp & 1) {
            result = (result * base) % mod;
        }
        exp >>= 1;
        base = (base * base) % mod;
    }
    return static_cast<uint32_t>(result);
}

// Modular inverse using extended Euclidean algorithm
uint32_t mod_inv(uint32_t a, uint32_t mod) {
    // Using Fermat's little theorem: a^(-1) = a^(p-2) mod p (for prime p)
    return mod_exp(a, mod - 2, mod);
}

// Modular multiplication for 32-bit values
uint32_t mod_mult(uint32_t a, uint32_t b, uint32_t mod) {
    return static_cast<uint32_t>((static_cast<uint64_t>(a) * b) % mod);
}

/**
 * Find a primitive 2N-th root of unity for negacyclic NTT
 *
 * For negacyclic convolution, we need psi where psi^N = -1 (mod P)
 * which means psi^(2N) = 1 (mod P) but psi^k != 1 for k < 2N
 *
 * For P = 655360001:
 * P - 1 = 655360000 = 2^20 * 625 = 2^20 * 5^4
 *
 * Since 2^20 divides (P-1), we have primitive 2^20-th roots of unity
 * For N = 1024 = 2^10, we need 2N = 2048 = 2^11, which divides 2^20
 *
 * The primitive (P-1)-th root is found by:
 * g = generator of Z_P^*
 * Then psi_k = g^((P-1)/k) is a primitive k-th root of unity
 */
uint32_t find_primitive_root(int log_n) {
    constexpr uint32_t P = small_ntt::P;  // 655360001
    constexpr uint32_t P_minus_1 = P - 1; // 655360000 = 2^20 * 5^4

    // For N = 2^log_n, we need primitive 2^(log_n+1)-th root of unity
    uint32_t two_n = 1U << (log_n + 1);

    // First, find a generator g of Z_P^*
    // P - 1 = 2^20 * 5^4, so we need g where g^((P-1)/2) != 1 and g^((P-1)/5) != 1
    // A simple trial: test small primes
    uint32_t g = 3;  // Common generator candidate

    // Verify g is a generator by checking g^((P-1)/q) != 1 for prime divisors q of P-1
    // Prime divisors of P-1: 2 and 5
    while (true) {
        bool is_generator = true;

        // Check g^((P-1)/2) != 1
        if (mod_exp(g, P_minus_1 / 2, P) == 1) {
            is_generator = false;
        }
        // Check g^((P-1)/5) != 1
        if (mod_exp(g, P_minus_1 / 5, P) == 1) {
            is_generator = false;
        }

        if (is_generator) break;
        g++;
        if (g > 1000) {
            throw std::runtime_error("Could not find generator for small modulus");
        }
    }

    // Now compute primitive 2N-th root of unity
    // psi = g^((P-1)/(2N))
    uint32_t exponent = P_minus_1 / two_n;
    uint32_t psi = mod_exp(g, exponent, P);

    // Verify: psi^N should be P-1 (which is -1 mod P)
    uint32_t psi_N = mod_exp(psi, 1U << log_n, P);
    if (psi_N != P - 1) {
        throw std::runtime_error("Computed root does not satisfy psi^N = -1");
    }

    return psi;
}

// Storage for root tables (host-side)
std::vector<uint32_t> g_small_forward_table;
std::vector<uint32_t> g_small_inverse_table;
uint32_t g_small_n_inverse;
int g_small_log_n = 0;

/**
 * Generate NTT root tables for small modulus
 */
void GenerateSmallRootTables(int log_n) {
    if (g_small_log_n == log_n && !g_small_forward_table.empty()) {
        return;  // Already generated
    }

    const int n = 1 << log_n;
    constexpr uint32_t P = small_ntt::P;

    // Find primitive 2N-th root of unity (psi)
    uint32_t psi = find_primitive_root(log_n);
    uint32_t psi_inv = mod_inv(psi, P);

    // Compute n^(-1) mod P
    g_small_n_inverse = mod_inv(static_cast<uint32_t>(n), P);

    // Generate forward root table: psi^0, psi^1, ..., psi^(n-1) in bit-reversed order
    // For Cooley-Tukey, we need powers of psi^2 (omega = psi^2 is the N-th root of unity)
    g_small_forward_table.resize(n);
    g_small_forward_table[0] = 1;
    for (int i = 1; i < n; i++) {
        g_small_forward_table[i] = mod_mult(g_small_forward_table[i - 1], psi, P);
    }

    // Generate inverse root table: psi_inv^0, psi_inv^1, ...
    g_small_inverse_table.resize(n);
    g_small_inverse_table[0] = 1;
    for (int i = 1; i < n; i++) {
        g_small_inverse_table[i] = mod_mult(g_small_inverse_table[i - 1], psi_inv, P);
    }

    // Convert to bit-reversed order for NTT algorithm
    std::vector<uint32_t> br_forward(n);
    std::vector<uint32_t> br_inverse(n);
    for (int i = 0; i < n; i++) {
        int br_idx = bitreverse(i, log_n);
        br_forward[i] = g_small_forward_table[br_idx];
        br_inverse[i] = g_small_inverse_table[br_idx];
    }
    g_small_forward_table = std::move(br_forward);
    g_small_inverse_table = std::move(br_inverse);

    g_small_log_n = log_n;
}

}  // anonymous namespace

//=============================================================================
// Static host functions for initialization
//=============================================================================

template <>
void CuSmallNTTHandler<TFHEpp::lvl1param::n>::Create() {
    constexpr int log_n = kLogLength;
    GenerateSmallRootTables(log_n);
}

template <>
void CuSmallNTTHandler<TFHEpp::lvl1param::n>::Destroy() {
    for (auto& params : g_small_ntt_params) {
        if (params.forward_root) {
            cudaFree(params.forward_root);
            params.forward_root = nullptr;
        }
        if (params.inverse_root) {
            cudaFree(params.inverse_root);
            params.inverse_root = nullptr;
        }
        params.initialized = false;
    }
    g_small_ntt_params.clear();
    g_small_forward_table.clear();
    g_small_inverse_table.clear();
    g_small_log_n = 0;
}

template <>
void CuSmallNTTHandler<TFHEpp::lvl1param::n>::SetDevicePointers(int device_id) {
    // Resize if needed
    if (g_small_ntt_params.size() <= static_cast<size_t>(device_id)) {
        g_small_ntt_params.resize(device_id + 1);
    }

    SmallNTTParams& params = g_small_ntt_params[device_id];

    // Initialize if not already done
    if (!params.initialized) {
        // Allocate device memory for root tables
        CuSafeCall(cudaMalloc(&params.forward_root, sizeof(uint32_t) * kLength));
        CuSafeCall(cudaMalloc(&params.inverse_root, sizeof(uint32_t) * kLength));

        // Copy root tables to device
        CuSafeCall(cudaMemcpy(params.forward_root, g_small_forward_table.data(),
                              sizeof(uint32_t) * kLength, cudaMemcpyHostToDevice));
        CuSafeCall(cudaMemcpy(params.inverse_root, g_small_inverse_table.data(),
                              sizeof(uint32_t) * kLength, cudaMemcpyHostToDevice));

        params.n_inverse = g_small_n_inverse;
        params.initialized = true;
    }

    // Set device pointers
    forward_root_ = params.forward_root;
    inverse_root_ = params.inverse_root;
    n_inverse_ = params.n_inverse;
}

// Explicit template instantiation
template class CuSmallNTTHandler<TFHEpp::lvl1param::n>;

}  // namespace cufhe
