/**
 * GPU-NTT based NTT implementation for cuFHE
 * Host-side initialization functions
 */

#include <include/ntt_gpu/ntt_gpuntt.cuh>
#include <include/details/error_gpu.cuh>
#include <cmath>
#include <vector>

namespace cufhe {

// Host-side storage for NTT parameters per GPU
std::vector<NTTParams> g_ntt_params;

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

// HEonGPU's TFHE parameters (optimized for N=1024)
// These are pre-computed for negacyclic NTT with N=1024
constexpr Data64 HEONGPU_TFHE_MODULUS = 1152921504606877697ULL;  // ~2^60
constexpr Data64 HEONGPU_TFHE_PSI = 1689264667710614ULL;  // Primitive 2048-th root of unity

// Use HEonGPU's TFHE parameters
constexpr Data64 ACTIVE_MODULUS = HEONGPU_TFHE_MODULUS;

// For HEonGPU's parameters, psi is already the correct 2N-th root for N=1024
Data64 ComputePsi(int log_n) {
    // HEonGPU provides psi directly for N=1024
    // The psi value 1689264667710614 is a primitive 2048-th root of unity
    return HEONGPU_TFHE_PSI;
}

// Generate root of unity tables for given length
void GenerateRootTables(
    int log_n,
    std::vector<Data64>& forward_table,
    std::vector<Data64>& inverse_table,
    DeviceModulus& modulus_params,
    Data64& n_inverse)
{
    const int n = 1 << log_n;
    NTTModulus modulus(ACTIVE_MODULUS);

    // Debug: print modulus parameters
    printf("NTTModulus computed: value=%lu, bit=%lu, mu=%lu\n",
           (unsigned long)modulus.value, (unsigned long)modulus.bit, (unsigned long)modulus.mu);

    // Store modulus parameters
    modulus_params.value = modulus.value;
    modulus_params.bit = modulus.bit;
    modulus_params.mu = modulus.mu;

    // Compute 2n-th root of unity (psi) for negacyclic NTT
    Data64 psi = ComputePsi(log_n);
    Data64 psi_inv = OPERATOR<Data64>::modinv(psi, modulus);

    // Compute n^{-1} mod p
    n_inverse = OPERATOR<Data64>::modinv(static_cast<Data64>(n), modulus);

    // Generate forward root table (psi^0, psi^1, ..., psi^{n-1})
    forward_table.resize(n);
    forward_table[0] = 1;
    for (int i = 1; i < n; i++) {
        forward_table[i] = OPERATOR<Data64>::mult(forward_table[i - 1], psi, modulus);
    }

    // Generate inverse root table (psi_inv^0, psi_inv^1, ...)
    inverse_table.resize(n);
    inverse_table[0] = 1;
    for (int i = 1; i < n; i++) {
        inverse_table[i] = OPERATOR<Data64>::mult(inverse_table[i - 1], psi_inv, modulus);
    }

    // Convert to bit-reversed order for GPU-NTT
    std::vector<Data64> br_forward(n);
    std::vector<Data64> br_inverse(n);
    for (int i = 0; i < n; i++) {
        int br_idx = bitreverse(i, log_n);
        br_forward[i] = forward_table[br_idx];
        br_inverse[i] = inverse_table[br_idx];
    }
    forward_table = std::move(br_forward);
    inverse_table = std::move(br_inverse);
}

// Storage for root tables (host-side)
std::vector<Data64> g_forward_table;
std::vector<Data64> g_inverse_table;
DeviceModulus g_modulus_params;
Data64 g_n_inverse;

} // anonymous namespace

//=============================================================================
// Static host functions for initialization
//=============================================================================

template <>
void CuNTTHandlerGPUNTT<TFHEpp::lvl1param::n>::Create() {
    constexpr int log_n = kLogLength;
    GenerateRootTables(log_n, g_forward_table, g_inverse_table, g_modulus_params, g_n_inverse);
}

template <>
void CuNTTHandlerGPUNTT<TFHEpp::lvl1param::n>::CreateConstant() {
    // Get current device ID
    int device_id;
    cudaGetDevice(&device_id);

    // Resize g_ntt_params if needed
    if (g_ntt_params.size() <= static_cast<size_t>(device_id)) {
        g_ntt_params.resize(device_id + 1);
    }

    NTTParams& params = g_ntt_params[device_id];

    // Allocate device memory for root tables
    CuSafeCall(cudaMalloc(&params.forward_root, sizeof(Data64) * kLength));
    CuSafeCall(cudaMalloc(&params.inverse_root, sizeof(Data64) * kLength));

    // Copy root tables to device memory
    CuSafeCall(cudaMemcpy(params.forward_root, g_forward_table.data(),
                          sizeof(Data64) * kLength, cudaMemcpyHostToDevice));
    CuSafeCall(cudaMemcpy(params.inverse_root, g_inverse_table.data(),
                          sizeof(Data64) * kLength, cudaMemcpyHostToDevice));

    // Store modulus and n_inverse in params
    params.modulus = g_modulus_params;
    params.n_inverse = g_n_inverse;
}

template <>
void CuNTTHandlerGPUNTT<TFHEpp::lvl1param::n>::SetDevicePointers(int device_id) {
    if (device_id < 0 || static_cast<size_t>(device_id) >= g_ntt_params.size()) {
        return;
    }

    const NTTParams& params = g_ntt_params[device_id];
    forward_root_ = params.forward_root;
    inverse_root_ = params.inverse_root;
    modulus_ = params.modulus;
    n_inverse_ = params.n_inverse;
}

template <>
void CuNTTHandlerGPUNTT<TFHEpp::lvl1param::n>::Destroy() {
    for (auto& params : g_ntt_params) {
        if (params.forward_root) {
            cudaFree(params.forward_root);
            params.forward_root = nullptr;
        }
        if (params.inverse_root) {
            cudaFree(params.inverse_root);
            params.inverse_root = nullptr;
        }
    }
    g_ntt_params.clear();
    g_forward_table.clear();
    g_inverse_table.clear();
}

// Explicit template instantiation
template class CuNTTHandlerGPUNTT<TFHEpp::lvl1param::n>;

} // namespace cufhe
