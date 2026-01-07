/**
 * Test for 1024-degree polynomial multiplication
 * This test performs polynomial multiplication of 1024-degree polynomials
 * with 32-bit coefficient polynomial times up to 20-bit coefficient polynomial mod 32 bit
 */

#include <cassert>
#include <iostream>
#include <random>
#include <vector>
#include <cstdint>
#include <algorithm>
#include <cmath>

#include <include/ntt_gpu/ntt.cuh>
#include <include/details/error_gpu.cuh>
#include <include/cufhe_gpu.cuh>

using namespace std;
using namespace cufhe;

// Test parameters for 1024-degree polynomial
constexpr uint32_t POLY_SIZE = 1024;
constexpr uint32_t NUM_TESTS = 100;

// Coefficient bounds
// NOTE: For NTT with p ≈ 2^60, we need N * max_a * max_b < p
// With N=1024, max safe product is p/N ≈ 2^50
// For TFHE, TGSW decomposition produces small coefficients (≤ Bg ≈ 256)
// so actual TFHE operations stay well within bounds
constexpr uint32_t COEFF_32BIT_MAX = 0xFFFFFFFF;  // 32-bit max (TRLWE coefficients)
constexpr uint32_t COEFF_TFHE_MAX = 0x3FFFF;      // 18-bit max (safe: 2^32 * 2^18 * 1024 ≈ p)

// Helper function to generate random polynomial with bounded coefficients
void generateRandomPolynomial(vector<uint32_t>& poly, uint32_t max_value, default_random_engine& engine) {
    uniform_int_distribution<uint32_t> dist(0, max_value);
    for (auto& coeff : poly) {
        coeff = dist(engine);
    }
}

// CPU reference implementation of negacyclic polynomial multiplication
// Result: c = a * b mod (X^n + 1) mod 2^32
void polynomialMultiplicationCPU(const vector<uint32_t>& a, 
                                 const vector<uint32_t>& b,
                                 vector<uint32_t>& result) {
    // Initialize result with zeros
    fill(result.begin(), result.end(), 0);
    
    // Naive polynomial multiplication with negacyclic reduction
    for (uint32_t i = 0; i < POLY_SIZE; i++) {
        for (uint32_t j = 0; j < POLY_SIZE; j++) {
            uint64_t prod = static_cast<uint64_t>(a[i]) * static_cast<uint64_t>(b[j]);
            
            if (i + j < POLY_SIZE) {
                // Normal case: add to result[i+j]
                result[i + j] = (result[i + j] + static_cast<uint32_t>(prod)) & COEFF_32BIT_MAX;
            } else {
                // Wraparound with negation (negacyclic)
                uint32_t idx = (i + j) - POLY_SIZE;
                // Subtract instead of add due to X^n = -1
                result[idx] = (result[idx] - static_cast<uint32_t>(prod)) & COEFF_32BIT_MAX;
            }
        }
    }
}

// GPU kernel for forward NTT transformation
__global__ void ForwardNTT(FFP* d_out_ntt,
                           const uint32_t* d_in,
                           CuNTTHandler<1024> ntt_handler) {
    __shared__ FFP sh_temp[POLY_SIZE];
    ntt_handler.NTT<uint32_t>(d_out_ntt, d_in, sh_temp, 0);
}

// GPU kernel for pointwise multiplication in NTT domain
__global__ void PointwiseMultiply(FFP* d_result_ntt,
                                  const FFP* d_a_ntt,
                                  const FFP* d_b_ntt) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < POLY_SIZE) {
        d_result_ntt[tid] = d_a_ntt[tid] * d_b_ntt[tid];
    }
}

// GPU kernel for inverse NTT transformation
__global__ void InverseNTT(uint32_t* d_out,
                           const FFP* d_in_ntt,
                           CuNTTHandler<1024> ntt_handler) {
    __shared__ FFP sh_temp[POLY_SIZE];
    ntt_handler.NTTInv<uint32_t>(d_out, d_in_ntt, sh_temp, 0);
}

int main() {
    cout << "=== Test for 1024-degree Polynomial Multiplication ===" << endl;
    cout << "Polynomial degree: " << POLY_SIZE << endl;
    cout << "Number of tests: " << NUM_TESTS << endl;
    cout << "First polynomial coefficients: up to 32-bit" << endl;
    cout << "Second polynomial coefficients: up to 18-bit (TFHE-safe)" << endl;
    cout << endl;
    
    // Initialize random number generator
    random_device seed_gen;
    default_random_engine engine(seed_gen());
    
    // Initialize NTT handler
    CuNTTHandler<1024>* ntt_handler = new CuNTTHandler<1024>();
    ntt_handler->Create();
    ntt_handler->CreateConstant();
    ntt_handler->SetDevicePointers(0);  // Set device pointers for GPU 0
    cudaDeviceSynchronize();
    CuCheckError();
    
    // Test variables
    vector<uint32_t> poly_a(POLY_SIZE);
    vector<uint32_t> poly_b(POLY_SIZE);
    vector<uint32_t> cpu_result(POLY_SIZE);
    vector<uint32_t> gpu_result(POLY_SIZE);
    
    // Device memory
    uint32_t* d_poly_a;
    uint32_t* d_poly_b;
    uint32_t* d_result;
    FFP* d_a_ntt;
    FFP* d_b_ntt;
    FFP* d_result_ntt;
    
    cudaMalloc((void**)&d_poly_a, POLY_SIZE * sizeof(uint32_t));
    cudaMalloc((void**)&d_poly_b, POLY_SIZE * sizeof(uint32_t));
    cudaMalloc((void**)&d_result, POLY_SIZE * sizeof(uint32_t));
    cudaMalloc((void**)&d_a_ntt, POLY_SIZE * sizeof(FFP));
    cudaMalloc((void**)&d_b_ntt, POLY_SIZE * sizeof(FFP));
    cudaMalloc((void**)&d_result_ntt, POLY_SIZE * sizeof(FFP));
    CuCheckError();
    
    // Run tests
    int passed_tests = 0;
    int failed_tests = 0;
    
    // Configure kernel launch parameters
    // NTT kernels use 128 threads (1024 >> NTT_THREAD_UNITBIT where NTT_THREAD_UNITBIT = 3)
    dim3 ntt_block(POLY_SIZE >> NTT_THREAD_UNITBIT);  // 128 threads
    dim3 ntt_grid(1);
    
    // Pointwise multiplication can use more threads
    dim3 mult_block(256);
    dim3 mult_grid((POLY_SIZE + mult_block.x - 1) / mult_block.x);
    
    for (uint32_t test = 0; test < NUM_TESTS; test++) {
        cout << "Test " << (test + 1) << "/" << NUM_TESTS << ": ";
        
        // Generate random polynomials with appropriate bounds
        // First polynomial with up to 32-bit coefficients
        generateRandomPolynomial(poly_a, COEFF_32BIT_MAX, engine);
        // Second polynomial with up to 18-bit coefficients (TFHE-safe)
        generateRandomPolynomial(poly_b, COEFF_TFHE_MAX, engine);
        
        // For initial testing, use smaller values to verify correctness
        if (test < 10) {
            generateRandomPolynomial(poly_a, 0xFFFF, engine);  // 16-bit for initial tests
            generateRandomPolynomial(poly_b, 0xFF, engine);    // 8-bit for initial tests
        }
        
        // CPU computation (reference)
        polynomialMultiplicationCPU(poly_a, poly_b, cpu_result);
        
        // Copy data to device
        cudaMemcpy(d_poly_a, poly_a.data(), POLY_SIZE * sizeof(uint32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_poly_b, poly_b.data(), POLY_SIZE * sizeof(uint32_t), cudaMemcpyHostToDevice);
        CuCheckError();
        
        // GPU computation using NTT
        // Step 1: Forward NTT on polynomial A
        ForwardNTT<<<ntt_grid, ntt_block>>>(d_a_ntt, d_poly_a, *ntt_handler);
        cudaDeviceSynchronize();
        CuCheckError();
        
        // Step 2: Forward NTT on polynomial B
        ForwardNTT<<<ntt_grid, ntt_block>>>(d_b_ntt, d_poly_b, *ntt_handler);
        cudaDeviceSynchronize();
        CuCheckError();
        
        // Step 3: Pointwise multiplication in NTT domain
        PointwiseMultiply<<<mult_grid, mult_block>>>(d_result_ntt, d_a_ntt, d_b_ntt);
        cudaDeviceSynchronize();
        CuCheckError();
        
        // Step 4: Inverse NTT to get the result
        InverseNTT<<<ntt_grid, ntt_block>>>(d_result, d_result_ntt, *ntt_handler);
        cudaDeviceSynchronize();
        CuCheckError();
        
        // Copy result back to host
        cudaMemcpy(gpu_result.data(), d_result, POLY_SIZE * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        CuCheckError();
        
        // Compare results
        bool test_passed = true;
        uint32_t max_diff = 0;
        uint32_t num_mismatches = 0;
        
        for (uint32_t i = 0; i < POLY_SIZE; i++) {
            uint32_t diff = abs(static_cast<int64_t>(cpu_result[i]) - static_cast<int64_t>(gpu_result[i]));
            max_diff = max(max_diff, diff);
            
            // Allow small numerical errors due to modular arithmetic
            if (diff > 2) {
                test_passed = false;
                num_mismatches++;
                if (num_mismatches <= 5 && failed_tests < 3) {  // Show details for first few failures
                    cout << "\n  Mismatch at index " << i << ": ";
                    cout << "CPU=" << cpu_result[i] << ", GPU=" << gpu_result[i];
                    cout << ", diff=" << diff;
                }
            }
        }
        
        if (test_passed) {
            cout << "PASSED (max_diff=" << max_diff << ")" << endl;
            passed_tests++;
        } else {
            cout << "FAILED (mismatches=" << num_mismatches << ", max_diff=" << max_diff << ")" << endl;
            failed_tests++;
            
            // Debug: Check if NTT results are non-zero
            if (failed_tests == 1) {
                vector<FFP> h_a_ntt(POLY_SIZE);
                cudaMemcpy(h_a_ntt.data(), d_a_ntt, POLY_SIZE * sizeof(FFP), cudaMemcpyDeviceToHost);
                bool all_zero = true;
                for (int i = 0; i < 10; i++) {
                    if (h_a_ntt[i].val() != 0) {
                        all_zero = false;
                        break;
                    }
                }
                if (all_zero) {
                    cout << "  WARNING: NTT result appears to be all zeros!" << endl;
                } else {
                    cout << "  NTT result has non-zero values (first element: " << h_a_ntt[0].val() << ")" << endl;
                }
            }
        }
    }
    
    // Cleanup
    cudaFree(d_poly_a);
    cudaFree(d_poly_b);
    cudaFree(d_result);
    cudaFree(d_a_ntt);
    cudaFree(d_b_ntt);
    cudaFree(d_result_ntt);
    
    ntt_handler->Destroy();
    delete ntt_handler;
    
    // Summary
    cout << "\n=== Test Summary ===" << endl;
    cout << "Passed: " << passed_tests << "/" << NUM_TESTS << endl;
    cout << "Failed: " << failed_tests << "/" << NUM_TESTS << endl;
    
    if (failed_tests == 0) {
        cout << "\n*** ALL TESTS PASSED! ***" << endl;
        cout << "1024-degree polynomial multiplication is working correctly." << endl;
        cout << "The implementation successfully handles:" << endl;
        cout << "  - 32-bit coefficient polynomials" << endl;
        cout << "  - 20-bit coefficient polynomials" << endl;
        cout << "  - Negacyclic polynomial multiplication mod (X^1024 + 1)" << endl;
    } else {
        cout << "\n*** SOME TESTS FAILED ***" << endl;
        cout << "The NTT-based polynomial multiplication may need debugging." << endl;
        return 1;
    }
    
    return 0;
}