# Option to use GPU-NTT instead of original FFP-based NTT
option(USE_GPUNTT "Use GPU-NTT library for NTT operations" ON)

if(USE_GPUNTT)
    add_library(cufhe_gpu
        bootstrap_gpu.cu
        cufhe_gates_gpu.cu
        cufhe_gpu.cu
        keyswitch_gpu.cu
        details/allocator_gpu.cu
        ntt_gpu/ntt_gpuntt.cu
        )
    target_compile_definitions(cufhe_gpu PUBLIC USE_GPUNTT)
    target_link_libraries(cufhe_gpu PUBLIC ntt)  # GPU-NTT library
else()
    add_library(cufhe_gpu
        bootstrap_gpu.cu
        cufhe_gates_gpu.cu
        cufhe_gpu.cu
        keyswitch_gpu.cu
        details/allocator_gpu.cu
        ntt_gpu/ntt_twiddle.cu
        )
endif()

target_link_libraries(cufhe_gpu INTERFACE tfhe++)
target_include_directories(cufhe_gpu
    PUBLIC ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/thirdparties/TFHEpp/include
    ${PROJECT_SOURCE_DIR}/thirdparties/TFHEpp/thirdparties/cereal/include
    ${PROJECT_SOURCE_DIR}/thirdparties/TFHEpp/thirdparties/randen
    ${PROJECT_SOURCE_DIR}/thirdparties/TFHEpp/thirdparties/spqlios
    ${PROJECT_SOURCE_DIR}/thirdparties/GPU-NTT/src/include
    )
set_target_properties(cufhe_gpu PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
set_target_properties(cufhe_gpu PROPERTIES CUDA_ARCHITECTURES 80) #A100

# Enable CUDA separable compilation for __device__ variable linking across TUs
if(USE_GPUNTT)
    set_target_properties(cufhe_gpu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    set_target_properties(cufhe_gpu PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()
