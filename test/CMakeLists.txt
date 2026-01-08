add_executable(test_gate_gpu test_gate_gpu.cc)
target_include_directories(test_gate_gpu PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(test_gate_gpu cufhe_gpu ${CUDA_LIBRARIES})

add_executable(test_gate_gpu_multi test_gate_gpu_multi.cc)
target_include_directories(test_gate_gpu_multi PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(test_gate_gpu_multi cufhe_gpu ${CUDA_LIBRARIES})

add_executable(test_intensive test_intensive.cc)
target_include_directories(test_intensive PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(test_intensive cufhe_gpu ${CUDA_LIBRARIES})

# Only build test_FFP for original FFP implementation (not GPU-NTT)
if(NOT USE_GPUNTT)
    add_executable(test_FFP test_FFP.cu)
    target_include_directories(test_FFP PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    find_library(gmp NAMES gmp gmpxx)
    target_link_libraries(test_FFP cufhe_gpu gmp ${CUDA_LIBRARIES})

    add_executable(test_cmux test_cmux.cc)
    target_include_directories(test_cmux PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    target_link_libraries(test_cmux cufhe_gpu ${CUDA_LIBRARIES})
endif()

add_executable(test_perf test_perf.cc)
target_include_directories(test_perf PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(test_perf cufhe_gpu ${CUDA_LIBRARIES})

# Only build polynomial mult test for GPU-NTT (uses GPU-NTT specific API)
if(USE_GPUNTT)
    add_executable(test_polynomial_mult_1024 test_polynomial_mult_1024.cu)
    target_include_directories(test_polynomial_mult_1024 PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    target_link_libraries(test_polynomial_mult_1024 cufhe_gpu ${CUDA_LIBRARIES})
    set_target_properties(test_polynomial_mult_1024 PROPERTIES
        CUDA_ARCHITECTURES 80
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()

