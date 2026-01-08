add_executable(test_gate_gpu test_gate_gpu.cc)
target_include_directories(test_gate_gpu PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(test_gate_gpu cufhe_gpu ${CUDA_LIBRARIES})

add_executable(test_gate_gpu_multi test_gate_gpu_multi.cc)
target_include_directories(test_gate_gpu_multi PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(test_gate_gpu_multi cufhe_gpu ${CUDA_LIBRARIES})

add_executable(test_intensive test_intensive.cc)
target_include_directories(test_intensive PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(test_intensive cufhe_gpu ${CUDA_LIBRARIES})

add_executable(test_perf test_perf.cc)
target_include_directories(test_perf PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(test_perf cufhe_gpu ${CUDA_LIBRARIES})

add_executable(test_polynomial_mult_1024 test_polynomial_mult_1024.cu)
target_include_directories(test_polynomial_mult_1024 PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(test_polynomial_mult_1024 cufhe_gpu ${CUDA_LIBRARIES})
set_target_properties(test_polynomial_mult_1024 PROPERTIES
    CUDA_ARCHITECTURES 80
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON)
