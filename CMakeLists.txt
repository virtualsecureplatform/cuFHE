cmake_minimum_required(VERSION 3.18)
project(cuFHE LANGUAGES CUDA CXX C)

set(CMAKE_CXX_STANDARD 20)
find_package(CUDAToolkit REQUIRED)
set(CMAKE_CUDA_FLAGS "--ptxas-options=-v")

option(USE_RANDEN "Use randen as CSPRNG" OFF)
option(USE_80BIT_SECURITY "Use 80bit security parameter(faster)" OFF)
option(USE_CGGI19 "Use the parameter set proposed in CGGI19" OFF)
option(USE_CONCRETE "Use the parameter set proposed in CONCRETE" OFF)
option(USE_SMALL_NTT_MODULUS "Use small NTT modulus with Torus discretization switching (RAINTT-style)" OFF)

if(USE_RANDEN)
  add_compile_definitions(USE_RANDEN)
endif()

if(USE_80BIT_SECURITY)
  add_compile_definitions(USE_80BIT_SECURITY)
elseif(USE_CGGI19)
  add_compile_definitions(USE_CGGI19)
elseif(USE_CONCRETE)
  add_compile_definitions(USE_CONCRETE)
endif()

if(USE_SMALL_NTT_MODULUS)
  add_compile_definitions(USE_SMALL_NTT_MODULUS)
  message(STATUS "Using small NTT modulus (RAINTT-style) with Torus discretization switching")
endif()

# For OpenMP
find_package(OpenMP)
if(OpenMP_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

OPTION(ENABLE_TEST "Build tests" OFF)
OPTION(ENABLE_TUTORIAL "Build tutorial" OFF)

add_subdirectory(src)
add_subdirectory(thirdparties)

if(ENABLE_TEST)
    add_subdirectory(test)
endif()
