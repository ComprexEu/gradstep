cmake_minimum_required(VERSION 3.18)
set(CMAKE_CXX_COMPILER clang++)
project(gradstep LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

add_executable(gradstep src/main.cpp src/scalar.cpp src/avx2.cpp)
target_include_directories(gradstep PRIVATE include)
target_compile_options(gradstep PRIVATE -Wall -Wextra)

# Only avx2.cpp may use AVX2 instructions; the rest runs on any x86-64 CPU.
set_source_files_properties(src/avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma")

# Without this clang would vectorize scalar.cpp on its own, and it would no
# longer be a scalar baseline for the AVX version.
option(NO_AUTOVEC "Disable auto-vectorization in scalar.cpp" ON)
if(NO_AUTOVEC)
    set_source_files_properties(src/scalar.cpp PROPERTIES COMPILE_OPTIONS "-fno-vectorize;-fno-slp-vectorize")
endif()

# ---------------------------------------------------------------------------
# Optional CUDA version.
#
# CUDA is detected, not required: without nvcc (or with -DGRADSTEP_CUDA=OFF)
# the program is built exactly as before, only with scalar + avx2. The
# GRADSTEP_HAS_CUDA define tells main.cpp whether src/cuda.cu and include/cuda.hpp
# exist in this build.
# ---------------------------------------------------------------------------
include(CheckLanguage)
check_language(CUDA)
option(GRADSTEP_CUDA "Build the CUDA version (needs nvcc and an NVIDIA GPU)" ON)

set(GRADSTEP_HAS_CUDA OFF)
if(GRADSTEP_CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)

        target_sources(gradstep PRIVATE src/cuda.cu)
        target_compile_definitions(gradstep PRIVATE GRADSTEP_HAS_CUDA=1)

        # "native" compiles for the GPU actually present at configure time
        # (CMake >= 3.24). Older CMake cannot understand it, so fall back to
        # the most common desktop architecture; override with
        # -DCMAKE_CUDA_ARCHITECTURES=86 (or 75/89) if needed.
        if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.24")
            set_target_properties(gradstep PROPERTIES CUDA_ARCHITECTURES native)
        else()
            set_target_properties(gradstep PROPERTIES CUDA_ARCHITECTURES 75)
        endif()

        set(GRADSTEP_HAS_CUDA ON)
        message(STATUS "gradstep: CUDA enabled (${CMAKE_CUDA_COMPILER})")
    else()
        message(STATUS "gradstep: nvcc not found, building CPU versions only")
    endif()
endif()

# The program returns 1 when a check fails, so it can serve as its own test.
enable_testing()
add_test(NAME gradient COMMAND gradstep -n 100000)
add_test(NAME training COMMAND gradstep -n 100000 --train 100)
if(GRADSTEP_HAS_CUDA)
    # Larger n so the GPU is actually bandwidth-bound and the CUDA check runs
    # on a meaningful amount of data. On a machine without a GPU the program
    # just skips the cuda row, so the test still passes.
    add_test(NAME gradient_cuda COMMAND gradstep -n 1048576)
endif()
