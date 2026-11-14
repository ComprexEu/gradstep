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

# The program returns 1 when a check fails, so it can serve as its own test.
enable_testing()
add_test(NAME gradient COMMAND gradstep -n 100000)
add_test(NAME training COMMAND gradstep -n 100000 --train 100)
