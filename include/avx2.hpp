#pragma once

constexpr int AVX_P = 17;   // same as P in common.hpp (F weights + bias)

// Gradient sums over n samples of column-major data (Xs[j*ld + i]),
// the same numbers grad_scalar computes.
// Plain pointers on purpose: avx2.cpp is compiled with -mavx2, so it must not
// include std:: headers whose inline code could then be reused by other files
// and crash on a CPU without AVX2.
void grad_avx2(const float* Xs, long long ld, const float* y, long long n, const float* w, float* g);

inline bool cpu_has_avx2() {
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
}
