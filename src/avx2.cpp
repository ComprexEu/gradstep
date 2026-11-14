#include "avx2.hpp"
#include <immintrin.h>

// Sum of the 8 floats in a register.
static inline float hsum(__m256 v) {
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_movehdup_ps(s));
    return _mm_cvtss_f32(s);
}

// Each register lane is a different sample: one iteration computes 8 predictions
// and 8 errors at once. Every parameter keeps its own 8-lane sum, which is
// added up across lanes (hsum) only once, at the very end.
void grad_avx2(const float* Xs, long long ld, const float* y, long long n, const float* w, float* g) {
    __m256 acc[AVX_P];
    for (int j = 0; j < AVX_P; j++) acc[j] = _mm256_setzero_ps();

    long long i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 pred = _mm256_set1_ps(w[AVX_P - 1]);
        for (int j = 0; j < AVX_P - 1; j++)
            pred = _mm256_fmadd_ps(_mm256_set1_ps(w[j]), _mm256_loadu_ps(Xs + j * ld + i), pred);
        __m256 e = _mm256_sub_ps(pred, _mm256_loadu_ps(y + i));
        for (int j = 0; j < AVX_P - 1; j++)
            acc[j] = _mm256_fmadd_ps(e, _mm256_loadu_ps(Xs + j * ld + i), acc[j]);
        acc[AVX_P - 1] = _mm256_add_ps(acc[AVX_P - 1], e);
    }
    for (int j = 0; j < AVX_P; j++) g[j] = hsum(acc[j]);

    // The last n % 8 samples don't fill a register; do them one by one.
    for (; i < n; i++) {
        float pred = w[AVX_P - 1];
        for (int j = 0; j < AVX_P - 1; j++) pred += w[j] * Xs[j * ld + i];
        float e = pred - y[i];
        for (int j = 0; j < AVX_P - 1; j++) g[j] += e * Xs[j * ld + i];
        g[AVX_P - 1] += e;
    }
}
