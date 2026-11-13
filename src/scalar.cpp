#include "scalar.hpp"

// gradient sums: g[j] = Σ e_i*x_ij (j < F), g[F] = Σ e_i
void grad_scalar(const Dataset& d, const float* w, float* g) {
    float acc[P] = {0};
    for (long long i = 0; i < d.n; i++) {
        const float* x = &d.X[i * F];          // sample i
        float pred = w[F];                     // bias
        for (int j = 0; j < F; j++) pred += w[j] * x[j];
        float e = pred - d.y[i];               // prediction error
        for (int j = 0; j < F; j++) acc[j] += e * x[j];
        acc[F] += e;
    }
    for (int j = 0; j < P; j++) g[j] = acc[j];
}

