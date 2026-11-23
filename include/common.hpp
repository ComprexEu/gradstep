#pragma once
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

constexpr int F = 16;       // features per sample
constexpr int P = F + 1;    // parameters: F weights, bias stored last (w[F])

struct Dataset {
    long long n = 0;
    std::vector<float> X;       // row-major: feature j of sample i is X[i*F + j]
    std::vector<float> y;
    std::vector<float> w_true;  // parameters the data was generated from (P values)
    std::vector<float> Xs;      // column-major copy for the AVX version: Xs[j*ld + i]
    long long ld = 0;           // distance between columns of Xs, see make_columns
};

// The AVX version wants one feature of many samples next to each other,
// so it reads this column-major copy of X instead.
// The columns are n + 16 floats apart, not n: for n = 2^20 all 16 columns would
// start at addresses that map to the same cache sets and evict each other.
inline void make_columns(Dataset& d) {
    d.ld = d.n + 1040; //d.n + 16;
    d.Xs.assign(F * d.ld, 0.0f);
    for (long long i = 0; i < d.n; i++)
        for (int j = 0; j < F; j++) d.Xs[j * d.ld + i] = d.X[i * F + j];
}

// y = w_true·x + b_true + noise. Because w_true is known, training can be checked
// against it; the noise (sigma = 0.1) puts the best reachable MSE at about 0.01.
// Fixed seed, so every run sees the same data.
inline Dataset make_dataset(long long n, unsigned seed = 42) {
    Dataset d;
    d.n = n;
    d.X.resize(n * F);
    d.y.resize(n);
    d.w_true.resize(P);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> uw(-2.0f, 2.0f), ux(-1.0f, 1.0f);
    std::normal_distribution<float> noise(0.0f, 0.1f);
    for (auto& w : d.w_true) w = uw(rng);
    for (long long i = 0; i < n; i++) {
        double acc = d.w_true[F];
        for (int j = 0; j < F; j++) {
            float x = ux(rng);
            d.X[i * F + j] = x;
            acc += (double)d.w_true[j] * x;
        }
        d.y[i] = (float)acc + noise(rng);
    }
    make_columns(d);
    return d;
}

// File layout: n, w_true[P], X[n*F], y[n]. Saving the data lets later versions
// (AVX) be measured on exactly the same input.
inline bool save_dataset(const Dataset& d, const std::string& path) {
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) return false;
    bool ok = fwrite(&d.n, sizeof d.n, 1, f) == 1 &&
              fwrite(d.w_true.data(), sizeof(float), P, f) == P &&
              fwrite(d.X.data(), sizeof(float), d.X.size(), f) == d.X.size() &&
              fwrite(d.y.data(), sizeof(float), d.y.size(), f) == d.y.size();
    return fclose(f) == 0 && ok;
}

inline bool load_dataset(Dataset& d, const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return false;
    bool ok = fread(&d.n, sizeof d.n, 1, f) == 1 && d.n > 0;
    if (ok) {
        d.w_true.resize(P);
        d.X.resize(d.n * F);
        d.y.resize(d.n);
        ok = fread(d.w_true.data(), sizeof(float), P, f) == P &&
             fread(d.X.data(), sizeof(float), d.X.size(), f) == d.X.size() &&
             fread(d.y.data(), sizeof(float), d.y.size(), f) == d.y.size();
    }
    fclose(f);
    if (ok) make_columns(d);
    return ok;
}

// Same gradient sums as grad_scalar, but accumulated in double: this is the
// ground truth every faster version is compared against.
inline std::vector<double> reference_grad(const Dataset& d, const std::vector<float>& w) {
    std::vector<double> g(P, 0.0);
    for (long long i = 0; i < d.n; i++) {
        const float* x = &d.X[i * F];
        double pred = w[F];
        for (int j = 0; j < F; j++) pred += (double)w[j] * x[j];
        double e = pred - d.y[i];
        for (int j = 0; j < F; j++) g[j] += e * x[j];
        g[F] += e;
    }
    return g;
}

inline double mse(const Dataset& d, const std::vector<float>& w) {
    double loss = 0.0;
    for (long long i = 0; i < d.n; i++) {
        double pred = w[F];
        for (int j = 0; j < F; j++) pred += (double)w[j] * d.X[i * F + j];
        loss += (pred - d.y[i]) * (pred - d.y[i]);
    }
    return loss / d.n;
}

// Float sums of a million terms never match the double reference exactly, and
// the AVX version adds in a different order, so we need a tolerance.
// The error is divided by the largest gradient component rather than by each
// component, so a component that happens to be near 0 can't inflate the result.
inline double max_rel_error(const float* g, const std::vector<double>& ref) {
    double err = 0.0, norm = 1e-12;
    for (int j = 0; j < P; j++) {
        err = std::max(err, std::fabs(g[j] - ref[j]));
        norm = std::max(norm, std::fabs(ref[j]));
    }
    return err / norm;
}
