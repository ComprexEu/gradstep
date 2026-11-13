#include "scalar.hpp"
#include <chrono>
#include <cstdlib>
#include <functional>
#include <string>

// Best of 10 runs, in ms. The minimum is the least disturbed by other processes
// and CPU clock changes; the first call is a warm-up (page faults, caches).
static double best_ms(const std::function<void()>& fn) {
    fn();
    double best = 1e30;
    for (int r = 0; r < 10; r++) {
        auto t0 = std::chrono::steady_clock::now();
        fn();
        best = std::min(best, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    }
    return best;
}

// The header is written only when the file is empty, so results of many runs
// collect in one file for plotting.
static void append_csv(const std::string& path, const char* version, long long n, double ms, double gbs, double err) {
    FILE* f = fopen(path.c_str(), "a");
    if (!f) { printf("Cannot open %s\n", path.c_str()); return; }
    if (ftell(f) == 0) fprintf(f, "version,n,f,time_ms,gb_s,max_rel_err\n");
    fprintf(f, "%s,%lld,%d,%.4f,%.3f,%.3e\n", version, n, F, ms, gbs, err);
    fclose(f);
}

struct Version {
    const char* name;
    std::function<void(const float* w, float* g)> grad;   // writes the P gradient sums to g
};

int main(int argc, char** argv) {
    long long n = 1 << 20;
    int train = 0;
    std::string save, load, csv;
    for (int i = 1; i + 1 < argc; i += 2) {
        std::string a = argv[i];
        if (a == "-n") n = atoll(argv[i + 1]);
        else if (a == "--train") train = atoi(argv[i + 1]);
        else if (a == "--save") save = argv[i + 1];
        else if (a == "--load") load = argv[i + 1];
        else if (a == "--csv") csv = argv[i + 1];
        else { printf("Unknown option: %s\n", a.c_str()); return 2; }
    }

    printf("== Batch Gradient Descent: one step ==\n");
    Dataset d;
    if (!load.empty()) {
        if (!load_dataset(d, load)) { printf("Cannot load %s\n", load.c_str()); return 1; }
        printf("Data loaded from %s\n", load.c_str());
    } else {
        d = make_dataset(n);
    }
    if (!save.empty()) {
        if (!save_dataset(d, save)) { printf("Cannot save %s\n", save.c_str()); return 1; }
        printf("Data saved to %s\n", save.c_str());
    }
    n = d.n;
    printf("N = %lld samples, F = %d features\n", n, F);

    // Every version gets the same data and weights and must produce the same sums.
    std::vector<Version> versions = {
        {"scalar", [&](const float* w, float* g) { grad_scalar(d, w, g); }},
    };

    std::vector<float> w(P, 0.0f);
    auto ref = reference_grad(d, w);
    // One step reads every x and y once, so this is the memory traffic per step.
    const double gb = n * (F + 1) * sizeof(float) / 1e9;

    printf("\nversion        error  check   time [ms]     GB/s   speedup\n");
    bool ok = true;
    double t_scalar = 0;
    for (const auto& v : versions) {
        float g[P];
        v.grad(w.data(), g);
        double err = max_rel_error(g, ref);
        bool v_ok = err < 1e-3;
        ok &= v_ok;

        double t = best_ms([&] { v.grad(w.data(), g); });
        if (t_scalar == 0) t_scalar = t;
        printf("%-12s %8.1e  %-5s %10.3f %8.2f %8.1fx\n", v.name, err, v_ok ? "OK" : "FAIL", t,
               gb / (t / 1e3), t_scalar / t);
        if (!csv.empty()) append_csv(csv, v.name, n, t, gb / (t / 1e3), err);
    }
    printf("(error vs. double reference must be < 1e-3; time is the best of 10)\n");
    if (!csv.empty()) printf("Results appended to %s\n", csv.c_str());

    // Repeated steps must recover the parameters the data was generated from.
    // Uses the last version in the list; its gradient was already checked above.
    if (train > 0) {
        const Version& v = versions.back();
        printf("\nTraining for %d steps with %s (lr = 0.5):\n", train, v.name);
        printf("  before:     MSE = %.6f\n", mse(d, w));
        for (int s = 1; s <= train; s++) {
            float g[P];
            v.grad(w.data(), g);
            for (int j = 0; j < P; j++) w[j] -= 0.5f * (2.0f / n) * g[j];   // w <- w - lr * (2/N) * g
            if (s % std::max(1, train / 5) == 0) printf("  step %4d:  MSE = %.6f\n", s, mse(d, w));
        }
        double max_dw = 0;
        for (int j = 0; j < P; j++) max_dw = std::max(max_dw, (double)std::fabs(w[j] - d.w_true[j]));
        bool train_ok = max_dw < 0.01;
        ok &= train_ok;
        printf("  max |w - w_true| = %.4f  %s  (noise sigma = 0.1, so the lowest MSE is ~0.01)\n",
               max_dw, train_ok ? "OK" : "FAIL");
    }

    printf("\n%s\n", ok ? "__SUCCESS__" : "__FAILED__");
    return ok ? 0 : 1;
}
