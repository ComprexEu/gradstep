#pragma once
#include <memory>

struct Dataset;

bool cuda_available();

const char* cuda_device_name();

class CudaGrad {
public:
    explicit CudaGrad(const Dataset& d);
    ~CudaGrad();

    CudaGrad(const CudaGrad&) = delete;
    CudaGrad& operator=(const CudaGrad&) = delete;

    void compute(const float* w, float* g);

private:
    struct Impl;
    std::unique_ptr<Impl> p;
};
