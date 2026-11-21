#include "cuda.hpp"
#include "common.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>

namespace {

// Passed to the kernel by value: 17 floats = 68 bytes, small enough that it
// lands in the kernel parameter space (constant bank) instead of global memory.
struct WParams {
    float v[P];
};

constexpr int kBlock = 256;   // threads per block
constexpr int kMaxWarps = 32; // a block has at most 1024 threads = 32 warps

// Sum of one value across the 32 lanes of a warp.
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
    return v;
}

__global__ void grad_kernel(const float* __restrict__ Xs, long long ld,
                            const float* __restrict__ y, long long n,
                            WParams w, float* __restrict__ part) {
    float acc[P];
#pragma unroll
    for (int j = 0; j < P; j++) acc[j] = 0.0f;

    const long long stride = (long long)gridDim.x * blockDim.x;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        const float* col = Xs + i;              // column j lives at col + j*ld
        // Keep the 16 features in registers.
        // Needed twice (prediction and gradient), so each element is read from memory only once.
        float xv[F];
        float pred = w.v[F];                   
#pragma unroll
        for (int j = 0; j < F; j++) {
            xv[j] = __ldg(col + (long long)j * ld);
            pred = fmaf(w.v[j], xv[j], pred);
        }
        const float e = pred - __ldg(y + i);    
#pragma unroll
        for (int j = 0; j < F; j++) acc[j] = fmaf(e, xv[j], acc[j]);
        acc[F] += e;
    }

    __shared__ float sm[P][kMaxWarps];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
#pragma unroll
    for (int j = 0; j < P; j++) {
        const float s = warp_sum(acc[j]);
        if (lane == 0) sm[j][warp] = s;
    }
    __syncthreads();

    const int warps = (blockDim.x + 31) >> 5;
    if (warp == 0) {
#pragma unroll
        for (int j = 0; j < P; j++) {
            float s = (lane < warps) ? sm[j][lane] : 0.0f;
            s = warp_sum(s);
            if (lane == 0) part[(long long)blockIdx.x * P + j] = s;
        }
    }
}

__global__ void reduce_kernel(const float* __restrict__ part, int nblocks,
                              float* __restrict__ g) {
    const int j = threadIdx.x;
    if (j >= P) return;
    float s = 0.0f;
    for (int b = 0; b < nblocks; b++) s += part[(long long)b * P + j];
    g[j] = s;
}

void cuda_check(cudaError_t err, const char* what) {
    if (err != cudaSuccess)
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
}

} 

struct CudaGrad::Impl {
    float* d_Xs = nullptr;
    float* d_y = nullptr;
    float* d_part = nullptr;
    float* d_g = nullptr;
    long long ld = 0;
    long long n = 0;
    int blocks = 0;

    ~Impl() {
        if (d_Xs) cudaFree(d_Xs);
        if (d_y) cudaFree(d_y);
        if (d_part) cudaFree(d_part);
        if (d_g) cudaFree(d_g);
    }
};

bool cuda_available() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess) {
        cudaGetLastError(); 
        return false;
    }
    if (count <= 0) return false;
    int dev = 0;
    cudaDeviceProp prop{};
    if (cudaGetDevice(&dev) != cudaSuccess || cudaGetDeviceProperties(&prop, dev) != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    return prop.major * 10 + prop.minor >= 50;
}

const char* cuda_device_name() {
    static std::string name = "unknown";
    int dev = 0;
    cudaDeviceProp prop{};
    if (cudaGetDevice(&dev) == cudaSuccess && cudaGetDeviceProperties(&prop, dev) == cudaSuccess)
        name = prop.name;
    else
        cudaGetLastError();
    return name.c_str();
}

CudaGrad::CudaGrad(const Dataset& d) : p(new Impl) {
    p->n = d.n;
    p->ld = d.ld;
    if (d.n <= 0) throw std::runtime_error("empty dataset");
    if ((long long)d.Xs.size() != (long long)F * d.ld)
        throw std::runtime_error("dataset has no column-major copy (call make_columns)");
    if ((long long)d.y.size() != d.n) throw std::runtime_error("dataset y has wrong size");

    cuda_check(cudaMalloc(reinterpret_cast<void**>(&p->d_Xs), sizeof(float) * F * p->ld),
               "cudaMalloc Xs");
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&p->d_y), sizeof(float) * p->n),
               "cudaMalloc y");
    cuda_check(cudaMemcpy(p->d_Xs, d.Xs.data(), sizeof(float) * F * p->ld, cudaMemcpyHostToDevice),
               "copy Xs to GPU");
    cuda_check(cudaMemcpy(p->d_y, d.y.data(), sizeof(float) * p->n, cudaMemcpyHostToDevice),
               "copy y to GPU");

    int dev = 0;
    cudaDeviceProp prop{};
    cuda_check(cudaGetDevice(&dev), "cudaGetDevice");
    cuda_check(cudaGetDeviceProperties(&prop, dev), "cudaGetDeviceProperties");
    int per_sm = 0;
    cuda_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, grad_kernel, kBlock, 0),
               "occupancy");
    const long long max_useful = (p->n + kBlock - 1) / kBlock;
    p->blocks = (int)std::min<long long>((long long)prop.multiProcessorCount * std::max(per_sm, 1),
                                         max_useful);
    if (p->blocks < 1) p->blocks = 1;

    cuda_check(cudaMalloc(reinterpret_cast<void**>(&p->d_part), sizeof(float) * P * p->blocks),
               "cudaMalloc part");
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&p->d_g), sizeof(float) * P), "cudaMalloc g");
}

CudaGrad::~CudaGrad() = default;

void CudaGrad::compute(const float* w, float* g) {
    WParams wp;
    std::memcpy(wp.v, w, sizeof wp.v);

    grad_kernel<<<p->blocks, kBlock>>>(p->d_Xs, p->ld, p->d_y, p->n, wp, p->d_part);
    cuda_check(cudaGetLastError(), "grad_kernel launch");

    reduce_kernel<<<1, 32>>>(p->d_part, p->blocks, p->d_g);
    cuda_check(cudaGetLastError(), "reduce_kernel launch");

    cuda_check(cudaMemcpy(g, p->d_g, sizeof(float) * P, cudaMemcpyDeviceToHost),
               "copy gradient to host");
}
