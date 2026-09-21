#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#define CUDA_CHECK(call)                                                \
    do {                                                                \
        cudaError_t error = call;                                       \
        if (error != cudaSuccess) {                                     \
            std::cerr << "CUDA error: "                                 \
                      << cudaGetErrorString(error)                       \
                      << " at " << __FILE__ << ":" << __LINE__           \
                      << std::endl;                                      \
            std::exit(EXIT_FAILURE);                                    \
        }                                                               \
    } while (0)


// SiLU(x) = x * sigmoid(x)
// SwiGLU output = SiLU(x) * gate
__global__ void swiglu_kernel(
    const float* x,
    const float* gate,
    float* output,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        float sigmoid = 1.0f / (1.0f + expf(-x[i]));
        float silu = x[i] * sigmoid;

        output[i] = silu * gate[i];
    }
}


int main(int argc, char** argv) {

    const int N = 1 << 22; // ~4 million elements
    const size_t bytes = N * sizeof(float);

    std::cout << "KernelPilot - Baseline SwiGLU Benchmark\n";
    std::cout << "Elements: " << N << "\n";

    // ----------------------------------
    // Allocate CPU memory
    // ----------------------------------

    std::vector<float> h_x(N);
    std::vector<float> h_gate(N);
    std::vector<float> h_output(N);

    for (int i = 0; i < N; i++) {
        h_x[i] = static_cast<float>(i % 100) / 100.0f;
        h_gate[i] = static_cast<float>((i + 1) % 100) / 100.0f;
    }


    // ----------------------------------
    // Allocate GPU memory
    // ----------------------------------

    float* d_x;
    float* d_gate;
    float* d_output;

    CUDA_CHECK(cudaMalloc(&d_x, bytes));
    CUDA_CHECK(cudaMalloc(&d_gate, bytes));
    CUDA_CHECK(cudaMalloc(&d_output, bytes));


    // ----------------------------------
    // Copy data to GPU
    // ----------------------------------

    CUDA_CHECK(cudaMemcpy(
        d_x,
        h_x.data(),
        bytes,
        cudaMemcpyHostToDevice
    ));

    CUDA_CHECK(cudaMemcpy(
        d_gate,
        h_gate.data(),
        bytes,
        cudaMemcpyHostToDevice
    ));


    // ----------------------------------
    // Kernel configuration
    // ----------------------------------

    int threads_per_block = 256;

    if (argc > 1) {
        threads_per_block = std::atoi(argv[1]);
    }

    const int blocks =
        (N + threads_per_block - 1) / threads_per_block;


    // ----------------------------------
    // Warmup
    // ----------------------------------

    for (int i = 0; i < 10; i++) {
        swiglu_kernel<<<blocks, threads_per_block>>>(
            d_x,
            d_gate,
            d_output,
            N
        );
    }

    CUDA_CHECK(cudaDeviceSynchronize());


    // ----------------------------------
    // Benchmark using CUDA Events
    // ----------------------------------

    cudaEvent_t start;
    cudaEvent_t stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    const int iterations = 100;

    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < iterations; i++) {

        swiglu_kernel<<<blocks, threads_per_block>>>(
            d_x,
            d_gate,
            d_output,
            N
        );

    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0.0f;

    CUDA_CHECK(cudaEventElapsedTime(
        &total_ms,
        start,
        stop
    ));

    float avg_ms = total_ms / iterations;


    // ----------------------------------
    // Check launch errors
    // ----------------------------------

    CUDA_CHECK(cudaGetLastError());


    // ----------------------------------
    // Copy result back
    // ----------------------------------

    CUDA_CHECK(cudaMemcpy(
        h_output.data(),
        d_output,
        bytes,
        cudaMemcpyDeviceToHost
    ));


    std::cout << "\nResults\n";
    std::cout << "-------\n";

    std::cout << "Threads/block: "
              << threads_per_block << "\n";

    std::cout << "Blocks: "
              << blocks << "\n";

    std::cout << "Average kernel latency: "
              << avg_ms
              << " ms\n";


    // ----------------------------------
    // Cleanup
    // ----------------------------------

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_gate));
    CUDA_CHECK(cudaFree(d_output));

    return 0;
}