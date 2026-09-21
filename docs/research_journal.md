## Entry 1 — First Baseline Result

The first CUDA SwiGLU benchmark executed successfully on the local NVIDIA GTX 1650.

Configuration:

- Elements: 4,194,304
- Threads per block: 256
- Blocks: 16,384
- Iterations: 100
- Optimization level: -O3
- Target architecture: sm_75

Measured average kernel latency:

0.456902 ms

This value is treated only as a baseline measurement.

No conclusion is made yet about the primary performance bottleneck.
The next experiment will vary thread-block size while keeping the workload
constant in order to observe how launch configuration affects performance.