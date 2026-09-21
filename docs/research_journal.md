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


## Entry 2 — Block Size Sweep Results

I performed a controlled block-size sweep while keeping the SwiGLU kernel, input size, data type, and iteration count unchanged.

Results:

| Threads/block | Blocks | Average latency (ms) |
| 64            | 65536  | 0.459644             |
| 128           | 32768  | 0.456745             |
| 256           | 16384  | 0.456115             |
| 512           | 8192   | 0.458967             |
| 1024          | 4096   | 0.471188             |

The lowest measured latency was observed at 256 threads per block.

Block sizes from 64 to 512 produced similar performance, while 1024 threads per block showed a noticeable slowdown.

No causal conclusion is made yet.

The next step is to profile these configurations using NVIDIA profiling tools to determine whether the slowdown is related to occupancy, register pressure, memory throughput, warp scheduling, or another architectural factor.