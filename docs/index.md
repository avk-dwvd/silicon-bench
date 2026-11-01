# Silicon Bench

Silicon Bench is a local, bounded benchmark suite for Apple Silicon. It runs workloads sequentially and produces detailed JSONL data plus JSON and optional standalone HTML summaries.

The suite measures CPU process-pool math, memory copying, accelerated CPU matrix multiplication, MLX/Metal matrix and precision throughput, synthetic transformer work, MLX 4-bit quantization, and optional Core ML prediction. These measurements help compare workload-specific behavior across runs; they are not theoretical hardware peak claims.

For a quick start on an Apple Silicon Mac with Homebrew and Xcode Command Line Tools installed:

```sh
git clone https://github.com/avk-dwvd/silicon-bench.git
cd silicon-bench
make run
```

Choose a guide to get started:

- [Setup](setup.md)
- [Running Benchmarks](running.md)
- [Methodology](methodology.md)
- [Contributing](contributing.md)

[View the source on GitHub](https://github.com/avk-dwvd/silicon-bench).
