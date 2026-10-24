# Apple Silicon and MLX benchmark suite

This project runs a bounded, sequential set of benchmarks and writes a detailed JSON Lines file that can be retained as raw data or summarized with the standalone parser. It is designed to run on Apple Silicon Macs, but can also be launched elsewhere: unsupported backends are recorded with a `skipped` status and a reason.

## Run

Use Python 3.10 or newer. On Apple Silicon, install dependencies and run:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python benchmark_mlx.py --all --output results/run.jsonl --report results/summary.json --html results/summary.html
```

To parse an existing raw file independently:

```sh
python report_mlx.py results/run.jsonl --json results/summary.json --html results/summary.html
```

The benchmark program runs each benchmark to completion before starting the next. It flushes each JSONL event immediately. A benchmark failure is recorded and later benchmarks continue. The raw file contains machine and software metadata, adaptive configuration, start/result events, durations, metrics, skip reasons, and bounded tracebacks for errors.

## Included measurements

- CPU-bound math across a bounded process pool.
- A native multithreaded memory triad kernel on macOS. The bandwidth is an estimated read-plus-write traffic rate, not a vendor-certified memory test.
- CPU matrix multiplication through NumPy/its installed BLAS.
- MLX matrix multiplication and `mx.compile` cold/warm call timings. MLX does not provide a portable compile-only timer, so compilation timing includes the first call and execution.
- A small synthetic transformer: forward prefill, autoregressive-shaped single-token decode, and one gradient plus SGD update step. It uses random weights and does not download a model; these rates are workload indicators, not model quality or production LLM performance.
- MLX 4-bit layer quantization with a small inference and output-error measurement.
- Optional Core ML generation and prediction of a tiny model. It asks Core ML for `ComputeUnit.ALL` and reports latency, while explicitly marking ANE placement as **unverified**. The public prediction call does not prove that any operation executed on the ANE. To compile this optional model, install `coremltools` and `torch` separately.

## Adaptive sizing and interpretation

`--all` detects OS, architecture, core counts, installed memory, and whether MLX imports. It chooses conservative allocation and model sizes based on available RAM/core counts. These are safe starting settings, not a thermal guarantee. Close other demanding workloads and keep the Mac ventilated for repeatable results. Activity, power mode, temperature, OS version, package builds, and BLAS thread settings can change results. The suite does not claim to detect unified-memory bandwidth specs, GPU core count, or ANE use from a successful Core ML prediction.

MLX/Metal hardware benchmarks require a compatible Apple Silicon Mac and a working MLX install. On other systems they are recorded as skipped. This code has not been run as a performance benchmark on an M4 Pro; results must be collected on the target Mac.
