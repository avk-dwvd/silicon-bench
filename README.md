# Apple Silicon Benchmarker

Apple Silicon Benchmarker runs bounded, sequential CPU, memory-copy, MLX/Metal, synthetic transformer, quantization, and optional Core ML measurements. It writes machine-readable JSONL plus human-readable JSON and standalone HTML reports. It does not upload results.

## Run from a fresh clone

On an Apple Silicon Mac, install Homebrew and Xcode Command Line Tools first. Then clone the repo and run `make run`:

```sh
git clone https://github.com/avk-dwvd/silicon-bench.git
cd benchmarker
make run
```

`make run` installs Homebrew Python 3.14 if needed, creates `.venv314`, installs the Python requirements and build tools, compiles Core ML Tools' native prediction bindings for that exact Python, then runs the full standard profile. Each run gets timestamped output paths so existing measurements are not overwritten:

- `results/run-<timestamp>.jsonl`: detailed raw benchmark events
- `results/summary-<timestamp>.json`: parsed human-readable summary
- `results/summary-<timestamp>.html`: standalone HTML summary

The first run takes longer because Core ML Tools and its native bindings are compiled locally. Its source/build tree is kept under `/private/tmp` so compiler linker flags work even when the repository path contains spaces. Later `make run` calls reuse the environment and build. PyTorch is not needed. Core ML is asked to use all compute units, but its public prediction API does not confirm that operations ran on the ANE.

To change output paths, pass Make variables, for example `make run RAW=results/m4pro.jsonl SUMMARY=results/m4pro.json HTML=results/m4pro.html`. The direct CLI supports `--profile quick|standard|thorough`, `--only cpu memory precision`, and `--skip-memory`; `--help` lists all choices. It refuses to replace an existing raw JSONL file unless `--overwrite` is supplied. `make setup` prepares the environment and Core ML bindings without running benchmarks.

The Makefile expects Apple Silicon macOS, Homebrew, Xcode Command Line Tools, Git, and network access for dependency/source downloads. Python requirements are in `requirements.txt`. No PyTorch, model download, administrator access, account connection, or telemetry is required.

To parse an existing raw file independently:

```sh
python report_mlx.py results/run.jsonl --json results/summary.json --html results/summary.html
```

The benchmark program runs each selected benchmark to completion before starting the next. It flushes each JSONL event immediately. A benchmark failure is recorded and later benchmarks continue; execution status and numerical correctness are separate fields. The raw file contains machine/software metadata, adaptive configuration, timings/statistics, correctness checks, skip reasons, and bounded tracebacks. Known local home and repository path prefixes are scrubbed from exported data.

## Included measurements

- CPU-bound square-root loops in a process pool; throughput includes Python execution, process scheduling, and IPC.
- A multithreaded NumPy byte-copy workload. It reports copy throughput separately from estimated read-plus-write traffic. Buffers are reused, so cache effects may influence the rate; it is not a physical DRAM bandwidth measurement.
- Accelerated CPU/BLAS matrix multiplication through NumPy. The active BLAS provider is recorded (NumPy 2.5.4 used Apple Accelerate in the reference environment); thread policy can remain runtime-controlled. This measures the installed BLAS path, not isolated scalar CPU-core FLOPS.
- MLX GPU matrix multiplication, explicitly synchronized with `mx.eval`; `mx.compile` records a lazy cold first call and warm timings. Timings include Python call overhead and device synchronization, rather than kernel-only duration.
- Observed MLX matrix throughput at FP32, FP16, BF16, MXFP8, MXFP4, and affine Q4. Each path is warmed and timed on a large adaptive matrix. FP32 reduced-precision math is disabled. The report gives median, p95, min, max, and standard deviation of synchronized end-to-end samples. The fastest sample is labeled as an observation, never as theoretical hardware peak. FP8, FP4, and Q4 use quantized weights with FP32 activations; their dense-equivalent operation counts are not conventional floating-point TFLOPS.
- A small synthetic transformer: full-sequence prefill, a sequential single-token recurrent proxy, and gradient plus SGD update steps. **Decode does not grow an attention context or use a KV cache**, and it is not real LLM decoding. The model uses random weights and is not downloaded; these are workload indicators, not production LLM performance or model quality.
- MLX 4-bit layer quantization with a small inference and output-error measurement. It uses `QuantizedLinear.from_linear` and records the quantized layer type and its weight plus scale/bias storage.
- Core ML generation and prediction of a tiny 128×128 dense-plus-ReLU model. It asks Core ML for `ComputeUnit.ALL`, reports end-to-end prediction latency, and checks output against a NumPy reference. ANE placement remains **unverified** because the public prediction call does not reveal per-operation placement.

## Adaptive sizing and interpretation

`--all` detects OS, architecture, core counts, installed memory, and a best-effort available-memory estimate. Unavailable chip/GPU-core, power, or thermal information is recorded as unavailable rather than inferred. It chooses conservative allocation and model sizes; each memory buffer is capped at 256 MiB. These are safe starting settings, not a thermal guarantee. Close other demanding workloads and keep the Mac ventilated for repeatable results. Activity, power mode, temperature, OS version, package builds, and BLAS thread settings can change results.

MLX/Metal hardware benchmarks require a compatible Apple Silicon Mac and a working MLX install. Other platforms may skip unsupported backends. METHODOLOGY.md documents formulas, timing, validation, cache effects, and comparison limits.

## Privacy

The program runs locally and does not send results anywhere. JSONL, JSON, and HTML outputs omit serial numbers, hardware UUIDs, and environment secrets. Known home and repository path prefixes are scrubbed; outputs identify the raw file by its basename. Review local benchmark files before publishing them.

## Troubleshooting

- If make run reports missing Xcode tools, run xcode-select --install, finish the installation, then retry.
- The first run needs network access to install Python packages and fetch Core ML Tools source. If a download fails, rerun make run.
- Core ML native modules are built for the active Python 3.14 venv. Do not copy them between Python versions or architectures.
- If memory is constrained, use --skip-memory and/or select fewer categories with --only. The suite records unsupported or unavailable measurements with a reason.
- Repeated runs use new timestamped raw and report files. Direct CLI use refuses to overwrite an existing raw file unless --overwrite is explicitly selected.

## Tests and contributions

Run portable tests from the repository root:

    python -m unittest discover -s tests

Most unit tests do not require MLX or a GPU. Hardware integration smoke tests automatically run the MLX GPU and Core ML prediction paths when those backends are available, and skip when they are not. They use small workloads and do not assert performance scores. Please include the tested macOS, Python, MLX version, benchmark profile, and sanitized JSONL when reporting a measurement issue. Do not submit unsanitized machine output.

## License

This project is licensed under the GNU General Public License, version 3. See LICENSE.
