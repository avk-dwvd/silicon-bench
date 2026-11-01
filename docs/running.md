# Running Benchmarks

## Run the suite

From the repository root, run the standard suite with:

```sh
make run
```

This prepares the environment if needed, then runs all supported categories sequentially. Each invocation uses timestamped paths under `results/`, so earlier measurements are not overwritten. To prepare without benchmarking, use `make setup`.

For direct control, use the Python environment created by the Makefile:

```sh
.venv314/bin/python benchmark_mlx.py --profile quick --only cpu memory
```

Run `python benchmark_mlx.py --help` (or `.venv314/bin/python benchmark_mlx.py --help`) for the current CLI help.

## Workloads and options

`--only` accepts one or more of these categories:

- `cpu`: square-root work in a process pool.
- `memory`: multithreaded NumPy byte copying.
- `cpu-matmul`: NumPy/BLAS matrix multiplication.
- `mlx-matmul`: MLX/Metal matrix multiplication and compile timing.
- `precision`: MLX dense and quantized precision paths.
- `transformer`: a small synthetic transformer’s prefill, recurrent decode proxy, and training/update work.
- `quantization`: MLX 4-bit layer quantization and output-error measurement.
- `coreml`: generated Core ML model prediction.

Options supported by `benchmark_mlx.py`:

| Option | Purpose |
| --- | --- |
| `--all` | Run every supported benchmark (the default). Cannot be combined with `--only`. |
| `--only CATEGORY [CATEGORY ...]` | Select categories listed above. |
| `--profile quick\|standard\|thorough` | Set measurement duration; default is `standard`. |
| `--output PATH` | Write detailed JSONL; default `benchmark_raw.jsonl`. Existing files are protected unless `--overwrite` is used. |
| `--report PATH` | Write JSON summary; default `benchmark_report.json`. |
| `--html PATH` | Also write a standalone HTML summary. |
| `--skip-ane` | Skip the Core ML prediction attempt. |
| `--skip-memory` | Skip the memory-copy benchmark. |
| `--overwrite` | Replace an existing raw JSONL output. |
| `--verbose` | Print progress as benchmarks start. |
| `--version` | Print the benchmark suite version. |

Examples:

```sh
# Short CPU and memory run
.venv314/bin/python benchmark_mlx.py --profile quick --only cpu memory

# Standard run excluding MLX and Core ML categories
.venv314/bin/python benchmark_mlx.py --only cpu memory cpu-matmul

# Save to explicitly named files
.venv314/bin/python benchmark_mlx.py --only cpu --output results/cpu.jsonl \
  --report results/cpu-summary.json --html results/cpu-summary.html

# Summarize an existing raw run independently
.venv314/bin/python report_mlx.py results/run.jsonl \
  --json results/summary.json --html results/summary.html
```

The Makefile supports output overrides too, for example:

```sh
make run RAW=results/m4pro.jsonl SUMMARY=results/m4pro.json HTML=results/m4pro.html
```

## Results and interpretation

Raw JSONL records machine and software metadata, configuration, per-benchmark timings and statistics, correctness results, skip reasons, and bounded tracebacks. JSON and optional HTML are summaries of those events. A benchmark error is recorded and later benchmarks continue; execution status and numerical correctness are separate. The command returns a failure status if any benchmark has an error or failed correctness.

Treat measurements as workload-specific observations. They include framework, Python, scheduling, and synchronization overhead as described in the [methodology](methodology.md). They are not hardware peak ratings.

## Limitations and troubleshooting

- MLX/Metal workloads need Apple Silicon with a working MLX installation; unsupported backends may be skipped elsewhere.
- The transformer uses random weights; decode is a recurrent single-token proxy without growing attention context or a KV cache. It is not real LLM decoding or a model-quality test.
- Core ML requests `ComputeUnit.ALL`, but per-operation ANE placement remains unverified.
- Memory buffers are reused and may benefit from cache effects; reported traffic is not a physical DRAM bandwidth measurement.
- If memory is constrained, use `--skip-memory` or select fewer categories with `--only`.
- Use a new `--output` path for repeated direct CLI runs, or deliberately pass `--overwrite`.
