# Repository Guidelines

## Project Structure & Module Organization

This repository is a small Python benchmark suite. `benchmark_mlx.py` runs bounded CPU, memory, MLX/Metal, transformer, quantization, and optional Core ML workloads and writes JSONL results. `report_mlx.py` parses JSONL into terminal, JSON, and optional standalone HTML summaries. `tests/` contains portable unit/parser tests and hardware smoke tests. `results/` holds example/generated run artifacts; avoid committing personal machine output. `README.md` covers setup and usage, while `METHODOLOGY.md` explains measurement limits and formulas.

## Build, Test, and Development Commands

- `make setup` prepares the Apple Silicon Python environment and Core ML native bindings.
- `make run` executes the full standard benchmark profile and writes timestamped results under `results/`.
- `python benchmark_mlx.py --help` shows direct CLI options; use `--profile quick` or `--only cpu memory` for focused runs.
- `python report_mlx.py results/run.jsonl --json results/summary.json --html results/summary.html` regenerates reports from an existing run.
- `python -m unittest discover -s tests -v` runs unit, parser, and available hardware smoke tests. CI also compiles both scripts with `py_compile` on Python 3.12/Linux.

## Coding Style & Naming Conventions

Use Python 3 syntax compatible with the CI runtime (Python 3.12); keep runtime dependencies minimal. Follow existing 4-space indentation, `snake_case` functions/modules, and `unittest.TestCase` class names in `tests/test_*.py`. Keep benchmark workloads bounded and sequential, record explicit skip/error reasons, and preserve result sanitization and correctness checks. No formatter or linter is configured; keep changes consistent with nearby code.

## Testing Guidelines

Add portable behavior tests to the relevant test module and name methods `test_<behavior>`. Hardware-dependent MLX/Core ML tests should remain small smoke checks, skip cleanly when backends are unavailable, and never assert performance thresholds. Run the full unittest command before submitting; on a non-Apple host, hardware skips are expected.

## Commit & Pull Request Guidelines

Recent history uses short, imperative, conventional prefixes such as `feat:`, `test:`, `docs:`, and `ci:`. Use a concise subject describing the change. PRs should explain the measurement or reporting impact, list validation performed and platform/backend availability, and include sanitized sample output when useful. Never attach unsanitized machine results or credentials; see the Privacy section in `README.md`.

## Measurement and Data Safety

Treat benchmark values as workload-specific observations, not hardware peak claims. Update `METHODOLOGY.md` and relevant README explanations when changing formulas, timing, adaptive sizing, or interpretation. Do not add telemetry or upload behavior; the suite is designed to run locally.
