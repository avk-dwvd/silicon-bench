# Contributing

Contributions are welcome. Keep changes focused and consistent with the existing project.

## General contributions

1. Fork the repository on GitHub and clone your fork:

   ```sh
   git clone https://github.com/<your-user>/silicon-bench.git
   cd silicon-bench
   git remote add upstream https://github.com/avk-dwvd/silicon-bench.git
   ```

2. Create a branch for your change:

   ```sh
   git switch -c describe-your-change
   ```

3. Make the change and validate it. For Python changes, run the test suite from the repository root:

   ```sh
   python -m unittest discover -s tests -v
   ```

4. Commit your work and open a pull request against the project’s default branch. Explain the change and include the validation performed. For measurement issues, include the macOS, Python and MLX versions, profile, and sanitized JSONL; do not submit unsanitized machine output.

There is no separate formatter or linter configured. Python code uses four-space indentation and `snake_case`; keep workloads bounded and sequential, record explicit skip or error reasons, and preserve sanitization and correctness checks. Hardware tests should remain small, skip cleanly when a backend is unavailable, and avoid performance thresholds.

## Documentation contributions

Documentation source files are in `docs/`; navigation and site settings are in `mkdocs.yml`. Install the built-in MkDocs tool locally using Homebrew:

```sh
brew install mkdocs
```

Preview and validate changes with:

```sh
mkdocs serve
mkdocs build --strict
```

Submit documentation changes through pull requests. GitHub Actions validates documentation changes on pull requests and publishes successful changes merged into the default branch to GitHub Pages.

## Benchmark contributions

Benchmark implementations live in `benchmark_mlx.py`; result parsing and presentation live in `report_mlx.py`, with portable and hardware smoke tests under `tests/`. Keep new workloads bounded and sequential, validate numerical results where applicable, and record unsupported or failed work with clear reasons. Update [methodology](methodology.md) and the README when changing formulas, timing, adaptive sizing, or result interpretation. Measurements should remain local; do not add telemetry or upload behavior.
