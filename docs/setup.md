# Setup

## Supported platform

The project’s full `make setup` and `make run` workflow expects an Apple Silicon Mac running macOS. It requires Homebrew, Xcode Command Line Tools, Git, and network access to download dependencies and Core ML Tools source. The benchmark script itself can run on other systems; unsupported MLX or Core ML workloads are skipped when their backends are unavailable.

## Install and prepare

1. Install [Homebrew](https://brew.sh/) and Xcode Command Line Tools. If the latter are missing, run:

   ```sh
   xcode-select --install
   ```

2. Clone the repository:

   ```sh
   git clone https://github.com/avk-dwvd/silicon-bench.git
   cd silicon-bench
   ```

3. Prepare the benchmark environment:

   ```sh
   make setup
   ```

   The Makefile installs Homebrew Python 3.14 if needed, creates `.venv314`, installs packages from the repository’s `requirements.txt` (NumPy, MLX, and Core ML Tools), and builds Core ML Tools’ native prediction bindings for that Python version. This initial build can take time. Its source and build tree are kept under `/private/tmp`.

4. Run a short benchmark as a functional check:

   ```sh
   .venv314/bin/python benchmark_mlx.py --profile quick --only cpu
   ```

   This creates `benchmark_raw.jsonl` and `benchmark_report.json` in the current directory. Choose a different `--output` path if that raw file already exists.

No PyTorch installation, model download, administrator access, account connection, or telemetry is required. Core ML Tools’ public prediction API does not confirm that operations ran on the ANE.

## Troubleshooting

- If setup reports missing Xcode tools, run `xcode-select --install`, finish installation, then retry.
- Setup needs network access to install Python packages and fetch the Core ML Tools 9.0 source. Retry `make setup` if a download fails.
- Native Core ML modules are built for Python 3.14 and the current architecture; do not copy them between Python versions or architectures.
- The Makefile stops on non-Apple-Silicon or non-macOS systems. You can use the Python script directly there, but hardware-specific backends may be skipped.
