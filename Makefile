SHELL := /bin/bash
.DEFAULT_GOAL := run

PYTHON_VERSION := 3.14
VENV := .venv314
VENV_PYTHON := $(VENV)/bin/python
VENV_CMAKE := $(VENV)/bin/cmake
DEPS_STAMP := $(VENV)/.benchmark-deps-installed
COREML_SOURCE := /private/tmp/benchmarker-coremltools-9.0
COREML_BUILD := $(COREML_SOURCE)/build
COREML_STAMP := .build/coremltools-py314-bindings-built

RUN_TAG := $(shell date +%Y%m%d-%H%M%S)
RAW ?= results/run-$(RUN_TAG).jsonl
SUMMARY ?= results/summary-$(RUN_TAG).json
HTML ?= results/summary-$(RUN_TAG).html

.PHONY: run setup benchmark report clean check-platform coreml-bindings

run: check-platform $(COREML_STAMP)
	@mkdir -p "$(dir $(RAW))" "$(dir $(SUMMARY))" "$(dir $(HTML))"
	@"$(VENV_PYTHON)" benchmark_mlx.py --all --verbose --output "$(RAW)" --report "$(SUMMARY)" --html "$(HTML)"

setup: check-platform $(COREML_STAMP)

benchmark: run

report: $(DEPS_STAMP)
	@"$(VENV_PYTHON)" report_mlx.py "$(RAW)" --json "$(SUMMARY)" --html "$(HTML)"

check-platform:
	@if [[ "$$(uname -s)" != "Darwin" || "$$(uname -m)" != "arm64" ]]; then \
		echo "This Makefile's full setup expects an Apple Silicon Mac running macOS." >&2; exit 1; \
	fi
	@if ! command -v brew >/dev/null 2>&1; then \
		echo "Homebrew is required to install Python 3.14. Install Homebrew, then run make run again." >&2; exit 1; \
	fi
	@if ! xcode-select -p >/dev/null 2>&1; then \
		echo "Xcode Command Line Tools are required. Run: xcode-select --install" >&2; exit 1; \
	fi

$(VENV_PYTHON):
	@if ! command -v brew >/dev/null 2>&1; then \
		echo "Homebrew is required to install Python 3.14." >&2; exit 1; \
	fi
	@if ! brew list --versions python@$(PYTHON_VERSION) >/dev/null 2>&1; then brew install python@$(PYTHON_VERSION); fi
	@PYTHON314="$$(brew --prefix python@$(PYTHON_VERSION))/bin/python$(PYTHON_VERSION)"; \
	if [[ ! -x "$$PYTHON314" ]]; then echo "Python $(PYTHON_VERSION) was not found at $$PYTHON314" >&2; exit 1; fi; \
	"$$PYTHON314" -m venv "$(VENV)"

$(DEPS_STAMP): requirements.txt $(VENV_PYTHON)
	@"$(VENV_PYTHON)" -m pip install --upgrade pip setuptools cmake
	@"$(VENV_PYTHON)" -m pip install -r requirements.txt
	@touch "$@"

$(COREML_STAMP): $(DEPS_STAMP) Makefile
	@if [[ ! -d "$(COREML_SOURCE)/.git" ]]; then \
		mkdir -p "$(dir $(COREML_SOURCE))"; \
		git clone --depth 1 --branch 9.0 https://github.com/apple/coremltools.git "$(COREML_SOURCE)"; \
	fi
	@PYTHON_INCLUDE="$$("$(VENV_PYTHON)" -c 'import sysconfig; print(sysconfig.get_path("include"))')"; \
	PYTHON_LIBRARY="$$("$(VENV_PYTHON)" -c 'import sysconfig; from pathlib import Path; print(Path(sysconfig.get_config_var("LIBDIR")) / "libpython3.14.dylib")')"; \
	PATH="$(CURDIR)/$(VENV)/bin:$$PATH" "$(VENV_CMAKE)" -S "$(COREML_SOURCE)" -B "$(COREML_BUILD)" \
		-DPYTHON_EXECUTABLE="$(CURDIR)/$(VENV_PYTHON)" \
		-DPYTHON_INCLUDE_DIRS="$$PYTHON_INCLUDE" \
		-DPYTHON_LIBRARY="$$PYTHON_LIBRARY" \
		-DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5
	@PATH="$(CURDIR)/$(VENV)/bin:$$PATH" "$(VENV_CMAKE)" --build "$(COREML_BUILD)" \
		--parallel "$$(/usr/sbin/sysctl -n hw.ncpu 2>/dev/null || echo 4)" --target coremlpython milstoragepython
	@COREMLTOOLS_DIR="$$("$(VENV_PYTHON)" -c 'import importlib.util; from pathlib import Path; spec=importlib.util.find_spec("coremltools"); assert spec and spec.origin; print(Path(spec.origin).parent)')"; \
	cp "$(COREML_SOURCE)/coremltools/libcoremlpython.so" "$(COREML_SOURCE)/coremltools/libmilstoragepython.so" "$$COREMLTOOLS_DIR/"; \
	"$(VENV_PYTHON)" -c 'import coremltools as ct; import coremltools.libcoremlpython, coremltools.libmilstoragepython; print("Core ML Tools", ct.__version__, "native prediction bindings loaded")'
	@mkdir -p "$(dir $@)"
	@touch "$@"

coreml-bindings: $(COREML_STAMP)

clean:
	rm -rf "$(VENV)" .build "$(COREML_SOURCE)"
