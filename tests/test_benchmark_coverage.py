import contextlib
import io
import json
import math
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import benchmark_mlx as bench
import report_mlx


class PlatformAndConfigurationTests(unittest.TestCase):
    def test_cmdout_returns_none_on_command_failure(self):
        with mock.patch.object(bench.subprocess, "check_output", side_effect=OSError("missing")):
            self.assertIsNone(bench.cmdout(["unavailable"]))

    def test_available_memory_linux_memavailable(self):
        with mock.patch.object(bench.sys, "platform", "linux"), mock.patch.object(
            bench.Path, "read_text", return_value="MemTotal: 1000 kB\nMemAvailable: 123 kB\n"
        ):
            self.assertEqual(bench.available_memory_bytes(), 123 * 1024)

    def test_available_memory_darwin_sums_reclaimable_pages(self):
        vm = "Pages free: 1.\nPages inactive: 2.\nPages speculative: 3.\nPages purgeable: 4.\n"
        with mock.patch.object(bench.sys, "platform", "darwin"), mock.patch.object(
            bench, "cmdout", side_effect=[vm, "4096"]
        ):
            self.assertEqual(bench.available_memory_bytes(), 10 * 4096)

    def test_available_memory_unknown_when_probe_is_unavailable(self):
        with mock.patch.object(bench.sys, "platform", "darwin"), mock.patch.object(
            bench, "cmdout", return_value=None
        ):
            self.assertIsNone(bench.available_memory_bytes())

    def test_detect_reports_expected_metadata_shape_without_assuming_hardware(self):
        prior_import_error = bench.MLX_IMPORT_ERROR
        try:
            with mock.patch.dict(sys.modules, {"mlx": None}), mock.patch.object(bench, "cmdout", return_value=None), mock.patch.object(
                bench, "available_memory_bytes", return_value=123456
            ):
                found = bench.detect()
        finally:
            bench.MLX_IMPORT_ERROR = prior_import_error
        for key in ("platform", "system", "machine", "logical_cpu_count", "ram_bytes", "mlx", "benchmark_suite_version"):
            self.assertIn(key, found)
        self.assertEqual(found["available_ram_bytes_estimate"], 123456)
        self.assertFalse(found["mlx"]["available"])
        self.assertIsNone(found["gpu_core_count"])

    def test_sizes_for_single_category_and_each_memory_class(self):
        low = {"ram_bytes": 1 * 1024**3, "available_ram_bytes_estimate": 256 * 1024**2, "logical_cpu_count": 4}
        cfg = bench.sizes(low, "cpu", "quick")
        self.assertEqual(cfg["cpu_iterations"], 500_000)
        self.assertEqual(cfg["precision_matmul_n"], 1024)
        self.assertEqual(cfg["duration_scale"], 0.25)
        mid = bench.sizes({**low, "ram_bytes": 8 * 1024**3, "available_ram_bytes_estimate": None}, "all")
        self.assertEqual(mid["precision_matmul_n"], 2048)
        self.assertEqual(mid["transformer_width"], 128)
        high = bench.sizes({**low, "ram_bytes": 24 * 1024**3, "available_ram_bytes_estimate": 4 * 1024**3, "physical_cpu_count": 8}, "all", "thorough")
        self.assertEqual(high["precision_matmul_n"], 4096)
        self.assertEqual(high["memory_threads"], 8)
        self.assertEqual(high["duration_scale"], 3.0)

    def test_size_memory_copy_cap_and_profile_timings(self):
        cfg = bench.sizes({"ram_bytes": 256 * 1024**3, "available_ram_bytes_estimate": 256 * 1024**3,
                           "logical_cpu_count": 100}, "all", "thorough")
        self.assertLessEqual(cfg["memory_bytes_per_buffer"], 256 * 1024**2)
        self.assertLessEqual(cfg["memory_threads"], 16)
        self.assertEqual(cfg["timing_seconds"]["precision_per_format"], 3.0)


class BenchmarkFunctionTests(unittest.TestCase):
    def test_cpu_benchmark_uses_worker_pool_and_checks_repeatable_checksum(self):
        class InlinePool:
            def __init__(self, max_workers):
                self.workers = max_workers
            def __enter__(self): return self
            def __exit__(self, *_): return False
            def map(self, fn, items): return [fn(item) for item in items]
        with mock.patch.object(bench.os, "cpu_count", return_value=2), mock.patch.object(
            bench.concurrent.futures, "ProcessPoolExecutor", InlinePool
        ):
            result = bench.bench_cpu(20, target_s=0)
        self.assertEqual(result["workers"], 2)
        self.assertEqual(result["correctness_status"], "passed")
        self.assertEqual(len(result["samples_seconds"]), 3)
        self.assertGreater(result["operations_per_second"], 0)

    @unittest.skipUnless(__import__("importlib").util.find_spec("numpy"), "NumPy is required")
    def test_memory_copy_benchmark_reports_copy_and_read_write_rates(self):
        result = bench.bench_memory(64, 2, target_s=1e-9)
        self.assertEqual(result["correctness_status"], "passed")
        self.assertEqual(result["checksum"], 64)
        self.assertEqual(result["estimated_bytes_read_plus_written"], result["bytes_moved_estimate"])
        self.assertAlmostEqual(result["estimated_memory_traffic_bytes_per_second"], 2 * result["copy_bytes_per_second"])

    @unittest.skipUnless(__import__("importlib").util.find_spec("numpy"), "NumPy is required")
    def test_numpy_matmul_benchmark_checks_output_and_flop_count(self):
        result = bench.bench_numpy(4, target_s=1e-8)
        self.assertEqual(result["correctness_status"], "passed")
        self.assertEqual(result["estimated_flops_per_call"], 2 * 4**3)
        self.assertEqual(result["checksum"], 4.0)
        self.assertGreater(result["timed_repetitions"], 0)

    def test_mlx_benchmarks_skip_when_mlx_import_failed(self):
        with mock.patch.object(bench, "MLX_IMPORT_ERROR", "ModuleNotFoundError: mlx"):
            cfg = {"matrix_n": 4, "precision_matmul_n": 4, "timing_seconds": {"matrix_aggregate": 0, "precision_per_format": 0}}
            for result in (bench.mlx_bench(cfg), bench.precision_throughput_bench(cfg),
                           bench.transformer_bench(cfg), bench.quant_bench(cfg)):
                self.assertEqual(result["status"], "skipped")
                self.assertIn("MLX unavailable", result["reason"])

    def test_mlx_benchmarks_skip_when_device_is_not_gpu(self):
        mlx = types.ModuleType("mlx")
        mlx.__path__ = []
        core = types.ModuleType("mlx.core")
        core.default_device = lambda: "Device(cpu, 0)"
        mlx.core = core
        nn = types.ModuleType("mlx.nn")
        mlx.nn = nn
        cfg = {"matrix_n": 4, "precision_matmul_n": 4, "transformer_seq": 4,
               "transformer_width": 4, "transformer_layers": 1, "quant_matrix_n": 4,
               "timing_seconds": {"matrix_aggregate": 0, "precision_per_format": 0, "transformer_phase": 0, "quantization_aggregate": 0}}
        with mock.patch.object(bench, "MLX_IMPORT_ERROR", None), mock.patch.dict(sys.modules, {"mlx": mlx, "mlx.core": core, "mlx.nn": nn}):
            for result in (bench.mlx_bench(cfg), bench.precision_throughput_bench(cfg),
                           bench.transformer_bench(cfg), bench.quant_bench(cfg)):
                self.assertEqual(result["status"], "skipped")
                self.assertIn("not the GPU", result["reason"])

    def test_coreml_benchmark_skips_off_macos(self):
        with mock.patch.object(bench.sys, "platform", "linux"):
            result = bench.ane_bench({}, Path("model.mlmodel"))
        self.assertEqual(result["status"], "skipped")
        self.assertIn("requires macOS", result["reason"])

    def test_coreml_benchmark_skips_when_native_package_is_missing(self):
        with mock.patch.object(bench.sys, "platform", "darwin"), mock.patch(
            "importlib.util.find_spec", return_value=None
        ):
            result = bench.ane_bench({}, Path("model.mlmodel"))
        self.assertEqual(result["status"], "skipped")
        self.assertIn("coremltools is not installed", result["reason"])


class ReporterTests(unittest.TestCase):
    def test_reporter_serializes_ok_skipped_failed_memory_and_error_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            reporter = bench.Reporter(path, {"machine": "test"}, {"profile": "quick"})
            reporter.run("good", lambda: {"correctness_status": "passed", "value": 2})
            reporter.run("bad_math", lambda: {"correctness_status": "failed"})
            reporter.run("skip", lambda: bench.skip("not supported"))
            reporter.run("no_memory", lambda: (_ for _ in ()).throw(MemoryError("allocation")))
            reporter.run("crash", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
            reporter.close()
            events = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual([event["sequence"] for event in events], list(range(1, len(events) + 1)))
            results = [event for event in events if event["type"] == "benchmark_result"]
            self.assertEqual([r["status"] for r in results], ["ok", "failed_correctness", "skipped", "skipped", "error"])
            self.assertEqual(results[-1]["execution_status"], "error")
            self.assertIn("RuntimeError: boom", results[-1]["error"])
            self.assertEqual(events[-1]["type"], "run_end")

    def test_reporter_nonfinite_result_is_captured_as_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            reporter = bench.Reporter(path, {}, {})
            reporter.run("nonfinite", lambda: {"value": math.nan})
            reporter.close()
            results = [json.loads(x) for x in path.read_text().splitlines()]
            self.assertEqual(results[-2]["status"], "error")
            self.assertIn("Out of range float", results[-2]["error"])

    def test_cli_dispatch_runs_selected_benchmark_and_builds_reports(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw, summary, html = root / "raw.jsonl", root / "summary.json", root / "report.html"
            spec = {"platform": "test", "system": "Darwin", "machine": "arm64", "processor": "arm",
                    "logical_cpu_count": 2, "physical_cpu_count": 2, "ram_bytes": 8 * 1024**3,
                    "available_ram_bytes_estimate": 4 * 1024**3, "mlx": {"available": False}}
            args = ["benchmark_mlx.py", "--only", "cpu", "--output", str(raw), "--report", str(summary), "--html", str(html)]
            with mock.patch.object(sys, "argv", args), mock.patch.object(bench, "detect", return_value=spec), mock.patch.object(
                bench, "bench_cpu", return_value={"correctness_status": "passed", "workers": 2, "operations_per_second": 9.0}
            ) as cpu_run, contextlib.redirect_stdout(io.StringIO()):
                status = bench.main()
            self.assertEqual(status, 0)
            cpu_run.assert_called_once()
            self.assertTrue(raw.exists() and summary.exists() and html.exists())
            parsed = json.loads(summary.read_text())
            self.assertEqual(parsed["counts"]["ok"], 1)
            self.assertIn("Training", html.read_text())
            self.assertEqual([e["type"] for e in map(json.loads, raw.read_text().splitlines())],
                             ["run_start", "benchmark_start", "benchmark_result", "run_end"])

    def test_cli_skip_flag_records_skip_without_calling_workload(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw.jsonl"
            spec = {"logical_cpu_count": 1, "physical_cpu_count": 1, "ram_bytes": 2 * 1024**3, "mlx": {"available": False}}
            args = ["benchmark_mlx.py", "--only", "memory", "--skip-memory", "--output", str(raw), "--report", str(root / "summary.json")]
            with mock.patch.object(sys, "argv", args), mock.patch.object(bench, "detect", return_value=spec), mock.patch.object(
                bench, "bench_memory"
            ) as memory_run, contextlib.redirect_stdout(io.StringIO()):
                status = bench.main()
            memory_run.assert_not_called()
            self.assertEqual(status, 0)
            result = next(json.loads(line) for line in raw.read_text().splitlines() if '"type": "benchmark_result"' in line)
            self.assertEqual(result["status"], "skipped")
            self.assertEqual(result["reason"], "disabled by --skip-memory")

    def test_cli_rejects_all_and_only_combination(self):
        with mock.patch.object(sys, "argv", ["benchmark_mlx.py", "--all", "--only", "cpu"]), self.assertRaises(SystemExit) as err, contextlib.redirect_stderr(io.StringIO()):
            bench.main()
        self.assertEqual(err.exception.code, 2)


class ParserTests(unittest.TestCase):
    def write_events(self, path, events):
        path.write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")

    def test_summarize_new_schema_counts_statuses_and_sanitizes_paths(self):
        project = str(Path(bench.__file__).resolve().parent)
        events = [
            {"type": "run_start", "run_id": "r1", "timestamp": "start", "machine": {"path": project + "/local"},
             "config": {"profile": "standard"}},
            {"type": "benchmark_result", "name": "ok", "status": "ok", "correctness_status": "passed"},
            {"type": "benchmark_result", "name": "skip", "status": "skipped", "reason": "optional"},
            {"type": "benchmark_result", "name": "error", "status": "error", "error": "failed"},
            {"type": "benchmark_result", "name": "failed", "status": "failed_correctness", "correctness_status": "failed"},
            {"type": "benchmark_result", "name": "unchecked", "status": "ok"},
            {"type": "run_end", "timestamp": "end", "run_duration_seconds": 3.5},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw.jsonl"
            self.write_events(raw, events)
            summary = report_mlx.summarize_file(raw)
        self.assertEqual(summary["source"], "raw.jsonl")
        self.assertEqual(summary["run_id"], "r1")
        self.assertEqual(summary["counts"], {"ok": 2, "skipped": 1, "error": 1, "failed_correctness": 1,
                                               "correctness_passed": 1, "correctness_not_checked": 3})
        self.assertNotIn(project, json.dumps(summary))
        self.assertEqual(summary["machine"]["path"], "./local")
        self.assertEqual(summary["run_duration_seconds"], 3.5)

    def test_summarize_accepts_legacy_rows_blank_lines_and_missing_run_markers(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "legacy.jsonl"
            raw.write_text('\n{"name":"old","status":"ok","type":"benchmark_result"}\n')
            summary = report_mlx.summarize_file(raw)
        self.assertIsNone(summary["run_id"])
        self.assertEqual(summary["counts"]["ok"], 1)
        self.assertEqual(summary["counts"]["correctness_not_checked"], 1)

    def test_summarize_reports_invalid_json_with_source_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "bad.jsonl"
            raw.write_text('{"type":"run_start"}\n{broken}\n')
            with self.assertRaisesRegex(ValueError, r"bad\.jsonl:2: invalid JSON"):
                report_mlx.summarize_file(raw)

    def test_summarize_rejects_missing_input_as_file_error(self):
        with self.assertRaises(FileNotFoundError):
            report_mlx.summarize_file(Path("missing-benchmark-file.jsonl"))

    def test_formatters_cover_time_units_and_json_values(self):
        self.assertEqual(report_mlx.fmt(True), "True")
        self.assertEqual(report_mlx.fmt(0.00005, "duration_s"), "50.00 µs")
        self.assertEqual(report_mlx.fmt(0.05, "median_seconds"), "50.00 ms")
        self.assertEqual(report_mlx.fmt(3.2, "duration_s"), "3.200 s")
        self.assertEqual(report_mlx.fmt(0.0, "duration_s"), "0.000 s")
        self.assertEqual(report_mlx.fmt({"a": 1}), '{"a": 1}')
        self.assertEqual(report_mlx.fmt_seconds(None), "unavailable")
        self.assertEqual(report_mlx.fmt_seconds(0.000001), "1.00 µs")

    def test_svg_chart_handles_empty_zero_and_escapes_labels(self):
        self.assertEqual(report_mlx._svg_bar_chart("empty", [], "x"), "")
        self.assertEqual(report_mlx._svg_bar_chart("zeros", [("a", 0)], "x"), "")
        svg = report_mlx._svg_bar_chart("<chart>", [("<FP16>", 2), ("FP32", 1)], "TFLOPS")
        self.assertIn("&lt;chart&gt;", svg)
        self.assertIn("&lt;FP16&gt;", svg)
        self.assertIn("linear scale", svg)
        self.assertIn("logarithmic scale", report_mlx._svg_bar_chart("log", [("small", 0.1), ("large", 10)], "x", "log"))

    def test_render_text_handles_skips_errors_and_reports_measurement_fields(self):
        summary = {"machine": {"mac_model": "Mac", "processor": "arm", "physical_cpu_count": 8,
                               "logical_cpu_count": 8, "ram_bytes": 8 * 1024**3, "mlx": {"version": "x"}},
                   "counts": {"ok": 1, "skipped": 1, "error": 1, "correctness_passed": 1, "failed_correctness": 0},
                   "benchmarks": [{"name": "skip", "status": "skipped", "reason": "not available"},
                                  {"name": "err", "status": "error", "error": "bad"},
                                  {"name": "ok", "status": "ok", "correctness_status": "passed", "median_seconds": 0.02}],
                   "run_duration_seconds": 1.5}
        text = report_mlx.render_text(summary)
        self.assertIn("Machine: Mac", text)
        self.assertIn("reason: not available", text)
        self.assertIn("error: bad", text)
        self.assertIn("20.00 ms", text)

    def test_write_outputs_produces_valid_json_full_html_and_safe_escaping(self):
        summary = {"schema_version": 2, "source": "input.jsonl", "run_id": "id", "started_at": "now",
                   "run_duration_seconds": 1.0, "benchmark_suite_version": "1.1.0",
                   "machine": {"mac_model": "Mac", "chip_name": "M4", "physical_cpu_count": 8,
                               "logical_cpu_count": 8, "ram_bytes": 8 * 1024**3, "mlx": {"version": "1"}},
                   "config": {"profile": "quick"}, "counts": {"ok": 1, "skipped": 0, "error": 0,
                   "failed_correctness": 0},
                   "benchmarks": [
                       {"name": "transformer_training_prefill_decode", "status": "ok", "correctness_status": "passed",
                        "training_tokens_per_second": 100, "training_steps": 2, "training_median_step_seconds": 0.01,
                        "prefill_tokens_per_second": 200, "decode_steps_per_second": 10, "decode_method": "proxy"},
                       {"name": "mlx_matmul_throughput_by_precision", "status": "ok", "formats": [
                           {"format": "FP32", "status": "ok", "median_tflops": 1.2, "fastest_observed_tflops": 1.3,
                            "correctness_status": "passed"}]},
                       {"name": "coreml_prediction", "status": "ok", "compute_units_requested": "ALL",
                        "warm_prediction_median_seconds": 0.001, "ane_execution_verified": False},
                       {"name": "cpu_multithreaded", "status": "ok", "workers": 8, "operations_per_second": 1,
                        "unit": "sqrt/s"},
                       {"name": "accelerated_cpu_blas_matmul", "status": "ok", "blas_provider": "Accelerate",
                        "matrix_size": [4, 4], "median_gflops": 1},
                       {"name": "metal_mlx_matmul_and_compile", "status": "ok", "device": "GPU",
                        "matrix_size": [4, 4], "median_gflops": 2},
                       {"name": "native_memory_bandwidth", "status": "ok", "threads": 2,
                        "buffer_bytes_each": 1024, "copy_gib_per_second": 1, "estimated_memory_traffic_gib_per_second": 2},
                       {"name": "mlx_4bit_quantization", "status": "ok", "compression_ratio_estimate": 4,
                        "relative_mean_absolute_output_error": 0.1, "inference_median_seconds": 0.001},
                   ]}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            json_path, html_path = root / "nested" / "summary.json", root / "nested" / "summary.html"
            with contextlib.redirect_stdout(io.StringIO()):
                report_mlx.write_outputs(summary, json_path, html_path)
            parsed = json.loads(json_path.read_text())
            page = html_path.read_text()
        self.assertEqual(parsed["run_id"], "id")
        for heading in ("Training", "Running", "Fine Tuning", "Other", "All recorded benchmark details"):
            self.assertIn(f"<h2>{heading}</h2>", page)
        self.assertIn("<svg", page)
        self.assertIn("not verified", page)
        self.assertIn("not a fine-tuning result", page)
        self.assertIn("Accelerate", page)
        self.assertIn("Memory-copy estimate", page)

    def test_html_escapes_untrusted_run_values(self):
        malicious = "</pre><script>alert(1)</script>"
        summary = {"benchmarks": [{"name": malicious, "status": "error", "error": malicious}],
                   "machine": {}, "config": {}, "counts": {"ok": 0, "skipped": 0, "error": 1}}
        with tempfile.TemporaryDirectory() as tmp:
            html_path = Path(tmp) / "report.html"
            with contextlib.redirect_stdout(io.StringIO()):
                report_mlx.write_outputs(summary, Path(tmp) / "report.json", html_path)
            page = html_path.read_text()
        self.assertNotIn("<script>alert(1)</script>", page)
        self.assertIn("&lt;script&gt;", page)

    def test_parser_cli_writes_outputs_and_returns_usage_error_for_bad_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "input.jsonl"
            self.write_events(raw, [{"type": "run_start", "machine": {}, "config": {}},
                                    {"type": "benchmark_result", "name": "x", "status": "ok"}])
            json_out, html_out = root / "out.json", root / "out.html"
            args = ["report_mlx.py", str(raw), "--json", str(json_out), "--html", str(html_out)]
            with mock.patch.object(sys, "argv", args), contextlib.redirect_stdout(io.StringIO()):
                report_mlx.main()
            self.assertEqual(json.loads(json_out.read_text())["counts"]["ok"], 1)
            self.assertTrue(html_out.read_text().startswith("<!doctype html>"))
            bad = root / "bad.jsonl"
            bad.write_text("{broken}\n")
            proc = subprocess.run([sys.executable, str(Path(report_mlx.__file__)), str(bad), "--json", str(root / "bad.json")],
                                 text=True, capture_output=True)
            self.assertEqual(proc.returncode, 2)
            self.assertIn("bad.jsonl:1: invalid JSON", proc.stderr)


if __name__ == "__main__":
    unittest.main()
