import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import benchmark_mlx as bench
import report_mlx


class BenchmarkMathTests(unittest.TestCase):
    def test_matmul_flop_count(self):
        self.assertEqual(bench.matrix_flops(2, 3, 4), 48)
        self.assertEqual(bench.matrix_flops(768, 768, 768), 2 * 768**3)

    def test_tflops_conversion(self):
        self.assertEqual(bench.tflops(1_000_000_000_000, 2), 0.5)

    def test_report_formatting_uses_units_only_for_timing_fields(self):
        self.assertEqual(report_mlx.fmt(0.03, "mean_absolute_output_error"), "0.030")
        self.assertEqual(report_mlx.fmt(0.00005, "median_seconds"), "50.00 µs")

    def test_sample_statistics(self):
        stats = bench.sample_stats([4, 1, 3, 2])
        self.assertEqual(stats["median_seconds"], 2.5)
        self.assertEqual(stats["min_seconds"], 1)
        self.assertEqual(stats["max_seconds"], 4)
        self.assertEqual(stats["p95_seconds"], 4)
        with self.assertRaises(ValueError):
            bench.sample_stats([])

    def test_profile_configuration_is_bounded(self):
        spec = {"ram_bytes": 24 * 1024**3, "available_ram_bytes_estimate": 8 * 1024**3,
                "physical_cpu_count": 12, "logical_cpu_count": 12}
        quick = bench.sizes(spec, "all", "quick")
        thorough = bench.sizes(spec, "all", "thorough")
        self.assertLessEqual(quick["memory_bytes_per_buffer"], 256 * 1024**2)
        self.assertEqual(quick["precision_matmul_n"], 4096)
        self.assertGreater(thorough["timing_seconds"]["precision_per_format"],
                           quick["timing_seconds"]["precision_per_format"])
        self.assertEqual(quick["memory_threads"], 12)
        constrained = bench.sizes({**spec, "available_ram_bytes_estimate": 512 * 1024**2}, "all")
        self.assertEqual(constrained["precision_matmul_n"], 1024)
        self.assertEqual(constrained["transformer_seq"], 64)

    def test_public_value_scrubs_local_roots(self):
        value = f"at {Path.home()}/private and {Path(__file__).resolve().parents[1]}/file"
        safe = bench.sanitize_public_value(value)
        self.assertNotIn(str(Path.home()), safe)
        self.assertNotIn(str(Path(__file__).resolve().parents[1]), safe)
        self.assertIn("~", safe)


class ReportingTests(unittest.TestCase):
    def test_explicit_skips_are_distinguished(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "skip.jsonl"
            reporter = bench.Reporter(raw, {}, {})
            reporter.run("unsupported", lambda: bench.skip("feature unavailable"))
            reporter.close()
            parsed = report_mlx.summarize_file(raw)
            self.assertEqual(parsed["counts"]["skipped"], 1)
            self.assertEqual(parsed["benchmarks"][0]["execution_status"], "skipped")
            self.assertEqual(parsed["benchmarks"][0]["reason"], "feature unavailable")

    def test_cli_refuses_to_overwrite_raw_jsonl_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "existing.jsonl"
            raw.write_text("preserve this")
            script = Path(bench.__file__)
            proc = subprocess.run([sys.executable, str(script), "--only", "cpu",
                                   "--output", str(raw)], text=True, capture_output=True)
            self.assertEqual(proc.returncode, 2)
            self.assertIn("choose a new --output path or pass --overwrite", proc.stderr)
            self.assertEqual(raw.read_text(), "preserve this")

    def test_jsonl_error_status_and_html_are_safe(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw.jsonl"
            reporter = bench.Reporter(raw, {"machine": str(Path.home())}, {"profile": "quick"})
            reporter.run("broken", lambda: (_ for _ in ()).throw(RuntimeError(str(Path.home() / "secret"))))
            reporter.run("mlx_matmul_throughput_by_precision", lambda: {
                "correctness_status": "passed",
                "formats": [{"format": "FP32", "status": "ok", "correctness_status": "passed",
                             "median_tflops": 1.0, "fastest_observed_tflops": 1.1}],
            })
            reporter.close()

            parsed = report_mlx.summarize_file(raw)
            self.assertEqual(parsed["counts"]["error"], 1)
            self.assertEqual(parsed["benchmarks"][0]["status"], "error")
            self.assertNotIn(str(Path.home()), raw.read_text())
            self.assertEqual(parsed["source"], "raw.jsonl")

            summary_path = Path(tmp) / "summary.json"
            html_path = Path(tmp) / "summary.html"
            with contextlib.redirect_stdout(io.StringIO()):
                report_mlx.write_outputs(parsed, summary_path, html_path)
            self.assertIn("Fine Tuning", html_path.read_text())
            self.assertIn("<svg", html_path.read_text())
            self.assertNotIn(str(Path.home()), summary_path.read_text())

    def test_invalid_jsonl_reports_line_number(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "invalid.jsonl"
            raw.write_text('{"type":"run_start"}\n{broken}\n')
            with self.assertRaisesRegex(ValueError, r"invalid\.jsonl:2"):
                report_mlx.summarize_file(raw)


if __name__ == "__main__":
    unittest.main()
