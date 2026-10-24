import importlib.util
import sys
import tempfile
import unittest
import warnings
from pathlib import Path

import benchmark_mlx as bench


def mlx_gpu_ready():
    if sys.platform != "darwin" or importlib.util.find_spec("mlx") is None:
        return False
    try:
        import mlx.core as mx
        return str(mx.default_device()).lower().startswith("device(gpu")
    except Exception:
        return False


def coreml_ready():
    if sys.platform != "darwin":
        return False
    try:
        spec = importlib.util.find_spec("coremltools")
        if spec is None or spec.origin is None:
            return False
        package = Path(spec.origin).parent
        if not all((package / name).is_file() for name in ("libcoremlpython.so", "libmilstoragepython.so")):
            return False
        import coremltools as ct
        return hasattr(ct, "ComputeUnit")
    except Exception:
        return False


@unittest.skipUnless(mlx_gpu_ready(), "requires an available MLX Metal GPU")
class MLXHardwareIntegrationTests(unittest.TestCase):
    def test_gpu_matmul_and_compile_smoke(self):
        result = bench.mlx_bench({"matrix_n": 16, "timing_seconds": {"matrix_aggregate": 0.001}})
        self.assertEqual(result["correctness_status"], "passed")
        self.assertGreater(result["timed_repetitions"], 0)
        self.assertIn("compile", result)

    def test_all_precision_formats_smoke(self):
        cfg = {"precision_matmul_n": 64, "timing_seconds": {"precision_per_format": 0.001}}
        result = bench.precision_throughput_bench(cfg)
        self.assertEqual(result["correctness_status"], "passed")
        self.assertEqual(result["format_count"], 6)
        self.assertGreaterEqual(result["successful_formats"], 3)
        for item in result["formats"]:
            if item["status"] == "ok":
                self.assertEqual(item["correctness_status"], "passed")
                self.assertGreater(item["median_tflops"], 0)

    def test_transformer_prefill_decode_and_gradient_update_smoke(self):
        cfg = {"transformer_seq": 8, "transformer_width": 16, "transformer_layers": 1,
               "timing_seconds": {"transformer_phase": 0.001}}
        result = bench.transformer_bench(cfg)
        self.assertEqual(result["correctness_status"], "passed")
        self.assertGreater(result["parameter_change_l1"], 0)
        self.assertGreater(result["prefill_tokens_per_second"], 0)
        self.assertGreater(result["decode_steps_per_second"], 0)
        self.assertGreater(result["training_tokens_per_second"], 0)

    def test_quantized_linear_smoke(self):
        result = bench.quant_bench({"quant_matrix_n": 64,
                                    "timing_seconds": {"quantization_aggregate": 0.001}})
        self.assertEqual(result["correctness_status"], "passed")
        self.assertTrue(result["output_shape_matches"])
        self.assertTrue(result["outputs_finite"])
        self.assertLessEqual(result["relative_mean_absolute_output_error"], result["relative_mae_tolerance"])


@unittest.skipUnless(coreml_ready(), "requires Core ML Tools with native bindings")
class CoreMLHardwareIntegrationTests(unittest.TestCase):
    def test_generated_model_prediction_and_reference_check(self):
        with tempfile.TemporaryDirectory() as tmp, warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ResourceWarning, message="Implicitly cleaning up")
            path = Path(tmp) / "generated.mlmodel"
            result = bench.ane_bench({}, path)
        self.assertEqual(result["correctness_status"], "passed")
        self.assertTrue(path.name == result["model_path"])
        self.assertFalse(result["ane_execution_verified"])
        self.assertGreater(result["model_file_bytes"], 0)
        self.assertGreater(result["median_seconds"], 0)
