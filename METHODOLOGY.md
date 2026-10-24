# Benchmark methodology

This document describes benchmark suite version 1.1.0. JSONL is the source record; JSON and HTML are summaries of those events.

## Timing and statistics

Durations use Python's time.perf_counter(). It is monotonic and high resolution. Timed MLX work includes Python dispatch and an explicit mx.eval() before the timer stops, so lazy device work is included. It is end-to-end call latency, not a GPU kernel-only profile. Warmup and compilation are outside the steady-state samples where the code records them separately. The first compiled MLX call includes tracing/compilation plus execution.

Each applicable workload collects samples until its profile-specific aggregate target is reached, subject to a repetition cap. Reports include median, minimum, maximum, p95, standard deviation, sample count, and aggregate measured duration where applicable. “Fastest observed” means the minimum measured latency converted to a throughput; it is not a theoretical or guaranteed peak.

Profiles scale measurement durations while keeping adaptive workload dimensions bounded:

| Profile | Scale | Intended use |
|---|---:|---|
| quick | 0.25× | Brief functional/performance check |
| standard | 1× | Default comparison run |
| thorough | 3× | More samples, still not a thermal endurance test |

Workloads remain sequential. Core ML model construction and loading are excluded from warm prediction timing. CPU and NumPy/BLAS timers include Python and runtime scheduling overhead.

## Operation counts

For an m×k matrix multiplied by a k×n matrix, the suite uses 2 × m × k × n floating-point operations, counting one multiply and one add per matrix-product term. For square n×n matrices this is 2n³. Throughput is this nominal count divided by median elapsed time. This convention is appropriate for dense multiply-accumulate work; quantized kernels use it as a dense-equivalent count and are labeled accordingly.

FP32, FP16, and BF16 measurements use dense MLX matmul with the input arrays cast to the named dtype. FP32 reduced-precision mode is disabled before MLX import. MXFP8, MXFP4, and affine Q4 quantize the right-hand weight matrix and use FP32 activations. They do not measure general native FP8/FP4 arithmetic.

## Workloads

### CPU process-pool math

Each worker evaluates a fixed number of sqrt expressions. The timed sample includes ProcessPoolExecutor.map, scheduling, result transfer, and Python work. The reported rate is square-root iterations per second across all workers, not CPU FLOPS. Pool startup is warmed before calibration and timed samples.

### NumPy / BLAS matrix multiplication

NumPy multiplies two n×n FP32 all-ones matrices. A warmup runs before timed samples. Every timed call creates and computes a fresh result. Correctness requires every output to equal n and be finite. The BLAS provider is read from NumPy's build configuration; thread counts are not forced because provider controls vary. On the captured reference environment, NumPy reported Apple Accelerate. This benchmark measures that installed accelerated BLAS path; it cannot separate ordinary CPU instructions from framework-level AMX or other acceleration.

### Memory copy

Two uint8 arrays of B bytes each are initialized; each timed pass copies B source bytes into the destination. copy_gib_per_second = B / median_pass_seconds / 2³⁰. Estimated read-plus-write traffic is 2B per pass, and its reported rate is twice the copy rate. The same buffers are reused across passes, the timing includes thread-pool dispatch, and cache reuse may affect the result. This estimates traffic for a CPU copy workload; it is not a direct physical DRAM or GPU-accessible unified-memory bandwidth measurement.

### MLX / Metal matrix multiplication

MLX uses seeded random FP32 inputs and requires the default MLX device to be a GPU. Inputs are evaluated before timing. Each iteration constructs a new matmul result and calls mx.eval() inside the timed interval. A small portion of the output is checked against a NumPy reference. The compiled run reports the first lazy compiled call separately from synchronized warm calls.

### Precision throughput

For each format, the suite warms the path, then repeatedly constructs and synchronizes a new matmul. Matrix inputs are reused; output arrays are recomputed. A full FP32 reference is computed outside the timed interval. Correctness checks output shape, finite values, and normalized RMSE against format-specific tolerances recorded in JSONL. Throughput uses median latency; the minimum latency is separately labeled fastest observed.

FP8, FP4, and Q4 use quantized weights with FP32 activations. Their dense-equivalent operation count is provided for workload comparison only. It should not be read as standard floating-point TFLOPS.

### Synthetic transformer

The model has adaptive sequence length and width, two layers, random weights, and batch size one. Prefill times a full sequence with self-attention. The decode proxy repeatedly applies the block to one hidden-state token and evaluates each step. It does not append tokens to a growing context and has no KV cache; its rate is reported as sequential step throughput, not representative LLM token generation.

Training uses a random input and target, mean-squared error, mx.value_and_grad, and SGD with learning rate 1e-4. One warmup update occurs before timing. Timed token throughput is input_sequence_length × batch_size / median_step_seconds; batch size is one. The output records parameter change, finite loss/gradients, and step counts. The synthetic loss is not a quality metric and the workload is not fine-tuning a pretrained model.

### MLX quantization

A seeded synthetic FP32 Linear layer is compared with its MLX 4-bit QuantizedLinear form. Reported storage counts quantized weights, scales, and biases when present. Compression is the original dense weight byte estimate divided by the quantized parameter/metadata byte estimate. Output error is measured against the original layer, and JSONL records the relative MAE tolerance and finite/shape checks.

### Core ML prediction

The suite creates a small 128-element input model with a 128×128 dense layer and ReLU, validates predictions against a NumPy reference, and requests ComputeUnit.ALL. Model creation, saving, load/compile, and two warmup predictions are excluded from the seven measured prediction calls. Core ML's public Python prediction API does not provide reliable per-operation placement confirmation; ANE execution remains unverified.

## Correctness and status

Each benchmark result separates execution status from correctness status. Numerical checks are workload-specific: exact all-ones BLAS output, finite checks and references for MLX matmuls, actual parameter changes and finite gradients for transformer training, output error for quantization, and NumPy-reference validation for Core ML. A completed workload can still be marked failed_correctness. Unsupported features are skipped; memory allocation failure is recorded as a resource skip; uncaught execution errors are error.

## Comparisons and limitations

Compare only runs with matching suite version, workload name, profile, matrix dimensions, precision, sequence/batch sizes, worker counts, quantization method, and timing fields. Hardware model, OS, power/thermal conditions, background load, BLAS provider/threading, and MLX version also matter. A matching unit alone does not make two results comparable.

The default standard profile uses bounded measurements; it does not measure sustained thermal performance. The synthetic transformer is not a proxy for Qwen3 or other real models. No real-model benchmark is currently included. GPU core count, active power mode, charger state, and thermal state are not reliably collected and are reported as unavailable.
