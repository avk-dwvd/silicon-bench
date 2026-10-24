#!/usr/bin/env python3
"""Sequential Apple Silicon / MLX benchmark suite. Results are measurements, not claims."""
from __future__ import annotations
import argparse, concurrent.futures, datetime as dt, json, math, os, platform, statistics, subprocess, sys, time, traceback, uuid
from pathlib import Path
from typing import Any, Callable

VERSION = "1.1.0"
MLX_IMPORT_ERROR: str | None = None
# Keep the FP32 benchmark on full precision; MLX can use reduced precision otherwise.
os.environ["MLX_ENABLE_TF32"] = "0"

def cmdout(args: list[str]) -> str | None:
    try: return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL, timeout=4).strip()
    except Exception: return None

def matrix_flops(m: int, k: int, n: int) -> int:
    """Conventional multiply-plus-add count for an m×k by k×n matmul."""
    return 2 * int(m) * int(k) * int(n)

def tflops(flops: int, seconds: float) -> float:
    return float(flops) / float(seconds) / 1e12

def sample_stats(samples: list[float]) -> dict[str, float]:
    if not samples: raise ValueError("at least one timing sample is required")
    ordered = sorted(float(x) for x in samples)
    p95_index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return {"median_seconds": statistics.median(ordered), "min_seconds": ordered[0],
            "max_seconds": ordered[-1], "p95_seconds": ordered[p95_index],
            "stdev_seconds": statistics.pstdev(ordered) if len(ordered) > 1 else 0.0}

def available_memory_bytes() -> int | None:
    """Best-effort estimate of immediately free/reclaimable memory; never requires privileges."""
    if sys.platform == "darwin":
        text = cmdout(["vm_stat"])
        page_size = int(cmdout(["sysctl", "-n", "vm.pagesize"]) or 0)
        if text and page_size:
            pages = 0
            for label in ("Pages free", "Pages inactive", "Pages speculative", "Pages purgeable"):
                match = __import__("re").search(rf"^{label}:\s+([\d.]+)", text, __import__("re").M)
                if match: pages += int(float(match.group(1).replace(".", "")))
            if pages: return pages * page_size
    elif sys.platform.startswith("linux"):
        try:
            for line in Path("/proc/meminfo").read_text().splitlines():
                if line.startswith("MemAvailable:"): return int(line.split()[1]) * 1024
        except OSError: pass
    return None

def sanitize_public_value(value: Any) -> Any:
    """Remove known local path roots from values written to shareable files."""
    if isinstance(value, dict): return {str(k): sanitize_public_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [sanitize_public_value(v) for v in value]
    if isinstance(value, Path): value = str(value)
    if isinstance(value, str):
        project_root = str(Path(__file__).resolve().parent)
        home = str(Path.home())
        return value.replace(project_root, ".").replace(home, "~")
    return value

def detect() -> dict[str, Any]:
    global MLX_IMPORT_ERROR
    mem = None
    try:
        if sys.platform == "darwin": mem = int(cmdout(["sysctl", "-n", "hw.memsize"]) or 0)
        else:
            for line in Path("/proc/meminfo").read_text().splitlines():
                if line.startswith("MemTotal:"): mem = int(line.split()[1])*1024; break
    except Exception: pass
    logical = os.cpu_count() or 1
    physical = None
    if sys.platform == "darwin":
        try: physical = int(cmdout(["sysctl", "-n", "hw.physicalcpu"]) or 0) or None
        except Exception: pass
    try:
        import mlx.core as mx
        import importlib.metadata
        mlx = {"available": True, "version": importlib.metadata.version("mlx"), "default_device": str(mx.default_device())}
    except Exception as e:
        MLX_IMPORT_ERROR=f"{type(e).__name__}: {e}"
        try:
            import importlib.metadata
            version=importlib.metadata.version("mlx")
        except Exception: version=None
        mlx = {"available": False, "version":version, "reason": f"MLX import/device check failed: {MLX_IMPORT_ERROR}"}
    return {"platform": platform.platform(), "system": platform.system(), "machine": platform.machine(), "processor": platform.processor() or None,
            "logical_cpu_count": logical, "physical_cpu_count": physical, "ram_bytes": mem,
            "available_ram_bytes_estimate": available_memory_bytes(),
            "mac_model": cmdout(["sysctl", "-n", "hw.model"]) if sys.platform == "darwin" else None,
            "chip_name": cmdout(["sysctl", "-n", "machdep.cpu.brand_string"]) if sys.platform == "darwin" else None,
            "gpu_core_count": None, "power_mode": None, "thermal_state": None,
            "mlx": mlx, "mlx_enable_tf32":os.environ.get("MLX_ENABLE_TF32"), "python": sys.version, "benchmark_suite_version": VERSION}

def sizes(spec: dict[str,Any], requested: str, profile: str = "standard") -> dict[str,Any]:
    ram=spec.get("ram_bytes") or 8*1024**3
    available=spec.get("available_ram_bytes_estimate")
    constrained_ram=min(ram,available) if available else ram
    # Keep matrix/model dimensions stable for a given installed-memory class.
    # Use transient available memory for buffer sizing and preflight instead.
    sizing_ram=ram
    cores=spec.get("physical_cpu_count") or spec["logical_cpu_count"]
    scale={"quick":0.25,"standard":1.0,"thorough":3.0}[profile]
    # Restrict allocations to a small fraction of RAM; --all chooses predictable, laptop-safe workloads.
    cap=max(16*1024**2,min(256*1024**2,constrained_ram//64))
    matrix_cap=max(16*1024**2,min(256*1024**2,ram//64))
    if constrained_ram<2*1024**3:
        sizing_ram=min(sizing_ram,constrained_ram)
    return {"cpu_iterations": 2_000_000 if requested=="all" else 500_000,
            "memory_bytes_per_buffer":cap, "memory_threads":max(1,min(cores,16)),
            "matrix_n":max(128,min(768,int((matrix_cap//16)**0.5))),
            "transformer_seq":128 if sizing_ram>=16*1024**3 else 64,
            "transformer_width":256 if sizing_ram>=16*1024**3 else 128,
            "transformer_layers":2, "quant_matrix_n":256 if sizing_ram>=16*1024**3 else 128,
            "precision_matmul_n":4096 if sizing_ram>=16*1024**3 else 2048 if sizing_ram>=2*1024**3 else 1024,
            "profile":profile,"duration_scale":scale,
            "timing_seconds":{"cpu_sample":0.35*scale,"memory_aggregate":0.75*scale,
                "matrix_aggregate":1.0*scale,"precision_per_format":1.0*scale,
                "transformer_phase":0.5*scale,"quantization_aggregate":0.5*scale},
            "estimated_precision_working_set_bytes":(4096 if sizing_ram>=16*1024**3 else 2048 if sizing_ram>=2*1024**3 else 1024)**2*4*5,
            "policy":"stable matrix/model sizes by installed-memory class; copy buffers use best-effort available memory; large allocation is preflighted; each copy buffer capped at 256 MiB"}

class Reporter:
    def __init__(self,path:Path,spec:dict[str,Any],config:dict[str,Any]):
        self.f=path.open("w",encoding="utf-8"); self.seq=0; self.started=time.perf_counter()
        self.run_id=uuid.uuid4().hex
        self.write({"type":"run_start","run_id":self.run_id,"timestamp":dt.datetime.now(dt.timezone.utc).isoformat(),"machine":spec,"config":config})
    def write(self,x):
        self.seq+=1; x={"schema_version":2,"sequence":self.seq,**x}
        x=sanitize_public_value(x)
        self.f.write(json.dumps(x,default=str,allow_nan=False)+"\n"); self.f.flush()
    def run(self,name:str,fn:Callable[[],dict[str,Any]]):
        start=time.perf_counter(); self.write({"type":"benchmark_start","name":name,"timestamp":dt.datetime.now(dt.timezone.utc).isoformat()})
        try:
            r=fn(); status=r.pop("status","ok"); correctness=r.pop("correctness_status","not_checked")
            if correctness=="failed" and status=="ok": status="failed_correctness"
            execution="completed" if status in {"ok","failed_correctness"} else "skipped" if status=="skipped" else "error"
            self.write({"type":"benchmark_result","name":name,"status":status,"execution_status":execution,
                "correctness_status":correctness,"duration_s":time.perf_counter()-start,**r})
        except MemoryError as e:
            self.write({"type":"benchmark_result","name":name,"status":"skipped","execution_status":"skipped",
                "correctness_status":"not_checked","duration_s":time.perf_counter()-start,
                "reason":f"insufficient memory for this workload: {e}"})
        except Exception as e:
            self.write({"type":"benchmark_result","name":name,"status":"error","execution_status":"error",
                "correctness_status":"not_checked","duration_s":time.perf_counter()-start,
                "error":f"{type(e).__name__}: {e}","traceback":traceback.format_exc(limit=5)})
    def close(self):
        self.write({"type":"run_end","run_duration_seconds":time.perf_counter()-self.started,
            "timestamp":dt.datetime.now(dt.timezone.utc).isoformat()}); self.f.close()

def skip(reason): return {"status":"skipped","reason":reason}

def _cpu_work(k):
    import math
    s=0.0
    for i in range(k): s+=math.sqrt((i%10000)+1.0)
    return s

def bench_cpu(n,target_s=0.35):
    # This measures Python math plus process-pool scheduling/IPC end to end.
    workers=max(1,min(os.cpu_count() or 1,16)); each=max(1,n//workers)
    samples=[]; checks=[]; max_total=100_000_000
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        # First call pays worker startup; subsequent calibration excludes that one-time startup.
        list(pool.map(_cpu_work,[each]*workers))
        for _ in range(4):
            t=time.perf_counter(); list(pool.map(_cpu_work,[each]*workers)); elapsed=time.perf_counter()-t
            if elapsed >= target_s*0.75 or each*workers >= max_total: break
            each=min(max_total//workers,max(each+1,int(each*target_s/max(elapsed,0.01))))
        for _ in range(3):
            t=time.perf_counter(); sums=list(pool.map(_cpu_work,[each]*workers)); samples.append(time.perf_counter()-t); checks.append(sum(sums))
    stats=sample_stats(samples); total_iterations=each*workers
    correct=all(math.isfinite(x) for x in checks) and len(set(checks))==1 and checks[0]>0
    return {"workers":workers,"iterations_per_sample":total_iterations,"samples_seconds":samples,**stats,
        "checksum":checks[-1],"operations_per_second":total_iterations/stats["median_seconds"],
        "unit":"sqrt iterations/s","calibration_target_seconds":target_s,
        "timing_method":"end-to-end ProcessPoolExecutor map, including scheduling and IPC",
        "correctness_status":"passed" if correct else "failed"}

def bench_memory(byte_count,threads,target_s=0.75):
    try: import numpy as np
    except Exception as e: return skip(f"NumPy unavailable for memory bandwidth benchmark: {e}")
    threads=max(1,min(int(threads),os.cpu_count() or 1)); elements=max(threads,int(byte_count))
    source=np.ones(elements,dtype=np.uint8); destination=np.zeros(elements,dtype=np.uint8)
    ranges=[(elements*i//threads,elements*(i+1)//threads) for i in range(threads)]
    def copy_range(bounds):
        lo,hi=bounds; np.copyto(destination[lo:hi],source[lo:hi])
    samples=[]; aggregate=0.0; max_passes=10000
    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as pool:
        list(pool.map(copy_range,ranges))
        while aggregate<target_s and len(samples)<max_passes:
            t=time.perf_counter(); list(pool.map(copy_range,ranges)); elapsed=time.perf_counter()-t
            samples.append(elapsed); aggregate+=elapsed
    stats=sample_stats(samples); copy_bytes=elements*len(samples); moved=copy_bytes*2
    checksum=int(destination.sum(dtype=np.uint64))
    sample_ok=all(int(destination[i])==1 for i in (0,elements//2,elements-1))
    copy_rate=elements/stats["median_seconds"]
    correct=checksum==elements and sample_ok
    return {"threads":threads,"buffer_bytes_each":elements,"buffer_bytes_total":elements*2,
        "timed_passes":len(samples),"aggregate_timed_seconds":aggregate,**stats,
        "median_seconds_per_pass":stats["median_seconds"],"copy_bytes_per_second":copy_rate,
        "copy_gib_per_second":copy_rate/1024**3,"estimated_bytes_read_plus_written":moved,
        "estimated_memory_traffic_bytes_per_second":copy_rate*2,
        "estimated_memory_traffic_gib_per_second":copy_rate*2/1024**3,
        "bytes_moved_estimate":moved,"bandwidth_bytes_per_second":copy_rate*2,
        "aggregate_bandwidth_bytes_per_second":moved/aggregate,
        "bandwidth_gib_per_second":copy_rate*2/1024**3,"checksum":checksum,
        "method":"end-to-end concurrent NumPy uint8 copies; copy rate counts one byte per element, estimated traffic counts one read plus one write; buffers reused so cache effects remain",
        "minimum_timed_seconds":target_s,"correctness_status":"passed" if correct else "failed"}

def bench_numpy(n,target_s=1.0):
    try: import numpy as np
    except Exception as e: return skip(f"NumPy unavailable: {e}")
    a=np.ones((n,n),dtype=np.float32); b=np.ones((n,n),dtype=np.float32)
    # This is BLAS-backed matmul; NumPy links the concrete provider reported below.
    c=a@b; repetitions=0; samples=[]; total=0.0; target=1.0
    target=target_s
    while total<target and repetitions<5000:
        t=time.perf_counter(); c=a@b; elapsed=time.perf_counter()-t
        samples.append(elapsed); total+=elapsed; repetitions+=1
    stats=sample_stats(samples); flops=matrix_flops(n,n,n)
    cfg=getattr(np.__config__,"CONFIG",{})
    blas=cfg.get("Build Dependencies",{}).get("blas",{})
    provider=blas.get("name","unknown") if isinstance(blas,dict) else "unknown"
    correct=bool(np.isfinite(c).all() and np.all(c==n))
    return {"matrix_size":[n,n],"timed_repetitions":repetitions,"aggregate_timed_seconds":total,**stats,
        "median_seconds_per_call":stats["median_seconds"],"p95_seconds_per_call":stats["p95_seconds"],
        "estimated_flops_per_call":flops,"median_gflops":flops/stats["median_seconds"]/1e9,
        "checksum":float(c[0,0]),"numpy_version":np.__version__,"blas_provider":provider,
        "blas_thread_policy":"runtime-dependent; not pinned",
        "backend":"NumPy/BLAS matrix multiplication; CPU-vs-accelerator execution details are provider/runtime dependent",
        "correctness_status":"passed" if correct else "failed"}

def mlx_bench(config):
    if MLX_IMPORT_ERROR: return skip(f"MLX unavailable: {MLX_IMPORT_ERROR}")
    try: import mlx.core as mx
    except Exception as e: return skip(f"MLX unavailable: {type(e).__name__}: {e}")
    if not str(mx.default_device()).lower().startswith("device(gpu"):
        return skip(f"MLX default device is {mx.default_device()}, not the GPU")
    mx.random.seed(0)
    n=config["matrix_n"]; a=mx.random.normal((n,n)); b=mx.random.normal((n,n)); mx.eval(a,b)
    c=a@b; mx.eval(c); samples=[]; total=0.0
    target=config["timing_seconds"]["matrix_aggregate"]
    while total<target and len(samples)<20000:
        t=time.perf_counter(); c=a@b; mx.eval(c); elapsed=time.perf_counter()-t; samples.append(elapsed); total+=elapsed
    stats=sample_stats(samples); flops=matrix_flops(n,n,n)
    try:
        import numpy as np
        ref=np.asarray(a[:min(16,n),:]) @ np.asarray(b)
        got=np.asarray(c[:min(16,n),:])
        correct=bool(np.isfinite(got).all() and np.allclose(got,ref,rtol=1e-4,atol=1e-4))
    except Exception as e: correct=False; validation_error=f"{type(e).__name__}: {e}"
    else: validation_error=None
    info={"matrix_size":[n,n],"timed_repetitions":len(samples),"aggregate_timed_seconds":total,**stats,
        "median_seconds_per_call":stats["median_seconds"],"estimated_flops_per_call":flops,
        "median_gflops":flops/stats["median_seconds"]/1e9,"device":str(mx.default_device()),
        "checksum":float(mx.sum(c).item()),"warmup_calls":1,
        "correctness_status":"passed" if correct else "failed","validation_method":"CPU NumPy reference for first 16 output rows",
        "validation_error":validation_error,"precision":"MLX default dtype"}
    # Compilation latency is isolated from warmed execution.
    try:
        f=mx.compile(lambda x,y:x@y); t=time.perf_counter(); z=f(a,b); mx.eval(z); compile_s=time.perf_counter()-t
        samples=[]; total=0.0
        target=config["timing_seconds"]["matrix_aggregate"]/2
        while total<target and len(samples)<20000:
            t=time.perf_counter(); z=f(a,b); mx.eval(z); elapsed=time.perf_counter()-t; samples.append(elapsed); total+=elapsed
        cstats=sample_stats(samples)
        info["compile"]={"cold_first_call_seconds":compile_s,"warm_timed_repetitions":len(samples),
            "warm_aggregate_seconds":total,"warm_median_seconds_per_call":cstats["median_seconds"],
            "warm_min_seconds_per_call":cstats["min_seconds"],"warm_max_seconds_per_call":cstats["max_seconds"],
            "warm_p95_seconds_per_call":cstats["p95_seconds"],
            "note":"MLX compilation is lazy; cold first call includes tracing/compilation and execution; warm timings are end-to-end Python calls with explicit GPU synchronization"}
    except Exception as e: info["compile"]={"status":"skipped","reason":f"MLX compile unavailable: {type(e).__name__}: {e}"}
    return info

def precision_throughput_bench(cfg):
    """Measure synchronized observed dense and quantized MLX GPU matmul throughput."""
    if MLX_IMPORT_ERROR: return skip(f"MLX unavailable: {MLX_IMPORT_ERROR}")
    try: import mlx.core as mx
    except Exception as e: return skip(f"MLX unavailable: {type(e).__name__}: {e}")
    if not str(mx.default_device()).lower().startswith("device(gpu"):
        return skip(f"MLX default device is {mx.default_device()}, not the GPU")
    mx.random.seed(0)
    n=cfg["precision_matmul_n"]; target=cfg["timing_seconds"]["precision_per_format"]; max_repetitions=2000
    estimated_working_set=n*n*4*5
    available=cfg.get("available_ram_bytes_estimate")
    if available is not None and available<estimated_working_set*2:
        return skip(f"estimated available memory ({available} bytes) is below the conservative working-set budget ({estimated_working_set*2} bytes)")
    a=mx.random.normal((n,n)).astype(mx.float32); b=mx.random.normal((n,n)).astype(mx.float32); mx.eval(a,b)
    flops=matrix_flops(n,n,n)
    reference=mx.matmul(a,b); mx.eval(reference)
    results=[]

    tolerances={"FP32":1e-5,"FP16":0.04,"BF16":0.12,"FP8 / MXFP8":0.30,"FP4 / MXFP4":0.60,"Q4 / affine":0.60}
    def measure(name,method,fn,input_dtype,weight_format=None):
        try:
            warm=fn(); mx.eval(warm)
            samples=[]; aggregate=0.0
            while aggregate<target and len(samples)<max_repetitions:
                t=time.perf_counter(); result=fn(); mx.eval(result); elapsed=time.perf_counter()-t
                samples.append(elapsed); aggregate+=elapsed
            if not samples: raise RuntimeError("no timed iterations completed")
            stats=sample_stats(samples)
            finite=bool(mx.all(mx.isfinite(result)).item())
            shape_ok=tuple(result.shape)==(n,n)
            nrmse=float((mx.sqrt(mx.mean((result.astype(mx.float32)-reference)*(result.astype(mx.float32)-reference))) /
                mx.maximum(mx.sqrt(mx.mean(reference*reference)),1e-12)).item())
            tolerance=tolerances[name]
            correct=finite and shape_ok and math.isfinite(nrmse) and nrmse<=tolerance
            results.append({"format":name,"status":"ok","correctness_status":"passed" if correct else "failed",
                "method":method,"input_dtype":input_dtype,"weight_format":weight_format,
                "matrix_shape":[n,n],"timed_repetitions":len(samples),"aggregate_timed_seconds":aggregate,
                **stats,"median_seconds_per_matmul":stats["median_seconds"],
                "fastest_observed_tflops":tflops(flops,stats["min_seconds"]),
                "median_tflops":tflops(flops,stats["median_seconds"]),
                "validation":{"finite":finite,"shape_matches":shape_ok,"normalized_rmse_vs_fp32":nrmse,
                    "relative_rmse_tolerance":tolerance},
                "nominal_flops_per_matmul":flops,"timing_target_seconds":target,
                "metric_note":"2 operations per multiply-accumulate; quantized formats are dense-equivalent throughput, not conventional IEEE floating-point TFLOPS" if weight_format else "2 operations per multiply-accumulate; observed sustained matrix throughput, not theoretical peak"})
        except Exception as e:
            results.append({"format":name,"status":"skipped","reason":f"{type(e).__name__}: {e}"})

    for name,dtype in (("FP32",mx.float32),("FP16",mx.float16),("BF16",mx.bfloat16)):
        try:
            lhs=a.astype(dtype); rhs=b.astype(dtype); mx.eval(lhs,rhs)
            measure(name,"dense MLX matmul",lambda lhs=lhs,rhs=rhs:mx.matmul(lhs,rhs),str(dtype))
        except Exception as e: results.append({"format":name,"status":"skipped","reason":f"{type(e).__name__}: {e}"})

    # MLX's MX formats are block-scaled quantized weights multiplied by FP32 activations.
    for name,mode,bits,group in (("FP8 / MXFP8","mxfp8",8,32),("FP4 / MXFP4","mxfp4",4,32),("Q4 / affine","affine",4,64)):
        try:
            packed,scales,*bias_list=mx.quantize(b,group_size=group,bits=bits,mode=mode)
            biases=bias_list[0] if bias_list else None
            mx.eval(packed,scales)
            if biases is not None: mx.eval(biases)
            measure(name,f"mx.quantized_matmul mode={mode}, bits={bits}, group_size={group}",
                lambda packed=packed,scales=scales,biases=biases,mode=mode,bits=bits,group=group:mx.quantized_matmul(a,packed,scales,biases,transpose=False,group_size=group,bits=bits,mode=mode),
                str(a.dtype),f"{bits}-bit {mode} weights; FP32 activations")
        except Exception as e: results.append({"format":name,"status":"skipped","reason":f"{type(e).__name__}: {e}"})

    passed=sum(r.get("status")=="ok" and r.get("correctness_status")=="passed" for r in results)
    if not passed: return {"status":"skipped","reason":"all requested precision paths were unsupported or failed","matrix_shape":[n,n],"formats":results}
    correct=all(r.get("status")!="ok" or r.get("correctness_status")=="passed" for r in results)
    return {"status":"ok" if correct else "failed_correctness","correctness_status":"passed" if correct else "failed",
        "device":str(mx.default_device()),"matrix_shape":[n,n],"target_seconds_per_format":target,
        "fp32_reduced_precision_disabled":os.environ.get("MLX_ENABLE_TF32")=="0","formats":results,
        "successful_formats":passed,"format_count":len(results),
        "note":"fastest_observed_tflops is a sample statistic, not hardware peak. Dense matmuls use the reported dtype; MXFP8/MXFP4/Q4 use quantized weights with FP32 activations and dense-equivalent operation counts."}

def transformer_bench(cfg):
    if MLX_IMPORT_ERROR: return skip(f"MLX unavailable: {MLX_IMPORT_ERROR}")
    try: import mlx.core as mx
    except Exception as e: return skip(f"MLX unavailable: {type(e).__name__}: {e}")
    if not str(mx.default_device()).lower().startswith("device(gpu"):
        return skip(f"MLX default device is {mx.default_device()}, not the GPU")
    mx.random.seed(1)
    seq,w,layers=cfg["transformer_seq"],cfg["transformer_width"],cfg["transformer_layers"]
    # Small synthetic decoder-only transformer; avoids downloads and reports synthetic workload.
    try:
        rng=mx.random
        weights=[(rng.normal((w,w))/w**0.5,rng.normal((w,w))/w**0.5,rng.normal((w,w))/w**0.5,rng.normal((w,w))/w**0.5) for _ in range(layers)]
        def block(x):
            for q,k,v,o in weights:
                a=x@q; b=x@k; c=x@v
                attn=mx.softmax((a@mx.swapaxes(b,-1,-2))/w**0.5,axis=-1)@c
                x=x+(attn@o)
            return x
        tokens=mx.random.normal((1,seq,w)); prefill_samples=[]
        for _ in range(1): y=block(tokens); mx.eval(y)
        prefill_total=0.0; phase_target=cfg["timing_seconds"]["transformer_phase"]
        while prefill_total<phase_target and len(prefill_samples)<20000:
            t=time.perf_counter(); y=block(tokens); mx.eval(y); elapsed=time.perf_counter()-t; prefill_samples.append(elapsed); prefill_total+=elapsed
        prefill_stats=sample_stats(prefill_samples)
        # Sequential one-token recurrent proxy; no KV cache and no growing attention context.
        steps=min(32,seq); decode_samples=[]; decode_total=0.0
        while decode_total<phase_target and len(decode_samples)<2000:
            x=tokens[:,:1,:]; t=time.perf_counter()
            for _ in range(steps): x=block(x); mx.eval(x)
            elapsed=time.perf_counter()-t; decode_samples.append(elapsed); decode_total+=elapsed
        decode_stats=sample_stats(decode_samples)
        # A real gradient/update pass on the same synthetic transformer weights, with random regression targets.
        train_x=mx.random.normal((1,min(seq,32),w)); target=mx.random.normal(train_x.shape); trainable=[z for tup in weights for z in tup]
        initial_trainable=list(trainable); mx.eval(initial_trainable)
        def loss_fn(params,x,y):
            # Explicit functional forward keeps this benchmark independent of downloaded models.
            h=x
            for i in range(layers):
                q,k,v,o=params[i*4:i*4+4]; a=h@q; b=h@k; c=h@v
                h=h+((mx.softmax((a@mx.swapaxes(b,-1,-2))/w**0.5,axis=-1)@c)@o)
            return mx.mean((h-y)*(h-y))
        grad_fn=mx.value_and_grad(loss_fn); training_samples=[]; training_total=0.0
        for _ in range(1):
            loss,grads=grad_fn(trainable,train_x,target); mx.eval(loss,grads); trainable=[p-1e-4*g for p,g in zip(trainable,grads)]; mx.eval(trainable)
        while training_total<phase_target and len(training_samples)<2000:
            t=time.perf_counter(); loss,grads=grad_fn(trainable,train_x,target); mx.eval(loss,grads); trainable=[p-1e-4*g for p,g in zip(trainable,grads)]; mx.eval(trainable); elapsed=time.perf_counter()-t; training_samples.append(elapsed); training_total+=elapsed
        training_stats=sample_stats(training_samples)
        parameter_change=sum(float(mx.sum(mx.abs(a-b)).item()) for a,b in zip(initial_trainable,trainable))
        finite=bool(mx.all(mx.isfinite(y)).item() and mx.all(mx.isfinite(x)).item() and
            mx.all(mx.isfinite(loss)).item() and all(bool(mx.all(mx.isfinite(g)).item()) for g in grads))
        correct=finite and parameter_change>0 and math.isfinite(float(loss.item()))
        return {"correctness_status":"passed" if correct else "failed",
            "model":"synthetic decoder transformer; random weights, no downloads",
            "sequence_length":seq,"width":w,"layers":layers,"batch_size":1,"precision":"MLX default (FP32 tensors)",
            "prefill_median_seconds":prefill_stats["median_seconds"],"prefill_samples":len(prefill_samples),
            "prefill_aggregate_seconds":prefill_total,**{f"prefill_{k}":v for k,v in prefill_stats.items()},
            "prefill_tokens_per_second":seq/prefill_stats["median_seconds"],
            "decode_steps_per_sample":steps,"decode_method":"sequential recurrent single-token forward; no KV cache or growing context",
            "decode_median_seconds":decode_stats["median_seconds"],"decode_samples":len(decode_samples),
            "decode_aggregate_seconds":decode_total,**{f"decode_{k}":v for k,v in decode_stats.items()},
            "decode_steps_per_second":steps/decode_stats["median_seconds"],
            "training_median_step_seconds":training_stats["median_seconds"],"training_steps":len(training_samples),
            "training_aggregate_seconds":training_total,**{f"training_{k}":v for k,v in training_stats.items()},
            "training_tokens_per_second":train_x.shape[1]/training_stats["median_seconds"],
            "training_tokens_per_step":int(train_x.shape[1]),"optimizer":"SGD","learning_rate":1e-4,
            "training_loss":float(loss.item()),"parameter_change_l1":parameter_change,
            "checksum":float(mx.sum(y).item()),
            "note":"Synthetic random-weight transformer. Training includes forward, loss, gradients, and SGD parameter updates. Prefill is full-sequence attention. Decode is only a recurrent single-token proxy without KV cache/context growth. No production model or quality metric."}
    except MemoryError as e: return skip(f"insufficient memory for synthetic transformer: {e}")
    except Exception as e: return {"status":"error","error":f"{type(e).__name__}: {e}"}

def quant_bench(cfg):
    if MLX_IMPORT_ERROR: return skip(f"MLX unavailable: {MLX_IMPORT_ERROR}")
    try: import mlx.core as mx; import mlx.nn as nn
    except Exception as e: return skip(f"MLX quantization unavailable: {type(e).__name__}: {e}")
    if not str(mx.default_device()).lower().startswith("device(gpu"):
        return skip(f"MLX default device is {mx.default_device()}, not the GPU")
    mx.random.seed(2)
    n=cfg["quant_matrix_n"]
    try:
        layer=nn.Linear(n,n,bias=False); x=mx.random.normal((1,n)); mx.eval(layer.weight,x)
        before=layer(x); mx.eval(before)
        # MLX Dtype intentionally has no itemsize attribute; infer common scalar widths by dtype name.
        dtype_name=str(layer.weight.dtype).lower(); scalar_bytes=2 if any(k in dtype_name for k in ("float16","bfloat16")) else 8 if "64" in dtype_name else 4
        original_bytes=layer.weight.size*scalar_bytes
        quantized=nn.QuantizedLinear.from_linear(layer,group_size=64,bits=4)
        after=quantized(x); mx.eval(after); samples=[]; total=0.0
        target=cfg["timing_seconds"]["quantization_aggregate"]
        while total<target and len(samples)<20000:
            t=time.perf_counter(); after=quantized(x); mx.eval(after); elapsed=time.perf_counter()-t; samples.append(elapsed); total+=elapsed
        stats=sample_stats(samples); elapsed=stats["median_seconds"]
        err=float(mx.mean(mx.abs(after-before)).item())
        def array_bytes(a):
            d=str(a.dtype).lower(); item=2 if any(k in d for k in ("float16","bfloat16")) else 8 if "64" in d else 4 if "32" in d else 1
            return int(a.size*item)
        qbytes=array_bytes(quantized.weight)+array_bytes(quantized.scales)+(array_bytes(quantized.biases) if quantized.biases is not None else 0)
        finite=bool(mx.all(mx.isfinite(before)).item() and mx.all(mx.isfinite(after)).item())
        shape_ok=tuple(before.shape)==tuple(after.shape)==(1,n)
        rel_mae=err/max(float(mx.mean(mx.abs(before)).item()),1e-12)
        correct=finite and shape_ok and math.isfinite(rel_mae) and rel_mae<=0.25
        return {"correctness_status":"passed" if correct else "failed",
            "bits":4,"group_size":64,"matrix_size":[n,n],"dense_weight_bytes_estimate":original_bytes,
            "quantized_weight_and_metadata_bytes_estimate":qbytes,"compression_ratio_estimate":original_bytes/qbytes,
            "inference_median_seconds":elapsed,"inference_timed_repetitions":len(samples),
            "inference_aggregate_seconds":total,**stats,"mean_absolute_output_error":err,
            "relative_mean_absolute_output_error":rel_mae,"relative_mae_tolerance":0.25,
            "output_shape_matches":shape_ok,"outputs_finite":finite,"checksum":float(mx.sum(after).item()),
            "quantized_layer_class":type(quantized).__name__,
            "note":"Uses MLX QuantizedLinear.from_linear on a synthetic FP32 layer; reports output error vs original layer, including weight/scale/bias storage; end-to-end Python calls with GPU synchronization"}
    except Exception as e: return skip(f"MLX quantization API unsupported: {type(e).__name__}: {e}")

def ane_bench(cfg,model_path:Path):
    if sys.platform!="darwin": return skip("Core ML / ANE benchmark requires macOS")
    # Source installs on newer Python versions may omit Core ML's native bindings.
    # Check for the bindings themselves so a locally compiled Python 3.14 install works.
    try:
        import importlib.util
        package_spec=importlib.util.find_spec("coremltools")
        if package_spec is None or package_spec.origin is None:
            return skip("coremltools is not installed")
        package_dir=Path(package_spec.origin).parent
        missing=[name for name in ("libcoremlpython.so","libmilstoragepython.so") if not (package_dir/name).is_file()]
        if missing:
            return skip("Core ML Tools native prediction bindings are missing ("+", ".join(missing)+"); compile the bindings for this Python environment as described in README.md")
    except Exception as e:
        return skip(f"could not inspect Core ML Tools native bindings: {type(e).__name__}: {e}")
    try:
        import contextlib, io
        # Some incomplete Core ML Tools installs print native-extension warnings during import.
        with contextlib.redirect_stderr(io.StringIO()): import coremltools as ct
    except Exception as e: return skip(f"coremltools unavailable: {type(e).__name__}: {e}")
    try:
        import numpy as np
        from coremltools.models import datatypes
        from coremltools.models.neural_network import NeuralNetworkBuilder
        # Build a tiny Core ML model directly. PyTorch is not needed for model generation.
        rng=np.random.default_rng(7); width=128
        builder=NeuralNetworkBuilder([("input",datatypes.Array(width))],[ ("output",datatypes.Array(width)) ],use_float_arraytype=True)
        weights=rng.normal(0,0.05,(width,width)).astype(np.float32); bias=np.zeros(width,dtype=np.float32)
        builder.add_inner_product(name="dense",W=weights,b=bias,input_channels=width,output_channels=width,has_bias=True,input_name="input",output_name="linear")
        builder.add_activation(name="relu",non_linearity="RELU",input_name="linear",output_name="output")
        ml=ct.models.MLModel(builder.spec,compute_units=ct.ComputeUnit.ALL)
        ml.save(str(model_path)); ml.predict({"input":np.zeros(width,dtype=np.float32)})
        # Core ML's public Python API requests allowed compute units but does not expose proof of per-op ANE placement.
        sample=np.ones(width,dtype=np.float32); timings=[]
        for _ in range(2): ml.predict({"input":sample})
        for _ in range(7):
            t=time.perf_counter(); pred=ml.predict({"input":sample}); timings.append(time.perf_counter()-t)
        stats=sample_stats(timings); actual=np.asarray(pred["output"])
        # Core ML inner-product weights are stored output-major: y = W @ x + b.
        expected=np.maximum(weights @ sample + bias,0.0)
        correct=actual.shape==(width,) and bool(np.isfinite(actual).all()) and bool(np.allclose(actual,expected,rtol=1e-4,atol=1e-4))
        return {"correctness_status":"passed" if correct else "failed",
            "model_path":model_path.name,"model_type":"generated Core ML neural-network MLP; dense 128×128 plus ReLU",
            "input_shape":[width],"compute_units_requested":"ALL (CPU, GPU, ANE allowed)",
            "ane_execution_verified":False,
            "verification":"Core ML Python prediction API provides no per-operation ANE placement confirmation; requested ALL does not prove ANE use",
            "warm_prediction_median_seconds":stats["median_seconds"],"warm_prediction_samples_seconds":timings,
            **stats,"predicted_output_keys":list(pred.keys()),"output_shape":list(actual.shape),
            "model_file_bytes":model_path.stat().st_size,
            "timing_method":"end-to-end Core ML Python predict call; model loading and first predictions excluded",
            "validation_method":"output compared with NumPy dense-plus-ReLU reference"}
    except Exception as e: return skip(f"Core ML model conversion/prediction failed: {type(e).__name__}: {e}")

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--version",action="version",version=f"%(prog)s {VERSION}")
    ap.add_argument("--all",action="store_true",help="run every supported benchmark (default)")
    ap.add_argument("--only",nargs="+",choices=("cpu","memory","cpu-matmul","mlx-matmul","precision","transformer","quantization","coreml"),help="run only selected benchmark categories")
    ap.add_argument("--profile",choices=("quick","standard","thorough"),default="standard",help="measurement duration profile (default: standard)")
    ap.add_argument("--output",default="benchmark_raw.jsonl",help="detailed JSON Lines output")
    ap.add_argument("--report",default="benchmark_report.json",help="human-readable summary JSON")
    ap.add_argument("--html",default=None,help="optional standalone HTML summary")
    ap.add_argument("--skip-ane",action="store_true",help="skip Core ML/ANE attempt")
    ap.add_argument("--skip-memory",action="store_true",help="skip the memory-copy benchmark")
    ap.add_argument("--overwrite",action="store_true",help="replace the raw output file if it already exists")
    ap.add_argument("--verbose",action="store_true",help="print progress as benchmarks start")
    args=ap.parse_args()
    if args.all and args.only: ap.error("--all and --only cannot be used together")
    selected=set(args.only or ("cpu","memory","cpu-matmul","mlx-matmul","precision","transformer","quantization","coreml"))
    if not selected: ap.error("no benchmarks selected; remove skip flags or select another category")
    out=Path(args.output).expanduser()
    if out.exists() and not args.overwrite: ap.error(f"raw output already exists: {out}; choose a new --output path or pass --overwrite")
    out=out.resolve(); out.parent.mkdir(parents=True,exist_ok=True)
    spec=detect(); cfg=sizes(spec,"all",args.profile); cfg["benchmarks"]=sorted(selected)
    model_path=out.parent/"generated_ane_model.mlmodel"
    rep=Reporter(out,spec,cfg)
    benchmarks=[
        ("cpu","cpu_multithreaded",lambda:bench_cpu(cfg["cpu_iterations"],cfg["timing_seconds"]["cpu_sample"])),
        ("memory","native_memory_bandwidth",lambda:bench_memory(cfg["memory_bytes_per_buffer"],cfg["memory_threads"],cfg["timing_seconds"]["memory_aggregate"])),
        ("cpu-matmul","accelerated_cpu_blas_matmul",lambda:bench_numpy(cfg["matrix_n"],cfg["timing_seconds"]["matrix_aggregate"])),
        ("mlx-matmul","metal_mlx_matmul_and_compile",lambda:mlx_bench(cfg)),
        ("precision","mlx_matmul_throughput_by_precision",lambda:precision_throughput_bench(cfg)),
        ("transformer","transformer_training_prefill_decode",lambda:transformer_bench(cfg)),
        ("quantization","mlx_4bit_quantization",lambda:quant_bench(cfg)),
        ("coreml","coreml_prediction",lambda:ane_bench(cfg,model_path)),
    ]
    try:
        for key,name,fn in benchmarks:
            if key not in selected: continue
            if args.verbose: print(f"Starting {name} ...",flush=True)
            if key=="memory" and args.skip_memory: fn=lambda:skip("disabled by --skip-memory")
            if key=="coreml" and args.skip_ane: fn=lambda:skip("disabled by --skip-ane")
            rep.run(name,fn)
    finally: rep.close()
    # Import local parser to ensure report schema and rendering logic stays independent.
    from report_mlx import summarize_file, write_outputs
    summary=summarize_file(out); write_outputs(summary,Path(args.report),Path(args.html) if args.html else None)
    print(f"Raw data: {out}\nSummary: {Path(args.report).resolve()}")
    if args.html: print(f"HTML: {Path(args.html).resolve()}")
    if summary["counts"]["error"] or summary["counts"]["failed_correctness"]: return 1
    return 0

if __name__=="__main__": raise SystemExit(main())
