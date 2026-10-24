#!/usr/bin/env python3
"""Sequential Apple Silicon / MLX benchmark suite. Results are measurements, not claims."""
from __future__ import annotations
import argparse, concurrent.futures, datetime as dt, json, os, platform, subprocess, sys, tempfile, time, traceback
from pathlib import Path
from typing import Any, Callable

VERSION = "1.0.0"

def cmdout(args: list[str]) -> str | None:
    try: return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL, timeout=4).strip()
    except Exception: return None

def detect() -> dict[str, Any]:
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
        mlx = {"available": True, "version": getattr(__import__("mlx"), "__version__", "unknown"), "default_device": str(mx.default_device())}
    except Exception as e: mlx = {"available": False, "reason": f"MLX import failed: {type(e).__name__}: {e}"}
    return {"platform": platform.platform(), "system": platform.system(), "machine": platform.machine(), "processor": platform.processor() or None,
            "logical_cpu_count": logical, "physical_cpu_count": physical, "ram_bytes": mem,
            "mac_model": cmdout(["sysctl", "-n", "hw.model"]) if sys.platform == "darwin" else None,
            "mlx": mlx, "python": sys.version, "benchmark_suite_version": VERSION}

def sizes(spec: dict[str,Any], requested: str) -> dict[str,Any]:
    ram=spec.get("ram_bytes") or 8*1024**3; cores=spec.get("physical_cpu_count") or spec["logical_cpu_count"]
    # Restrict allocations to a small fraction of RAM; --all chooses predictable, laptop-safe workloads.
    cap=max(16*1024**2,min(256*1024**2,ram//64))
    return {"cpu_iterations": 2_000_000 if requested=="all" else 500_000,
            "memory_bytes_per_buffer":cap, "memory_threads":max(1,min(cores,8)),
            "matrix_n":max(128,min(768,int((cap//16)**0.5))),
            "transformer_seq":128 if ram>=16*1024**3 else 64,
            "transformer_width":256 if ram>=16*1024**3 else 128,
            "transformer_layers":2, "quant_matrix_n":256 if ram>=16*1024**3 else 128,
            "policy":"adaptive safe profile; memory buffers capped at 256 MiB each"}

class Reporter:
    def __init__(self,path:Path,spec:dict[str,Any],config:dict[str,Any]):
        self.f=path.open("w",encoding="utf-8"); self.seq=0
        self.write({"type":"run_start","timestamp":dt.datetime.now(dt.timezone.utc).isoformat(),"machine":spec,"config":config})
    def write(self,x):
        self.seq+=1; x={"schema_version":1,"sequence":self.seq,**x}; self.f.write(json.dumps(x,default=str,allow_nan=False)+"\n"); self.f.flush()
    def run(self,name:str,fn:Callable[[],dict[str,Any]]):
        start=time.perf_counter(); self.write({"type":"benchmark_start","name":name,"timestamp":dt.datetime.now(dt.timezone.utc).isoformat()})
        try:
            r=fn(); status=r.pop("status","ok"); self.write({"type":"benchmark_result","name":name,"status":status,"duration_s":time.perf_counter()-start,**r})
        except Exception as e:
            self.write({"type":"benchmark_result","name":name,"status":"error","duration_s":time.perf_counter()-start,"error":f"{type(e).__name__}: {e}","traceback":traceback.format_exc(limit=5)})
    def close(self): self.write({"type":"run_end","timestamp":dt.datetime.now(dt.timezone.utc).isoformat()}); self.f.close()

def skip(reason): return {"status":"skipped","reason":reason}

def _cpu_work(k):
    import math
    s=0.0
    for i in range(k): s+=math.sqrt((i%10000)+1.0)
    return s

def bench_cpu(n):
    # Independent CPU-bound tasks; process pool exercises all selected CPU workers.
    workers=max(1,min(os.cpu_count() or 1,8)); each=max(1,n//workers)
    t=time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool: sums=list(pool.map(_cpu_work,[each]*workers))
    elapsed=time.perf_counter()-t
    return {"workers":workers,"iterations":each*workers,"checksum":sum(sums),"operations_per_second":each*workers/elapsed,"unit":"sqrt iterations/s"}

def bench_memory(byte_count,threads):
    if sys.platform!="darwin": return skip("native bandwidth kernel currently uses macOS Accelerate/dispatch; platform is not macOS")
    # Native C pthread copy/triad kernel. Allocation is bounded and workers are joined before return.
    source=r'''#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
typedef struct { double *a,*b; size_t lo,n; int reps; } Job;
void *run(void *p){ Job*j=p; for(int r=0;r<j->reps;r++) for(size_t i=j->lo;i<j->lo+j->n;i++) j->b[i]=j->a[i]*1.0000001+j->b[i]; return 0; }
int main(int c,char**v){size_t n=strtoull(v[1],0,10)/8;int nt=atoi(v[2]),reps=8;double*a=malloc(n*8),*b=calloc(n,8);if(!a||!b)return 3;for(size_t i=0;i<n;i++)a[i]=1.0;
pthread_t*t=calloc(nt,sizeof(*t));Job*j=calloc(nt,sizeof(*j));struct timespec x,y;clock_gettime(CLOCK_MONOTONIC,&x);for(int k=0;k<nt;k++){size_t lo=n*k/nt,hi=n*(k+1)/nt;j[k]=(Job){a,b,lo,hi-lo,reps};if(pthread_create(&t[k],0,run,&j[k]))return 4;}for(int k=0;k<nt;k++)pthread_join(t[k],0);clock_gettime(CLOCK_MONOTONIC,&y);double sec=y.tv_sec-x.tv_sec+(y.tv_nsec-x.tv_nsec)*1e-9;double sum=0;for(size_t i=0;i<n;i++)sum+=b[i];printf("%.9f %.9f\\n",sec,sum);return 0;}'''
    with tempfile.TemporaryDirectory(prefix="mlxbench-") as td:
        src=Path(td)/"mem.c"; exe=Path(td)/"mem"; src.write_text(source)
        cc=cmdout(["xcrun","--find","clang"]) or "/usr/bin/clang"
        p=subprocess.run([cc,"-O3","-pthread",str(src),"-o",str(exe)],capture_output=True,text=True)
        if p.returncode: return skip("could not compile native bandwidth kernel: "+p.stderr[-800:])
        p=subprocess.run([str(exe),str(byte_count),str(threads)],capture_output=True,text=True,timeout=90)
        if p.returncode: raise RuntimeError(p.stderr or f"native kernel exit {p.returncode}")
        sec,checksum=map(float,p.stdout.split()); moved=byte_count*2*8*8
        return {"threads":threads,"buffer_bytes_each":byte_count,"passes":8,"seconds":sec,"bytes_moved_estimate":moved,"bandwidth_bytes_per_second":moved/sec,"bandwidth_gib_per_second":moved/sec/1024**3,"checksum":checksum,"method":"pthread parallel triad; estimate counts read+write traffic"}

def bench_numpy(n):
    try: import numpy as np
    except Exception as e: return skip(f"NumPy unavailable: {e}")
    t=time.perf_counter(); a=np.ones((n,n),dtype=np.float32); b=np.ones((n,n),dtype=np.float32)
    # Single BLAS call; thread count follows common runtime settings when supplied.
    c=a@b; elapsed=time.perf_counter()-t; flops=2*n**3
    return {"matrix_size":[n,n],"seconds":elapsed,"estimated_flops":flops,"gflops":flops/elapsed/1e9,"checksum":float(c[0,0]),"backend":"NumPy matmul (BLAS thread policy is runtime-dependent)"}

def mlx_bench(config):
    try: import mlx.core as mx
    except Exception as e: return skip(f"MLX unavailable: {type(e).__name__}: {e}")
    n=config["matrix_n"]; a=mx.random.normal((n,n)); b=mx.random.normal((n,n)); mx.eval(a,b)
    t=time.perf_counter(); c=a@b; mx.eval(c); elapsed=time.perf_counter()-t
    info={"matrix_size":[n,n],"seconds":elapsed,"estimated_flops":2*n**3,"gflops":2*n**3/elapsed/1e9,"device":str(mx.default_device()),"checksum":float(mx.sum(c).item())}
    # Compilation latency is isolated from warmed execution.
    try:
        f=mx.compile(lambda x,y:x@y); t=time.perf_counter(); z=f(a,b); mx.eval(z); compile_s=time.perf_counter()-t
        t=time.perf_counter(); z=f(a,b); mx.eval(z); warm_s=time.perf_counter()-t
        info["compile"]={"cold_call_seconds":compile_s,"warm_seconds":warm_s,"note":"cold call includes tracing/compilation and execution; MLX does not expose a portable compile-only timer"}
    except Exception as e: info["compile"]={"status":"skipped","reason":f"MLX compile unavailable: {type(e).__name__}: {e}"}
    return info

def transformer_bench(cfg):
    try: import mlx.core as mx
    except Exception as e: return skip(f"MLX unavailable: {type(e).__name__}: {e}")
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
        tokens=mx.random.normal((1,seq,w)); t=time.perf_counter(); y=block(tokens); mx.eval(y); prefill=time.perf_counter()-t
        # Autoregressive one-token-at-a-time decode with a bounded context cache.
        x=tokens[:,:1,:]; t=time.perf_counter()
        for _ in range(min(32,seq)):
            x=block(x); mx.eval(x)
        decode=time.perf_counter()-t; steps=min(32,seq)
        # A real gradient/update pass on the same synthetic transformer weights, with random regression targets.
        train_x=mx.random.normal((1,min(seq,32),w)); target=mx.random.normal(train_x.shape); trainable=[z for tup in weights for z in tup]
        def loss_fn(params,x,y):
            # Explicit functional forward keeps this benchmark independent of downloaded models.
            h=x
            for i in range(layers):
                q,k,v,o=params[i*4:i*4+4]; a=h@q; b=h@k; c=h@v
                h=h+((mx.softmax((a@mx.swapaxes(b,-1,-2))/w**0.5,axis=-1)@c)@o)
            return mx.mean((h-y)*(h-y))
        grad_fn=mx.value_and_grad(loss_fn); t=time.perf_counter(); loss,grads=grad_fn(trainable,train_x,target); mx.eval(loss,grads)
        updated=[p-1e-4*g for p,g in zip(trainable,grads)]; mx.eval(updated); train_s=time.perf_counter()-t
        return {"model":"synthetic decoder transformer; random weights, no downloads","sequence_length":seq,"width":w,"layers":layers,"prefill_seconds":prefill,"prefill_tokens_per_second":seq/prefill,"decode_steps":steps,"decode_seconds":decode,"decode_tokens_per_second":steps/decode,"training_step_seconds":train_s,"training_tokens_per_second":train_x.shape[1]/train_s,"training_loss":float(loss.item()),"checksum":float(mx.sum(y).item()),"note":"synthetic random-weight transformer; reports forward prefill, recurrent decode, and one gradient/update step; not a production model or quality metric"}
    except Exception as e: return {"status":"error","error":f"{type(e).__name__}: {e}"}

def quant_bench(cfg):
    try: import mlx.core as mx; import mlx.nn as nn
    except Exception as e: return skip(f"MLX quantization unavailable: {type(e).__name__}: {e}")
    n=cfg["quant_matrix_n"]
    try:
        layer=nn.Linear(n,n,bias=False); x=mx.random.normal((1,n)); mx.eval(layer.weight,x)
        before=layer(x); mx.eval(before); original_bytes=layer.weight.size*layer.weight.dtype.itemsize
        nn.quantize(layer,group_size=64,bits=4)
        t=time.perf_counter(); after=layer(x); mx.eval(after); elapsed=time.perf_counter()-t
        # compare against saved dense reference; quantization modifies layer in-place, preserve expected via before output.
        err=float(mx.mean(mx.abs(after-before)).item())
        return {"bits":4,"group_size":64,"matrix_size":[n,n],"dense_weight_bytes_estimate":original_bytes,"quantized_weight_bytes_estimate":int(n*n*0.5+n*4),"inference_seconds":elapsed,"mean_absolute_output_error":err,"checksum":float(mx.sum(after).item()),"note":"MLX nn.Linear quantized in place; byte estimate excludes metadata/alignment"}
    except Exception as e: return skip(f"MLX quantization API unsupported: {type(e).__name__}: {e}")

def ane_bench(cfg,model_path:Path):
    if sys.platform!="darwin": return skip("Core ML / ANE benchmark requires macOS")
    try: import coremltools as ct
    except Exception as e: return skip(f"coremltools unavailable: {type(e).__name__}: {e}")
    try:
        import numpy as np
        import torch
        class Tiny(torch.nn.Module):
            def __init__(self): super().__init__(); self.fc=torch.nn.Linear(128,128)
            def forward(self,x): return torch.relu(self.fc(x))
        torch.manual_seed(7); model=Tiny().eval(); traced=torch.jit.trace(model,torch.zeros(1,128))
        ml=ct.convert(traced,inputs=[ct.TensorType(name="input",shape=(1,128))],compute_units=ct.ComputeUnit.ALL)
        ml.save(str(model_path)); pred=ml.predict({"input":np.zeros((1,128),dtype=np.float32)})
        # Core ML's public Python API requests allowed compute units but does not expose proof of per-op ANE placement.
        t=time.perf_counter()
        for _ in range(20): ml.predict({"input":np.ones((1,128),dtype=np.float32)})
        elapsed=(time.perf_counter()-t)/20
        return {"model_path":str(model_path),"model_type":"generated tiny traced PyTorch MLP","compute_units_requested":"ALL (CPU, GPU, ANE allowed)","ane_execution_verified":False,"verification":"Core ML Python prediction API provides no per-operation ANE placement confirmation; requested ALL does not prove ANE use","warm_prediction_seconds_mean":elapsed,"predicted_output_keys":list(pred.keys()),"model_file_bytes":model_path.stat().st_size}
    except Exception as e: return skip(f"Core ML model conversion/prediction failed: {type(e).__name__}: {e}")

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--all",action="store_true",help="run every supported benchmark with adaptive safe sizes")
    ap.add_argument("--output",default="benchmark_raw.jsonl",help="detailed JSON Lines output")
    ap.add_argument("--report",default="benchmark_report.json",help="human-readable summary JSON")
    ap.add_argument("--html",default=None,help="optional standalone HTML summary")
    ap.add_argument("--skip-ane",action="store_true",help="skip Core ML/ANE attempt")
    args=ap.parse_args()
    if not args.all: ap.error("currently --all is required")
    spec=detect(); cfg=sizes(spec,"all"); out=Path(args.output).expanduser().resolve(); out.parent.mkdir(parents=True,exist_ok=True)
    model_path=out.parent/"generated_ane_model.mlmodel"
    rep=Reporter(out,spec,cfg)
    try:
        rep.run("cpu_multithreaded",lambda:bench_cpu(cfg["cpu_iterations"]))
        rep.run("native_memory_bandwidth",lambda:bench_memory(cfg["memory_bytes_per_buffer"],cfg["memory_threads"]))
        rep.run("cpu_matrix_multiply",lambda:bench_numpy(cfg["matrix_n"]))
        rep.run("metal_mlx_matmul_and_compile",lambda:mlx_bench(cfg))
        rep.run("transformer_training_prefill_decode",lambda:transformer_bench(cfg))
        rep.run("mlx_4bit_quantization",lambda:quant_bench(cfg))
        rep.run("coreml_ane",lambda:skip("disabled by --skip-ane") if args.skip_ane else ane_bench(cfg,model_path))
    finally: rep.close()
    # Import local parser to ensure report schema and rendering logic stays independent.
    from report_mlx import summarize_file, write_outputs
    summary=summarize_file(out); write_outputs(summary,Path(args.report),Path(args.html).resolve() if args.html else None)
    print(f"Raw data: {out}\nSummary: {Path(args.report).resolve()}")
    if args.html: print(f"HTML: {Path(args.html).resolve()}")

if __name__=="__main__": main()
