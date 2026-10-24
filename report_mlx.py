#!/usr/bin/env python3
"""Parse benchmark_mlx.py JSONL into terminal, JSON and optional standalone HTML reports."""
from __future__ import annotations
import argparse, html, json, math
from pathlib import Path
from typing import Any

def summarize_file(path:Path)->dict[str,Any]:
    events=[]
    with path.open(encoding="utf-8") as f:
        for lineno,line in enumerate(f,1):
            if not line.strip(): continue
            try: events.append(json.loads(line))
            except json.JSONDecodeError as e: raise ValueError(f"{path}:{lineno}: invalid JSON: {e}") from e
    start=next((e for e in events if e.get("type")=="run_start"),{})
    end=next((e for e in reversed(events) if e.get("type")=="run_end"),{})
    results=[e for e in events if e.get("type")=="benchmark_result"]
    root=Path(__file__).resolve().parent; home=Path.home()
    def scrub(value):
        if isinstance(value,dict): return {str(k):scrub(v) for k,v in value.items()}
        if isinstance(value,list): return [scrub(v) for v in value]
        if isinstance(value,str): return value.replace(str(root),".").replace(str(home),"~")
        return value
    results=scrub(results)
    counts={
        "ok":sum(e.get("status")=="ok" for e in results),
        "skipped":sum(e.get("status")=="skipped" for e in results),
        "error":sum(e.get("status")=="error" for e in results),
        "failed_correctness":sum(e.get("status")=="failed_correctness" or e.get("correctness_status")=="failed" for e in results),
        "correctness_passed":sum(e.get("correctness_status")=="passed" for e in results),
        "correctness_not_checked":sum(e.get("correctness_status","not_checked")=="not_checked" for e in results),
    }
    machine=scrub(start.get("machine",{})); config=scrub(start.get("config",{}))
    return {"schema_version":2,"source":path.name,"run_id":start.get("run_id"),"started_at":start.get("timestamp"),
            "finished_at":end.get("timestamp"),"run_duration_seconds":end.get("run_duration_seconds"),
            "benchmark_suite_version":machine.get("benchmark_suite_version"),
            "machine":machine,"config":config,"benchmarks":results,"counts":counts}

def fmt(v,key=None):
    if isinstance(v,bool): return str(v)
    if isinstance(v,float):
        if key and ("seconds" in key or key.endswith("_s")):
            if 0 < abs(v) < 0.001: return f"{v*1_000_000:,.2f} µs"
            if 0 < abs(v) < 1: return f"{v*1_000:,.2f} ms"
            return f"{v:,.3f} s"
        if 0 < abs(v) < 0.001: return f"{v:.6g}"
        return f"{v:,.3f}"
    if isinstance(v,(dict,list)): return json.dumps(v,ensure_ascii=False)
    return str(v)

def fmt_seconds(v, digits=3):
    if v is None: return "unavailable"
    v=float(v)
    if 0 < abs(v) < 0.001: return f"{v*1_000_000:,.2f} µs"
    if 0 < abs(v) < 1: return f"{v*1_000:,.2f} ms"
    return f"{v:.{digits}f} s"

def render_text(s):
    m=s["machine"]; lines=["Apple Silicon / MLX Benchmark Summary", "="*37,
      f"Machine: {m.get('mac_model') or m.get('machine') or 'unknown'} | {m.get('processor') or m.get('platform','unknown')}",
      f"CPU: {m.get('physical_cpu_count') or '?'} physical / {m.get('logical_cpu_count','?')} logical cores | RAM: "+(f"{m['ram_bytes']/1024**3:.1f} GiB" if m.get("ram_bytes") else "unknown"),
      f"MLX: {m.get('mlx',{}).get('version','unavailable')} | status ok {s['counts']['ok']}, skipped {s['counts']['skipped']}, errors {s['counts']['error']}",
      f"Suite: {s.get('benchmark_suite_version','unknown')} | started: {s.get('started_at','unknown')} | duration: {fmt_seconds(s.get('run_duration_seconds'))}",
      f"Run ID: {s.get('run_id') or 'not recorded'} | correctness passed: {s['counts'].get('correctness_passed',0)} | failed correctness: {s['counts'].get('failed_correctness',0)}",""]
    for r in s["benchmarks"]:
        lines.append(f"{r.get('name')}: {r.get('status')}")
        if r.get("status")=="skipped": lines.append(f"  reason: {r.get('reason','not reported')}"); continue
        if r.get("status")=="error": lines.append(f"  error: {r.get('error','unknown')}"); continue
        exclude={"schema_version","sequence","type","name","status","duration_s","traceback","correctness_status","execution_status"}
        lines.append(f"  execution: {r.get('execution_status','unknown')} | correctness: {r.get('correctness_status','not_checked')}")
        for k,v in r.items():
            if k not in exclude and v is not None: lines.append(f"  {k.replace('_',' ')}: {fmt(v,k)}")
    return "\n".join(lines)+"\n"

def _svg_bar_chart(title, points, unit, scale="linear"):
    if not points: return ""
    values=[max(0.0,float(v)) for _,v in points]
    positives=[v for v in values if v>0]
    if not positives: return ""
    row_h=38; top=48; height=top+row_h*len(points)+18; bar_x=220; bar_max=410
    if scale=="log":
        lo=max(min(positives),1e-12); hi=max(positives)
        widths=[bar_max*(math.log10(max(v,lo))-math.log10(lo)+0.08)/(math.log10(hi)-math.log10(lo)+0.08) for v in values]
        scale_note="logarithmic scale"
    else:
        hi=max(positives); widths=[bar_max*v/hi for v in values]; scale_note="linear scale"
    shapes=[]
    for i,((label,value),width) in enumerate(zip(points,widths)):
        y=top+i*row_h
        shapes.append(f"<text x='8' y='{y+19}' class='chart-label'>{html.escape(str(label))}</text>"
            f"<rect x='{bar_x}' y='{y+2}' width='{max(2,width):.1f}' height='23' rx='5' class='bar'/>"
            f"<text x='{bar_x+width+8:.1f}' y='{y+19}' class='chart-value'>{value:,.3f} {html.escape(unit)}</text>")
    return ("<figure><figcaption><strong>"+html.escape(title)+"</strong><span>"+scale_note+
        "</span></figcaption><svg viewBox='0 0 760 "+str(height)+"' role='img' aria-label='"+
        html.escape(title)+"'>"+"".join(shapes)+"</svg></figure>")

def _benchmark_table(results):
    rows=[]
    for r in results:
        details={k:v for k,v in r.items() if k not in {"traceback","schema_version","sequence","type","name"}}
        rows.append("<tr><td>"+html.escape(str(r.get("name")))+"</td><td>"+html.escape(str(r.get("status")))+
            "<br><small>correctness: "+html.escape(str(r.get("correctness_status","not_checked")))+
            "</small></td><td><pre>"+html.escape(json.dumps(details,indent=2,ensure_ascii=False))+"</pre></td></tr>")
    return "".join(rows)

def write_outputs(s,path:Path,html_path:Path|None=None):
    path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(s,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(render_text(s),end="")
    if html_path:
        html_path.parent.mkdir(parents=True,exist_ok=True)
        r=s["benchmarks"]; by={x.get("name"):x for x in r}; m=s.get("machine",{}); c=s.get("config",{})
        transformer=by.get("transformer_training_prefill_decode",{})
        training=(f"<p>{html.escape(fmt(transformer['training_tokens_per_second']))} synthetic tokens/s; "
            f"{transformer.get('training_steps','?')} timed steps, median {html.escape(fmt_seconds(transformer.get('training_median_step_seconds')))}. "
            "Training is synthetic and is not a fine-tuning result.</p>") if transformer.get("training_tokens_per_second") else "<p>Training workload not recorded.</p>"
        running_points=[]
        if transformer.get("prefill_tokens_per_second"): running_points.append(("Synthetic prefill",transformer["prefill_tokens_per_second"]))
        decode_rate=transformer.get("decode_steps_per_second",transformer.get("decode_tokens_per_second"))
        if decode_rate: running_points.append(("Synthetic single-token decode proxy",decode_rate))
        running=(f"<p>Prefill and sequential decode are different workloads. Decode methodology: {html.escape(str(transformer.get('decode_method','not recorded')))}.</p>"+
            _svg_bar_chart("Synthetic transformer inference",running_points,"steps/s","log" if len(running_points)>1 else "linear"))
        precision=by.get("mlx_matmul_throughput_by_precision",by.get("mlx_peak_tflops_by_precision",{}))
        formats=precision.get("formats",[])
        points=[(f.get("format","unknown"),f.get("median_tflops",0)) for f in formats if f.get("status")=="ok"]
        precision_chart=_svg_bar_chart("Observed matrix throughput by precision",points,"TFLOPS")
        fmt_rows="".join("<tr><td>"+html.escape(str(f.get("format")))+"</td><td>"+html.escape(fmt(f.get("median_tflops")))+
            "</td><td>"+html.escape(fmt(f.get("fastest_observed_tflops",f.get("peak_observed_tflops"))))+
            "</td><td>"+html.escape(str(f.get("correctness_status","not_checked")))+"</td></tr>" for f in formats)
        precision_table=("<table><tr><th>Format</th><th>Median TFLOPS</th><th>Fastest sample</th><th>Correctness</th></tr>"+
            fmt_rows+"</table><p class='note'>Fastest sample is an observed timing, not theoretical hardware peak. Quantized paths use dense-equivalent operation counts when reported.</p>") if fmt_rows else "<p>Precision-specific MLX results not recorded.</p>"
        mem=by.get("native_memory_bandwidth",{})
        cpu=by.get("accelerated_cpu_blas_matmul",by.get("cpu_matrix_multiply",{}))
        mlxmm=by.get("metal_mlx_matmul_and_compile",{})
        cpumath=by.get("cpu_multithreaded",{})
        quant=by.get("mlx_4bit_quantization",{})
        mem_note=("Copy throughput counts one byte per copied element; estimated traffic counts both a read and a write. "
            "Buffers are reused, so results may include cache effects and do not certify physical DRAM bandwidth.")
        ft="<p><strong>Not measured.</strong> This suite has no pretrained-model, dataset, LoRA, or adapter fine-tuning benchmark. Synthetic gradient/update work is reported under Training only.</p>"
        metadata={
            "Machine":m.get("mac_model") or m.get("machine","unavailable"),
            "Chip":m.get("chip_name") or "unavailable",
            "CPU cores":f"{m.get('physical_cpu_count','?')} physical / {m.get('logical_cpu_count','?')} logical",
            "GPU cores":m.get("gpu_core_count") or "unavailable",
            "Memory":(f"{m['ram_bytes']/1024**3:.1f} GiB installed" if m.get("ram_bytes") else "unavailable"),
            "Estimated available memory":(f"{m['available_ram_bytes_estimate']/1024**3:.1f} GiB" if m.get("available_ram_bytes_estimate") else "unavailable"),
            "Platform":m.get("platform","unavailable"),"Python":m.get("python","unavailable"),
            "MLX":m.get("mlx",{}).get("version","unavailable"),
            "Power mode":m.get("power_mode") or "unavailable",
            "Thermal state":m.get("thermal_state") or "unavailable",
            "Suite":s.get("benchmark_suite_version","unavailable"),"Profile":c.get("profile","unavailable"),
            "Run duration":fmt_seconds(s.get("run_duration_seconds")),
            "Status":f"{s['counts']['ok']} passed · {s['counts']['skipped']} skipped · {s['counts']['error']} errors · {s['counts'].get('failed_correctness',0)} correctness failures",
        }
        cards="".join("<div class='card'><span>"+html.escape(k)+"</span><strong>"+html.escape(str(v))+"</strong></div>" for k,v in metadata.items())
        mem_detail=(f"<p>Memory-copy estimate: {fmt(mem.get('copy_gib_per_second'))} GiB/s copy throughput; "
            f"{fmt(mem.get('estimated_memory_traffic_gib_per_second',mem.get('bandwidth_gib_per_second')))} GiB/s estimated read-plus-write traffic. "
            f"{mem.get('threads','?')} workers, {mem.get('buffer_bytes_each',0)/1024**2:.0f} MiB per buffer. {mem_note}</p>") if mem else "<p>Memory copy benchmark not recorded.</p>"
        other_rows=[]
        if cpumath:
            other_rows.append(f"<tr><th>CPU process-pool math</th><td>{fmt(cpumath.get('operations_per_second'))} {html.escape(str(cpumath.get('unit','operations/s')))}</td><td>{cpumath.get('workers','?')} workers; includes Python, scheduling, and IPC</td></tr>")
        if cpu:
            other_rows.append(f"<tr><th>NumPy / BLAS matmul</th><td>{fmt(cpu.get('median_gflops'))} GFLOPS</td><td>provider {html.escape(str(cpu.get('blas_provider','unknown')))}; matrix {html.escape(str(cpu.get('matrix_size','unknown')))}</td></tr>")
        if mlxmm:
            other_rows.append(f"<tr><th>MLX GPU matmul</th><td>{fmt(mlxmm.get('median_gflops'))} GFLOPS</td><td>matrix {html.escape(str(mlxmm.get('matrix_size','unknown')))}; device {html.escape(str(mlxmm.get('device','unknown')))}</td></tr>")
        if quant:
            other_rows.append(f"<tr><th>MLX 4-bit layer</th><td>{fmt(quant.get('compression_ratio_estimate'))}× estimated storage reduction</td><td>relative MAE {fmt(quant.get('relative_mean_absolute_output_error',quant.get('mean_absolute_output_error')))}; median latency {html.escape(fmt_seconds(quant.get('inference_median_seconds')))}</td></tr>")
        other_table="<table><tr><th>Workload</th><th>Result</th><th>Method / context</th></tr>"+"".join(other_rows)+"</table>" if other_rows else ""
        coreml=by.get("coreml_prediction",by.get("coreml_ane",{}))
        ane_note=("Core ML prediction requested "+html.escape(str(coreml.get("compute_units_requested","unknown")))+
            f"; median latency {html.escape(fmt_seconds(coreml.get('warm_prediction_median_seconds')))}. "+
            ("ANE placement was verified." if coreml.get("ane_execution_verified") else "ANE placement was not verified by the public prediction API.")) if coreml else "Core ML prediction not recorded."
        rows=_benchmark_table(r)
        doc="<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>Apple Silicon benchmark report</title>"
        doc+="<style>body{font:15px/1.55 system-ui,-apple-system,sans-serif;max-width:1100px;margin:2rem auto;padding:0 1rem;color:#17202a}h2{margin-top:2.3rem;border-bottom:1px solid #dce3eb;padding-bottom:.4rem}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:10px}.card,figure{background:#f5f8fb;border:1px solid #dce3eb;border-radius:10px;padding:14px}.card span,figcaption span,.note{display:block;color:#5d6b7b;font-size:.9rem}.card strong{display:block;margin-top:5px}.chart-label{font:13px system-ui;fill:#18212f}.chart-value{font:12px system-ui;fill:#18212f}.bar{fill:#3478f6}figure{overflow-x:auto}svg{width:100%;min-width:640px;height:auto}table{border-collapse:collapse;width:100%;margin:1rem 0}td,th{border-bottom:1px solid #dce3eb;padding:.65rem;text-align:left;vertical-align:top}pre{white-space:pre-wrap;overflow-wrap:anywhere;margin:0;font-size:.83rem}details{margin:1rem 0}small{color:#5d6b7b}</style></head><body>"
        doc+="<h1>Apple Silicon / MLX benchmark</h1><p>Workload-specific results; this report does not combine measurements into a universal score.</p><div class='grid'>"+cards+"</div>"
        doc+="<h2>Training</h2>"+training
        doc+="<h2>Running</h2>"+running
        doc+="<h2>Fine Tuning</h2>"+ft
        doc+="<h2>Other</h2>"+precision_chart+precision_table+other_table+mem_detail+f"<p>Core ML prediction: {ane_note}</p>"
        doc+="<details><summary>Adaptive run configuration</summary><pre>"+html.escape(json.dumps(c,indent=2,ensure_ascii=False))+"</pre></details>"
        doc+="<h2>All recorded benchmark details</h2><table><tr><th>Benchmark</th><th>Status</th><th>Data</th></tr>"+rows+"</table>"
        doc+="<p class='note'>Exported source file: "+html.escape(str(s.get("source","unknown")))+" · Run ID: "+html.escape(str(s.get("run_id") or "not recorded"))+"</p></body></html>"
        html_path.write_text(doc,encoding="utf-8")

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("input",type=Path); p.add_argument("--json",type=Path,default=Path("benchmark_report.json")); p.add_argument("--html",type=Path)
    a=p.parse_args()
    try: write_outputs(summarize_file(a.input),a.json,a.html)
    except (OSError,ValueError) as e: p.error(str(e))
if __name__=="__main__": main()
