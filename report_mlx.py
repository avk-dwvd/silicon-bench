#!/usr/bin/env python3
"""Parse benchmark_mlx.py JSONL into terminal, JSON and optional standalone HTML reports."""
from __future__ import annotations
import argparse, html, json
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
    results=[e for e in events if e.get("type")=="benchmark_result"]
    return {"schema_version":1,"source":str(path.resolve()),"machine":start.get("machine",{}),"config":start.get("config",{}),"benchmarks":results,
            "counts":{"ok":sum(e.get("status")=="ok" for e in results),"skipped":sum(e.get("status")=="skipped" for e in results),"error":sum(e.get("status")=="error" for e in results)}}

def fmt(v):
    if isinstance(v,bool): return str(v)
    if isinstance(v,float): return f"{v:,.3f}"
    if isinstance(v,(dict,list)): return json.dumps(v,ensure_ascii=False)
    return str(v)

def render_text(s):
    m=s["machine"]; lines=["Apple Silicon / MLX Benchmark Summary", "="*37,
      f"Machine: {m.get('mac_model') or m.get('machine') or 'unknown'} | {m.get('processor') or m.get('platform','unknown')}",
      f"CPU: {m.get('physical_cpu_count') or '?'} physical / {m.get('logical_cpu_count','?')} logical cores | RAM: "+(f"{m['ram_bytes']/1024**3:.1f} GiB" if m.get("ram_bytes") else "unknown"),
      f"MLX: {m.get('mlx',{}).get('version','unavailable')} | status ok {s['counts']['ok']}, skipped {s['counts']['skipped']}, errors {s['counts']['error']}",""]
    for r in s["benchmarks"]:
        lines.append(f"{r.get('name')}: {r.get('status')}")
        if r.get("status")=="skipped": lines.append(f"  reason: {r.get('reason','not reported')}"); continue
        if r.get("status")=="error": lines.append(f"  error: {r.get('error','unknown')}"); continue
        exclude={"schema_version","sequence","type","name","status","duration_s","traceback"}
        for k,v in r.items():
            if k not in exclude and v is not None: lines.append(f"  {k.replace('_',' ')}: {fmt(v)}")
    return "\n".join(lines)+"\n"

def write_outputs(s,path:Path,html_path:Path|None=None):
    path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(s,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(render_text(s),end="")
    if html_path:
        html_path.parent.mkdir(parents=True,exist_ok=True)
        rows="".join("<tr><td>"+html.escape(str(r.get("name")))+"</td><td>"+html.escape(str(r.get("status")))+"</td><td><pre>"+html.escape(json.dumps({k:v for k,v in r.items() if k not in {"traceback","schema_version","sequence","type"}},indent=2))+"</pre></td></tr>" for r in s["benchmarks"])
        html_path.write_text("<!doctype html><meta charset='utf-8'><title>Benchmark report</title><style>body{font:15px system-ui;max-width:1100px;margin:3rem auto;padding:0 1rem;color:#17202a}table{border-collapse:collapse;width:100%}td,th{border-bottom:1px solid #ddd;padding:1rem;text-align:left;vertical-align:top}pre{white-space:pre-wrap;margin:0}.meta{background:#f4f6f7;padding:1rem}</style><h1>Apple Silicon / MLX Benchmark</h1><div class='meta'><pre>"+html.escape(render_text(s))+"</pre></div><h2>Measurements</h2><table><tr><th>Benchmark</th><th>Status</th><th>Details</th></tr>"+rows+"</table>",encoding="utf-8")

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("input",type=Path); p.add_argument("--json",type=Path,default=Path("benchmark_report.json")); p.add_argument("--html",type=Path)
    a=p.parse_args()
    try: write_outputs(summarize_file(a.input),a.json,a.html)
    except (OSError,ValueError) as e: p.error(str(e))
if __name__=="__main__": main()
