#!/usr/bin/env python3
"""Benchmark Qwen2.5-0.5B-Instruct locally with llama.cpp (llama-cli), CPU-only.

Two settings: FP16 ("f16", high precision) vs Q4_K_M (4-bit quantised).
Same 10 prompts, same sampling params in both settings:
  temperature 0.0, seed 42, max 128 generated tokens, 4 threads, ctx 512.

Metrics per prompt, parsed from llama-cli's own per-turn speed line on stdout
("[ Prompt: X t/s | Generation: Y t/s ]"):
  - prompt-eval tokens/sec and generation tokens/sec
  - peak RSS (MB): polled from /proc/<pid>/status VmRSS every 50 ms while
    the child runs (GNU /usr/bin/time -v is not installed on this box, so
    /proc polling is used instead; see README).

Usage:
  python3 bench.py            # full run: download missing files, run both settings
  python3 bench.py --setting f16        # only one setting
  python3 bench.py --skip-download       # assume binary + models already present

Outputs (in this directory):
  raw/<setting>_<prompt_id>.log   raw llama-cli stdout+stderr per prompt
  outputs_<setting>.json          per-prompt rows (output text + timings)
  results_<setting>.json          aggregate stats + model size + peak RSS
"""
import argparse
import glob
import json
import os
import re
import statistics
import subprocess
import sys
import threading
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
BIN_DIR = os.path.join(HERE, "bin")
MODELS_DIR = os.path.join(HERE, "models")
RAW_DIR = os.path.join(HERE, "raw")

LLAMA_RELEASE = "b12441"
LLAMA_URL = (
    "https://github.com/ggml-org/llama.cpp/releases/download/"
    f"{LLAMA_RELEASE}/llama-{LLAMA_RELEASE}-bin-ubuntu-x64.tar.gz"
)
HF_BASE = "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/"

SETTINGS = {
    "f16": "qwen2.5-0.5b-instruct-fp16.gguf",
    "q4_k_m": "qwen2.5-0.5b-instruct-q4_k_m.gguf",
}

TEMP = 0.0
SEED = 42
MAX_TOKENS = 128
THREADS = 4
CTX = 512

BRACKET_RE = re.compile(r"\[ Prompt:\s*([\d.]+) t/s \| Generation:\s*([\d.]+) t/s \]")


def find_llama_cli():
    cands = [os.path.join(BIN_DIR, "llama-cli")]
    cands += sorted(glob.glob(os.path.join(BIN_DIR, "llama-*/llama-cli")))
    for c in cands:
        if os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    return None


def download(url, dest):
    print(f"  downloading {url}\n    -> {dest}", flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "curl/8.5"})
    with urllib.request.urlopen(req) as r, open(dest, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    print(f"    done: {os.path.getsize(dest) / 1e6:.1f} MB", flush=True)


def ensure_files(skip_download):
    os.makedirs(BIN_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RAW_DIR, exist_ok=True)
    cli = find_llama_cli()
    if cli is None:
        if skip_download:
            sys.exit("llama-cli not found under bin/ and --skip-download given.")
        tarball = os.path.join(BIN_DIR, f"llama-{LLAMA_RELEASE}-bin-ubuntu-x64.tar.gz")
        if not os.path.exists(tarball):
            download(LLAMA_URL, tarball)
        print("  extracting llama.cpp tarball ...", flush=True)
        subprocess.run(["tar", "xzf", tarball, "-C", BIN_DIR], check=True)
        cli = find_llama_cli()
        if cli is None:
            sys.exit("llama-cli still not found after extracting the tarball.")
    print(f"llama-cli: {cli}")
    for setting, fname in SETTINGS.items():
        path = os.path.join(MODELS_DIR, fname)
        if not os.path.exists(path):
            if skip_download:
                sys.exit(f"model missing: {path} (and --skip-download given)")
            download(HF_BASE + fname, path)
        print(f"model {setting}: {path} ({os.path.getsize(path) / 1e6:.1f} MB)")
    return cli


def poll_rss(proc, peak):
    """Poll /proc/<pid>/status VmRSS; keep the max seen in peak[0] (MB)."""
    pid = proc.pid
    while proc.poll() is None:
        try:
            with open(f"/proc/{pid}/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        peak[0] = max(peak[0], int(line.split()[1]) / 1024.0)
                        break
        except (FileNotFoundError, ProcessLookupError):
            break
        time.sleep(0.05)


def run_one(cli, model_path, prompt_text):
    """Run one prompt through llama-cli; return timings + output text.

    llama-cli (build 12441) applies the GGUF's embedded chat template to the
    plain user text itself, so the raw prompt is passed as-is. It prints the
    generated text to stdout and a per-turn speed line like
    "[ Prompt: 54.5 t/s | Generation: 10.0 t/s ]" at the end of stdout.
    NOTE: this build does not print per-prompt token counts in single-turn
    mode, only the two rates, so token counts are recorded as null.
    """
    cmd = [
        cli, "-m", model_path,
        "-p", prompt_text,
        "-n", str(MAX_TOKENS),
        "--temp", str(TEMP),
        "--seed", str(SEED),
        "-t", str(THREADS),
        "-c", str(CTX),
        "--no-display-prompt",
        "-st",  # --single-turn: exit after one turn (llama-cli is interactive by default)
    ]
    peak = [0.0]
    t0 = time.time()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            stdin=subprocess.DEVNULL, text=True)
    watcher = threading.Thread(target=poll_rss, args=(proc, peak), daemon=True)
    watcher.start()
    out, err = proc.communicate()
    watcher.join(timeout=1.0)
    wall = time.time() - t0
    m = BRACKET_RE.search(out)
    prompt_tps = float(m.group(1)) if m else None
    gen_tps = float(m.group(2)) if m else None
    # Generated text = stdout after the echoed "> <prompt>", up to the speed line.
    text = out
    idx = text.find(prompt_text)
    if idx != -1:
        text = text[idx + len(prompt_text):]
    bidx = text.find("[ Prompt:")
    if bidx != -1:
        text = text[:bidx]
    return text.strip(), prompt_tps, None, gen_tps, None, peak[0], wall, err, out


def run_setting(cli, setting):
    model_path = os.path.join(MODELS_DIR, SETTINGS[setting])
    model_size_mb = os.path.getsize(model_path) / 1e6
    with open(os.path.join(HERE, "prompts.json")) as f:
        prompts = json.load(f)
    rows = []
    for item in prompts:
        print(f"[{setting}] {item['id']}: {item['prompt'][:60]}...", flush=True)
        text, pe_tps, pe_n, gen_tps, gen_n, peak_rss, wall, err, out = run_one(
            cli, model_path, item["prompt"])
        log_path = os.path.join(RAW_DIR, f"{setting}_{item['id']}.log")
        with open(log_path, "w") as f:
            f.write(f"CMD settings: temp={TEMP} seed={SEED} n={MAX_TOKENS} "
                    f"threads={THREADS} ctx={CTX}\nPROMPT: {item['prompt']}\n\n"
                    f"--- STDOUT (raw llama-cli output) ---\n{out}\n\n"
                    f"--- STDERR ---\n{err}\n"
                    f"--- PARSED ---\nprompt_eval_tps={pe_tps} gen_tps={gen_tps} "
                    f"peak_rss_mb={peak_rss:.1f} wall_s={wall:.2f}\n")
        rows.append({
            "id": item["id"], "category": item["category"], "prompt": item["prompt"],
            "output": text,
            "prompt_eval_tokens": pe_n, "prompt_eval_tps": pe_tps,
            "gen_tokens": gen_n, "gen_tps": gen_tps,
            "peak_rss_mb": round(peak_rss, 1), "wall_s": round(wall, 2),
        })
        print(f"    -> gen {gen_tps} tok/s, prompt-eval {pe_tps} tok/s, "
              f"peak RSS {peak_rss:.0f} MB, {wall:.1f}s wall", flush=True)

    with open(os.path.join(HERE, f"outputs_{setting}.json"), "w") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    def agg(key):
        vals = [r[key] for r in rows if r[key] is not None]
        return {"mean": round(statistics.mean(vals), 2),
                "median": round(statistics.median(vals), 2),
                "min": round(min(vals), 2), "max": round(max(vals), 2),
                "n": len(vals)} if vals else None

    summary = {
        "setting": setting,
        "model_file": SETTINGS[setting],
        "model_size_mb": round(model_size_mb, 1),
        "sampling": {"temperature": TEMP, "seed": SEED, "max_tokens": MAX_TOKENS,
                     "threads": THREADS, "ctx": CTX},
        "n_prompts": len(rows),
        "prompt_eval_tps": agg("prompt_eval_tps"),
        "gen_tps": agg("gen_tps"),
        "peak_rss_mb": round(max(r["peak_rss_mb"] for r in rows), 1),
        "peak_rss_mb_mean": round(statistics.mean(r["peak_rss_mb"] for r in rows), 1),
        "total_wall_s": round(sum(r["wall_s"] for r in rows), 1),
        "per_prompt": [
            {"id": r["id"], "prompt_eval_tps": r["prompt_eval_tps"],
             "gen_tps": r["gen_tps"], "gen_tokens": r["gen_tokens"],
             "peak_rss_mb": r["peak_rss_mb"]} for r in rows
        ],
    }
    with open(os.path.join(HERE, f"results_{setting}.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[{setting}] mean gen {summary['gen_tps']['mean']} tok/s, "
          f"mean prompt-eval {summary['prompt_eval_tps']['mean']} tok/s, "
          f"peak RSS {summary['peak_rss_mb']} MB", flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setting", choices=list(SETTINGS) + ["both"], default="both")
    ap.add_argument("--skip-download", action="store_true")
    args = ap.parse_args()
    cli = ensure_files(args.skip_download)
    settings = list(SETTINGS) if args.setting == "both" else [args.setting]
    for s in settings:
        run_setting(cli, s)
    print("Done. See outputs_<setting>.json and results_<setting>.json.")


if __name__ == "__main__":
    main()
