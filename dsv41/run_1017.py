"""Explicit single-A100 test entry point: python -m dsv41.run_1017 --help.

This module deliberately imports no GPU runtime before setting the 1017 numerical
mode. Legacy launchers retain their defaults. No model download is performed.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import sys


CACHE_EPOCH = "fullstate-v5-index-engram-dspark-1017-o-qdq-v1"


def make_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=("bench", "chat", "serve"))
    ap.add_argument("--ckpt", type=Path, default=Path(os.environ.get(
        "DSV41_1017_CKPT", "~/models/DeepSeek-V4.1-Flash-1017")))
    ap.add_argument("--offload-experts", choices=("cpu", "gpu"), default="cpu")
    ap.add_argument("--hot-experts", type=int, default=0)
    ap.add_argument("--hot-stats", type=Path, default=Path("results/route_stats_1017.pt"))
    ap.add_argument("--route-stats", type=Path, help="bench: save calibration counts")
    ap.add_argument("--max-seq-len", type=int, default=8192)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float)
    ap.add_argument("--prompt", default="Write a Python LRU cache with tests.")
    ap.add_argument("--profile", action="store_true", help="bench only; not a speed measurement")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--dry-run", action="store_true", help="print command/settings; do not load or validate files")
    return ap


def build_launch(a: argparse.Namespace) -> tuple[list[str], dict[str, str]]:
    """Build argv and only our environment overrides; do not import torch."""
    if a.max_seq_len <= 0 or a.max_new_tokens <= 0:
        raise ValueError("token limits must be positive")
    if a.max_new_tokens >= a.max_seq_len:
        raise ValueError("max-new-tokens must leave room for the prompt")
    if not 0 <= a.hot_experts <= 384:
        raise ValueError("hot-experts must be between 0 and 384 (VRAM may impose a lower limit)")
    if a.offload_experts != "cpu" and a.hot_experts:
        raise ValueError("hot-experts requires CPU expert offload")
    if a.mode != "bench" and (a.route_stats is not None or a.profile):
        raise ValueError("route-stats and profile are bench-only options")
    if a.mode == "serve" and a.temperature is not None:
        raise ValueError("set sampling temperature in each API request")
    if a.temperature is not None and not 0 <= a.temperature < float("inf"):
        raise ValueError("temperature must be finite and nonnegative")
    if not 1 <= a.port <= 65535:
        raise ValueError("port must be between 1 and 65535")
    ckpt = str(a.ckpt.expanduser().resolve())
    module = {"bench": "run", "chat": "chat", "serve": "serve"}[a.mode]
    command = [sys.executable, "-m", f"dsv41.{module}", "--ckpt", ckpt,
               "--devices", "0", "--offload-experts", a.offload_experts,
               "--max-seq-len", str(a.max_seq_len), "--hot-experts", str(a.hot_experts)]
    if a.hot_experts:
        # Never fall back to the repository's older route_stats.pt implicitly.
        command += ["--hot-stats", str(a.hot_stats.expanduser().resolve())]
    if a.mode == "serve":
        command += ["--max-seqs", "1", "--mtp", "0", "--host", a.host, "--port", str(a.port)]
    else:
        temp = a.temperature if a.temperature is not None else (0.0 if a.mode == "bench" else 1.0)
        command += ["--max-new-tokens", str(a.max_new_tokens), "--temperature", str(temp)]
        if a.mode == "chat":
            command += ["--top-p", "1.0"]
        else:
            # Upstream ROUTE_STATS is updated only by the eager MoE path;
            # graph/offload decode bypasses it and would save an empty histogram.
            decode = "eager" if a.route_stats is not None else "graph"
            command += ["--decode", decode, "--chat", "--prompt", a.prompt]
            if a.route_stats is not None:
                command += ["--route-stats", str(a.route_stats.expanduser().resolve())]
            if a.profile:
                command += ["--profile"]
    env = {"DSV41_O_FP8_QDQ": "1", "DSV41_CKPT": ckpt,
           "DSV41_PREFIX_CACHE_EPOCH": CACHE_EPOCH}
    if a.mode == "serve":
        env["DSV41_INTERACTIVE_MAX_NEW"] = str(a.max_new_tokens)
    return command, env


def check_checkpoint(a: argparse.Namespace) -> None:
    """Check the original HF layout without reading weight contents or executing code.

    Layout validation is not authentication of checkpoint identity or weights.
    """
    root = a.ckpt.expanduser().resolve()
    required = ("inference/config.json", "model.safetensors.index.json",
                "tokenizer.json", "tokenizer_config.json", "encoding/encoding.py")
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise ValueError(f"incomplete HF checkpoint at {root}: {', '.join(missing)}")
    with (root / "model.safetensors.index.json").open() as f:
        index = json.load(f)
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("checkpoint index has no weight_map")
    if not all(isinstance(name, str) for name in weight_map.values()):
        raise ValueError("checkpoint shard names must be strings")
    for name in sorted(set(weight_map.values())):
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"invalid checkpoint shard path: {name}")
        # HF cache snapshots legitimately symlink shards into a sibling blobs/.
        shard = root / relative
        if not shard.is_file() or shard.stat().st_size == 0:
            raise ValueError(f"missing or empty checkpoint shard: {name}")
    if a.hot_experts and not a.hot_stats.expanduser().is_file():
        raise ValueError("missing 1017 hot-stats; calibrate with bench --hot-experts 0 "
                         "--route-stats results/route_stats_1017.pt first")


def main(argv: list[str] | None = None) -> int:
    ap = make_parser()
    a = ap.parse_args(argv)
    try:
        command, overrides = build_launch(a)
        if a.dry_run:
            print(json.dumps({"command": command, "environment": overrides}, indent=2))
            return 0
        check_checkpoint(a)
        nvcc = os.environ.get("DSV41_NVCC") or shutil.which("nvcc")
        if not nvcc and Path("/usr/local/cuda-12.8/bin/nvcc").is_file():
            nvcc = "/usr/local/cuda-12.8/bin/nvcc"
        if not nvcc or not shutil.which(nvcc):
            raise ValueError("nvcc is required; install the CUDA toolkit or set DSV41_NVCC")
        overrides["DSV41_NVCC"] = nvcc
        if a.offload_experts == "cpu" and not shutil.which("g++"):
            raise ValueError("g++ is required by CPU MoE (-march=sapphirerapids)")
        if a.route_stats is not None:
            a.route_stats.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    except (OSError, ValueError, TypeError) as e:
        ap.error(str(e))
    print("1017 FP8 output QDQ enabled; single GPU, MTP disabled.", flush=True)
    if a.route_stats is not None:
        print("Calibration uses eager decode; do not report this as optimized throughput.", flush=True)
    print(shlex.join(command), flush=True)
    # A fresh interpreter reads the numerical mode before capturing any graphs.
    os.execve(sys.executable, command, {**os.environ, **overrides})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
