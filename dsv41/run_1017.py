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
import struct
import sys


CACHE_EPOCH = "fullstate-v5-index-engram-dspark-1017-o-qdq-v1"
GIB = 2**30
# Room for Python, CUDA host buffers, tokenizer state and allocation peaks. The
# weight estimate below is a lower bound, not a promise about peak RSS.
HOST_HEADROOM = 8 * GIB


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
    ap.add_argument("--prefix-cache-dir", type=Path,
                    help="opt in to process-restart prefix persistence in a dedicated 1017 directory")
    ap.add_argument("--preflight", action="store_true",
                    help="validate files, host RAM and toolchain; print requirements without importing torch")
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
           "DSV41_W8": "1", "DSV41_MAX_SEQS": "1",
           "DSV41_MODEL_NAME": "deepseek-v4.1-flash-1017",
           "DSV41_PREFIX_CACHE_EPOCH": CACHE_EPOCH,
           # Do not scan, evict or write the legacy server's shared tmpfs cache.
           # In-memory prefix reuse remains enabled. Persistence is explicit.
           "DSV41_PREFIX_CACHE_DIR": (str(a.prefix_cache_dir.expanduser().resolve())
                                       if a.prefix_cache_dir is not None else "")}
    if a.mode == "serve":
        env["DSV41_INTERACTIVE_MAX_NEW"] = str(a.max_new_tokens)
    return command, env


def host_requirements(cfg: dict) -> dict[str, int]:
    """Resident host weight copies made by load_model, without touching shards.

    Both CPU-compute and GPU-streaming offload retain packed experts in RAM.
    Engram weights and scales are cloned into RAM in either mode.
    """
    def positive(name: str) -> int:
        value = cfg.get(name)
        if type(value) is not int or value <= 0:
            raise ValueError(f"checkpoint config requires a positive integer {name}")
        return value

    layers, experts = positive("n_layers"), positive("n_routed_experts")
    dim, inter = positive("dim"), positive("moe_inter_dim")
    if cfg.get("expert_dtype") != "fp4" or cfg.get("dtype") != "fp8":
        raise ValueError("1017 requires the original FP8/FP4 inference config")
    if dim % 32 or inter % 32:
        raise ValueError("checkpoint expert dimensions must be multiples of 32")
    expert_parameters = layers * experts * 3 * dim * inter
    expert_bytes = expert_parameters // 2 + expert_parameters // 32
    engram_bytes = 0
    engram_layers = cfg.get("engram_layer_ids", [])
    if engram_layers:
        rows = cfg.get("engram_num_embeddings")
        width = positive("engram_head_dim")
        if (not isinstance(rows, list) or len(rows) != len(engram_layers)
                or any(type(n) is not int or n <= 0 for n in rows) or width % 32):
            raise ValueError("checkpoint has invalid Engram table dimensions")
        engram_bytes = sum(rows) * (width + width // 32)
    return {"expert_bytes": expert_bytes, "engram_bytes": engram_bytes,
            "resident_weight_bytes": expert_bytes + engram_bytes,
            "headroom_bytes": HOST_HEADROOM,
            "required_host_bytes": expert_bytes + engram_bytes + HOST_HEADROOM}


def host_memory() -> tuple[int, int]:
    """Return effective (capacity, available) bytes, respecting Linux cgroups."""
    mem = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        mem[key] = int(value.split()[0]) * 1024
    capacity, available = mem["MemTotal"], mem["MemAvailable"]
    # cgroup v2 limits can be lower than the host values visible in /proc.
    try:
        for line in Path("/proc/self/cgroup").read_text().splitlines():
            if line.startswith("0::"):
                group = Path("/sys/fs/cgroup") / line[3:].lstrip("/")
                for directory in (group, *group.parents):
                    if directory == Path("/sys/fs"):
                        break
                    try:
                        limit_text = (directory / "memory.max").read_text().strip()
                        if limit_text != "max":
                            limit = int(limit_text)
                            used = int((directory / "memory.current").read_text())
                            capacity = min(capacity, limit)
                            available = min(available, max(0, limit - used))
                    except (OSError, ValueError):
                        continue
                break
    except (OSError, ValueError):
        pass  # Bare-metal and cgroup-v1 hosts still have /proc/meminfo.
    return capacity, available


def check_host_memory(cfg: dict) -> dict[str, int]:
    report = host_requirements(cfg)
    capacity, available = host_memory()
    report.update(host_capacity_bytes=capacity, host_available_bytes=available)
    required = report["required_host_bytes"]
    if required > capacity:
        raise ValueError(f"insufficient host RAM capacity: offloaded expert/Engram copies need at least "
                         f"{report['resident_weight_bytes'] / GIB:.1f} GiB plus "
                         f"{HOST_HEADROOM / GIB:.0f} GiB headroom; host limit is {capacity / GIB:.1f} GiB. "
                         "Freeing VRAM or disk space cannot resolve this host RAM limit")
    if required > available:
        raise ValueError(f"insufficient available host RAM: need at least {required / GIB:.1f} GiB, "
                         f"currently {available / GIB:.1f} GiB; wait for other jobs to finish")
    return report


def check_checkpoint(a: argparse.Namespace) -> dict:
    """Check the original HF layout without reading weight contents or executing code.

    Layout validation is not authentication of checkpoint identity or weights.
    """
    root = a.ckpt.expanduser().resolve()
    required = ("inference/config.json", "model.safetensors.index.json",
                "tokenizer.json", "tokenizer_config.json", "encoding/encoding.py")
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise ValueError(f"incomplete HF checkpoint at {root}: {', '.join(missing)}")
    with (root / "inference/config.json").open() as f:
        cfg = json.load(f)
    if not isinstance(cfg, dict):
        raise ValueError("checkpoint inference config must be an object")
    host_requirements(cfg)
    if a.hot_experts > cfg["n_routed_experts"]:
        raise ValueError("hot-experts exceeds the checkpoint's routed expert count")
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
        # A nonempty partial download is not a usable checkpoint. Check its
        # safetensors envelope without allocating or reading any tensor data.
        size = shard.stat().st_size
        with shard.open("rb") as f:
            raw_length = f.read(8)
            if len(raw_length) != 8:
                raise ValueError(f"invalid safetensors header: {name}")
            header_length = struct.unpack("<Q", raw_length)[0]
            if not 0 < header_length <= min(100_000_000, size - 8):
                raise ValueError(f"invalid or truncated safetensors header: {name}")
            try:
                header = json.loads(f.read(header_length))
            except (ValueError, UnicodeError) as e:
                raise ValueError(f"invalid safetensors header: {name}") from e
        if not isinstance(header, dict):
            raise ValueError(f"invalid safetensors header: {name}")
        offsets = []
        for key, meta in header.items():
            if key == "__metadata__":
                continue
            span = meta.get("data_offsets") if isinstance(meta, dict) else None
            if (not isinstance(span, list) or len(span) != 2
                    or any(type(v) is not int for v in span) or not 0 <= span[0] <= span[1]):
                raise ValueError(f"invalid safetensors offsets: {name}")
            offsets.append(span[1])
        if not offsets or max(offsets) != size - 8 - header_length:
            raise ValueError(f"truncated or incomplete checkpoint shard: {name}")
        if any(key not in header for key, filename in weight_map.items() if filename == name):
            raise ValueError(f"checkpoint index/header mismatch: {name}")
    if a.hot_experts and not a.hot_stats.expanduser().is_file():
        raise ValueError("missing 1017 hot-stats; calibrate with bench --hot-experts 0 "
                         "--route-stats results/route_stats_1017.pt first")
    return cfg


def preflight(a: argparse.Namespace) -> dict:
    """Validate a launch without importing torch, allocating CUDA or executing it."""
    command, overrides = build_launch(a)
    cfg = check_checkpoint(a)
    resources = check_host_memory(cfg)
    nvcc = os.environ.get("DSV41_NVCC") or shutil.which("nvcc")
    if not nvcc and Path("/usr/local/cuda-12.8/bin/nvcc").is_file():
        nvcc = "/usr/local/cuda-12.8/bin/nvcc"
    if not nvcc or not shutil.which(nvcc):
        raise ValueError("nvcc is required; install the CUDA toolkit or set DSV41_NVCC")
    overrides["DSV41_NVCC"] = nvcc
    if a.offload_experts == "cpu" and not shutil.which("g++"):
        raise ValueError("g++ is required by CPU MoE")
    return {"command": command, "environment": overrides, "resources": resources}


def main(argv: list[str] | None = None) -> int:
    ap = make_parser()
    a = ap.parse_args(argv)
    try:
        command, overrides = build_launch(a)
        if a.dry_run:
            print(json.dumps({"command": command, "environment": overrides}, indent=2))
            return 0
        report = preflight(a)
        command, overrides = report["command"], report["environment"]
        if a.preflight:
            print(json.dumps(report, indent=2))
            return 0
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
