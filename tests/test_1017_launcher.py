"""Launcher tests need no model, GPU, CUDA toolkit, or transformers."""
import json
import os
from pathlib import Path
import subprocess
import struct
import sys

import pytest

from dsv41 import run_1017 as launch


def args(*extra):
    return launch.make_parser().parse_args(list(extra))


@pytest.mark.parametrize("mode,module", [("bench", "run"), ("chat", "chat"), ("serve", "serve")])
def test_launch_modes(mode, module, tmp_path):
    a = args(mode, "--ckpt", str(tmp_path / "model with spaces"))
    command, env = launch.build_launch(a)
    assert command[:3] == [sys.executable, "-m", f"dsv41.{module}"]
    assert command[command.index("--devices") + 1] == "0"
    assert command[command.index("--ckpt") + 1] == env["DSV41_CKPT"]
    assert env["DSV41_O_FP8_QDQ"] == "1"
    assert env["DSV41_W8"] == "1"
    assert env["DSV41_MAX_SEQS"] == "1"
    assert env["DSV41_MODEL_NAME"] == "deepseek-v4.1-flash-1017"
    assert env["DSV41_PREFIX_CACHE_DIR"] == ""
    assert "1017" in env["DSV41_PREFIX_CACHE_EPOCH"]
    if mode == "serve":
        assert command[command.index("--mtp") + 1] == "0"
        assert command[command.index("--host") + 1] == "127.0.0.1"
        assert env["DSV41_INTERACTIVE_MAX_NEW"] == "512"
    if mode == "bench":
        assert command[command.index("--temperature") + 1] == "0.0"


@pytest.mark.parametrize("extra", [
    ["bench", "--hot-experts", "-1"], ["bench", "--hot-experts", "385"],
    ["bench", "--max-seq-len", "0"], ["bench", "--max-new-tokens", "0"],
    ["bench", "--max-new-tokens", "8192"],
    ["bench", "--offload-experts", "gpu", "--hot-experts", "8"],
    ["chat", "--route-stats", "x.pt"], ["serve", "--profile"],
    ["serve", "--temperature", "1"], ["bench", "--temperature", "nan"],
    ["bench", "--temperature", "inf"], ["serve", "--port", "0"],
])
def test_invalid_options(extra):
    with pytest.raises(ValueError):
        launch.build_launch(args(*extra))


@pytest.fixture
def checkpoint(tmp_path):
    # Synthetic layout only: this is not a safetensors or inference test.
    for name in ("inference/config.json", "tokenizer.json", "tokenizer_config.json", "encoding/encoding.py"):
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("{}")
    (tmp_path / "inference/config.json").write_text(json.dumps(model_config()))
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"weight": "part.safetensors"}}))
    write_shard(tmp_path / "part.safetensors")
    return tmp_path


def model_config():
    return {"n_layers": 40, "n_routed_experts": 384, "dim": 5120,
            "moe_inter_dim": 2304, "dtype": "fp8", "expert_dtype": "fp4",
            "engram_layer_ids": [1, 14], "engram_head_dim": 256,
            "engram_num_embeddings": [384006168, 384016682]}


def write_shard(path):
    # A real safetensors envelope with one U8 value; no safetensors dependency.
    header = json.dumps({"weight": {"dtype": "U8", "shape": [1], "data_offsets": [0, 1]}}).encode()
    path.write_bytes(struct.pack("<Q", len(header)) + header + b"x")


def test_layout_and_missing_shard(checkpoint):
    a = args("bench", "--ckpt", str(checkpoint))
    launch.check_checkpoint(a)
    (checkpoint / "part.safetensors").unlink()
    with pytest.raises(ValueError, match="checkpoint shard"):
        launch.check_checkpoint(a)


@pytest.mark.parametrize("weight_map", [{}, {"x": ["invalid"]}, {"x": "../outside.safetensors"}])
def test_bad_index(checkpoint, weight_map):
    (checkpoint / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    with pytest.raises(ValueError):
        launch.check_checkpoint(args("bench", "--ckpt", str(checkpoint)))


def test_hot_stats_must_be_explicit_1017_calibration(checkpoint):
    missing = checkpoint / "absent.pt"
    a = args("bench", "--ckpt", str(checkpoint), "--hot-experts", "16", "--hot-stats", str(missing))
    with pytest.raises(ValueError, match="calibrate"):
        launch.check_checkpoint(a)
    missing.write_bytes(b"synthetic stats")
    launch.check_checkpoint(a)
    command, _ = launch.build_launch(a)
    assert command[command.index("--hot-stats") + 1] == str(missing)


def test_dry_run_does_not_import_gpu_runtime_or_mutate_env(monkeypatch, capsys):
    monkeypatch.setenv("DSV41_O_FP8_QDQ", "0")
    assert launch.main(["bench", "--ckpt", "/nonexistent/model", "--dry-run"]) == 0
    assert json.loads(capsys.readouterr().out)["environment"]["DSV41_O_FP8_QDQ"] == "1"
    assert os.environ["DSV41_O_FP8_QDQ"] == "0"
    code = "from dsv41.run_1017 import main; import sys; assert 'torch' not in sys.modules; main(['bench','--dry-run']); assert 'torch' not in sys.modules"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_incomplete_layout_fails_before_exec(tmp_path, monkeypatch):
    def forbidden(*unused):
        pytest.fail("must not launch an incomplete model")
    monkeypatch.setattr(os, "execve", forbidden)
    with pytest.raises(SystemExit) as exc:
        launch.main(["bench", "--ckpt", str(tmp_path)])
    assert exc.value.code == 2


def test_exec_receives_1017_environment(checkpoint, monkeypatch):
    seen = {}
    monkeypatch.setenv("DSV41_O_FP8_QDQ", "0")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    monkeypatch.setenv("DSV41_MAX_SEQS", "65")
    monkeypatch.setenv("DSV41_W8", "0")
    monkeypatch.setenv("DSV41_PREFIX_CACHE_DIR", "/old-shared-cache")
    monkeypatch.setattr(launch, "host_memory", lambda: (1024 * launch.GIB, 800 * launch.GIB))
    monkeypatch.setattr(launch.shutil, "which", lambda executable: f"/usr/bin/{Path(executable).name}")
    def capture(executable, command, env):
        seen.update(command=command, env=env)
    monkeypatch.setattr(os, "execve", capture)
    assert launch.main(["serve", "--ckpt", str(checkpoint)]) == 0
    assert seen["env"]["DSV41_O_FP8_QDQ"] == "1"
    assert seen["env"]["CUDA_VISIBLE_DEVICES"] == "3"
    assert seen["env"]["DSV41_PREFIX_CACHE_EPOCH"] == launch.CACHE_EPOCH
    assert seen["env"]["DSV41_MAX_SEQS"] == "1"
    assert seen["env"]["DSV41_W8"] == "1"
    assert seen["env"]["DSV41_PREFIX_CACHE_DIR"] == ""


def test_calibration_uses_instrumented_eager_path():
    calibration, _ = launch.build_launch(args("bench", "--route-stats", "results/1017.pt"))
    benchmark, _ = launch.build_launch(args("bench"))
    assert calibration[calibration.index("--decode") + 1] == "eager"
    assert benchmark[benchmark.index("--decode") + 1] == "graph"


def test_hf_snapshot_symlinks_are_supported(checkpoint, tmp_path):
    shard = checkpoint / "part.safetensors"
    shard.unlink()
    blobs = tmp_path / "hf-blobs"
    blobs.mkdir()
    external = blobs / "content-hash"
    write_shard(external)
    # Move the fake checkpoint into snapshots/ so the symlink leaves that root.
    snapshot = tmp_path / "snapshots" / "revision"
    snapshot.mkdir(parents=True)
    for name in ("inference", "encoding", "tokenizer.json", "tokenizer_config.json", "model.safetensors.index.json"):
        (checkpoint / name).rename(snapshot / name)
    (snapshot / "part.safetensors").symlink_to(external)
    launch.check_checkpoint(args("bench", "--ckpt", str(snapshot)))


@pytest.mark.parametrize("content", [b"x", struct.pack("<Q", 1000) + b"{}"])
def test_nonempty_partial_shard_is_rejected(checkpoint, content):
    (checkpoint / "part.safetensors").write_bytes(content)
    with pytest.raises(ValueError, match="header"):
        launch.check_checkpoint(args("bench", "--ckpt", str(checkpoint)))


def test_truncated_shard_payload_is_rejected(checkpoint):
    shard = checkpoint / "part.safetensors"
    shard.write_bytes(shard.read_bytes()[:-1])
    with pytest.raises(ValueError, match="incomplete checkpoint shard"):
        launch.check_checkpoint(args("bench", "--ckpt", str(checkpoint)))


def test_index_header_mismatch_is_rejected(checkpoint):
    (checkpoint / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"other_weight": "part.safetensors"}}))
    with pytest.raises(ValueError, match="index/header mismatch"):
        launch.check_checkpoint(args("bench", "--ckpt", str(checkpoint)))


def test_host_weight_estimate_matches_full_1017_layout():
    report = launch.host_requirements(model_config())
    # Expert nibble weights + one scale per 32 columns; Engram FP8 rows + scales.
    expected_experts = 40 * 384 * (2 * 2304 * (5120 // 2 + 5120 // 32)
                                 + 5120 * (2304 // 2 + 2304 // 32))
    expected_engram = (384006168 + 384016682) * (256 + 8)
    assert report["expert_bytes"] == expected_experts
    assert report["engram_bytes"] == expected_engram
    assert 457 < report["resident_weight_bytes"] / launch.GIB < 458
    assert report["required_host_bytes"] == expected_experts + expected_engram + launch.HOST_HEADROOM


@pytest.mark.parametrize("mode", ["cpu", "gpu"])
def test_insufficient_host_ram_stops_before_exec(checkpoint, monkeypatch, mode):
    monkeypatch.setattr(launch, "host_memory", lambda: (235 * launch.GIB, 230 * launch.GIB))
    monkeypatch.setattr(os, "execve", lambda *unused: pytest.fail("must not start a doomed model load"))
    with pytest.raises(SystemExit) as exc:
        launch.main(["bench", "--ckpt", str(checkpoint), "--offload-experts", mode])
    assert exc.value.code == 2


def test_temporary_host_ram_shortage_is_reported(monkeypatch):
    monkeypatch.setattr(launch, "host_memory", lambda: (1024 * launch.GIB, 400 * launch.GIB))
    with pytest.raises(ValueError, match="wait for other jobs"):
        launch.check_host_memory(model_config())


@pytest.mark.parametrize("change", [
    {"n_layers": 0}, {"dim": 5119}, {"expert_dtype": "bf16"},
    {"engram_num_embeddings": [123]}, {"engram_head_dim": 31},
])
def test_invalid_resource_config_is_rejected(change):
    with pytest.raises(ValueError):
        launch.host_requirements({**model_config(), **change})


def test_preflight_reports_requirements_without_launching(checkpoint, monkeypatch, capsys):
    monkeypatch.delenv("DSV41_NVCC", raising=False)
    monkeypatch.setattr(launch, "host_memory", lambda: (1024 * launch.GIB, 800 * launch.GIB))
    monkeypatch.setattr(launch.shutil, "which", lambda executable: f"/usr/bin/{Path(executable).name}")
    monkeypatch.setattr(os, "execve", lambda *unused: pytest.fail("preflight must not exec"))
    assert launch.main(["bench", "--ckpt", str(checkpoint), "--preflight"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["resources"]["required_host_bytes"] > 465 * launch.GIB
    assert report["environment"]["DSV41_NVCC"] == "/usr/bin/nvcc"
    code = ("import sys; from dsv41 import run_1017 as launch; "
            "launch.host_memory=lambda: (1024*2**30,800*2**30); "
            "launch.shutil.which=lambda executable: '/usr/bin/'+executable; "
            "launch.main(['bench','--ckpt',sys.argv[1],'--preflight']); "
            "assert 'torch' not in sys.modules")
    result = subprocess.run([sys.executable, "-c", code, str(checkpoint)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_prefix_persistence_requires_explicit_directory(tmp_path):
    cache = tmp_path / "dedicated-1017"
    _, env = launch.build_launch(args("serve", "--prefix-cache-dir", str(cache)))
    assert env["DSV41_PREFIX_CACHE_DIR"] == str(cache)
    assert not cache.exists()  # launch construction and dry run are read-only.


def test_host_memory_honors_nested_cgroup_limits(monkeypatch):
    files = {
        "/proc/meminfo": "MemTotal: 1073741824 kB\nMemAvailable: 838860800 kB\n",
        "/proc/self/cgroup": "0::/parent/job\n",
        "/sys/fs/cgroup/parent/job/memory.max": str(32 * launch.GIB),
        "/sys/fs/cgroup/parent/job/memory.current": str(10 * launch.GIB),
        "/sys/fs/cgroup/parent/memory.max": str(40 * launch.GIB),
        "/sys/fs/cgroup/parent/memory.current": str(30 * launch.GIB),
        "/sys/fs/cgroup/memory.max": "max",
    }
    def read(path, *args, **kwargs):
        try:
            return files[str(path)]
        except KeyError:
            raise FileNotFoundError(str(path))
    monkeypatch.setattr(Path, "read_text", read)
    # Both the job's capacity and its parent's remaining allowance apply.
    assert launch.host_memory() == (32 * launch.GIB, 10 * launch.GIB)


def test_host_memory_checks_mount_root_if_child_cgroup_is_hidden(monkeypatch):
    files = {
        "/proc/meminfo": "MemTotal: 1073741824 kB\nMemAvailable: 838860800 kB\n",
        "/proc/self/cgroup": "0::/host-only-child\n",
        "/sys/fs/cgroup/memory.max": str(24 * launch.GIB),
        "/sys/fs/cgroup/memory.current": str(4 * launch.GIB),
    }
    def read(path, *args, **kwargs):
        try:
            return files[str(path)]
        except KeyError:
            raise FileNotFoundError(str(path))
    monkeypatch.setattr(Path, "read_text", read)
    assert launch.host_memory() == (24 * launch.GIB, 20 * launch.GIB)
