"""Launcher tests need no model, GPU, CUDA toolkit, or transformers."""
import json
import os
from pathlib import Path
import subprocess
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
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"weight": "part.safetensors"}}))
    (tmp_path / "part.safetensors").write_bytes(b"synthetic placeholder")
    return tmp_path


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
    monkeypatch.setattr(launch.shutil, "which", lambda executable: f"/usr/bin/{Path(executable).name}")
    def capture(executable, command, env):
        seen.update(command=command, env=env)
    monkeypatch.setattr(os, "execve", capture)
    assert launch.main(["serve", "--ckpt", str(checkpoint)]) == 0
    assert seen["env"]["DSV41_O_FP8_QDQ"] == "1"
    assert seen["env"]["CUDA_VISIBLE_DEVICES"] == "3"
    assert seen["env"]["DSV41_PREFIX_CACHE_EPOCH"] == launch.CACHE_EPOCH


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
    external.write_bytes(b"synthetic shard in HF cache")
    # Move the fake checkpoint into snapshots/ so the symlink leaves that root.
    snapshot = tmp_path / "snapshots" / "revision"
    snapshot.mkdir(parents=True)
    for name in ("inference", "encoding", "tokenizer.json", "tokenizer_config.json", "model.safetensors.index.json"):
        (checkpoint / name).rename(snapshot / name)
    (snapshot / "part.safetensors").symlink_to(external)
    launch.check_checkpoint(args("bench", "--ckpt", str(snapshot)))
