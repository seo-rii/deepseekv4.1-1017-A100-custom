"""Resource waits must never allocate CUDA or start an incomplete model."""
import json
import os
import signal
import subprocess
import sys

from dsv41 import one_server


def test_cancellation_at_ready_transition_does_not_launch(monkeypatch, tmp_path):
    status = tmp_path / "status.json"
    monkeypatch.setattr(one_server.run_1017, "preflight", lambda _a: {"resources": {}})
    original = one_server.write_status
    def interrupt(path, state):
        original(path, state)
        if state["phase"] == "ready_for_gpu_queue":
            os.kill(os.getpid(), signal.SIGTERM)
    monkeypatch.setattr(one_server, "write_status", interrupt)
    def forbidden(*_args):
        raise AssertionError("must not launch a cancelled job")
    monkeypatch.setattr(one_server.os, "execv", forbidden)
    assert one_server.main(["--status", str(status), "--", "bench"]) == 143
    assert json.loads(status.read_text())["exit_code"] == 143


def test_missing_checkpoint_check_does_not_import_torch(tmp_path):
    status = tmp_path / "status.json"
    command = [sys.executable, "-c",
               "from dsv41.one_server import main; import sys; "
               "code=main(sys.argv[1:]); assert 'torch' not in sys.modules; sys.exit(code)",
               "--status", str(status), "--check-only", "--", "bench", "--ckpt", str(tmp_path / "missing")]
    result = subprocess.run(command, capture_output=True, text=True, timeout=20)
    assert result.returncode == 2, result.stderr
    state = json.loads(status.read_text())
    assert state["phase"] == "blocked"
    assert "incomplete HF checkpoint" in state["reason"]


def test_resources_rechecked_before_gpu_queue(monkeypatch, tmp_path):
    calls = []
    def preflight(_args):
        calls.append("check")
        if calls.count("check") == 1:
            raise ValueError("insufficient available host RAM")
        return {"resources": {"required_host_bytes": 1}}
    monkeypatch.setattr(one_server.run_1017, "preflight", preflight)
    def capture(_exe, argv):
        calls.append(argv)
        raise SystemExit(0)  # exec normally does not return.
    monkeypatch.setattr(one_server.os, "execv", capture)
    import pytest
    with pytest.raises(SystemExit) as exc:
        one_server.main(["--status", str(tmp_path / "status.json"),
                         "--resource-poll-seconds", "0.001", "--lock-file", str(tmp_path / "shared.lock"),
                         "--", "bench", "--ckpt", str(tmp_path / "model")])
    assert exc.value.code == 0
    assert calls[:2] == ["check", "check"]
    queue = calls[-1]
    assert queue[1:3] == ["-m", "dsv41.wait_gpu"]
    assert queue[queue.index("--lock-file") + 1] == str(tmp_path / "shared.lock")
    child = queue[queue.index("--") + 1:]
    assert child[:4] == [sys.executable, "-u", "-m", "dsv41.run_1017"]


def test_sigterm_during_resource_wait_persists_exit_status(tmp_path):
    import signal
    import time
    status = tmp_path / "status.json"
    process = subprocess.Popen([sys.executable, "-m", "dsv41.one_server", "--status", str(status),
                                "--", "bench", "--ckpt", str(tmp_path / "missing")],
                               stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        deadline = time.monotonic() + 10
        while not status.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert status.exists()
        process.send_signal(signal.SIGTERM)
        _, stderr = process.communicate(timeout=10)
        assert process.returncode == 143, stderr
        state = json.loads(status.read_text())
        assert state["phase"] == "failed" and state["exit_code"] == 143
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
