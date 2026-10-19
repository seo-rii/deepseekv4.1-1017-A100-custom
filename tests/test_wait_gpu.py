"""GPU queue supervision tests use a synthetic nvidia-smi, never CUDA or torch."""
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

from dsv41 import wait_gpu as queue


UUID = "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
OTHER_UUID = "GPU-11111111-2222-3333-4444-555555555555"


def test_gpu_queries_validate_identity_memory_and_processes(monkeypatch):
    replies = iter([[["2", UUID, "81000"]], [[UUID, "123"], [OTHER_UUID, "456"], [UUID, "123"]]])
    monkeypatch.setattr(queue, "_query", lambda *args: next(replies))
    assert queue.read_gpu("2", "nvidia-smi", 1) == queue.GpuSample(2, UUID, 81000)
    assert queue.compute_pids(UUID, "nvidia-smi", 1) == [123]


@pytest.mark.parametrize("rows", [[], [["0", UUID, "N/A"]], [["0", "N/A", "81000"]],
                                   [["1", UUID, "81000"]], [["0", UUID, "-1"]],
                                   [["0", UUID, "81000"], ["1", OTHER_UUID, "81000"]]])
def test_gpu_queries_fail_closed(monkeypatch, rows):
    monkeypatch.setattr(queue, "_query", lambda *args: rows)
    with pytest.raises(queue.NvidiaError):
        queue.read_gpu("0", "nvidia-smi", 1)


@pytest.mark.parametrize("rows", [[[UUID, "N/A"]], [["N/A", "10"]], [[UUID, "0"]], [[UUID]]])
def test_unverifiable_process_list_fails_closed(monkeypatch, rows):
    monkeypatch.setattr(queue, "_query", lambda *args: rows)
    with pytest.raises(queue.NvidiaError):
        queue.compute_pids(UUID, "nvidia-smi", 1)


def test_query_failure_and_timeout_are_explicit(monkeypatch):
    monkeypatch.setattr(queue.subprocess, "run", lambda *a, **kw: subprocess.CompletedProcess(a, 9, "", "driver unavailable"))
    with pytest.raises(queue.NvidiaError, match="exited 9"):
        queue.read_gpu("0", "nvidia-smi", 1)
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], 1)
    monkeypatch.setattr(queue.subprocess, "run", timeout)
    with pytest.raises(queue.NvidiaError, match="query failed"):
        queue.read_gpu("0", "nvidia-smi", 1)


@pytest.fixture
def fake_smi(tmp_path):
    executable = tmp_path / "nvidia-smi"
    executable.write_text(f"""#!{sys.executable}
import json, os, pathlib, sys
root = pathlib.Path(__file__).parent
if (root / 'fail').exists():
    sys.exit(9)
if any('query-gpu=' in a for a in sys.argv):
    counter = root / 'polls'
    poll = int(counter.read_text()) if counter.exists() else 0
    counter.write_text(str(poll + 1))
    values = json.loads((root / 'memory.json').read_text())
    print('0, {UUID}, ' + str(values[min(poll, len(values) - 1)]))
else:
    process_file = root / 'processes'
    if process_file.exists():
        print(process_file.read_text(), end='')
""")
    executable.chmod(0o700)
    (tmp_path / "memory.json").write_text("[81000]")
    return executable


def command(fake_smi, *child, extra=()):
    root = fake_smi.parent
    return [sys.executable, "-m", "dsv41.wait_gpu", "--gpu", "0", "--min-free-mib", "70000",
            "--poll-seconds", "0.01", "--stable-polls", "2", "--status", str(root / "status.json"),
            "--lock-dir", str(root / "locks"), "--nvidia-smi", str(fake_smi), *extra, "--", *child]


def status(fake_smi):
    return json.loads((fake_smi.parent / "status.json").read_text())


def wait_until(predicate, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    pytest.fail("timed out waiting for synthetic queue state")


def test_stability_resets_and_uuid_reaches_child(fake_smi):
    root = fake_smi.parent
    # The first query resolves identity; low memory later resets the idle count.
    (root / "memory.json").write_text("[81000, 81000, 1000, 81000, 81000]")
    output = root / "child.json"
    child = f"import json, os; from pathlib import Path; Path({str(output)!r}).write_text(json.dumps(dict(os.environ)))"
    result = subprocess.run(command(fake_smi, sys.executable, "-c", child), capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert int((root / "polls").read_text()) == 5
    assert json.loads(output.read_text())["CUDA_VISIBLE_DEVICES"] == UUID
    assert status(fake_smi)["phase"] == "finished"
    assert status(fake_smi)["exit_code"] == 0
    assert (root / "status.json").stat().st_mode & 0o777 == 0o600
    assert not list(root.glob(".status.json.*"))


def test_process_presence_blocks_launch_and_timeout(fake_smi):
    root = fake_smi.parent
    (root / "processes").write_text(f"{UUID}, 1234\n")
    output = root / "should-not-exist"
    child = f"from pathlib import Path; Path({str(output)!r}).touch()"
    result = subprocess.run(command(fake_smi, sys.executable, "-c", child,
                                    extra=("--timeout-seconds", "1")), capture_output=True, text=True)
    assert result.returncode == 124, result.stderr
    assert not output.exists()
    assert status(fake_smi)["phase"] == "failed"
    assert status(fake_smi)["reason"] == "wait_timeout"
    assert status(fake_smi)["compute_pids"] == [1234]


def test_other_gpu_process_does_not_block_and_child_failure_propagates(fake_smi):
    (fake_smi.parent / "processes").write_text(f"{OTHER_UUID}, 1234\n")
    result = subprocess.run(command(fake_smi, sys.executable, "-c", "raise SystemExit(7)"),
                            capture_output=True, text=True)
    assert result.returncode == 7, result.stderr
    assert status(fake_smi)["phase"] == "failed"
    assert status(fake_smi)["exit_code"] == status(fake_smi)["child_exit_code"] == 7


def test_driver_failure_prevents_child(fake_smi):
    root = fake_smi.parent
    (root / "fail").touch()
    output = root / "should-not-exist"
    result = subprocess.run(command(fake_smi, sys.executable, "-c", f"open({str(output)!r}, 'w').close()"),
                            capture_output=True, text=True)
    assert result.returncode == 1
    assert not output.exists()
    assert status(fake_smi)["phase"] == "failed"
    assert "nvidia-smi exited 9" in status(fake_smi)["reason"]


def test_full_uuid_selector_and_missing_command_error(fake_smi):
    args = command(fake_smi, "/missing/command")
    args[args.index("--gpu") + 1] = UUID
    result = subprocess.run(args, capture_output=True, text=True)
    assert result.returncode == 1
    assert status(fake_smi)["gpu_uuid"] == UUID
    assert status(fake_smi)["child_pid"] is None
    assert status(fake_smi)["phase"] == "failed"


def test_failed_atomic_replace_keeps_previous_status(tmp_path, monkeypatch):
    output = tmp_path / "status.json"
    queue.write_status(output, {"phase": "waiting"})
    def fail(*args):
        raise OSError("synthetic failed replacement")
    monkeypatch.setattr(queue.os, "replace", fail)
    with pytest.raises(OSError, match="replacement"):
        queue.write_status(output, {"phase": "running"})
    assert json.loads(output.read_text()) == {"phase": "waiting"}
    assert not list(tmp_path.glob(".status.json.*"))


def test_lock_contention_waits_without_starting_child(fake_smi):
    root = fake_smi.parent
    (root / "locks").mkdir()
    with (root / "locks" / f"{UUID}.lock").open("w") as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = subprocess.run(command(fake_smi, sys.executable, "-c", "raise SystemExit(99)",
                                        extra=("--timeout-seconds", "0.15")), capture_output=True, text=True)
    assert result.returncode == 124, result.stderr
    assert status(fake_smi)["child_pid"] is None
    # Only identity resolution should have queried memory while another queue owns the lock.
    assert (root / "polls").read_text() == "1"


def test_explicit_shared_lock_cooperates_with_other_queues(fake_smi):
    shared = fake_smi.parent / "shared-gpu.lock"
    with shared.open("w") as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = subprocess.run(command(fake_smi, sys.executable, "-c", "raise SystemExit(99)",
                                        extra=("--lock-file", str(shared), "--timeout-seconds", "0.5")),
                                capture_output=True, text=True)
    assert result.returncode == 124, result.stderr
    assert status(fake_smi)["child_pid"] is None
    assert status(fake_smi)["lock_file"] == str(shared)
    assert not (fake_smi.parent / "locks").exists()


def test_sigterm_while_waiting_never_touches_other_processes(fake_smi):
    root = fake_smi.parent
    (root / "processes").write_text(f"{UUID}, 1234\n")
    process = subprocess.Popen(command(fake_smi, sys.executable, "-c", "raise SystemExit(99)"),
                               stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    try:
        wait_until(lambda: (root / "status.json").exists() and status(fake_smi).get("compute_pids") == [1234])
        process.send_signal(signal.SIGTERM)
        _, stderr = process.communicate(timeout=5)
        assert process.returncode == 143, stderr
        assert status(fake_smi)["reason"] == "interrupted_while_waiting"
        assert status(fake_smi)["child_pid"] is None
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def test_sigterm_reaches_child_and_releases_lock(fake_smi):
    root = fake_smi.parent
    marker = root / "signaled"
    ready = root / "child-ready"
    child = ("import signal, time; from pathlib import Path; "
             f"signal.signal(signal.SIGTERM, lambda *a: (Path({str(marker)!r}).touch(), exit(0))); "
             f"Path({str(ready)!r}).touch(); time.sleep(60)")
    process = subprocess.Popen(command(fake_smi, sys.executable, "-c", child),
                               stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    try:
        wait_until(lambda: ready.exists())
        with (root / "locks" / f"{UUID}.lock").open("r") as held:
            with pytest.raises(BlockingIOError):
                fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        process.send_signal(signal.SIGTERM)
        _, stderr = process.communicate(timeout=5)
        assert process.returncode == 143, stderr
        assert marker.exists()
        assert status(fake_smi)["reason"] == "command_interrupted"
        with (root / "locks" / f"{UUID}.lock").open("r") as held:
            fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def test_ignored_signal_is_escalated_only_for_child_group(fake_smi):
    root = fake_smi.parent
    ready = root / "child-ready"
    child = ("import signal, time; from pathlib import Path; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
             f"Path({str(ready)!r}).touch(); time.sleep(60)")
    process = subprocess.Popen(command(fake_smi, sys.executable, "-c", child,
                                       extra=("--stop-timeout-seconds", "0.1")),
                               stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    try:
        wait_until(lambda: ready.exists())
        process.send_signal(signal.SIGTERM)
        _, stderr = process.communicate(timeout=5)
        assert process.returncode == 143, stderr
        assert status(fake_smi)["child_exit_code"] == -signal.SIGKILL
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


@pytest.mark.parametrize("extra", [("--poll-seconds", "nan"), ("--poll-seconds", "0"),
                                     ("--stable-polls", "0"), ("--gpu", "MIG-bad"),
                                     ("--timeout-seconds", "-1"), ("--min-free-mib", "0")])
def test_invalid_options_fail_before_query(tmp_path, extra):
    with pytest.raises(SystemExit) as exc:
        queue.main(["--min-free-mib", "70000", "--status", str(tmp_path / "status"),
                    *extra, "--", "unused"])
    assert exc.value.code == 2


def test_module_imports_no_gpu_runtime():
    result = subprocess.run([sys.executable, "-c", "import dsv41.wait_gpu, sys; assert 'torch' not in sys.modules"],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
