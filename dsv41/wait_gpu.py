"""Wait for an idle physical GPU, then run a command with that GPU as CUDA device 0.

Example: python -m dsv41.wait_gpu --gpu 0 --min-free-mib 70000 \
    --status /tmp/1017-status.json -- python -m dsv41.run_1017 bench

Linux flock serializes cooperating launches by GPU UUID. Other programs do not
honor this lock: polling cannot reserve VRAM against an unrelated GPU job.
The caller owns backgrounding and stdout/stderr log redirection.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time


@dataclass(frozen=True)
class GpuSample:
    index: int
    uuid: str
    free_mib: int


class NvidiaError(RuntimeError):
    """GPU availability could not be verified; never launch on uncertain data."""


def _query(executable: str, arguments: list[str], timeout: float) -> list[list[str]]:
    try:
        result = subprocess.run([executable, *arguments], capture_output=True,
                                text=True, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise NvidiaError(f"nvidia-smi query failed: {exc}") from exc
    if result.returncode:
        detail = (result.stderr.strip() or result.stdout.strip())[:400]
        raise NvidiaError(f"nvidia-smi exited {result.returncode}: {detail}")
    return [[field.strip() for field in row] for row in csv.reader(result.stdout.splitlines())
            if row and any(field.strip() for field in row)]


def read_gpu(selector: str, executable: str, timeout: float) -> GpuSample:
    rows = _query(executable, [f"--id={selector}", "--query-gpu=index,uuid,memory.free",
                               "--format=csv,noheader,nounits"], timeout)
    if len(rows) != 1 or len(rows[0]) != 3:
        raise NvidiaError("nvidia-smi did not return exactly one physical GPU")
    index, uuid, free = rows[0]
    if not index.isdecimal() or not free.isdecimal() or not re.fullmatch(r"GPU-[0-9a-fA-F-]+", uuid):
        raise NvidiaError("nvidia-smi returned invalid GPU identity or free-memory data")
    if (selector.isdecimal() and int(selector) != int(index)) or (
            selector.startswith("GPU-") and selector != uuid):
        raise NvidiaError("nvidia-smi returned a different GPU than requested")
    return GpuSample(int(index), uuid, int(free))


def compute_pids(uuid: str, executable: str, timeout: float) -> list[int]:
    rows = _query(executable, ["--query-compute-apps=gpu_uuid,pid",
                               "--format=csv,noheader,nounits"], timeout)
    pids = []
    for row in rows:
        if (len(row) != 2 or not re.fullmatch(r"GPU-[0-9a-fA-F-]+", row[0])
                or not row[1].isdecimal() or int(row[1]) <= 0):
            raise NvidiaError("nvidia-smi returned invalid compute-process data")
        if row[0] == uuid:
            pids.append(int(row[1]))
    return sorted(set(pids))


def write_status(path: Path, state: dict) -> None:
    """Readers always see a complete JSON document; status files are private."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", delete=False) as output:
            temporary = Path(output.name)
            os.fchmod(output.fileno(), 0o600)
            json.dump(state, output, sort_keys=True, indent=2)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", default="0", help="physical nvidia-smi index or full GPU UUID")
    parser.add_argument("--min-free-mib", type=int, required=True)
    parser.add_argument("--stable-polls", type=int, default=3)
    parser.add_argument("--poll-seconds", type=float, default=30)
    parser.add_argument("--timeout-seconds", type=float, default=0,
                        help="maximum queue wait; 0 waits indefinitely")
    parser.add_argument("--query-timeout-seconds", type=float, default=15)
    parser.add_argument("--stop-timeout-seconds", type=float, default=30,
                        help="seconds before escalating a forwarded signal to SIGKILL")
    parser.add_argument("--log-seconds", type=float, default=60,
                        help="minimum interval between waiting log messages")
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--lock-dir", type=Path,
                        default=Path(tempfile.gettempdir()) / f"dsv41-gpu-locks-{os.getuid()}")
    parser.add_argument("--lock-file", type=Path,
                        help="use an existing shared flock path instead of the per-UUID lock")
    parser.add_argument("--nvidia-smi", default="nvidia-smi")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="command after --")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a command after -- is required")
    if not (args.gpu.isdecimal() or re.fullmatch(r"GPU-[0-9a-fA-F-]+", args.gpu)):
        parser.error("gpu must be a physical index or full GPU UUID")
    if args.min_free_mib <= 0 or args.stable_polls <= 0:
        parser.error("min-free-mib and stable-polls must be positive")
    for name in ("poll_seconds", "query_timeout_seconds", "stop_timeout_seconds", "log_seconds"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(f"{name.replace('_', '-')} must be finite and positive")
    if not math.isfinite(args.timeout_seconds) or args.timeout_seconds < 0:
        parser.error("timeout-seconds must be finite and nonnegative")

    started = time.monotonic()
    wake = threading.Event()
    stop_signal = None
    stop_started = None
    child = None
    lock_fd = None
    have_lock = False
    last_log = float("-inf")
    state = {"phase": "waiting", "pid": os.getpid(), "child_pid": None,
             "requested_gpu": args.gpu, "gpu_uuid": None,
             "started_at": datetime.now(timezone.utc).isoformat(),
             "min_free_mib": args.min_free_mib, "stable_polls": 0,
             "required_stable_polls": args.stable_polls, "exit_code": None,
             "reason": "checking_gpu"}

    def publish(phase: str, reason: str, **updates) -> None:
        nonlocal last_log
        changed = state["phase"] != phase
        state.update(phase=phase, reason=reason,
                     updated_at=datetime.now(timezone.utc).isoformat(), **updates)
        write_status(args.status, state)
        now = time.monotonic()
        if changed or now - last_log >= args.log_seconds:
            print(f"[{state['updated_at']}] {phase}: {reason}; "
                  f"GPU={state['gpu_uuid'] or args.gpu}, free_mib={state.get('free_mib')}, "
                  f"stable={state['stable_polls']}/{args.stable_polls}", flush=True)
            last_log = now

    def forward(signum: int, _frame) -> None:
        nonlocal stop_signal, stop_started
        if stop_signal is None:
            stop_signal = signum
            stop_started = time.monotonic()
        wake.set()
        if child is not None and child.poll() is None:
            try:
                os.killpg(child.pid, signum)
            except ProcessLookupError:
                pass

    previous_handlers = {signum: signal.signal(signum, forward)
                         for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
    try:
        publish("waiting", "checking_gpu")
        selected = read_gpu(args.gpu, args.nvidia_smi, args.query_timeout_seconds)
        state.update(gpu_uuid=selected.uuid, gpu_index=selected.index, free_mib=selected.free_mib)
        lock_path = (args.lock_file or args.lock_dir / f"{selected.uuid}.lock").expanduser()
        lock_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        state["lock_file"] = str(lock_path.resolve())
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        stable = 0
        while True:
            if stop_signal is not None:
                code = 128 + stop_signal
                publish("failed", "interrupted_while_waiting", exit_code=code, signal=stop_signal)
                return code
            elapsed = time.monotonic() - started
            if args.timeout_seconds and elapsed >= args.timeout_seconds:
                publish("failed", "wait_timeout", exit_code=124)
                return 124
            if not have_lock:
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    have_lock = True
                except BlockingIOError:
                    publish("waiting", "cooperating_job_holds_lock", stable_polls=0)
            if have_lock:
                sample = read_gpu(selected.uuid, args.nvidia_smi, args.query_timeout_seconds)
                pids = compute_pids(selected.uuid, args.nvidia_smi, args.query_timeout_seconds)
                ready = sample.free_mib >= args.min_free_mib and not pids
                stable = stable + 1 if ready else 0
                reason = "confirming_idle" if ready else (
                    "compute_processes_present" if pids else "insufficient_free_vram")
                publish("waiting", reason, free_mib=sample.free_mib,
                        compute_pids=pids, stable_polls=stable)
                if stop_signal is not None:
                    continue
                if stable >= args.stable_polls:
                    # Queries can take time; do not start after the queue timeout.
                    if args.timeout_seconds and time.monotonic() - started >= args.timeout_seconds:
                        continue
                    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": selected.uuid,
                                   "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}
                    child = subprocess.Popen(command, env=environment, start_new_session=True,
                                             pass_fds=(lock_fd,))
                    if stop_signal is not None:
                        forward(stop_signal, None)
                    publish("running", "command_started", child_pid=child.pid)
                    break
            delay = args.poll_seconds
            if args.timeout_seconds:
                delay = min(delay, max(0, args.timeout_seconds - (time.monotonic() - started)))
            wake.wait(delay)

        while child.poll() is None:
            if stop_started is not None and time.monotonic() - stop_started >= args.stop_timeout_seconds:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            try:
                child.wait(timeout=0.25)
            except subprocess.TimeoutExpired:
                pass
        child_code = child.returncode
        code = (128 + stop_signal if stop_signal is not None else
                child_code if child_code >= 0 else 128 - child_code)
        publish("finished" if code == 0 else "failed",
                "command_finished" if stop_signal is None else "command_interrupted",
                exit_code=code, child_exit_code=child_code, signal=stop_signal)
        return code
    except (OSError, NvidiaError) as exc:
        # Reap our child on any supervision/status failure; never touch other jobs.
        if child is not None and child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGTERM)
                child.wait(timeout=args.stop_timeout_seconds)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            except ProcessLookupError:
                child.wait()
        print(f"GPU queue failed: {exc}", file=sys.stderr, flush=True)
        try:
            publish("failed", str(exc), exit_code=1)
        except OSError as status_error:
            print(f"Unable to write queue status: {status_error}", file=sys.stderr, flush=True)
        return 1
    finally:
        if lock_fd is not None:
            # Closing, rather than LOCK_UN, preserves an inherited child's lock
            # if the supervisor is killed before it can reap the child.
            os.close(lock_fd)
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


if __name__ == "__main__":
    raise SystemExit(main())
