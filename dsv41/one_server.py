"""Wait for checkpoint/host resources, then queue a single-GPU 1017 launch.

No dependencies or weights are downloaded. The caller owns backgrounding and
private log redirection. Pass the run_1017 arguments after --.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import re
import signal
import sys
import threading

from . import run_1017
from .wait_gpu import write_status


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--min-free-mib", type=int, default=70000)
    parser.add_argument("--stable-polls", type=int, default=3)
    parser.add_argument("--gpu-poll-seconds", type=float, default=30)
    parser.add_argument("--resource-poll-seconds", type=float, default=300)
    parser.add_argument("--lock-file", type=Path)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("launcher_args", nargs=argparse.REMAINDER)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)
    launcher_args = args.launcher_args[1:] if args.launcher_args[:1] == ["--"] else args.launcher_args
    launch = run_1017.make_parser().parse_args(launcher_args)
    if launch.dry_run or launch.preflight:
        parser.error("use --check-only before -- for a readiness check")
    try:
        run_1017.build_launch(launch)  # Invalid arguments must not wait forever.
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    for value in (args.resource_poll_seconds, args.gpu_poll_seconds):
        if not math.isfinite(value) or value <= 0:
            parser.error("poll intervals must be finite and positive")
    if args.min_free_mib <= 0 or args.stable_polls <= 0:
        parser.error("memory threshold and stable poll count must be positive")
    # Validate queue options before potentially waiting for a missing checkpoint.
    from .wait_gpu import make_parser as gpu_parser
    queue = [sys.executable, "-m", "dsv41.wait_gpu", "--gpu", args.gpu,
             "--min-free-mib", str(args.min_free_mib), "--stable-polls", str(args.stable_polls),
             "--poll-seconds", str(args.gpu_poll_seconds), "--status", str(args.status)]
    if args.lock_file is not None:
        queue += ["--lock-file", str(args.lock_file)]
    queue += ["--", sys.executable, "-u", "-m", "dsv41.run_1017", *launcher_args]
    # The full runtime repeats preflight immediately before loading weights.
    gpu_parser().parse_args(queue[3:])
    if not (args.gpu.isdecimal() or re.fullmatch(r"GPU-[0-9a-fA-F-]+", args.gpu)):
        parser.error("gpu must be a physical index or full GPU UUID")

    wake = threading.Event()
    stopped = None
    state = {"phase": "checking_resources", "pid": os.getpid(),
             "checkpoint": str(launch.ckpt.expanduser().resolve()), "exit_code": None,
             "started_at": datetime.now(timezone.utc).isoformat()}

    def publish(phase: str, reason: str, **updates) -> None:
        state.update(phase=phase, reason=reason,
                     updated_at=datetime.now(timezone.utc).isoformat(), **updates)
        write_status(args.status, state)

    def stop(signum, _frame):
        nonlocal stopped
        if stopped is None:
            stopped = signum
        wake.set()

    previous = {sig: signal.signal(sig, stop)
                for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)}
    last_error = None
    try:
        while stopped is None:
            try:
                report = run_1017.preflight(launch)
            except (ValueError, OSError, TypeError) as exc:
                error = str(exc)
                publish("waiting_for_resources", error)
                if error != last_error:
                    print(f"[{state['updated_at']}] waiting_for_resources: {error}", flush=True)
                    last_error = error
                if args.check_only:
                    publish("blocked", error, exit_code=2)
                    return 2
                wake.wait(args.resource_poll_seconds)
                continue
            if stopped is not None:
                break
            publish("ready_for_gpu_queue", "checkpoint_and_host_preflight_passed",
                    resources=report["resources"])
            if stopped is not None:
                break
            if args.check_only:
                publish("ready", "GPU availability has not been checked", exit_code=0)
                return 0
            print(f"[{state['updated_at']}] host ready; waiting for idle GPU {args.gpu}", flush=True)
            if stopped is not None:
                break
            try:
                os.execv(sys.executable, queue)
            except OSError as exc:
                publish("failed", str(exc), exit_code=1)
                return 1
        code = 128 + stopped
        publish("failed", "interrupted_while_waiting_for_resources", exit_code=code, signal=stopped)
        return code
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    raise SystemExit(main())
