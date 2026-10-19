#!/usr/bin/env bash
# Run with stdout/stderr redirected from process start, e.g. via nohup.
set -euo pipefail
umask 077
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

PYTHON="${DSV41_PYTHON:-python3}"
CKPT="${DSV41_1017_CKPT:-$HOME/models/DeepSeek-V4.1-Flash-1017}"
: "${DSV41_JOB_DIR:?Set DSV41_JOB_DIR to a unique private status directory}"
mkdir -p -- "$DSV41_JOB_DIR"
chmod 700 -- "$DSV41_JOB_DIR"
DSV41_JOB_DIR="$(cd -- "$DSV41_JOB_DIR" && pwd)"
trap 'code=$?; printf "%s\n" "$code" > "$DSV41_JOB_DIR/job.exit"' EXIT

gpu=(--gpu "${DSV41_GPU:-0}" --min-free-mib "${DSV41_MIN_FREE_MIB:-70000}"
     --stable-polls 3)
if [[ -n "${DSV41_GPU_LOCK_FILE:-}" ]]; then
    gpu+=(--lock-file "$DSV41_GPU_LOCK_FILE")
fi

# Record missing weights or host RAM independently of the GPU queue. Never
# download hundreds of GiB or allocate model memory on an unprepared host.
if "$PYTHON" -m dsv41.one_server "${gpu[@]}" \
        --status "$DSV41_JOB_DIR/resources.json" --check-only -- \
        bench --ckpt "$CKPT" --max-new-tokens 128; then
    :
else
    code=$?
    [[ "$code" == 2 ]] || exit "$code"
fi

# Synthetic kernels need no model weights. Respect the existing fleet lock.
"$PYTHON" -m dsv41.wait_gpu "${gpu[@]}" \
    --status "$DSV41_JOB_DIR/gpu-tests.json" -- \
    "$PYTHON" -m pytest -q tests/test_1017_cuda.py

# Release the GPU lock while waiting for checkpoint/RAM; reacquire and verify
# idle VRAM before each real launch. Benchmark throughput is printed to the log.
"$PYTHON" -m dsv41.one_server "${gpu[@]}" \
    --status "$DSV41_JOB_DIR/model.json" -- \
    bench --ckpt "$CKPT" --max-new-tokens 128

"$PYTHON" -m dsv41.one_server "${gpu[@]}" \
    --status "$DSV41_JOB_DIR/model.json" -- \
    serve --ckpt "$CKPT" --host 127.0.0.1 --port "${DSV41_PORT:-8107}"
