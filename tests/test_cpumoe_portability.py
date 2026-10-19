"""Small native CPU tests; no CUDA context, checkpoint, or large allocations."""
import ctypes
import os

import pytest
import torch

from dsv41 import cpumoe
from dsv41.quant import dequant_fp4, fake_quant_fp8


@pytest.fixture(scope="module")
def native():
    cpu = min(os.sched_getaffinity(0))
    old = os.environ.get("DSV41_CPU_LIST")
    old_cores = os.environ.get("DSV41_CORES_PER_NODE")
    os.environ["DSV41_CPU_LIST"] = str(cpu)
    os.environ["DSV41_CORES_PER_NODE"] = "1"
    try:
        yield cpumoe.lib()
    finally:
        for name, value in (("DSV41_CPU_LIST", old), ("DSV41_CORES_PER_NODE", old_cores)):
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def int8_qdq(x):
    groups = x.float().reshape(-1, 32)
    scale = groups.abs().amax(-1, keepdim=True) / 127
    scale = torch.where(scale > 0, scale, 1)
    return (torch.round(groups / scale).clamp(-127, 127) * scale).reshape_as(x)


@pytest.mark.parametrize("mode", [0, 1, 2])
def test_native_experts_match_reference(native, mode):
    torch.manual_seed(1017)
    native.cpumoe_set_int8(mode)
    E, inter, dim = 2, 256, 128
    host = cpumoe.HostExperts(E, inter, dim)
    shapes = ((inter, dim // 2), (inter, dim // 2), (inter, dim // 32),
              (inter, dim // 32), (dim, inter // 2), (dim, inter // 32))
    tensors = [[torch.randint(0, 256, shape, dtype=torch.uint8) if i in (0, 1, 4)
                else torch.randint(119, 123, shape, dtype=torch.uint8) for _ in range(E)]
               for i, shape in enumerate(shapes)]
    # Interleaved storage creates noncontiguous checkpoint views. Loading must
    # retain their contiguous copies until the native copy has completed.
    sources = []
    for tensors_for_matrix in tensors:
        views = []
        for t in tensors_for_matrix:
            storage = torch.empty(t.shape[0], t.shape[1] * 2, dtype=t.dtype)
            storage[:, ::2] = t
            views.append(storage[:, ::2])
        sources.append(views)
    host.load_layer(*sources)
    w13, s13, w2, s2 = host.views()
    for e in range(E):
        torch.testing.assert_close(w13[e], torch.cat((tensors[0][e], tensors[1][e])))
        torch.testing.assert_close(s13[e], torch.cat((tensors[2][e], tensors[3][e])))
        torch.testing.assert_close(w2[e], tensors[4][e])
        torch.testing.assert_close(s2[e], tensors[5][e])
    x = (torch.randn(dim) * .25).to(torch.bfloat16)
    weights = [.7, .3]
    expected = torch.zeros(dim)
    activation = int8_qdq(x) if mode else x.float()
    for e, weight in enumerate(weights):
        gate = dequant_fp4(tensors[0][e], tensors[2][e]).float() @ activation
        up = dequant_fp4(tensors[1][e], tensors[3][e]).float() @ activation
        h = (weight * torch.nn.functional.silu(gate.clamp(max=2)) * up.clamp(-2, 2)).to(torch.bfloat16)
        h = fake_quant_fp8(h)
        hidden = int8_qdq(h) if mode else h.float()
        expected += dequant_fp4(tensors[4][e], tensors[5][e]).float() @ hidden
    torch.testing.assert_close(host.forward(x, [0, 1], weights, 2), expected, rtol=2e-4, atol=2e-5)
    ids = torch.tensor([0, 1], dtype=torch.int32)
    routing = torch.tensor(weights, dtype=torch.float32)
    actual = torch.empty(dim, dtype=torch.float32)
    host.forward_ids(x.data_ptr(), ids.data_ptr(), routing.data_ptr(), 2, actual.data_ptr(), 2)
    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-5)
    actual.fill_(123)
    host.forward_ids(x.data_ptr(), ids.data_ptr(), routing.data_ptr(), 0, actual.data_ptr(), 2)
    assert torch.count_nonzero(actual) == 0


def test_default_workers_honor_process_affinity(monkeypatch):
    allowed = sorted(os.sched_getaffinity(0))[:2]
    monkeypatch.setattr(os, "sched_getaffinity", lambda _: set(allowed))
    monkeypatch.delenv("DSV41_CORES_PER_NODE", raising=False)
    workers = cpumoe._expand(cpumoe.default_cpu_list(reserve_cores=0))
    assert workers
    assert set(workers).issubset(allowed)


def test_worker_groups_are_checked_before_native_array_writes(native):
    cpu = min(os.sched_getaffinity(0))
    cpus = (ctypes.c_int * 1)(cpu)
    assert native.cpumoe_init_list(cpus, 257, 1) == -1
    assert native.cpumoe_init_list(cpus, 1, 0) == -1
    assert native.cpumoe_init_list(cpus, 1, 2) == -1


def test_forward_preserves_main_thread_affinity_set_after_init(native):
    allowed = os.sched_getaffinity(0)
    reserved = {max(allowed)}
    host = cpumoe.HostExperts(1, 256, 128)
    try:
        os.sched_setaffinity(0, reserved)
        host.forward(torch.zeros(128, dtype=torch.bfloat16), [], [], 2)
        assert os.sched_getaffinity(0) == reserved
    finally:
        os.sched_setaffinity(0, allowed)


@pytest.mark.parametrize("inter,dim", [(255, 128), (4352, 128), (256, 96), (256, 8320)])
def test_scratch_buffer_shapes_rejected_before_allocation(inter, dim):
    with pytest.raises(ValueError, match="CPU MoE requires"):
        cpumoe.HostExperts(2, inter, dim)
