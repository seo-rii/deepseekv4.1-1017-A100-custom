"""Optional real-kernel A100 smoke tests. No model weights are required.

These tests intentionally skip on CPU-only hosts; CPU test success is not CUDA
numerical validation or a model throughput result.
"""
import pytest
import torch

from dsv41 import quant, w8

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU and CUDA toolkit")


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("tokens", [1, 5, 513])
def test_real_projection_and_cuda_graph(monkeypatch, packed, tokens):
    from dsv41.fused import fake_quant_fp8
    monkeypatch.setattr(w8, "O_FP8_QDQ", True)
    torch.manual_seed(1017)
    device = torch.device("cuda:0")
    x = (torch.randn(1, tokens, 2, 64, device=device) * 3).to(torch.bfloat16)
    if packed:
        weight = w8.W8(torch.randn(64, 64, device=device).to(torch.float8_e4m3fn).view(torch.uint8),
                       torch.full((2, 2), 124, dtype=torch.uint8, device=device))
        ref_weight = weight.bf16()
        # Fail rather than silently using w8.oproj_a's existing GEMM fallback.
        def no_fallback(self):
            raise AssertionError("unexpected CUDA GEMM fallback")
        monkeypatch.setattr(w8.W8, "bf16", no_fallback)
    else:
        weight = ref_weight = torch.randn(64, 64, device=device).to(torch.bfloat16)
    torch.testing.assert_close(fake_quant_fp8(x), quant.fake_quant_fp8(x), rtol=0, atol=0)
    for _ in range(3):
        w8.oproj_a(x, weight, 2, 32)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = w8.oproj_a(x, weight, 2, 32)
    for factor in (1.0, 0.5):
        x.mul_(factor)
        graph.replay()
        torch.cuda.synchronize()
        reference = torch.einsum("bsgd,grd->bsgr", quant.fake_quant_fp8(x),
                                 ref_weight.view(2, 32, 64)).flatten(2)
        # Different GEMM accumulation order is not expected to be bit-identical.
        relative_l2 = (result.float() - reference.float()).norm() / reference.float().norm().clamp_min(1e-6)
        assert relative_l2.item() < 0.01
