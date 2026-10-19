"""Optional real-kernel A100 smoke tests. No model weights are required.

These tests intentionally skip on CPU-only hosts; CPU test success is not CUDA
numerical validation or a model throughput result.
"""
import pytest
import torch

from dsv41 import quant, w8

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU and CUDA toolkit")


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("tokens,rank,dim", [(1, 32, 64), (5, 32, 64), (513, 32, 64),
                                          (17, 128, 128), (65, 128, 128), (513, 128, 128)])
def test_real_projection_and_cuda_graph(monkeypatch, packed, tokens, rank, dim):
    from dsv41.fused import fake_quant_fp8
    monkeypatch.setattr(w8, "O_FP8_QDQ", True)
    torch.manual_seed(1017)
    device = torch.device("cuda:0")
    x = (torch.randn(1, tokens, 2, dim, device=device) * 3).to(torch.bfloat16)
    if packed:
        weight = w8.W8(torch.randn(2 * rank, dim, device=device).to(torch.float8_e4m3fn).view(torch.uint8),
                       torch.full((2 * rank // 32, dim // 32), 124, dtype=torch.uint8, device=device))
        ref_weight = weight.bf16()
        # Fail rather than silently using w8.oproj_a's existing GEMM fallback.
        def no_fallback(self):
            raise AssertionError("unexpected CUDA GEMM fallback")
        monkeypatch.setattr(w8.W8, "bf16", no_fallback)
    else:
        weight = ref_weight = torch.randn(2 * rank, dim, device=device).to(torch.bfloat16)
    torch.testing.assert_close(fake_quant_fp8(x), quant.fake_quant_fp8(x), rtol=0, atol=0)
    for _ in range(3):
        w8.oproj_a(x, weight, 2, rank)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = w8.oproj_a(x, weight, 2, rank)
    for factor in (1.0, 0.5):
        x.mul_(factor)
        graph.replay()
        torch.cuda.synchronize()
        reference = torch.einsum("bsgd,grd->bsgr", quant.fake_quant_fp8(x),
                                 ref_weight.view(2, rank, dim)).flatten(2)
        # Different GEMM accumulation order is not expected to be bit-identical.
        relative_l2 = (result.float() - reference.float()).norm() / reference.float().norm().clamp_min(1e-6)
        assert relative_l2.item() < 0.01


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_cuda_qdq_scale_boundaries_and_rounding_ties(dtype):
    from dsv41.fused import fake_quant_fp8
    # Exercise exact scale boundaries, tiny/zero groups and ties on the E4M3
    # grid. Wider grouped projections above also dispatch the tcw/tcg kernels.
    maxes = torch.tensor([0.0, 1e-7, 1e-4, 7 * 2.0**-20, 3.5, 7.0, 448.0])
    maxes = torch.cat((maxes, torch.nextafter(torch.tensor([7.0]), torch.tensor([0.0])),
                      torch.nextafter(torch.tensor([7.0]), torch.tensor([float("inf")]))))
    values = torch.tensor([0.0, -0.0, 1 / 1024, 3 / 1024, 5 / 1024,
                           1.0625, 1.1875, 1.9375, 248.0, 432.0, -432.0, -1.0625])
    rows = maxes[:, None] * torch.linspace(-1, 1, 32)[None, :]
    ties = torch.zeros(1, 32)
    ties[0, :len(values)] = values
    ties[0, -1] = 448.0  # Scale is exactly one for the tie cases.
    # A strided input also covers the contiguous copy in the fused wrapper.
    source = torch.zeros(len(rows) + 1, 64, device="cuda", dtype=dtype)
    source[:, ::2] = torch.cat((rows, ties)).to(device="cuda", dtype=dtype)
    x = source[:, ::2]
    torch.testing.assert_close(fake_quant_fp8(x), quant.fake_quant_fp8(x), rtol=0, atol=0)
