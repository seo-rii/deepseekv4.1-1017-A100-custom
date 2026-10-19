"""CPU numerical/control-flow tests; the CUDA GEMM is mocked where stated."""
import sys
from types import SimpleNamespace

import pytest
import torch

from dsv41 import quant, w8


def reference_qdq(x):
    """Independent per-32 reference: pow2 scale, E4M3 rounding, original dtype."""
    rows = x.float().reshape(-1, 32)
    scale = torch.exp2(torch.ceil(torch.log2(rows.abs().amax(-1, keepdim=True).clamp_min(1e-4) / 448)))
    return ((rows / scale).clamp(-448, 448).to(torch.float8_e4m3fn).float() * scale).reshape(x.shape).to(x.dtype)


def projection(x, weight, groups=2, rank=32):
    return torch.einsum("bsgd,grd->bsgr", x, weight.view(groups, rank, -1)).flatten(2)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("shape", [(1, 1, 2, 64), (2, 3, 2, 64), (1, 7, 2, 64)])
@pytest.mark.parametrize("enabled", [False, True])
def test_bf16_weight_path(monkeypatch, dtype, shape, enabled):
    torch.manual_seed(1017)
    # A non-contiguous view catches incorrect flattening/group boundaries.
    x = (torch.randn(*shape[:-1], shape[-1] * 2) * 3).to(dtype)[..., ::2]
    weight = torch.randn(64, 64).to(dtype)
    before = x.clone()
    monkeypatch.setattr(w8, "O_FP8_QDQ", enabled)
    expected = projection(reference_qdq(x) if enabled else x, weight)
    actual = w8.oproj_a(x, weight, 2, 32)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(x, before, rtol=0, atol=0)
    assert actual.dtype == dtype
    if enabled:
        assert not torch.equal(actual, projection(x, weight))


@pytest.mark.parametrize("scale", [0.0, 1e-7, 1.0, 512.0])
def test_qdq_scale_and_zero_groups(monkeypatch, scale):
    torch.manual_seed(1017)
    x = (torch.randn(1, 3, 2, 64) * scale).to(torch.bfloat16)
    x[..., :32] = 0
    weight = torch.randn(64, 64).to(torch.bfloat16)
    monkeypatch.setattr(w8, "O_FP8_QDQ", True)
    torch.testing.assert_close(w8.oproj_a(x, weight, 2, 32),
                               projection(reference_qdq(x), weight), rtol=0, atol=0)


@pytest.mark.parametrize("shape", [(1, 1, 2, 64), (2, 3, 2, 64), (3, 1, 2, 64)])
@pytest.mark.parametrize("fallback", [False, True])
def test_packed_weight_and_chunk_fallback_paths(monkeypatch, shape, fallback):
    torch.manual_seed(1017)
    x = (torch.randn(*shape) * 3).to(torch.bfloat16)
    weight = w8.W8(torch.randn(64, 64).to(torch.float8_e4m3fn).view(torch.uint8),
                   torch.full((2, 2), 124, dtype=torch.uint8))
    monkeypatch.setattr(w8, "O_FP8_QDQ", True)
    monkeypatch.setattr(w8, "MAX_TC_ROWS", 0 if fallback else 2)
    calls = []

    def cpu_gemm(activation, packed, scales, group_cols):
        # This is a CPU stand-in, NOT validation of the CUDA tensor-core kernel.
        calls.append(activation.clone())
        if fallback:
            raise RuntimeError("exercise the existing BF16 fallback")
        assert group_cols == 32
        w = quant.dequant_fp8_block(w8.unpermute_k(packed).view(torch.float8_e4m3fn), scales)
        return projection(activation.view(-1, 1, 2, 64), w).view(-1, 64)

    monkeypatch.setitem(sys.modules, "dsv41.cukern", SimpleNamespace(fp8_gemm_tc=cpu_gemm))
    actual = w8.oproj_a(x, weight, 2, 32)
    torch.testing.assert_close(actual, projection(reference_qdq(x), weight.bf16()), rtol=0, atol=0)
    assert calls
    # QDQ must happen before either kernel dispatch or fallback.
    torch.testing.assert_close(calls[0], reference_qdq(x[:, :calls[0].shape[0] // (shape[0] * 2)]).reshape_as(calls[0]), rtol=0, atol=0)


def test_quantization_failure_is_not_swallowed(monkeypatch):
    weight = w8.W8(torch.zeros(64, 64, dtype=torch.uint8), torch.full((2, 2), 127, dtype=torch.uint8))
    monkeypatch.setattr(w8, "O_FP8_QDQ", True)
    monkeypatch.setattr(w8, "MAX_TC_ROWS", 1)
    def fail(*args):
        raise RuntimeError("QDQ failed")
    monkeypatch.setattr(quant, "fake_quant_fp8", fail)
    with pytest.raises(RuntimeError, match="QDQ failed"):
        w8.oproj_a(torch.ones(1, 3, 2, 64), weight, 2, 32)


def test_group_alignment(monkeypatch):
    monkeypatch.setattr(w8, "O_FP8_QDQ", True)
    with pytest.raises(ValueError, match="32 activations"):
        w8.oproj_a(torch.ones(1, 1, 2, 48), torch.ones(64, 48), 2, 32)
    monkeypatch.setattr(w8, "O_FP8_QDQ", False)
    assert w8.oproj_a(torch.ones(1, 1, 2, 48), torch.ones(64, 48), 2, 32).shape == (1, 1, 64)


@pytest.mark.parametrize("value, expected", [("0", False), ("1", True)])
def test_mode_parser(monkeypatch, value, expected):
    monkeypatch.setenv("DSV41_O_FP8_QDQ", value)
    assert w8._read_o_fp8_qdq() is expected


def test_mode_default_and_bad_value(monkeypatch):
    monkeypatch.delenv("DSV41_O_FP8_QDQ", raising=False)
    assert not w8._read_o_fp8_qdq()
    monkeypatch.setenv("DSV41_O_FP8_QDQ", "typo")
    with pytest.raises(ValueError, match="must be 0"):
        w8._read_o_fp8_qdq()


def test_mode_is_frozen_after_import(monkeypatch):
    monkeypatch.setattr(w8, "O_FP8_QDQ", True)
    monkeypatch.setenv("DSV41_O_FP8_QDQ", "0")
    torch.manual_seed(1017)
    x = torch.randn(1, 1, 2, 64)
    weight = torch.randn(64, 64)
    torch.testing.assert_close(w8.oproj_a(x, weight, 2, 32),
                               projection(reference_qdq(x), weight), rtol=0, atol=0)
