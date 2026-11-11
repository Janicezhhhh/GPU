import pytest
import torch

from rmsnorm_lab import reference, rmsnorm


def test_reference_known_answer():
    x = torch.tensor([[3.0, 4.0]])
    weight = torch.tensor([2.0, 0.5])
    expected = torch.tensor([[6.0, 2.0]]) / (12.5 + 1e-6) ** 0.5
    torch.testing.assert_close(reference(x, weight), expected)


def test_reference_residual_rounding():
    x = torch.tensor([[1.0, 2.0]], dtype=torch.float16)
    r = torch.tensor([[0.0006, -0.2]], dtype=torch.float16)
    w = torch.ones(2, dtype=x.dtype)
    torch.testing.assert_close(reference(x, w, r), reference(x + r, w))


@pytest.mark.parametrize("shape", [(1,), (2, 7), (2, 3, 1025)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_reference_against_double_oracle(shape, dtype):
    torch.manual_seed(23)
    x = torch.randn(shape).to(dtype)
    r = torch.randn(shape).to(dtype)
    w = torch.randn(shape[-1]).to(dtype)
    z = (x + r).double()
    expected = (z / (z.square().mean(-1, keepdim=True) + 1e-6).sqrt() * w.double()).to(dtype)
    tolerance = {torch.float32: 2e-6, torch.float16: 2e-3, torch.bfloat16: 2e-2}[dtype]
    torch.testing.assert_close(reference(x, w, r), expected, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("eps", [0, -1, float("inf"), 1e-50, 1e40])
def test_epsilon_outside_supported_range(eps):
    with pytest.raises(ValueError, match="eps"):
        reference(torch.ones(2, 3), torch.ones(3), eps=eps)


@pytest.mark.parametrize("case", ["shape", "dtype", "noncontiguous", "grad", "eps", "residual", "empty_hidden"])
def test_invalid_inputs(case):
    x, w, r, eps = torch.ones(2, 3), torch.ones(3), None, 1e-6
    if case == "shape": w = torch.ones(1, 3)
    if case == "dtype": w = w.double()
    if case == "noncontiguous": x = torch.ones(3, 2).t()
    if case == "grad": x.requires_grad_()
    if case == "eps": eps = float("nan")
    if case == "residual": r = torch.ones(1, 3)
    if case == "empty_hidden": x, w = torch.empty(2, 0), torch.empty(0)
    with pytest.raises((ValueError, TypeError)):
        reference(x, w, r, eps)


def test_cpu_kernel_rejected():
    with pytest.raises(ValueError, match="requires CUDA"):
        rmsnorm(torch.ones(2, 3), torch.ones(3))


@pytest.mark.parametrize("input_name", ["x", "weight", "residual"])
def test_gradients_explicitly_rejected(input_name):
    inputs = {"x": torch.ones(2, 3), "weight": torch.ones(3), "residual": torch.ones(2, 3)}
    inputs[input_name].requires_grad_()
    with pytest.raises(ValueError, match="inference only"):
        reference(**inputs)


@pytest.mark.parametrize("dtype", [torch.int64, torch.float64, torch.complex64])
def test_unsupported_dtype(dtype):
    with pytest.raises(TypeError, match="supported dtypes"):
        reference(torch.ones(2, 3, dtype=dtype), torch.ones(3, dtype=dtype))


def test_empty_rows_and_scalar_input():
    assert reference(torch.empty(0, 3), torch.ones(3)).shape == (0, 3)
    with pytest.raises(ValueError, match="nonempty last dimension"):
        reference(torch.tensor(1.0), torch.ones(1))


def test_unknown_implementation():
    with pytest.raises(ValueError, match="implementation"):
        rmsnorm(torch.ones(2, 3), torch.ones(3), implementation="unknown")


cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")


@cuda
@pytest.mark.parametrize("shape", [(1,), (2, 7), (3, 31), (2, 33), (4, 127),
                                   (3, 256), (2, 1025), (2, 3, 4096), (1, 8192), (0, 32)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("implementation", ["serial", "block"])
def test_cuda_matches_reference(shape, dtype, fused, implementation):
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("GPU does not support bf16")
    torch.manual_seed(17)
    x = torch.randn(shape, device="cuda", dtype=dtype)
    w = torch.randn(shape[-1], device="cuda", dtype=dtype)
    r = torch.randn_like(x) if fused else None
    before = [t.clone() for t in (x, w, r) if t is not None]
    actual = rmsnorm(x, w, r, implementation=implementation)
    expected = reference(x, w, r)
    tolerance = {torch.float32: 2e-5, torch.float16: 3e-3, torch.bfloat16: 2e-2}[dtype]
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    for original, snapshot in zip((t for t in (x, w, r) if t is not None), before):
        torch.testing.assert_close(original, snapshot, atol=0, rtol=0)


@cuda
def test_nondefault_stream_and_zero_input():
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        x = torch.zeros(5, 513, device="cuda")
        w = torch.ones(513, device="cuda")
        out = rmsnorm(x, w)
        expected = reference(x, w)
    stream.synchronize()
    torch.testing.assert_close(out, expected, atol=0, rtol=0)


@cuda
def test_nonzero_storage_offset():
    # Contiguous views need not have an aligned base address.
    x = torch.randn(1 + 3 * 127, device="cuda")[1:].view(3, 127)
    w = torch.randn(128, device="cuda")[1:]
    torch.testing.assert_close(rmsnorm(x, w), reference(x, w), atol=2e-5, rtol=2e-5)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="two GPUs required")
def test_device_guard():
    with torch.cuda.device(0):
        x = torch.randn(3, 257, device="cuda:1")
        w = torch.ones(257, device="cuda:1")
        torch.testing.assert_close(rmsnorm(x, w), reference(x, w), atol=2e-5, rtol=2e-5)
        assert torch.cuda.current_device() == 0
