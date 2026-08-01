import pytest
import torch

from ai.encoding import ACTION_SIZE, INPUT_CHANNELS
from ai.network import PolicyValueNetwork, ResidualBlock, configure_device


def test_residual_block_preserves_tensor_shape() -> None:
    block = ResidualBlock(channels=8)

    result = block(torch.zeros(2, 8, 10, 9))

    assert result.shape == (2, 8, 10, 9)


def test_network_outputs_policy_and_bounded_value() -> None:
    model = PolicyValueNetwork(channels=16, residual_blocks=1)

    policy, value = model(torch.zeros(2, INPUT_CHANNELS, 10, 9))

    assert policy.shape == (2, ACTION_SIZE)
    assert value.shape == (2, 1)
    assert torch.all(value >= -1)
    assert torch.all(value <= 1)


def test_cpu_device_applies_requested_thread_count() -> None:
    device = configure_device("cpu", torch_threads=2)

    assert device == torch.device("cpu")
    assert torch.get_num_threads() == 2


@pytest.mark.parametrize("name", ["cuda", "cuda:0", "cuda:3"])
def test_explicit_cuda_fails_instead_of_falling_back(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA"):
        configure_device(name, torch_threads=1)


def test_auto_uses_cpu_when_cuda_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    assert configure_device("auto", torch_threads=1) == torch.device("cpu")


def test_auto_uses_cuda_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    assert configure_device("auto", torch_threads=1) == torch.device("cuda")


def test_numbered_cuda_device_is_preserved(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)

    assert configure_device("cuda:1", torch_threads=1) == torch.device("cuda:1")


def test_cuda_index_must_exist(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)

    with pytest.raises(ValueError, match="CUDA.*2"):
        configure_device("cuda:2", torch_threads=1)


@pytest.mark.parametrize("name", ["mps", "cuda:-1", "cuda:abc", "cpu:0", ""])
def test_unsupported_device_is_rejected(name: str) -> None:
    with pytest.raises(ValueError, match="设备"):
        configure_device(name, torch_threads=1)


@pytest.mark.parametrize("torch_threads", [0, -1, True, 1.5])
def test_torch_threads_must_be_a_positive_integer(torch_threads: object) -> None:
    with pytest.raises(ValueError, match="torch_threads"):
        configure_device("cpu", torch_threads=torch_threads)  # type: ignore[arg-type]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="当前环境没有可用 CUDA")
def test_network_can_run_a_cuda_forward_pass() -> None:
    device = configure_device("cuda", torch_threads=1)
    model = PolicyValueNetwork(channels=8, residual_blocks=1).to(device)
    inputs = torch.zeros(1, INPUT_CHANNELS, 10, 9, device=device)

    policy, value = model(inputs)

    assert policy.device.type == "cuda"
    assert value.device.type == "cuda"
    assert policy.shape == (1, ACTION_SIZE)
    assert value.shape == (1, 1)
