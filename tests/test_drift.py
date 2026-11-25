from pathlib import Path
from typing import Any, cast

import pytest
import torch
from dinov3_vhr.monitor.drift import DriftMonitor, linear_cka, patch_features_by_layer


def test_linear_cka_is_one_for_identical_features():
    x = torch.randn(32, 64)
    assert linear_cka(x, x) == pytest.approx(1.0, abs=1e-5)


def test_linear_cka_is_invariant_to_scale_and_shift():
    x = torch.randn(48, 96)
    assert linear_cka(x, 3.0 * x + 5.0) == pytest.approx(1.0, abs=1e-5)


def test_linear_cka_below_one_for_independent_features():
    torch.manual_seed(0)
    x = torch.randn(64, 32)
    y = torch.randn(64, 32)
    assert linear_cka(x, y) < 0.5


class _StubViT:
    """Minimal ViT surface: the input batch flows through one hookable block."""

    def __init__(self) -> None:
        self.num_prefix_tokens = 1
        self.blocks = [torch.nn.Identity()]

    def forward_features(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.blocks[0](tokens)


def test_patch_features_pool_over_patch_tokens_only():
    tokens = torch.randn(8, 1 + 5, 16)
    features = patch_features_by_layer(cast(Any, _StubViT()), tokens, "cpu", [0], batch=4)
    expected = tokens[:, 1:, :].mean(dim=1)
    torch.testing.assert_close(features[0], expected)


class _StubModule:
    def __init__(self) -> None:
        self.training = True

    def eval(self) -> None:
        self.training = False

    def train(self) -> None:
        self.training = True


class _StubTrainer:
    def __init__(self, step: int) -> None:
        self.global_step = step


def test_drift_monitor_logs_only_on_interval(monkeypatch):
    import dinov3_vhr.monitor.drift as drift

    class _StubPlModule:
        def __init__(self) -> None:
            self.student_backbone = _StubModule()
            self.device = torch.device("cpu")
            self.grad_checkpointing = False
            self.logged: dict[str, float] = {}

        def log(self, name: str, value: float, **_: object) -> None:
            self.logged[name] = value

    monkeypatch.setattr(drift, "DinoV3ContinuedSSL", _StubPlModule)
    monkeypatch.setattr(drift, "patch_features_by_layer", lambda *a, **k: {0: torch.randn(4, 16)})

    monitor = DriftMonitor(tiles_dir=Path("."), model_name="m", layers=[0], every_n_steps=250)
    monitor.probe_images = torch.randn(4, 3, 8, 8)
    monitor.reference = {0: torch.randn(4, 16)}
    pl_module = _StubPlModule()

    monitor.on_train_batch_end(cast(Any, _StubTrainer(10)), cast(Any, pl_module), None, None, 0)
    assert pl_module.logged == {}
    monitor.on_train_batch_end(cast(Any, _StubTrainer(250)), cast(Any, pl_module), None, None, 0)
    assert "monitor/cka_patch_l0" in pl_module.logged


def test_drift_monitor_runs_once_per_step_under_accumulation(monkeypatch):
    import dinov3_vhr.monitor.drift as drift

    class _StubPlModule:
        def __init__(self) -> None:
            self.student_backbone = _StubModule()
            self.device = torch.device("cpu")
            self.grad_checkpointing = False

        def log(self, name: str, value: float, **_: object) -> None:
            pass

    calls = {"n": 0}

    def _counting_probe(*_a, **_k):
        calls["n"] += 1
        return {0: torch.randn(4, 16)}

    monkeypatch.setattr(drift, "DinoV3ContinuedSSL", _StubPlModule)
    monkeypatch.setattr(drift, "patch_features_by_layer", _counting_probe)

    monitor = DriftMonitor(tiles_dir=Path("."), model_name="m", layers=[0], every_n_steps=250)
    monitor.probe_images = torch.randn(4, 3, 8, 8)
    monitor.reference = {0: torch.randn(4, 16)}
    pl_module = _StubPlModule()

    for _ in range(32):
        monitor.on_train_batch_end(
            cast(Any, _StubTrainer(250)), cast(Any, pl_module), None, None, 0
        )
    assert calls["n"] == 1
