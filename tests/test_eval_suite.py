from pathlib import Path

import pytest
import timm
import torch
from dinov3_vhr.eval.run_suite import (
    evenly_spaced_layers,
    validate_checkpoint,
    write_encoder_config,
)

VIT_SMALL = "vit_small_patch16_dinov3.lvd1689m"


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    backbone = timm.create_model(VIT_SMALL, pretrained=False, num_classes=0)
    path = tmp_path_factory.mktemp("ckpt") / "vits.pth"
    torch.save(backbone.state_dict(), path)
    return path


def test_layer_selection_matches_pangaea_encoder_configs():
    assert evenly_spaced_layers(12) == [2, 5, 8, 11]
    assert evenly_spaced_layers(24) == [5, 11, 17, 23]


def test_validate_checkpoint_returns_block_count(checkpoint: Path):
    assert validate_checkpoint(VIT_SMALL, checkpoint) == 12


def test_validate_checkpoint_rejects_mismatched_architecture(checkpoint: Path):
    with pytest.raises(RuntimeError):
        validate_checkpoint("vit_large_patch16_dinov3.lvd1689m", checkpoint)


def test_write_encoder_config_emits_weights_and_layers(checkpoint: Path, tmp_path: Path):
    (tmp_path / "configs" / "encoder").mkdir(parents=True)
    path = write_encoder_config(tmp_path, "vits_test", VIT_SMALL, checkpoint, 256)
    text = path.read_text()
    assert f"encoder_weights: {checkpoint.resolve()}" in text
    assert "model_name: " + VIT_SMALL in text
    assert "  - 2\n  - 5\n  - 8\n  - 11\n" in text
