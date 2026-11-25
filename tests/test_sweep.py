"""Unit tests for the sweep config expansion (pure logic, no SLURM)."""

from pathlib import Path

from dinov3_vhr.ssl.sweep import (
    eval_combos,
    load_sweep,
    run_name,
    train_command,
    train_parameters,
    training_runs,
)
from omegaconf import OmegaConf


def sample_config():
    return OmegaConf.create(
        {
            "name_prefix": "vits",
            "base": {
                "model": "vit_small_patch16_dinov3.lvd1689m",
                "data_dir": "data/tiles",
                "batch_size": 32,
                "anchor_target": "frozen",
            },
            "schedule": {"epochs": 8},
            "seeds": [42, 1234],
            "learning_rates": [5.0e-5],
            "arms": {
                "control": {"gram_weight": 0.0, "cls_anchor_weight": 0.0},
                "winner": {"gram_weight": 2.0, "cls_anchor_weight": 1.0},
            },
            "eval": {"datasets": ["flair2", "opencanopy"], "steps": [2000, 8000]},
        }
    )


def test_run_name_is_stable_and_safe():
    assert run_name("vits", "winner", 5e-5, 42) == "vits_winner_lr5e-05_s42"


def test_train_parameters_merge_base_epochs_and_arm():
    params = train_parameters(sample_config(), "winner", 5e-5, 42)
    assert params["batch_size"] == 32
    assert params["epochs"] == 8
    assert params["gram_weight"] == 2.0
    assert params["cls_anchor_weight"] == 1.0
    assert params["lr"] == 5e-5
    assert params["run_name"] == "vits_winner_lr5e-05_s42"
    assert params["out_ckpt"] == "out/vits_winner_lr5e-05_s42.pth"


def test_train_command_hyphenates_keys():
    command = train_command({"data_dir": "data/tiles", "gram_weight": 2.0})
    assert command[:3] == ["python", "-m", "dinov3_vhr.ssl.train"]
    assert "--data-dir" in command
    assert "--gram-weight" in command


def test_training_runs_expand_arm_lr_seed_grid():
    runs = training_runs(sample_config())
    assert len(runs) == 2 * 1 * 2
    names = [name for name, _ in runs]
    assert "vits_control_lr5e-05_s42" in names
    assert "vits_winner_lr5e-05_s1234" in names


def test_eval_combos_cover_run_step_dataset():
    combos = eval_combos(sample_config())
    assert len(combos) == 4 * 2 * 2  # (arm x seed) x steps x datasets
    assert combos[0] == (
        "vits_control_lr5e-05_s42_2000 flair2 segmentation seg_upernet seg_default cross_entropy"
    )


def test_real_config_expands_to_nine_runs():
    config = load_sweep(Path("configs/sweep_vits_confirm.yaml"))
    assert len(training_runs(config)) == 9
