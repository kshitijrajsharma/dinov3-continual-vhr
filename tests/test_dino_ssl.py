from typing import cast

import pytest
import torch
from dinov3_vhr.dino_ssl import (
    DinoV3ContinuedSSL,
    cls_anchor_loss,
    gram_loss,
    multicrop_collate,
    patch_grid_mask,
)


@pytest.fixture(scope="module")
def model() -> DinoV3ContinuedSSL:
    return DinoV3ContinuedSSL(
        model_name="vit_small_patch16_dinov3.lvd1689m",
        freeze_backbone_steps=10,
        backbone_warmup_steps=5,
    )


def test_backbone_frozen_holds_blocks_but_releases_patch_embed(model):
    model._freeze_backbone(True)
    grads = {name: param.requires_grad for name, param in model.student_backbone.named_parameters()}
    assert not grads["blocks.0.attn.qkv.weight"]
    assert not grads["norm.weight"]
    assert all(trainable for name, trainable in grads.items() if name.startswith("patch_embed"))


def test_backbone_unfrozen_releases_every_parameter(model):
    model._freeze_backbone(False)
    assert all(param.requires_grad for param in model.student_backbone.parameters())


def test_backbone_lr_scale_is_zero_at_unfreeze_then_ramps_to_one(model):
    assert model._backbone_lr_scale(0) == 1.0
    assert model._backbone_lr_scale(9) == 1.0
    assert model._backbone_lr_scale(10) == 0.0
    assert model._backbone_lr_scale(12) == pytest.approx(0.4)
    assert model._backbone_lr_scale(15) == 1.0
    assert model._backbone_lr_scale(100) == 1.0


def test_cls_anchor_loss_is_zero_for_identical_embeddings():
    cls = torch.randn(4, 16)
    assert float(cls_anchor_loss(cls, cls)) == pytest.approx(0.0, abs=1e-6)


def test_cls_anchor_loss_is_positive_for_different_embeddings():
    assert float(cls_anchor_loss(torch.randn(4, 16), torch.randn(4, 16))) > 0.0


def test_cls_anchor_loss_is_two_for_opposite_embeddings():
    cls = torch.randn(4, 16)
    assert float(cls_anchor_loss(cls, -cls)) == pytest.approx(2.0, abs=1e-5)


def test_anchor_refresh_copies_ema_teacher_only_after_unfreeze():
    m = DinoV3ContinuedSSL(
        model_name="vit_small_patch16_dinov3.lvd1689m",
        freeze_backbone_steps=10,
        anchor_target="refresh",
        anchor_refresh_steps=5,
        anchor_max_refreshes=2,
    )
    # Drift the EMA teacher so a refresh is detectable.
    with torch.no_grad():
        for param in m.teacher_backbone.parameters():
            param.add_(1.0)
    anchor = m.anchor_teacher_backbone.get_parameter("blocks.0.attn.qkv.weight")
    before = anchor.detach().clone()

    m._maybe_refresh_anchor(5)  # still inside the frozen stage -> no refresh
    assert torch.equal(anchor, before)
    assert m.refreshes_done == 0

    m._maybe_refresh_anchor(15)  # past unfreeze and on the interval -> refresh
    assert not torch.equal(anchor, before)
    assert m.refreshes_done == 1


def test_anchor_refresh_stops_at_max():
    m = DinoV3ContinuedSSL(
        model_name="vit_small_patch16_dinov3.lvd1689m",
        freeze_backbone_steps=10,
        anchor_target="refresh",
        anchor_refresh_steps=5,
        anchor_max_refreshes=1,
    )
    m._maybe_refresh_anchor(15)
    m._maybe_refresh_anchor(20)
    assert m.refreshes_done == 1


def test_anchor_frozen_target_never_refreshes():
    m = DinoV3ContinuedSSL(
        model_name="vit_small_patch16_dinov3.lvd1689m",
        freeze_backbone_steps=10,
        anchor_target="frozen",
        anchor_refresh_steps=5,
    )
    m._maybe_refresh_anchor(15)
    m._maybe_refresh_anchor(20)
    assert m.refreshes_done == 0


def test_invalid_anchor_target_raises():
    with pytest.raises(ValueError, match="anchor_target"):
        DinoV3ContinuedSSL(model_name="vit_small_patch16_dinov3.lvd1689m", anchor_target="bogus")


def _step_once(model: DinoV3ContinuedSSL) -> None:
    """One real backward and optimizer step on the student backbone."""
    config = cast(dict, model.configure_optimizers())
    images = torch.randn(2, 3, 64, 64)
    model._embed(model.student_backbone, images).square().mean().backward()
    config["optimizer"].step()


def test_frozen_backbone_blocks_do_not_move_but_patch_embed_does():
    model = DinoV3ContinuedSSL(
        model_name="vit_small_patch16_dinov3.lvd1689m", freeze_backbone_steps=10
    )
    block = model.student_backbone.get_parameter("blocks.0.attn.qkv.weight")
    stem = model.student_backbone.get_parameter("patch_embed.proj.weight")
    before_block, before_stem = block.detach().clone(), stem.detach().clone()

    model._freeze_backbone(True)
    _step_once(model)

    assert torch.equal(block, before_block), "frozen block moved"
    assert not torch.equal(stem, before_stem), "patch_embed should stay trainable"


def test_unfrozen_backbone_blocks_move():
    model = DinoV3ContinuedSSL(
        model_name="vit_small_patch16_dinov3.lvd1689m", freeze_backbone_steps=10
    )
    block = model.student_backbone.get_parameter("blocks.0.attn.qkv.weight")
    before = block.detach().clone()

    model._freeze_backbone(False)
    _step_once(model)

    assert not torch.equal(block, before), "unfrozen block did not move"


def test_corr_byol_trains_and_reaches_shallow_backbone():
    layer = 2
    model = DinoV3ContinuedSSL(
        model_name="vit_small_patch16_dinov3.lvd1689m",
        freeze_backbone_steps=0,
        corr_weight=1.0,
        corr_layers=(layer,),
        corr_byol=True,
        global_crop_size=64,
    )
    model._freeze_backbone(False)
    view_a, view_b = torch.randn(2, 3, 64, 64), torch.randn(2, 3, 64, 64)
    box = torch.tensor([0.0, 0.0, 64.0, 64.0, 0.0])
    boxes = box.expand(2, 2, 5).contiguous()  # full-overlap identity boxes, 2 images

    _, tokens_a = model._capture_corr_layers(model.student_backbone, view_a)
    _, tokens_b = model._capture_corr_layers(model.student_backbone, view_b)
    loss = model._correspondence_loss([view_a, view_b], [tokens_a, tokens_b], boxes)

    assert torch.isfinite(loss) and float(loss) > 0.0
    loss.backward()

    projector_grad = model.corr_projector[0].net[0].weight.grad
    predictor_grad = model.corr_predictor[0].net[0].weight.grad
    block_grad = model.student_backbone.get_parameter(f"blocks.{layer}.attn.qkv.weight").grad
    assert projector_grad is not None and float(projector_grad.abs().sum()) > 0
    assert predictor_grad is not None and float(predictor_grad.abs().sum()) > 0
    assert block_grad is not None and float(block_grad.abs().sum()) > 0
    assert model.corr_teacher_projector[0].net[0].weight.grad is None
    assert layer in model._corr_std and float(model._corr_std[layer]) > 0.0


def test_gram_loss_is_zero_for_identical_features():
    features = torch.randn(2, 16, 8)
    assert float(gram_loss(features, features)) == pytest.approx(0.0, abs=1e-6)


def test_gram_loss_is_positive_for_different_features():
    student = torch.randn(2, 16, 8)
    teacher = torch.randn(2, 16, 8)
    assert float(gram_loss(student, teacher)) > 0.0


def test_patch_grid_mask_shape_and_dtype():
    mask = patch_grid_mask(batch_size=4, side=14, mask_ratio=0.3, device="cpu")
    assert mask.shape == (4, 14, 14)
    assert mask.dtype == torch.bool


def test_patch_grid_mask_masks_equal_count_per_image():
    mask = patch_grid_mask(batch_size=4, side=14, mask_ratio=0.3, device="cpu")
    expected = int(0.3 * 14 * 14)
    counts = mask.flatten(1).sum(dim=1)
    assert counts.tolist() == [expected] * 4


def test_multicrop_collate_groups_views_and_labels():
    crops = [torch.zeros(3, 4, 4), torch.zeros(3, 4, 4), torch.zeros(3, 2, 2)]
    other = [torch.ones(3, 4, 4), torch.ones(3, 4, 4), torch.ones(3, 2, 2)]
    views, labels = multicrop_collate([(crops, 0), (other, 1)])
    assert len(views) == 3
    assert views[0].shape == (2, 3, 4, 4)
    assert views[2].shape == (2, 3, 2, 2)
    assert labels.tolist() == [0, 1]
