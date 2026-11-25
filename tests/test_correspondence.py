import torch
from dinov3_vhr.ssl.correspondence import (
    PatchMLP,
    byol_correspondence_loss,
    correspondence_loss,
    patch_correspondence,
)

GRID = 16
SIZE = 256


def _box(top, left, h, w, flip=0.0):
    return torch.tensor([[top, left, h, w, flip]], dtype=torch.float32)


def _identity_grid(count):
    axis = 2 * torch.arange(GRID, dtype=torch.float32) / (GRID - 1) - 1
    gy, gx = torch.meshgrid(axis, axis, indexing="ij")
    return torch.stack([gx, gy], dim=-1).expand(count, GRID, GRID, 2)


def test_byol_loss_is_zero_for_aligned_identical_features():
    count = 2
    feat = torch.randn(count, GRID * GRID, 8)
    grid = _identity_grid(count)
    mask = torch.ones(count, GRID, GRID, dtype=torch.bool)
    loss = byol_correspondence_loss(feat, feat.clone(), grid, mask, GRID)
    assert float(loss) < 1e-5


def test_byol_loss_is_four_for_antiparallel_features():
    count = 2
    student = torch.ones(count, GRID * GRID, 4)
    teacher = -torch.ones(count, GRID * GRID, 4)
    grid = _identity_grid(count)
    mask = torch.ones(count, GRID, GRID, dtype=torch.bool)
    loss = byol_correspondence_loss(student, teacher, grid, mask, GRID)
    assert float(loss) == 4.0


def test_byol_loss_grads_reach_student_not_teacher():
    count = 2
    student = torch.randn(count, GRID * GRID, 8, requires_grad=True)
    teacher = torch.randn(count, GRID * GRID, 8, requires_grad=True)
    grid = _identity_grid(count)
    mask = torch.ones(count, GRID, GRID, dtype=torch.bool)
    byol_correspondence_loss(student, teacher, grid, mask, GRID).backward()
    assert student.grad is not None and float(student.grad.abs().sum()) > 0
    assert teacher.grad is None


def test_byol_loss_ignores_invalid_patches():
    count = 1
    student = torch.randn(count, GRID * GRID, 8)
    teacher = torch.randn(count, GRID * GRID, 8)
    grid = _identity_grid(count)
    mask = torch.zeros(count, GRID, GRID, dtype=torch.bool)
    assert float(byol_correspondence_loss(student, teacher, grid, mask, GRID)) == 0.0


def test_patch_mlp_preserves_token_layout_and_flows_grad():
    mlp = PatchMLP(16, 32, 8)
    x = torch.randn(4, GRID * GRID, 16, requires_grad=True)
    y = mlp(x)
    assert y.shape == (4, GRID * GRID, 8)
    y.sum().backward()
    assert x.grad is not None and float(x.grad.abs().sum()) > 0


def test_identity_maps_each_patch_to_itself():
    box = _box(0, 0, SIZE, SIZE)
    sampling, mask = patch_correspondence(box, box, GRID, SIZE)
    assert bool(mask.all())
    # normalized grid should equal the canonical identity grid (align_corners=True)
    axis = 2 * torch.arange(GRID, dtype=torch.float32) / (GRID - 1) - 1
    gy, gx = torch.meshgrid(axis, axis, indexing="ij")
    torch.testing.assert_close(sampling[0, ..., 0], gx, atol=1e-5, rtol=0)
    torch.testing.assert_close(sampling[0, ..., 1], gy, atol=1e-5, rtol=0)


def test_one_patch_translation_shifts_and_drops_edge():
    a = _box(0, 0, SIZE, SIZE)
    b = _box(0, 16, SIZE, SIZE)  # B shifted right by exactly one patch in source
    _, mask = patch_correspondence(a, b, GRID, SIZE)
    # A's leftmost column falls outside B (invalid); the rest overlaps
    assert not bool(mask[0, :, 0].any())
    assert bool(mask[0, :, 1:].all())


def test_disjoint_crops_have_no_overlap():
    a = _box(0, 0, 100, 100)
    b = _box(400, 400, 100, 100)
    _, mask = patch_correspondence(a, b, GRID, SIZE)
    assert not bool(mask.any())


def test_horizontal_flip_reverses_columns():
    a = _box(0, 0, SIZE, SIZE, flip=0.0)
    b = _box(0, 0, SIZE, SIZE, flip=1.0)
    sampling, mask = patch_correspondence(a, b, GRID, SIZE)
    assert bool(mask.all())
    # column c of A should map to column (grid-1-c) of B
    col_index = (sampling[0, 0, :, 0] + 1) / 2 * (GRID - 1)
    torch.testing.assert_close(
        col_index,
        torch.arange(GRID - 1, -1, -1, dtype=torch.float32),
        atol=1e-4,
        rtol=0,
    )


def test_loss_zero_for_identical_features_under_identity():
    box = _box(0, 0, SIZE, SIZE)
    sampling, mask = patch_correspondence(box, box, GRID, SIZE)
    feat = torch.randn(1, GRID * GRID, 32)
    loss = correspondence_loss(feat, feat.clone(), sampling, mask, GRID)
    assert loss.item() < 1e-5


def test_loss_positive_for_mismatched_features():
    box = _box(0, 0, SIZE, SIZE)
    sampling, mask = patch_correspondence(box, box, GRID, SIZE)
    torch.manual_seed(0)
    a = torch.randn(1, GRID * GRID, 32)
    b = torch.randn(1, GRID * GRID, 32)
    assert correspondence_loss(a, b, sampling, mask, GRID).item() > 0.5


def test_loss_gradient_flows_to_student_only():
    box = _box(0, 0, SIZE, SIZE)
    sampling, mask = patch_correspondence(box, box, GRID, SIZE)
    student = torch.randn(1, GRID * GRID, 32, requires_grad=True)
    teacher = torch.randn(1, GRID * GRID, 32)
    correspondence_loss(student, teacher, sampling, mask, GRID).backward()
    assert student.grad is not None and student.grad.abs().sum() > 0
