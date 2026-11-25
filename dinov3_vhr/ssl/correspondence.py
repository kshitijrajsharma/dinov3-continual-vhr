"""Cross-view dense correspondence for continued pretraining.

Two global crops of the same tile are related by a known crop-and-resize geometry, so for
every patch in view A its true location in view B is computable. This maps A's patch grid
into B's, warps the (stop-grad) teacher's patch features from B into A's frame, and pulls
the student's patch features toward them over the overlapping region. It is the geometric,
cross-view analogue of iBOT's same-view masked objective, and it targets the dense features
that the global DINO objective leaves unmoved.

A crop box is (top, left, height, width, hflip) in the source-image pixel frame, with the
flip applied after the resize, matching torchvision RandomResizedCrop then RandomHorizontalFlip.
"""

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class PatchMLP(nn.Module):
    """Two-layer MLP applied per patch token, with batch norm over the flattened patches.

    Serves as both the BYOL projector (backbone feature to latent) and predictor (latent to
    latent). Batch norm treats every patch as a sample, giving the batch statistics that,
    together with the EMA teacher and stop-grad, keep the regression objective off the trivial
    constant solution."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        count, tokens, channels = x.shape
        flat = self.net(x.reshape(count * tokens, channels))
        return flat.reshape(count, tokens, -1)


def patch_correspondence(
    boxes_a: Tensor, boxes_b: Tensor, grid: int = 16, size: int = 256
) -> tuple[Tensor, Tensor]:
    """Map view A's patch grid into view B. Returns a grid_sample sampling grid of shape
    (N, grid, grid, 2) normalized to [-1, 1] and a boolean validity mask (N, grid, grid)
    that is true where an A patch center falls inside B's crop."""
    device = boxes_a.device
    count = boxes_a.shape[0]
    index = torch.arange(grid, device=device, dtype=torch.float32)
    rows, cols = torch.meshgrid(index, index, indexing="ij")
    patch = size / grid
    x = ((cols + 0.5) * patch).expand(count, grid, grid)
    y = ((rows + 0.5) * patch).expand(count, grid, grid)

    top_a, left_a, h_a, w_a, flip_a = (boxes_a[:, i].view(count, 1, 1) for i in range(5))
    top_b, left_b, h_b, w_b, flip_b = (boxes_b[:, i].view(count, 1, 1) for i in range(5))

    x_pre = torch.where(flip_a > 0.5, size - x, x)
    src_x = left_a + x_pre / size * w_a
    src_y = top_a + y / size * h_a

    x_b = (src_x - left_b) / w_b * size
    y_b = (src_y - top_b) / h_b * size
    valid = (x_b >= 0) & (x_b <= size) & (y_b >= 0) & (y_b <= size)
    x_b = torch.where(flip_b > 0.5, size - x_b, x_b)

    col_b = x_b / patch - 0.5
    row_b = y_b / patch - 0.5
    grid_x = 2 * col_b / (grid - 1) - 1
    grid_y = 2 * row_b / (grid - 1) - 1
    return torch.stack([grid_x, grid_y], dim=-1), valid


def warp_patch_features(
    patch: Tensor, sampling_grid: Tensor, grid: int = 16, mode: str = "nearest"
) -> Tensor:
    """Warp patch features (N, grid*grid, C) from their own view into the target view's
    frame using the sampling grid. Nearest mode picks a real patch's features (no blur),
    which the correspondence objective needs to avoid smoothing the target."""
    count, _tokens, channels = patch.shape
    source = patch.reshape(count, grid, grid, channels).permute(0, 3, 1, 2).float()
    warped = F.grid_sample(
        source,
        sampling_grid.float(),
        mode=mode,
        padding_mode="border",
        align_corners=True,
    )
    return warped.permute(0, 2, 3, 1).reshape(count, grid * grid, channels)


def correspondence_loss(
    student_patch_a: Tensor,
    teacher_patch_b: Tensor,
    sampling_grid: Tensor,
    mask: Tensor,
    grid: int = 16,
) -> Tensor:
    """Mean 1 - cosine between student patches on A and teacher patches on B warped into A's
    frame, over the overlapping patches. Patch tokens are (N, grid*grid, C); the teacher must
    be detached by the caller. Returns a zero-dim tensor; zero valid overlap yields 0."""
    student_patch_a = student_patch_a.float()
    teacher_patch_b = teacher_patch_b.float()
    sampling_grid = sampling_grid.float()
    count, tokens, channels = student_patch_a.shape
    teacher = teacher_patch_b.reshape(count, grid, grid, channels).permute(0, 3, 1, 2)
    warped = F.grid_sample(
        teacher,
        sampling_grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    warped = warped.permute(0, 2, 3, 1).reshape(count, tokens, channels)
    cosine = (F.normalize(student_patch_a, dim=-1) * F.normalize(warped, dim=-1)).sum(-1)
    weight = mask.reshape(count, tokens).to(cosine.dtype)
    return ((1 - cosine) * weight).sum() / weight.sum().clamp(min=1.0)


def byol_correspondence_loss(
    student_pred_a: Tensor,
    teacher_proj_b: Tensor,
    sampling_grid: Tensor,
    mask: Tensor,
    grid: int = 16,
) -> Tensor:
    """BYOL cross-view dense loss: warp the teacher's projected patches on B into A's frame
    (nearest, no blur), then 2 - 2*cosine between the student's predicted patches on A and the
    stop-grad teacher target, over the overlapping patches. Unlike the prototype cross-entropy
    this cannot stall at a uniform floor, and the predictor-plus-EMA asymmetry (supplied by the
    caller) is what prevents the trivial collapse that a bare cosine loss suffers. Patch tokens
    are (N, grid*grid, C); returns a zero-dim tensor, and zero valid overlap yields 0."""
    teacher_warped = warp_patch_features(
        teacher_proj_b.float(), sampling_grid, grid, "nearest"
    ).detach()
    prediction = F.normalize(student_pred_a.float(), dim=-1)
    target = F.normalize(teacher_warped, dim=-1)
    cosine = (prediction * target).sum(-1)
    weight = mask.reshape(mask.shape[0], -1).to(cosine.dtype)
    return ((2 - 2 * cosine) * weight).sum() / weight.sum().clamp(min=1.0)
