import numpy as np
import torch
from dinov3_vhr.figures._common import pca_rgb
from dinov3_vhr.figures.patch_pca import cls_patch_similarity


def test_pca_rgb_shape_and_range():
    patches = torch.randn(64, 32)
    rgb = pca_rgb(patches)
    assert rgb.shape == (8, 8, 3)
    assert rgb.min() >= 0.0 and rgb.max() <= 1.0


def test_cls_patch_similarity_is_one_when_cls_equals_patches():
    patches = torch.randn(2, 16, 8)
    cls = patches[:, 0, :]
    identical = torch.ones(2, 16, 8) * patches[:, :1, :]
    assert cls_patch_similarity(identical, cls) == 1.0


def test_cls_patch_similarity_in_valid_cosine_range():
    patches = torch.randn(4, 16, 8)
    cls = torch.randn(4, 8)
    value = cls_patch_similarity(patches, cls)
    assert -1.0 <= value <= 1.0


def test_cls_patch_similarity_near_zero_for_orthogonal_directions():
    patches = torch.zeros(1, 2, 2)
    patches[0, 0] = torch.tensor([1.0, 0.0])
    patches[0, 1] = torch.tensor([0.0, 1.0])
    cls = torch.tensor([[1.0, 0.0]])
    assert cls_patch_similarity(patches, cls) == np.float32(0.5)
