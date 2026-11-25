"""Typed access to the timm DINOv3 ViT surface used across training and probing.

timm models are dynamically typed, so the static checker sees attribute access on an
nn.Module as `Tensor | Module`. The `ViT` protocol declares the exact subset this
project relies on, and `as_vit` views a backbone through it without copying.
"""

from collections.abc import Sequence
from typing import Protocol, cast

import timm
import torch
from torch import Tensor


class PatchEmbed(Protocol):
    patch_size: tuple[int, int]

    def __call__(self, x: Tensor) -> Tensor: ...


class Block(Protocol):
    def __call__(self, x: Tensor, rope: Tensor | None = None) -> Tensor: ...


class ViT(Protocol):
    """The subset of the timm DINOv3 ViT interface this project relies on."""

    embed_dim: int
    num_prefix_tokens: int
    patch_embed: PatchEmbed
    blocks: Sequence[Block]

    def forward_features(self, x: Tensor) -> Tensor: ...

    def forward_head(self, x: Tensor, pre_logits: bool = False) -> Tensor: ...

    def no_weight_decay(self) -> set[str]: ...

    def set_grad_checkpointing(self, enable: bool = True) -> None: ...

    def _pos_embed(self, x: Tensor) -> tuple[Tensor, Tensor | None]: ...

    def norm_pre(self, x: Tensor) -> Tensor: ...

    def norm(self, x: Tensor) -> Tensor: ...


def build_backbone(model_name: str, pretrained: bool = True) -> torch.nn.Module:
    """DINOv3 ViT backbone as an nn.Module, suitable for EMA copies and optimizers."""
    return timm.create_model(model_name, pretrained=pretrained, num_classes=0)


def as_vit(backbone: torch.nn.Module) -> ViT:
    """View an nn.Module backbone through the timm DINOv3 ViT interface."""
    return cast(ViT, backbone)
