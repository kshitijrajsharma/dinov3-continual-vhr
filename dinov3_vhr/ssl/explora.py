"""ExPLoRA adaptation of a timm DINOv3 ViT for continued pretraining.

Extended pre-training with low-rank adapters (Khanna et al., ICML 2025, arXiv 2406.10973):
unfreeze the last one or two blocks and all normalization layers, and adapt every other
block's attention and MLP linears with LoRA, then continue the self-supervised objective.

This is a minimal LoRA wrapper rather than the PEFT library on purpose. PEFT renames modules
to `base_model.model.*`, which would break three timm-native contracts this codebase relies
on: the layer-wise LR grouping in `timm.optim.param_groups_layer_decay` (keys on `blocks.N`),
the EMA/anchor teacher built by structural `deepcopy`, and the checkpoint contract (pangaea
eval and the drift figures load a plain timm ViT `state_dict`). `LoRALinear` keeps the
original `weight`/`bias` parameter names in place and adds `lora_A`/`lora_B`, so the base
keys are unchanged and `merged_state_dict` folds the adapters back into a clean timm
`state_dict` for evaluation.
"""

import copy
import math
from typing import cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from dinov3_vhr.backbone import as_vit

LORA_TARGETS = ("qkv", "proj", "fc1", "fc2")


class LoRALinear(nn.Module):
    """A frozen Linear with a trainable low-rank update, preserving the base parameter names.

    The base ``weight``/``bias`` are the original module's parameters, held frozen; only
    ``lora_A``/``lora_B`` train. ``lora_B`` starts at zero so the initial output equals the
    base Linear exactly, leaving the released model unperturbed at step zero.
    """

    def __init__(self, linear: nn.Linear, rank: int, alpha: float) -> None:
        super().__init__()
        out_features, in_features = linear.weight.shape
        self.weight = linear.weight
        self.bias = linear.bias
        self.weight.requires_grad_(False)
        if self.bias is not None:
            self.bias.requires_grad_(False)
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.scaling = alpha / rank
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: Tensor) -> Tensor:
        base = F.linear(x, self.weight, self.bias)
        update = F.linear(F.linear(x, self.lora_A), self.lora_B)
        return base + update * self.scaling

    def merged_linear(self) -> nn.Linear:
        """A plain Linear with the low-rank update folded into the weight."""
        out_features, in_features = self.weight.shape
        linear = nn.Linear(in_features, out_features, bias=self.bias is not None)
        with torch.no_grad():
            delta = (self.lora_B @ self.lora_A) * self.scaling
            linear.weight.copy_(self.weight + delta)
            if self.bias is not None:
                linear.bias.copy_(self.bias)
        return linear


def _wrap_block_linears(block: nn.Module, rank: int, alpha: float) -> None:
    for parent in block.modules():
        for name, child in list(parent.named_children()):
            if isinstance(child, nn.Linear) and name in LORA_TARGETS:
                setattr(parent, name, LoRALinear(child, rank, alpha))


def apply_explora(
    backbone: nn.Module, rank: int, alpha: float, unfreeze_blocks: list[int]
) -> set[str]:
    """Freeze the backbone, then make the ExPLoRA subset trainable: LoRA on the attention
    and MLP linears of the frozen blocks, plus the unfrozen blocks, all LayerNorms, the
    prefix tokens, and the patch-embedding stem. Returns the trainable parameter names."""
    for param in backbone.parameters():
        param.requires_grad_(False)

    for index, block in enumerate(as_vit(backbone).blocks):
        block = cast(nn.Module, block)
        if index in unfreeze_blocks:
            for param in block.parameters():
                param.requires_grad_(True)
        else:
            _wrap_block_linears(block, rank, alpha)

    for module in backbone.modules():
        if isinstance(module, nn.LayerNorm):
            for param in module.parameters():
                param.requires_grad_(True)
    for name, param in backbone.named_parameters():
        if name.endswith("_token") or name.startswith("patch_embed"):
            param.requires_grad_(True)

    return {name for name, param in backbone.named_parameters() if param.requires_grad}


def merged_state_dict(backbone: nn.Module) -> dict[str, Tensor]:
    """A plain timm ViT state_dict with every LoRA update folded into its base weight."""
    clone = copy.deepcopy(backbone)
    for parent in clone.modules():
        for name, child in list(parent.named_children()):
            if isinstance(child, LoRALinear):
                setattr(parent, name, child.merged_linear())
    return clone.state_dict()
