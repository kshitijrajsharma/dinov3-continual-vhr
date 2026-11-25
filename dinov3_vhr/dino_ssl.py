"""Continued DINO self-supervised pretraining of a released DINOv3 backbone.

The student starts from the released DINOv3 weights and trains on VHR imagery with five
objectives: DINO global self-distillation, iBOT masked-patch self-distillation, DKoleo
embedding spread, Gram anchoring to preserve dense patch geometry, and a CLS anchor to
preserve the global embedding classification reads. Both anchors target the same teacher,
which by default is periodically re-snapshotted from the EMA teacher (a VHR-adapted
reference) rather than pinned to the frozen released model.

The released checkpoints carry backbone weights only, so the DINO and iBOT projection
heads start from random init. Training therefore runs in two stages: the heads converge
against a held backbone, which then unfreezes behind a linear learning-rate ramp.
"""

import copy
from typing import Any, cast

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from lightly.loss import DINOLoss, IBOTPatchLoss, KoLeoLoss
from lightly.models.modules import DINOProjectionHead
from lightly.models.utils import (
    deactivate_requires_grad,
    get_weight_decay_parameters,
    update_momentum,
)
from lightly.utils.scheduler import CosineWarmupScheduler, cosine_schedule
from pytorch_lightning.utilities.types import OptimizerLRScheduler
from timm.optim import param_groups_layer_decay
from torch import Tensor

from dinov3_vhr.backbone import as_vit, build_backbone
from dinov3_vhr.ssl.correspondence import (
    PatchMLP,
    byol_correspondence_loss,
    correspondence_loss,
    patch_correspondence,
    warp_patch_features,
)
from dinov3_vhr.ssl.explora import apply_explora, merged_state_dict


def multicrop_collate(
    batch: list[tuple[Any, int]],
) -> tuple[Any, ...]:
    """Group a batch of per-image crop lists into per-view batched tensors. When the
    transform also returns per-image global-crop boxes (correspondence mode), each item is
    ((views, boxes), label) and the boxes are stacked into the returned tuple."""
    labels = torch.tensor([item[1] for item in batch])
    if isinstance(batch[0][0], tuple):
        views_per_image = [item[0][0] for item in batch]
        boxes = torch.stack([item[0][1] for item in batch])
        batched_views = [torch.stack(v) for v in zip(*views_per_image, strict=True)]
        return batched_views, boxes, labels
    views_per_image = [item[0] for item in batch]
    batched_views = [torch.stack(v) for v in zip(*views_per_image, strict=True)]
    return batched_views, labels


def gram_loss(student_patches: Tensor, teacher_patches: Tensor) -> Tensor:
    """MSE between L2-normalized patch-similarity matrices (DINOv3 Eq. 2)."""
    student = F.normalize(student_patches.float(), dim=-1)
    teacher = F.normalize(teacher_patches.float(), dim=-1)
    gram_student = student @ student.transpose(-1, -2)
    gram_teacher = teacher @ teacher.transpose(-1, -2)
    return F.mse_loss(gram_student, gram_teacher)


def cls_anchor_loss(student_cls: Tensor, teacher_cls: Tensor) -> Tensor:
    """One minus mean cosine similarity between the student and anchor class embeddings.

    The global counterpart of the Gram anchor: it preserves the class-token representation
    downstream classification reads, distilling to the same self-checkpoint teacher so it
    holds structure without freezing out domain adaptation (CaSSLe, arXiv 2112.04215)."""
    return 1.0 - F.cosine_similarity(student_cls.float(), teacher_cls.float(), dim=-1).mean()


def patch_grid_mask(
    batch_size: int, side: int, mask_ratio: float, device: torch.device | str
) -> Tensor:
    """Per-image boolean mask selecting mask_ratio of the side*side patches."""
    count = int(mask_ratio * side * side)
    order = torch.rand(batch_size, side * side, device=device).argsort(dim=1)
    mask = torch.zeros(batch_size, side * side, dtype=torch.bool, device=device)
    mask.scatter_(1, order[:, :count], True)
    return mask.view(batch_size, side, side)


class DinoV3ContinuedSSL(pl.LightningModule):
    def __init__(
        self,
        model_name: str = "vit_small_patch16_dinov3.lvd1689m",
        learning_rate: float = 3e-5,
        weight_decay: float = 0.04,
        layer_decay: float = 0.9,
        warmup_fraction: float = 0.1,
        min_lr_ratio: float = 0.01,
        max_epochs: int = 10,
        steps_per_epoch: int = 1000,
        output_dim: int = 8192,
        teacher_temp: float = 0.04,
        momentum_start: float = 0.996,
        n_global_crops: int = 2,
        gram_weight: float = 2.0,
        ibot_weight: float = 1.0,
        koleo_weight: float = 0.1,
        cls_anchor_weight: float = 1.0,
        corr_weight: float = 0.0,
        corr_layers: tuple[int, ...] = (),
        corr_ibot: bool = False,
        corr_byol: bool = False,
        corr_proj_hidden: int = 2048,
        corr_proj_dim: int = 256,
        global_crop_size: int = 256,
        mask_ratio: float = 0.3,
        freeze_backbone_steps: int = 1000,
        backbone_warmup_steps: int = 500,
        anchor_target: str = "refresh",
        anchor_refresh_steps: int = 1500,
        anchor_max_refreshes: int = 2,
        grad_checkpointing: bool = False,
        explora: bool = False,
        lora_rank: int = 64,
        lora_alpha: float = 64.0,
        unfreeze_blocks: tuple[int, ...] = (23,),
    ) -> None:
        super().__init__()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.layer_decay = layer_decay
        self.warmup_fraction = warmup_fraction
        self.min_lr_ratio = min_lr_ratio
        self.max_epochs = max_epochs
        self.steps_per_epoch = steps_per_epoch
        self.momentum_start = momentum_start
        self.n_global_crops = n_global_crops
        self.gram_weight = gram_weight
        self.ibot_weight = ibot_weight
        self.koleo_weight = koleo_weight
        self.cls_anchor_weight = cls_anchor_weight
        self.corr_weight = corr_weight
        self.corr_layers = corr_layers
        self.corr_ibot = corr_ibot
        self.corr_byol = corr_byol
        self.global_crop_size = global_crop_size
        self.mask_ratio = mask_ratio
        self.freeze_backbone_steps = freeze_backbone_steps
        self.backbone_warmup_steps = backbone_warmup_steps
        if anchor_target not in {"frozen", "refresh"}:
            raise ValueError(f"anchor_target must be frozen or refresh, got {anchor_target}")
        self.anchor_target = anchor_target
        self.anchor_refresh_steps = anchor_refresh_steps
        self.anchor_max_refreshes = anchor_max_refreshes
        self.refreshes_done = 0

        self.student_backbone = build_backbone(model_name, pretrained=True)
        # Only the student holds backward activations, so checkpointing it is what lets ViT-L fit.
        self.grad_checkpointing = grad_checkpointing
        if grad_checkpointing:
            as_vit(self.student_backbone).set_grad_checkpointing(True)
        # ExPLoRA restricts adaptation to LoRA plus the last block, norms, and prefix tokens;
        # apply it before the teacher copies so they mirror the student's structure.
        self.explora = explora
        if explora:
            self.adapt_param_names: set[str] | None = apply_explora(
                self.student_backbone, lora_rank, lora_alpha, list(unfreeze_blocks)
            )
        else:
            self.adapt_param_names = None
        embed_dim = as_vit(self.student_backbone).embed_dim
        self.student_head = DINOProjectionHead(
            input_dim=embed_dim, output_dim=output_dim, freeze_last_layer=1
        )
        self.student_ibot_head = DINOProjectionHead(
            input_dim=embed_dim, output_dim=output_dim, freeze_last_layer=1
        )
        # Learnable token that replaces masked patch embeddings for the iBOT objective.
        self.mask_token = torch.nn.Parameter(torch.zeros(1, 1, 1, embed_dim))

        # Reference for the patch (Gram) and CLS anchors; when refreshed it tracks VHR
        # adaptation rather than pinning to natural-image structure (CaSSLe 2112.04215).
        self.anchor_teacher_backbone = copy.deepcopy(self.student_backbone)
        deactivate_requires_grad(self.anchor_teacher_backbone)

        self.teacher_backbone = copy.deepcopy(self.student_backbone)
        # DINOProjectionHead uses weight_norm, which cannot be deepcopied; build fresh
        # and load the student weights so the teacher starts identical to the student.
        self.teacher_head = DINOProjectionHead(
            input_dim=embed_dim, output_dim=output_dim, freeze_last_layer=1
        )
        self.teacher_head.load_state_dict(self.student_head.state_dict())
        self.teacher_ibot_head = DINOProjectionHead(
            input_dim=embed_dim, output_dim=output_dim, freeze_last_layer=1
        )
        self.teacher_ibot_head.load_state_dict(self.student_ibot_head.state_dict())
        deactivate_requires_grad(self.teacher_backbone)
        deactivate_requires_grad(self.teacher_head)
        deactivate_requires_grad(self.teacher_ibot_head)

        self.criterion = DINOLoss(
            output_dim=output_dim,
            warmup_teacher_temp=teacher_temp,
            teacher_temp=teacher_temp,
            warmup_teacher_temp_epochs=0,
        )
        self.ibot_criterion = IBOTPatchLoss(output_dim=output_dim, teacher_temp=teacher_temp)
        # iBOT-head correspondence gives each layer its own projection head, EMA teacher, and
        # centered criterion, so a shared head never starves the mid-layers and each layer's
        # cross-entropy can descend on its own features.
        corr_head_layers = corr_layers if corr_ibot else ()
        self.corr_ibot_criteria = torch.nn.ModuleList(
            IBOTPatchLoss(output_dim=output_dim, teacher_temp=teacher_temp)
            for _ in corr_head_layers
        )
        self.corr_student_heads = torch.nn.ModuleList(
            DINOProjectionHead(input_dim=embed_dim, output_dim=output_dim, freeze_last_layer=1)
            for _ in corr_head_layers
        )
        self.corr_teacher_heads = torch.nn.ModuleList(
            DINOProjectionHead(input_dim=embed_dim, output_dim=output_dim, freeze_last_layer=1)
            for _ in corr_head_layers
        )
        for student_head, teacher_head in zip(
            self.corr_student_heads, self.corr_teacher_heads, strict=True
        ):
            teacher_head.load_state_dict(student_head.state_dict())
            deactivate_requires_grad(teacher_head)
        # BYOL-style correspondence gives each layer a projector plus predictor on the student
        # and an EMA projector (no predictor) on the teacher. The predictor-EMA asymmetry with
        # stop-grad keeps the regression objective off the trivial constant, where a bare cosine
        # loss collapses.
        corr_byol_layers = corr_layers if corr_byol else ()
        self.corr_projector = torch.nn.ModuleList(
            PatchMLP(embed_dim, corr_proj_hidden, corr_proj_dim) for _ in corr_byol_layers
        )
        self.corr_predictor = torch.nn.ModuleList(
            PatchMLP(corr_proj_dim, corr_proj_hidden, corr_proj_dim) for _ in corr_byol_layers
        )
        self.corr_teacher_projector = torch.nn.ModuleList(
            PatchMLP(embed_dim, corr_proj_hidden, corr_proj_dim) for _ in corr_byol_layers
        )
        for projector, teacher_projector in zip(
            self.corr_projector, self.corr_teacher_projector, strict=True
        ):
            teacher_projector.load_state_dict(projector.state_dict())
            deactivate_requires_grad(teacher_projector)
        # DINOv3 DKoleo regularizer on the L2-normalized class-token embedding.
        self.koleo_criterion = KoLeoLoss()
        # Per-layer BYOL feature std, stashed each step by the correspondence term and logged as a
        # collapse canary; a value trending to zero means the projected features are degenerating.
        self._corr_std: dict[int, Tensor] = {}

    def _embed(self, backbone: torch.nn.Module, x: Tensor) -> Tensor:
        vit = as_vit(backbone)
        return vit.forward_head(vit.forward_features(x), pre_logits=True)

    def _patch_tokens(self, backbone: torch.nn.Module, features: Tensor) -> Tensor:
        return features[:, as_vit(backbone).num_prefix_tokens :, :]

    def _student_masked_patches(self, x: Tensor, mask: Tensor) -> Tensor:
        """Student patch tokens with masked positions replaced by the mask token."""
        vit = as_vit(self.student_backbone)
        tokens = vit.patch_embed(x)
        tokens = torch.where(mask.unsqueeze(-1), self.mask_token, tokens)
        tokens, rope = vit._pos_embed(tokens)
        tokens = vit.norm_pre(tokens)
        for block in vit.blocks:
            tokens = block(tokens, rope=rope)
        tokens = vit.norm(tokens)
        return tokens[:, vit.num_prefix_tokens :, :]

    def _ibot_loss(self, view: Tensor) -> Tensor:
        """Masked-patch self-distillation against the EMA teacher (iBOT)."""
        vit = as_vit(self.student_backbone)
        side = view.shape[-1] // vit.patch_embed.patch_size[0]
        mask = patch_grid_mask(view.shape[0], side, self.mask_ratio, view.device)
        student_patches = self._student_masked_patches(view, mask)
        with torch.no_grad():
            teacher_patches = self._patch_tokens(
                self.teacher_backbone,
                as_vit(self.teacher_backbone).forward_features(view),
            )
        flat_mask = mask.flatten(1)
        student_out = self.student_ibot_head(student_patches)[flat_mask]
        teacher_out = self.teacher_ibot_head(teacher_patches)[flat_mask]
        return self.ibot_criterion(teacher_out, student_out, mask)

    def _freeze_backbone(self, frozen: bool) -> None:
        """Gate gradients on the pretrained backbone. During head warmup only patch_embed
        trains; after it, the whole backbone trains, or, under ExPLoRA, only the adapter
        subset (LoRA, the last block, norms, and prefix tokens)."""
        for name, param in self.student_backbone.named_parameters():
            if frozen:
                param.requires_grad = name.startswith("patch_embed")
            elif self.adapt_param_names is None:
                param.requires_grad = True
            else:
                param.requires_grad = name in self.adapt_param_names

    def export_backbone_state_dict(self) -> dict[str, Tensor]:
        """The student backbone as a plain timm state_dict; under ExPLoRA the low-rank
        updates are folded into the base weights so downstream eval loads it unchanged."""
        if self.explora:
            return merged_state_dict(self.student_backbone)
        return self.student_backbone.state_dict()

    def _backbone_lr_scale(self, step: int) -> float:
        """Linear 0 to 1 ramp over backbone_warmup_steps once the backbone unfreezes."""
        if step < self.freeze_backbone_steps:
            return 1.0
        ramp = step - self.freeze_backbone_steps
        if ramp >= self.backbone_warmup_steps:
            return 1.0
        return ramp / self.backbone_warmup_steps

    def _maybe_refresh_anchor(self, step: int) -> None:
        """Re-snapshot the anchor teacher from the current EMA teacher, up to a fixed count.

        Only after the backbone unfreezes (a snapshot while frozen equals the released
        model), so each refresh captures a VHR-adapted-but-not-degraded reference."""
        if self.anchor_target != "refresh" or step <= self.freeze_backbone_steps:
            return
        if self.refreshes_done >= self.anchor_max_refreshes:
            return
        if step % self.anchor_refresh_steps != 0:
            return
        self.anchor_teacher_backbone.load_state_dict(self.teacher_backbone.state_dict())
        deactivate_requires_grad(self.anchor_teacher_backbone)
        self.refreshes_done += 1

    def on_train_batch_start(self, batch: object, batch_idx: int) -> None:
        """Two-stage schedule. The released checkpoint ships no projection heads, so the
        randomly initialised DINO and iBOT heads train against a held backbone before it
        is released; unfreezing them together destabilises the pretrained weights."""
        step = self.trainer.global_step
        self._freeze_backbone(step < self.freeze_backbone_steps)
        self._maybe_refresh_anchor(step)
        scale = self._backbone_lr_scale(step)
        if scale == 1.0:
            return
        for group in self.trainer.optimizers[0].param_groups:
            if group.get("is_backbone"):
                group["lr"] *= scale

    def forward(self, x: Tensor) -> Tensor:
        return self.student_head(self._embed(self.student_backbone, x))

    @torch.no_grad()
    def forward_teacher(self, x: Tensor) -> Tensor:
        return self.teacher_head(self._embed(self.teacher_backbone, x))

    def training_step(self, batch: tuple[list[Tensor], Tensor], batch_idx: int) -> Tensor:
        momentum = cosine_schedule(self.current_epoch, self.max_epochs, self.momentum_start, 1.0)
        update_momentum(self.student_backbone, self.teacher_backbone, m=momentum)
        update_momentum(self.student_head, self.teacher_head, m=momentum)
        update_momentum(self.student_ibot_head, self.teacher_ibot_head, m=momentum)
        for student_head, teacher_head in zip(
            self.corr_student_heads, self.corr_teacher_heads, strict=True
        ):
            update_momentum(student_head, teacher_head, m=momentum)
        for projector, teacher_projector in zip(
            self.corr_projector, self.corr_teacher_projector, strict=True
        ):
            update_momentum(projector, teacher_projector, m=momentum)

        views = batch[0]
        boxes = batch[1] if len(batch) == 3 else None
        n_global = self.n_global_crops
        global_views = views[:n_global]
        local_views = views[n_global:]
        teacher_out = [self.forward_teacher(v) for v in global_views]

        student_out = []
        gram = views[0].new_zeros(())
        ibot = views[0].new_zeros(())
        koleo = views[0].new_zeros(())
        cls_anchor = views[0].new_zeros(())
        corr = views[0].new_zeros(())
        student_corr: list[Any] = []
        anchor = as_vit(self.anchor_teacher_backbone)
        for view in global_views:
            vit = as_vit(self.student_backbone)
            if self.corr_weight > 0 and self.corr_layers:
                features, layer_tokens = self._capture_corr_layers(self.student_backbone, view)
                student_corr.append(layer_tokens)
            else:
                features = vit.forward_features(view)
                if self.corr_weight > 0:
                    student_corr.append(self._patch_tokens(self.student_backbone, features))
            embedding = vit.forward_head(features, pre_logits=True)
            student_out.append(self.student_head(embedding))
            koleo = koleo + self.koleo_criterion(embedding)
            # One anchor forward serves both preservation losses; weight 0 disables each.
            if self.gram_weight > 0 or self.cls_anchor_weight > 0:
                with torch.no_grad():
                    anchor_features = anchor.forward_features(view)
                if self.gram_weight > 0:
                    gram = gram + gram_loss(
                        self._patch_tokens(self.student_backbone, features),
                        self._patch_tokens(self.anchor_teacher_backbone, anchor_features),
                    )
                if self.cls_anchor_weight > 0:
                    with torch.no_grad():
                        anchor_cls = anchor.forward_head(anchor_features, pre_logits=True)
                    cls_anchor = cls_anchor + cls_anchor_loss(embedding, anchor_cls)
            ibot = ibot + self._ibot_loss(view)
        if self.corr_weight > 0 and boxes is not None:
            corr = self._correspondence_loss(global_views, student_corr, boxes)
        gram = gram / n_global
        ibot = ibot / n_global
        koleo = koleo / n_global
        cls_anchor = cls_anchor / n_global
        student_out.extend(self.forward(v) for v in local_views)

        dino = self.criterion(teacher_out, student_out, epoch=self.current_epoch)
        loss = (
            dino
            + self.gram_weight * gram
            + self.ibot_weight * ibot
            + self.koleo_weight * koleo
            + self.cls_anchor_weight * cls_anchor
            + self.corr_weight * corr
        )
        self.log("dino_loss", dino, prog_bar=True)
        self.log("gram_loss", gram, prog_bar=True)
        self.log("ibot_loss", ibot, prog_bar=True)
        self.log("koleo_loss", koleo, prog_bar=True)
        self.log("cls_anchor_loss", cls_anchor, prog_bar=True)
        self.log("corr_loss", corr, prog_bar=True)
        for layer, std in self._corr_std.items():
            self.log(f"corr_std_l{layer}", std)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def _capture_corr_layers(
        self, backbone: torch.nn.Module, view: Tensor
    ) -> tuple[Tensor, dict[int, Tensor]]:
        """forward_features once, capturing patch tokens at each corr layer via hooks.
        Returns the final features and a per-layer patch-token map (prefix stripped)."""
        vit = as_vit(backbone)
        store: dict[int, Tensor] = {}
        handles = [
            cast(torch.nn.Module, vit.blocks[i]).register_forward_hook(
                lambda _m, _inp, out, i=i: store.__setitem__(i, out)
            )
            for i in self.corr_layers
        ]
        final = vit.forward_features(view)
        for handle in handles:
            handle.remove()
        prefix = vit.num_prefix_tokens
        return final, {i: store[i][:, prefix:, :] for i in self.corr_layers}

    def _correspondence_loss(
        self, global_views: list[Tensor], student: list[Any], boxes: Tensor
    ) -> Tensor:
        """Symmetric cross-view dense correspondence between the two global crops: the teacher
        patch features on each view are warped into the other view's frame and the student's
        patches are pulled toward them over the overlap. With corr_layers set the loss is
        applied at each of those intermediate layers (direct pressure on mid dense features);
        otherwise only on the final patch tokens."""
        with torch.no_grad():
            if self.corr_layers:
                teacher = [
                    self._capture_corr_layers(self.teacher_backbone, view)[1]
                    for view in global_views
                ]
            else:
                teacher_vit = as_vit(self.teacher_backbone)
                teacher = [
                    self._patch_tokens(self.teacher_backbone, teacher_vit.forward_features(view))
                    for view in global_views
                ]
        sample = student[0][self.corr_layers[0]] if self.corr_layers else student[0]
        side = int(round(sample.shape[1] ** 0.5))
        box_a, box_b = boxes[:, 0], boxes[:, 1]
        grid_ab, mask_ab = patch_correspondence(box_a, box_b, side, self.global_crop_size)
        grid_ba, mask_ba = patch_correspondence(box_b, box_a, side, self.global_crop_size)
        if not self.corr_layers:
            t0, t1 = cast(Tensor, teacher[0]), cast(Tensor, teacher[1])
            return 0.5 * (
                correspondence_loss(student[0], t1, grid_ab, mask_ab, side)
                + correspondence_loss(student[1], t0, grid_ba, mask_ba, side)
            )
        terms = []
        for index, layer in enumerate(self.corr_layers):
            if self.corr_ibot:
                terms.append(
                    self._corr_ibot_term(
                        student[0][layer],
                        teacher[1][layer],
                        grid_ab,
                        mask_ab,
                        side,
                        index,
                    )
                )
                terms.append(
                    self._corr_ibot_term(
                        student[1][layer],
                        teacher[0][layer],
                        grid_ba,
                        mask_ba,
                        side,
                        index,
                    )
                )
            elif self.corr_byol:
                terms.append(
                    self._corr_byol_term(
                        student[0][layer],
                        teacher[1][layer],
                        grid_ab,
                        mask_ab,
                        side,
                        index,
                    )
                )
                terms.append(
                    self._corr_byol_term(
                        student[1][layer],
                        teacher[0][layer],
                        grid_ba,
                        mask_ba,
                        side,
                        index,
                    )
                )
            else:
                terms.append(
                    correspondence_loss(
                        student[0][layer], teacher[1][layer], grid_ab, mask_ab, side
                    )
                )
                terms.append(
                    correspondence_loss(
                        student[1][layer], teacher[0][layer], grid_ba, mask_ba, side
                    )
                )
        return sum(terms) / len(terms)  # ty: ignore[invalid-return-type]

    def _corr_ibot_term(
        self,
        student_feat: Tensor,
        teacher_feat: Tensor,
        sampling_grid: Tensor,
        mask: Tensor,
        side: int,
        index: int,
    ) -> Tensor:
        """One layer, one direction of iBOT-head correspondence: warp the teacher patches
        (nearest, no blur) into the student's frame, project both through this layer's own
        head, and take the centered cross-entropy over the overlapping patches. The per-layer
        head and centered criterion supply the anti-collapse that raw cosine lacks and let each
        layer's features actually move."""
        teacher_warped = warp_patch_features(teacher_feat, sampling_grid, side, "nearest")
        flat = mask.reshape(mask.shape[0], -1)
        student_out = self.corr_student_heads[index](student_feat)[flat]
        teacher_out = self.corr_teacher_heads[index](teacher_warped)[flat]
        return self.corr_ibot_criteria[index](teacher_out, student_out, mask)

    def _corr_byol_term(
        self,
        student_feat: Tensor,
        teacher_feat: Tensor,
        sampling_grid: Tensor,
        mask: Tensor,
        side: int,
        index: int,
    ) -> Tensor:
        """One layer, one direction of BYOL correspondence: project and predict the student patches,
        project the teacher patches with the EMA projector, and regress the student prediction onto
        the warped stop-grad teacher target over the overlap. Logs the per-layer feature std as a
        collapse canary, so the objective is provably learning rather than degenerating to a
        constant."""
        student_pred = self.corr_predictor[index](self.corr_projector[index](student_feat))
        with torch.no_grad():
            teacher_proj = self.corr_teacher_projector[index](teacher_feat)
        self._corr_std[self.corr_layers[index]] = (
            F.normalize(student_pred.detach().float().flatten(0, 1), dim=-1).std(dim=0).mean()
        )
        return byol_correspondence_loss(student_pred, teacher_proj, sampling_grid, mask, side)

    def on_after_backward(self) -> None:
        self.student_head.cancel_last_layer_gradients(current_epoch=self.current_epoch)
        self.student_ibot_head.cancel_last_layer_gradients(current_epoch=self.current_epoch)
        for head in self.corr_student_heads:
            head.cancel_last_layer_gradients(current_epoch=self.current_epoch)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """AdamW with layer-wise LR decay on the backbone, warmup-cosine LR, no weight
        decay on norms and biases. The EMA teacher and anchor teacher are frozen, so they
        are not in any group; the EMA teacher tracks the student in the training step."""
        backbone_groups = param_groups_layer_decay(
            self.student_backbone,
            weight_decay=self.weight_decay,
            no_weight_decay_list=as_vit(self.student_backbone).no_weight_decay(),
            layer_decay=self.layer_decay,
            weight_decay_exclude_1d=True,
        )
        for group in backbone_groups:
            group["lr"] = self.learning_rate * group["lr_scale"]
            group["is_backbone"] = True

        decay, no_decay = get_weight_decay_parameters(
            [
                self.student_head,
                self.student_ibot_head,
                *self.corr_student_heads,
                *self.corr_projector,
                *self.corr_predictor,
            ]
        )
        head_groups = [
            {
                "params": decay,
                "weight_decay": self.weight_decay,
                "lr": self.learning_rate,
            },
            {
                "params": [*no_decay, self.mask_token],
                "weight_decay": 0.0,
                "lr": self.learning_rate,
            },
        ]
        optimizer = torch.optim.AdamW(
            [*backbone_groups, *head_groups], lr=self.learning_rate, betas=(0.9, 0.99)
        )

        total_steps = self.steps_per_epoch * self.max_epochs
        scheduler = CosineWarmupScheduler(
            optimizer,
            warmup_epochs=max(1, int(self.warmup_fraction * total_steps)),
            max_epochs=total_steps,
            end_value=self.min_lr_ratio,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
