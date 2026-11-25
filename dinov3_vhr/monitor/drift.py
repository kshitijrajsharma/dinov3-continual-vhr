"""Live per-patch feature-drift canary as a Lightning callback.

The continued-pretraining question is whether the dense features move, not just the
class token. This measures it during training: linear CKA of the student's mean-pooled
patch tokens against the frozen released backbone, at the same layers the pangaea probes
read. The released reference is fixed, so its features on a held-out VHR probe batch are
computed once at setup and cached; each interval only runs the student. CKA 1.0 means the
dense geometry is unchanged; a drop means the patch features have moved.
"""

from pathlib import Path
from typing import cast

import pytorch_lightning as pl
import torch
from torch import Tensor
from torchvision import transforms
from webdataset.compat import WebDataset

from dinov3_vhr.backbone import ViT, as_vit, build_backbone
from dinov3_vhr.dino_ssl import DinoV3ContinuedSSL

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def linear_cka(x: Tensor, y: Tensor) -> float:
    """Linear CKA: 1.0 means identical representation geometry, lower means it moved."""
    x = (x - x.mean(0)).float()
    y = (y - y.mean(0)).float()
    cross = (x.T @ y).norm() ** 2
    return float(cross / ((x.T @ x).norm() * (y.T @ y).norm()))


def sample_probe_tiles(
    tiles_dir: Path, per_source: int, image_size: int = 256, seed: int = 0
) -> Tensor:
    """A fixed, source-balanced batch of normalized VHR tiles, in a stable order so the
    reference and the student always see identical inputs."""
    to_tensor = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    sources = sorted(
        p.name for p in tiles_dir.iterdir() if p.is_dir() and not p.name.startswith(".")
    )
    images: list[Tensor] = []
    for index, source in enumerate(sources):
        shards = sorted(str(p) for p in (tiles_dir / source).glob("*.tar"))
        if not shards:
            continue
        stream = (
            WebDataset(shards, shardshuffle=False)
            .shuffle(1000, seed=seed + index)
            .decode("pil")
            .to_tuple("png")
            .map_tuple(to_tensor)
        )
        taken = 0
        for (image,) in stream:  # ty: ignore[not-iterable]
            images.append(image)
            taken += 1
            if taken >= per_source:
                break
    if not images:
        raise FileNotFoundError(f"no probe tiles sampled under {tiles_dir}")
    return torch.stack(images)


@torch.no_grad()
def patch_features_by_layer(
    vit: ViT, images: Tensor, device: str, layers: list[int], batch: int = 16
) -> dict[int, Tensor]:
    """Mean-pooled patch tokens per requested block, one row per tile, on CPU."""
    store: dict[int, Tensor] = {}
    handles = [
        cast(torch.nn.Module, vit.blocks[i]).register_forward_hook(
            lambda _m, _inp, out, i=i: store.__setitem__(i, out.detach())
        )
        for i in layers
    ]
    prefix = vit.num_prefix_tokens
    pooled: dict[int, list[Tensor]] = {i: [] for i in layers}
    for start in range(0, len(images), batch):
        vit.forward_features(images[start : start + batch].to(device))
        for i in layers:
            pooled[i].append(store[i][:, prefix:, :].mean(dim=1).cpu())
    for handle in handles:
        handle.remove()
    return {i: torch.cat(pooled[i]) for i in layers}


class DriftMonitor(pl.Callback):
    """Log patch-token CKA to the frozen released backbone every N steps.

    Caches the released reference's patch features on a fixed VHR probe batch once, so each
    interval runs only the student. Emits monitor/cka_patch_l{layer}; watched together with
    the ImageNet-kNN forgetting canary, the target is CKA falling at mid layers while kNN
    holds (dense features moving without catastrophic forgetting).
    """

    def __init__(
        self,
        tiles_dir: Path,
        model_name: str,
        layers: list[int],
        per_source: int = 16,
        image_size: int = 256,
        every_n_steps: int = 250,
        seed: int = 0,
        infer_batch: int = 16,
    ) -> None:
        self.tiles_dir = Path(tiles_dir)
        self.model_name = model_name
        self.layers = layers
        self.per_source = per_source
        self.image_size = image_size
        self.every_n_steps = every_n_steps
        self.seed = seed
        self.infer_batch = infer_batch
        self.probe_images: Tensor | None = None
        self.reference: dict[int, Tensor] | None = None
        self._last_logged_step = -1

    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str) -> None:
        if self.reference is not None:
            return
        device = str(pl_module.device)
        self.probe_images = sample_probe_tiles(
            self.tiles_dir, self.per_source, self.image_size, self.seed
        )
        released = build_backbone(self.model_name, pretrained=True).eval().to(device)
        self.reference = patch_features_by_layer(
            as_vit(released), self.probe_images, device, self.layers, self.infer_batch
        )
        del released
        torch.cuda.empty_cache()

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: object,
        batch: object,
        batch_idx: int,
    ) -> None:
        step = trainer.global_step
        # on_train_batch_end fires once per micro-batch, so under gradient accumulation the
        # step is unchanged across the window; log once per optimizer step, not per micro-batch.
        if step == 0 or step % self.every_n_steps != 0 or step == self._last_logged_step:
            return
        self._last_logged_step = step
        assert self.probe_images is not None and self.reference is not None
        assert isinstance(pl_module, DinoV3ContinuedSSL)
        backbone = pl_module.student_backbone
        was_training = backbone.training
        backbone.eval()
        # A no_grad forward through grad-checkpointed blocks hangs, so disable checkpointing
        # for the probe and restore it before training resumes.
        checkpointing = pl_module.grad_checkpointing
        if checkpointing:
            as_vit(backbone).set_grad_checkpointing(False)
        current = patch_features_by_layer(
            as_vit(backbone),
            self.probe_images,
            str(pl_module.device),
            self.layers,
            self.infer_batch,
        )
        if checkpointing:
            as_vit(backbone).set_grad_checkpointing(True)
        if was_training:
            backbone.train()
        for layer in self.layers:
            cka = linear_cka(current[layer], self.reference[layer])
            pl_module.log(f"monitor/cka_patch_l{layer}", cka, sync_dist=True)
