"""In-loop forgetting and collapse canary as a Lightning callback.

Every N steps it reads two cheap signals off the frozen student features:

- forgetting: weighted kNN top-1 on ImageNet-1k val (the source domain LVD was trained on),
- collapse: the standard deviation of the L2-normalized embeddings.

These are diagnostics, not benchmark numbers. VHR learning is measured separately by
running the real pangaea suite on saved checkpoints, with its own decoders and splits.
Keeping the two apart stops a cheap proxy from being mistaken for a result.
"""

from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from dinov3_vhr.backbone import ViT, as_vit
from dinov3_vhr.dino_ssl import DinoV3ContinuedSSL
from dinov3_vhr.monitor.knn import weighted_knn_accuracy

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _normalize() -> transforms.Compose:
    return transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    )


@torch.no_grad()
def _cls_features(model: ViT, loader: DataLoader, device: str) -> tuple[Tensor, Tensor]:
    features, labels = [], []
    for images, targets in loader:
        embedding = model.forward_head(model.forward_features(images.to(device)), pre_logits=True)
        features.append(embedding.cpu())
        labels.append(targets)
    return torch.cat(features), torch.cat(labels)


class SourceCoverageMonitor(pl.Callback):
    """Tally per-source tile draws so every sensor is provably seen and in what mix.

    Sample labels carry the source index (set by the tile loader). Counts accumulate and
    log every N steps, giving a running per-sensor draw share for the reproducibility audit.
    """

    def __init__(self, source_names: list[str], every_n_steps: int = 250) -> None:
        self.source_names = source_names
        self.every_n_steps = every_n_steps
        self.counts = np.zeros(len(source_names), dtype="int64")

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx: int,
    ) -> None:
        labels = batch[-1]
        for index in labels.tolist():
            self.counts[index] += 1
        step = trainer.global_step
        if step == 0 or step % self.every_n_steps != 0:
            return
        total = max(1, int(self.counts.sum()))
        for index, name in enumerate(self.source_names):
            pl_module.log(f"source/{name}", self.counts[index] / total, sync_dist=True)


class ForgettingMonitor(pl.Callback):
    def __init__(
        self,
        probe_root: Path,
        every_n_steps: int = 1000,
        imagenet_classes: int = 1000,
        batch_size: int = 64,
        knn_k: int = 20,
    ) -> None:
        self.probe_root = Path(probe_root)
        self.every_n_steps = every_n_steps
        self.imagenet_classes = imagenet_classes
        self.batch_size = batch_size
        self.knn_k = knn_k
        self._last_logged_step = -1

    def _imagenet_knn(self, model: ViT, device: str) -> tuple[float, float]:
        root = self.probe_root / "imagenet_val"

        def loader(split: str) -> DataLoader:
            return DataLoader(
                ImageFolder(str(root / split), transform=_normalize()),
                batch_size=self.batch_size,
                num_workers=4,
            )

        bank_x, bank_y = _cls_features(model, loader("bank"), device)
        query_x, query_y = _cls_features(model, loader("query"), device)
        accuracy = weighted_knn_accuracy(
            bank_x, bank_y, query_x, query_y, self.imagenet_classes, k=self.knn_k
        )
        feature_std = float(F.normalize(query_x, dim=1).std(0).mean())
        return accuracy, feature_std

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx: int,
    ) -> None:
        step = trainer.global_step
        # on_train_batch_end fires once per micro-batch, so under gradient accumulation the
        # step is unchanged across the window; run the kNN probe once per optimizer step.
        if step == 0 or step % self.every_n_steps != 0 or step == self._last_logged_step:
            return
        self._last_logged_step = step
        assert isinstance(pl_module, DinoV3ContinuedSSL)
        backbone = pl_module.student_backbone
        was_training = backbone.training
        backbone.eval()
        # A no_grad forward through grad-checkpointed blocks hangs, so disable checkpointing
        # for the probe and restore it before training resumes.
        checkpointing = pl_module.grad_checkpointing
        if checkpointing:
            as_vit(backbone).set_grad_checkpointing(False)
        knn, feature_std = self._imagenet_knn(as_vit(backbone), str(pl_module.device))
        if checkpointing:
            as_vit(backbone).set_grad_checkpointing(True)
        if was_training:
            backbone.train()
        pl_module.log("monitor/imagenet_knn", knn, prog_bar=True, sync_dist=True)
        pl_module.log("monitor/feature_std", feature_std, sync_dist=True)
