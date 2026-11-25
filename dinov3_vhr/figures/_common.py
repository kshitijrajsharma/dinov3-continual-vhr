"""Shared helpers for the paper figure generators: backbone loading, normalization,
device selection, patch-PCA, and the balanced kNN split."""

import io
import tarfile
from pathlib import Path

import numpy as np
import timm
import torch
from PIL import Image
from sklearn.decomposition import PCA
from torchvision import transforms

from dinov3_vhr.backbone import ViT, as_vit
from dinov3_vhr.monitor.knn import weighted_knn_accuracy

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def select_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_encoder(model_name: str, weights: Path | None, device: str) -> ViT:
    """Stock timm backbone when weights is None, else a continued-pretrained checkpoint."""
    backbone = timm.create_model(model_name, pretrained=weights is None, num_classes=0)
    if weights is not None:
        backbone.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    return as_vit(backbone.to(device).eval())


def parse_models(stock_name: str, checkpoint_specs: list[str]) -> dict[str, Path | None]:
    """Map display name to weights: 'stock' (None, the released model) plus each name=path spec."""
    models: dict[str, Path | None] = {stock_name: None}
    for spec in checkpoint_specs:
        name, sep, path = spec.partition("=")
        if not sep:
            raise ValueError(f"checkpoint spec must be name=path.pth, got {spec!r}")
        models[name] = Path(path)
    return models


def pca_rgb(patches: torch.Tensor) -> np.ndarray:
    """Top-3 PCA components of one image's patch tokens as a (side, side, 3) RGB map."""
    side = int(patches.shape[0] ** 0.5)
    reduced = PCA(n_components=3).fit_transform(patches.float().numpy())
    lo, hi = reduced.min(0), reduced.max(0)
    normalized = (reduced - lo) / np.clip(hi - lo, 1e-6, None)
    return normalized.reshape(side, side, 3)


def balanced_knn(features: torch.Tensor, labels: np.ndarray, classes: int, k: int = 20) -> float:
    """Weighted kNN top-1 with a class-balanced half/half bank/query split."""
    target = torch.from_numpy(labels)
    bank = torch.zeros(len(labels), dtype=torch.bool)
    for label in np.unique(labels):
        idx = np.flatnonzero(labels == label)
        bank[idx[: len(idx) // 2]] = True
    return weighted_knn_accuracy(
        features[bank], target[bank], features[~bank], target[~bank], classes, k=k
    )


def load_tiles_from_tar(
    tar_path: Path, n: int, size: int
) -> tuple[list[Image.Image], torch.Tensor]:
    """First n PNG tiles from a WebDataset shard as display images and a normalized batch."""
    to_tensor = transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    displays, tensors = [], []
    with tarfile.open(tar_path) as tar:
        names = sorted(m for m in tar.getnames() if m.endswith(".png"))[:n]
        for name in names:
            member = tar.extractfile(name)
            if member is None:
                raise FileNotFoundError(f"cannot read {name} in {tar_path}")
            image = Image.open(io.BytesIO(member.read())).convert("RGB")
            displays.append(image.resize((size, size)))
            tensors.append(to_tensor(image))
    return displays, torch.stack(tensors)
