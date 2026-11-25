"""DINO multi-crop transform that records the two global crops' geometry.

lightly's DINOTransform discards the RandomResizedCrop box, but the cross-view correspondence
objective needs it. This reuses lightly's exact per-view transforms (so the photometric
pipeline and the local crops are identical to the baseline) and only replaces each global
crop's RandomResizedCrop with a box-capturing equivalent. Flips are not tracked, so it runs
flip-free (hf_prob is 0); the box is (top, left, height, width, 0) in source pixels.
"""

from typing import Any, cast

import torch
from lightly.transforms import DINOTransform
from PIL.Image import Image
from torch import Tensor
from torchvision.transforms import Compose, RandomResizedCrop
from torchvision.transforms import functional as TF


class CorrespondenceDINOTransform:
    def __init__(
        self,
        global_crop_size: int = 256,
        global_crop_scale: tuple[float, float] = (0.32, 1.0),
        local_crop_size: int = 112,
        local_crop_scale: tuple[float, float] = (0.05, 0.32),
        n_local_views: int = 8,
    ) -> None:
        base = DINOTransform(
            global_crop_size=global_crop_size,
            global_crop_scale=global_crop_scale,
            local_crop_size=local_crop_size,
            local_crop_scale=local_crop_scale,
            n_local_views=n_local_views,
            hf_prob=0.0,
        )
        views = cast(list[Any], base.transforms)
        self._global_views = views[:2]
        self._local_views = views[2:]

    def _global(self, view_transform: Any, image: Image) -> tuple[Tensor, Tensor]:
        steps = cast(Compose, view_transform.transform).transforms
        crop = cast(RandomResizedCrop, steps[0])
        top, left, height, width = cast(Any, RandomResizedCrop).get_params(
            image, crop.scale, crop.ratio
        )
        cropped = TF.resized_crop(
            cast(Any, image), top, left, height, width, crop.size, crop.interpolation
        )
        view = cast(Tensor, Compose(list(steps[1:]))(cropped))
        box = torch.tensor([top, left, height, width, 0.0], dtype=torch.float32)
        return view, box

    def __call__(self, image: Image) -> tuple[list[Tensor], Tensor]:
        views: list[Tensor] = []
        boxes: list[Tensor] = []
        for view_transform in self._global_views:
            view, box = self._global(view_transform, image)
            views.append(view)
            boxes.append(box)
        views.extend(cast(Tensor, view_transform(image)) for view_transform in self._local_views)
        return views, torch.stack(boxes)
