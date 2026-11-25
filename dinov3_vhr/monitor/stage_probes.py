"""Stage the forgetting probe into a compact local folder.

Run once. Downloads ImageNet-1k validation, the source domain the released DINOv3 was
trained on, and writes a fixed deterministic subset to disk so every training checkpoint
probes the identical images. Training-time loading then needs only torchvision and PIL.

Layout:
  <out>/imagenet_val/{bank,query}/<label>/<i>.jpg
"""

import argparse
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset
from PIL import Image


def _resize_center_crop(image: Image.Image, size: int) -> Image.Image:
    """Resize the short side to size, then centre-crop size x size, no aspect distortion."""
    width, height = image.size
    scale = size / min(width, height)
    resized = image.resize((round(width * scale), round(height * scale)), Image.Resampling.BICUBIC)
    left = (resized.width - size) // 2
    top = (resized.height - size) // 2
    return resized.crop((left, top, left + size, top + size))


def stage_imagenet(out: Path, per_class_bank: int, per_class_query: int, size: int) -> None:
    """Fixed balanced subset of ImageNet-1k val, split into a kNN bank and query set."""
    dataset = load_dataset("evanarlian/imagenet_1k_resized_256", split="val")
    counts: dict[int, int] = defaultdict(int)
    need = per_class_bank + per_class_query
    saved = 0
    for example in dataset:
        label = int(example["label"])
        seen = counts[label]
        if seen >= need:
            continue
        split = "bank" if seen < per_class_bank else "query"
        folder = out / "imagenet_val" / split / f"{label:04d}"
        folder.mkdir(parents=True, exist_ok=True)
        image = _resize_center_crop(example["image"].convert("RGB"), size)
        image.save(folder / f"{seen:03d}.jpg", quality=95)
        counts[label] = seen + 1
        saved += 1
    print(f"imagenet: {saved} images across {len(counts)} classes -> {out / 'imagenet_val'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("data/probes"))
    parser.add_argument("--imagenet-bank", type=int, default=40)
    parser.add_argument("--imagenet-query", type=int, default=10)
    parser.add_argument("--imagenet-size", type=int, default=256)
    args = parser.parse_args()

    stage_imagenet(args.out, args.imagenet_bank, args.imagenet_query, args.imagenet_size)


if __name__ == "__main__":
    main()
