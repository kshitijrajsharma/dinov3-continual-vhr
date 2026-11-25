"""Run continued DINO pretraining on the VHR tile shards and export the backbone."""

import argparse
from pathlib import Path

import pytorch_lightning as pl
import torch
from lightly.transforms import DINOTransform
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.strategies import DDPStrategy

from dinov3_vhr.data.corr_transform import CorrespondenceDINOTransform
from dinov3_vhr.data.tile_loader import build_tile_loader
from dinov3_vhr.dino_ssl import DinoV3ContinuedSSL
from dinov3_vhr.monitor.callback import ForgettingMonitor, SourceCoverageMonitor
from dinov3_vhr.monitor.drift import DriftMonitor


class BackboneCheckpoint(Callback):
    """Export the student backbone state_dict every `every_n_epochs` epochs, and every
    `every_n_steps` optimizer steps when set (for a dose curve inside a single long epoch).
    Saves only on the global-zero rank so concurrent DDP ranks do not race on the file."""

    def __init__(self, out_ckpt: Path, every_n_epochs: int, every_n_steps: int = 0) -> None:
        self.out_ckpt = out_ckpt
        self.every_n_epochs = every_n_epochs
        self.every_n_steps = every_n_steps
        self._last_saved_step = -1

    def _save(self, pl_module: pl.LightningModule, tag: str) -> None:
        assert isinstance(pl_module, DinoV3ContinuedSSL)
        path = self.out_ckpt.with_name(f"{self.out_ckpt.stem}_{tag}.pth")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(pl_module.export_backbone_state_dict(), path)
        print(f"saved backbone checkpoint to {path}")

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: object,
        batch: object,
        batch_idx: int,
    ) -> None:
        if not self.every_n_steps or not trainer.is_global_zero:
            return
        step = trainer.global_step
        if step > 0 and step % self.every_n_steps == 0 and step != self._last_saved_step:
            self._last_saved_step = step
            self._save(pl_module, f"step{step}")

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        epoch = trainer.current_epoch + 1
        if not trainer.is_global_zero or epoch % self.every_n_epochs != 0:
            return
        self._save(pl_module, f"epoch{epoch}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/tiles"))
    parser.add_argument("--model", default="vit_small_patch16_dinov3.lvd1689m")
    parser.add_argument("--epoch-tiles", type=int, default=1_018_163)
    parser.add_argument("--out-ckpt", type=Path, default=Path("out/dinov3_vits_oam.pth"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--layer-decay", type=float, default=0.9)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--ckpt-every", type=int, default=5)
    parser.add_argument("--gram-weight", type=float, default=2.0)
    parser.add_argument("--ibot-weight", type=float, default=1.0)
    parser.add_argument("--cls-anchor-weight", type=float, default=1.0)
    parser.add_argument("--corr-weight", type=float, default=0.0)
    parser.add_argument("--corr-layers", default="")
    parser.add_argument("--corr-ibot", action="store_true")
    parser.add_argument("--corr-byol", action="store_true")
    parser.add_argument("--freeze-backbone-steps", type=int, default=1000)
    parser.add_argument("--backbone-warmup-steps", type=int, default=500)
    parser.add_argument("--anchor-target", choices=["frozen", "refresh"], default="refresh")
    parser.add_argument("--anchor-refresh-steps", type=int, default=1500)
    parser.add_argument("--anchor-max-refreshes", type=int, default=2)
    parser.add_argument("--probe-root", type=Path, default=Path("data/probes"))
    parser.add_argument("--monitor-every", type=int, default=1000)
    parser.add_argument("--drift-monitor", action="store_true")
    parser.add_argument("--drift-layers", default="5,11,17,23")
    parser.add_argument("--drift-per-source", type=int, default=16)
    parser.add_argument("--grad-clip", type=float, default=3.0)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--accumulate", type=int, default=1)
    parser.add_argument("--grad-checkpointing", action="store_true")
    parser.add_argument("--explora", action="store_true")
    parser.add_argument("--lora-rank", type=int, default=64)
    parser.add_argument("--lora-alpha", type=float, default=64.0)
    parser.add_argument("--unfreeze-blocks", default="23")
    parser.add_argument("--full-coverage", action="store_true")
    parser.add_argument("--ckpt-every-steps", type=int, default=0)
    parser.add_argument("--run-name", default="dino")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    pl.seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision("high")
    global_crop_size = 256
    if args.corr_weight > 0:
        transform: object = CorrespondenceDINOTransform(
            global_crop_size=global_crop_size,
            global_crop_scale=(0.32, 1.0),
            local_crop_size=112,
            local_crop_scale=(0.05, 0.32),
            n_local_views=8,
        )
    else:
        transform = DINOTransform(
            global_crop_size=global_crop_size,
            global_crop_scale=(0.32, 1.0),
            local_crop_size=112,
            local_crop_scale=(0.05, 0.32),
            n_local_views=8,
            hf_prob=0.0,
        )
    world = max(1, args.devices)
    batches_per_epoch = args.epoch_tiles // (args.batch_size * world)
    optimizer_steps_per_epoch = max(1, batches_per_epoch // args.accumulate)
    loader, source_names = build_tile_loader(
        args.data_dir,
        transform=transform,
        batch_size=args.batch_size,
        steps_per_epoch=batches_per_epoch,
        num_workers=args.num_workers,
        seed=args.seed,
        full_coverage=args.full_coverage,
    )
    print(
        f"training on shards in {args.data_dir}, {batches_per_epoch} batches/epoch/rank, "
        f"{optimizer_steps_per_epoch} optimizer steps/epoch"
    )

    model = DinoV3ContinuedSSL(
        model_name=args.model,
        learning_rate=args.lr,
        layer_decay=args.layer_decay,
        max_epochs=args.epochs,
        steps_per_epoch=optimizer_steps_per_epoch,
        gram_weight=args.gram_weight,
        ibot_weight=args.ibot_weight,
        cls_anchor_weight=args.cls_anchor_weight,
        corr_weight=args.corr_weight,
        corr_layers=tuple(int(x) for x in args.corr_layers.split(",") if x.strip()),
        corr_ibot=args.corr_ibot,
        corr_byol=args.corr_byol,
        global_crop_size=global_crop_size,
        freeze_backbone_steps=args.freeze_backbone_steps,
        backbone_warmup_steps=args.backbone_warmup_steps,
        anchor_target=args.anchor_target,
        anchor_refresh_steps=args.anchor_refresh_steps,
        anchor_max_refreshes=args.anchor_max_refreshes,
        grad_checkpointing=args.grad_checkpointing,
        explora=args.explora,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        unfreeze_blocks=tuple(int(b) for b in args.unfreeze_blocks.split(",")),
    )
    callbacks: list[Callback] = [
        BackboneCheckpoint(args.out_ckpt, args.ckpt_every, every_n_steps=args.ckpt_every_steps),
        SourceCoverageMonitor(source_names, every_n_steps=args.monitor_every),
    ]
    if args.probe_root.exists():
        callbacks.append(ForgettingMonitor(args.probe_root, every_n_steps=args.monitor_every))
    else:
        print(f"no probe data at {args.probe_root}; forgetting monitor disabled")
    if args.drift_monitor:
        callbacks.append(
            DriftMonitor(
                args.data_dir,
                args.model,
                layers=[int(layer) for layer in args.drift_layers.split(",")],
                per_source=args.drift_per_source,
                every_n_steps=args.monitor_every,
                seed=args.seed,
            )
        )
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu",
        devices=args.devices,
        # The two-stage freeze toggles requires_grad off on the backbone during warmup, so
        # those params yield no gradient and DDP's reducer needs unused-parameter detection.
        strategy=DDPStrategy(find_unused_parameters=True) if args.devices > 1 else "auto",
        use_distributed_sampler=False,
        accumulate_grad_batches=args.accumulate,
        precision="bf16-mixed",
        gradient_clip_val=args.grad_clip,
        log_every_n_steps=10,
        logger=CSVLogger("out", name=args.run_name),
        enable_checkpointing=False,
        callbacks=callbacks,
    )
    trainer.fit(model, loader)

    args.out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.export_backbone_state_dict(), args.out_ckpt)
    print(f"saved continued backbone to {args.out_ckpt}")


if __name__ == "__main__":
    main()
