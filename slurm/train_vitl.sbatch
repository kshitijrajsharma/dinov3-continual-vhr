#!/usr/bin/env bash
# Continual DINOv3 ViT-L/16 pretraining on the VHR tiles, 2x A100 80GB, DDP.
# Usage:  sbatch slurm/train_vitl.sbatch <arm> <seed> [layer_decay]
#   arm = gram0   plain continued, no anchors            (control)
#         gram2f  patch anchor to frozen released model  (old method)
#         gram2r  patch anchor to refreshed self-snapshot
#         full    patch + CLS anchor, refreshed snapshot (proposed method)
#   layer_decay = 1.0 uniform LR (reported model, default); 0.9 is the layer-decay ablation
# One srun task per GPU so SLURM_PROCID/SLURM_NTASKS drive per-rank shard partitioning.

#SBATCH --job-name=dinov3-vhr
#SBATCH --partition=longrun
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00
#SBATCH --output=slurm/logs/%x-%j.out

set -euo pipefail

ARM="${1:-full}"
SEED="${2:-42}"
LD="${3:-1.0}"
case "$ARM" in
  gram0)  GRAM=0.0; CLS=0.0; TARGET=frozen  ;;
  gram2f) GRAM=2.0; CLS=0.0; TARGET=frozen  ;;
  gram2r) GRAM=2.0; CLS=0.0; TARGET=refresh ;;
  full)   GRAM=2.0; CLS=1.0; TARGET=refresh ;;
  winner) GRAM=2.0; CLS=1.0; TARGET=frozen  ;;
  *) echo "usage: sbatch train_vitl.sbatch [gram0|gram2f|gram2r|full|winner] <seed> [layer_decay]"; exit 1 ;;
esac

REPO="${REPO:-$HOME/dinov3-vhr}"
DATA="${DATA:-$REPO/data/tiles}"
PROBES="${PROBES:-$REPO/data/probes}"
RUN="vitl_${ARM}_s${SEED}"

cd "$REPO"
mkdir -p slurm/logs
export PYTHONIOENCODING=utf-8
export PATH="$HOME/.local/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# effective batch 32 x 8 accum x 2 gpu = 512; ~2000 optimizer steps per epoch over the
# 1.0M tiles; 3 epochs checkpoints the dose response at 2k / 4k / 6k steps.
srun uv run python -m dinov3_vhr.ssl.train \
  --model vit_large_patch16_dinov3.lvd1689m \
  --data-dir "$DATA" \
  --probe-root "$PROBES" \
  --devices 2 \
  --batch-size 32 \
  --accumulate 8 \
  --lr 5e-5 \
  --layer-decay "$LD" \
  --epochs 3 \
  --epoch-tiles 1018163 \
  --num-workers 8 \
  --monitor-every 250 \
  --grad-clip 3.0 \
  --gram-weight "$GRAM" \
  --cls-anchor-weight "$CLS" \
  --anchor-target "$TARGET" \
  --anchor-refresh-steps 2000 \
  --anchor-max-refreshes 3 \
  --freeze-backbone-steps 1000 \
  --backbone-warmup-steps 500 \
  --seed "$SEED" \
  --run-name "$RUN" \
  --out-ckpt "out/${RUN}.pth" \
  --ckpt-every 1
