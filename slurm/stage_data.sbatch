#!/usr/bin/env bash
# One-time data staging: download tile shards from HF + build the probe sets.
# Runs as a SLURM job (survives login-session logout) on a CPU slot, no GPU.
#   sbatch slurm/stage_data.sbatch

#SBATCH --job-name=dinov3-stage
#SBATCH --partition=shortrun
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:0
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=slurm/logs/stage-%j.out

set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd "$HOME/dinov3-vhr"
mkdir -p slurm/logs

echo "[$(date)] downloading tiles from HF"
uv run hf download kshitijrajsharma/dinov3-vhr-pretrain \
  --repo-type dataset --local-dir data/tiles

echo "[$(date)] staging probe sets"
uv run python -m dinov3_vhr.monitor.stage_probes --out data/probes

echo "[$(date)] STAGE_COMPLETE"
