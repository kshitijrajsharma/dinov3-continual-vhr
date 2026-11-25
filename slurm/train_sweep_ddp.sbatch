#!/usr/bin/env bash
# DDP launcher for the ViT-L continued-pretraining sweep (configs/vitl_control.yaml).
# sweep.py submit passes the full "python -m dinov3_vhr.ssl.train --devices 2 ..." argv as
# positional args; one srun task per GPU, and Lightning's SLURMEnvironment assigns the ranks.
# Resource flags (partition, gres, cpus-per-task, mem, time, job-name, output) come from
# sweep.py on the sbatch command line; only the per-node task count is set here.
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
set -uo pipefail
cd "$HOME/dinov3-vhr"
mkdir -p slurm/logs
export PATH="$HOME/.local/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
srun uv run --no-sync "$@"
echo "==== ${SLURM_JOB_NAME:-train_sweep_ddp} DONE $(date) ===="
