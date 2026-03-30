#!/bin/bash
# Sync offline wandb run for merge job 32802.

#SBATCH -p yukaichenglab
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH -J wandb_sync_32802
#SBATCH -o /storage/yukaichengLab/lishiwen/jiayusun/openpi/trainlogs/wandb_sync_32802_%j.log
#SBATCH -e /storage/yukaichengLab/lishiwen/jiayusun/openpi/trainlogs/wandb_sync_32802_%j.err

set -euo pipefail

cd /storage/yukaichengLab/lishiwen/jiayusun/openpi
uv run wandb sync /storage/yukaichengLab/lishiwen/jiayusun/openpi/wandb/wandb/offline-run-20260330_112557-cdx0xqkp
