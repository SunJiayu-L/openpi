#!/bin/bash
# 4-suite balanced dump at ~10% total volume (target based on 273,465 total frames).
# batch_size=32, total_samples = 4 * NUM_BATCHES * 32
# choose NUM_BATCHES=214 per suite => 27,392 samples (~10.02%)

#SBATCH -p yukaichenglab
#SBATCH --gres=gpu:0
#SBATCH --exclude=gnho009
#SBATCH -J jiayu_dump_val_10pct
#SBATCH -o trainlogs/dump_val_10pct_%j.log
#SBATCH -e trainlogs/dump_val_10pct_%j.err
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G

set -euo pipefail
echo "Job ID: $SLURM_JOB_ID  Node: $SLURMD_NODENAME  Start: $(date)"

cd /storage/yukaichengLab/lishiwen/jiayusun/openpi
mkdir -p trainlogs pkl

export HF_HOME=~/.cache/huggingface
export HF_HUB_OFFLINE=1
export JAX_PLATFORMS=cpu
export CUDA_VISIBLE_DEVICES=""

uv run python -c "
import pickle, sys
sys.path.insert(0, 'src')
import dataclasses
from openpi.training import config as _config, data_loader as _data_loader

SUITES = ['pi05_libero_10', 'pi05_libero_goal', 'pi05_libero_object', 'pi05_libero_spatial']
NUM_BATCHES = 214
BATCH_SIZE = 32

loaders = []
for name in SUITES:
    cfg = _config.get_config(name)
    cfg = dataclasses.replace(cfg, batch_size=BATCH_SIZE)
    loaders.append(iter(_data_loader.create_data_loader(cfg, sharding=None, shuffle=True)))
    print(f'  loader ready: {name}')

batches = []
suite_labels = []
suite_names = ['libero_10', 'libero_goal', 'libero_object', 'libero_spatial']

for i in range(NUM_BATCHES):
    for loader, suite in zip(loaders, suite_names):
        batches.append(next(loader))
        suite_labels.append(suite)
    if (i + 1) % 25 == 0:
        print(f'  {i+1}/{NUM_BATCHES} rounds done ({len(batches)} batches total)')

out_pkl = 'pkl/libero_val_balanced_10pct.pkl'
out_meta = 'pkl/libero_val_balanced_10pct_meta.json'
with open(out_pkl, 'wb') as f:
    pickle.dump(batches, f)

import json
with open(out_meta, 'w') as f:
    json.dump(
        {
            'suite_labels': suite_labels,
            'batches_per_suite': NUM_BATCHES,
            'total_batches': len(batches),
            'batch_size': BATCH_SIZE,
            'total_samples': len(batches) * BATCH_SIZE,
            'target_fraction': 0.1,
            'reference_total_samples': 273465,
            'actual_fraction': (len(batches) * BATCH_SIZE) / 273465,
        },
        f,
        indent=2
    )

print(f'Done: {len(batches)} batches, {len(batches) * BATCH_SIZE} samples')
print(f'Output pkl: {out_pkl}')
print(f'Output meta: {out_meta}')
"

echo "Finished at $(date)"

