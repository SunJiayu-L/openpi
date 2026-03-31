#!/bin/bash
#SBATCH -p yukaichenglab
#SBATCH --gres=gpu:0
#SBATCH --exclude=gnho009
#SBATCH -J dump_libero_val_balanced
#SBATCH -o trainlogs/dump_libero_val_balanced_%j.log
#SBATCH -e trainlogs/dump_libero_val_balanced_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -euo pipefail
echo "Job ID: $SLURM_JOB_ID  Node: $SLURMD_NODENAME  Start: $(date)"

cd /storage/yukaichengLab/lishiwen/jiayusun/openpi
mkdir -p trainlogs

export HF_HOME=~/.cache/huggingface
export HF_HUB_OFFLINE=1
export JAX_PLATFORMS=cpu
export CUDA_VISIBLE_DEVICES=""

# Dump 13 batches per suite (4 * 13 = 52 total, then trim to 50)
echo "Dumping libero_10 ..."
uv run scripts/dump_data.py \
    --dataset pi05_libero_10 \
    --output libero_val_10.pkl \
    --num_batches 13 \
    --batch_size 32

echo "Dumping libero_goal ..."
uv run scripts/dump_data.py \
    --dataset pi05_libero_goal \
    --output libero_val_goal.pkl \
    --num_batches 13 \
    --batch_size 32

echo "Dumping libero_object ..."
uv run scripts/dump_data.py \
    --dataset pi05_libero_object \
    --output libero_val_object.pkl \
    --num_batches 13 \
    --batch_size 32

echo "Dumping libero_spatial ..."
uv run scripts/dump_data.py \
    --dataset pi05_libero_spatial \
    --output libero_val_spatial.pkl \
    --num_batches 13 \
    --batch_size 32

echo "Merging into balanced val pkl ..."
uv run python -c "
import pickle, json

suite_files = [
    ('libero_10',      'libero_val_10.pkl'),
    ('libero_goal',    'libero_val_goal.pkl'),
    ('libero_object',  'libero_val_object.pkl'),
    ('libero_spatial', 'libero_val_spatial.pkl'),
]

# Load per-suite batches
per_suite = {}
for suite, path in suite_files:
    with open(path, 'rb') as f:
        per_suite[suite] = pickle.load(f)
    print(f'  {suite}: {len(per_suite[suite])} batches')

# Interleave: suite0[0], suite1[0], suite2[0], suite3[0], suite0[1], ...
# Use the minimum batch count so all suites contribute equally
min_batches = min(len(v) for v in per_suite.values())
suites = [s for s, _ in suite_files]

batches = []
suite_labels = []
for i in range(min_batches):
    for suite in suites:
        batches.append(per_suite[suite][i])
        suite_labels.append(suite)

with open('libero_val_balanced.pkl', 'wb') as f:
    pickle.dump(batches, f)
with open('libero_val_balanced_meta.json', 'w') as f:
    json.dump({'suite_labels': suite_labels}, f)

n = len(batches)
print(f'Saved libero_val_balanced.pkl: {n} batches ({n*32} samples), {min_batches} per suite')
print(f'Saved libero_val_balanced_meta.json')
"

echo "Done at $(date)"
