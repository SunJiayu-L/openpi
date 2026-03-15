#!/bin/bash
# Download pi05_base checkpoint from GCS to openpi cache
# Run on login node with proxy

export http_proxy=http://localhost:7897
export https_proxy=http://localhost:7897

cd /storage/yukaichengLab/lishiwen/jiayusun/openpi

# Remove incomplete cache
#rm -rf ~/.cache/openpi/openpi-assets/checkpoints/pi0_base

uv run python -c "
from openpi.shared.download import maybe_download
maybe_download('gs://openpi-assets/checkpoints/pi0_base/params', force_download=True)
print('Download complete!')
"
