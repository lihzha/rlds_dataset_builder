#!/bin/bash
#SBATCH --job-name=agibot_merge
#SBATCH --output=logs/agibot_merge_%j.out
#SBATCH --error=logs/agibot_merge_%j.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#
# Merges per-tar staging shards into final TFDS dataset.
# Idempotent: skips destinations that already exist, safe to re-run.

set -euo pipefail

export CURL_CA_BUNDLE=/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem
export SSL_CERT_FILE=/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem
export REQUESTS_CA_BUNDLE=/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem
export PYTHONPATH=/n/fs/robot-data/rlds_multithread:${PYTHONPATH:-}

mkdir -p logs

source /n/fs/robot-data/miniconda3/etc/profile.d/conda.sh
conda activate rlds_zarr3

cd /n/fs/robot-data/rlds_multithread

python agibot_pipeline/merge_shards.py \
    --gcs-staging gs://pi0-cot/agibot_staging \
    --gcs-final gs://pi0-cot/agibot_final \
    --copy
