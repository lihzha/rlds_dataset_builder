#!/bin/bash
#SBATCH --job-name=agibot_bootstrap
#SBATCH --output=logs/agibot_bootstrap_%j.out
#SBATCH --error=logs/agibot_bootstrap_%j.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#
# One-time download of task_info + proprio_stats + manifest generation.
# Outputs <BASE>/tasks.json which the array job consumes.
#
# Submit with:
#   sbatch scripts/agibot/submit_bootstrap.sbatch
#
# Required env (set in your shell or override here):
#   AGIBOT_BASE   - shared scratch dir (e.g. /scratch/agibot)
#   HF_TOKEN      - optional, for higher HF rate limits

set -euo pipefail

: "${AGIBOT_BASE:?Set AGIBOT_BASE to the shared scratch dir}"

mkdir -p logs

# Activate the conda env that has huggingface_hub + tensorflow_datasets.
# Adjust this block for your cluster.
if [ -n "${CONDA_PREFIX:-}" ]; then
    echo "Using already-active conda env: $CONDA_PREFIX"
else
    source "${CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV:-rlds}"
fi

cd "$(dirname "$0")/../.."   # repo root
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

python -m agibot_pipeline.bootstrap_hf_data \
    --base "$AGIBOT_BASE" \
    --workers 8

echo "Bootstrap complete. Manifest: $AGIBOT_BASE/tasks.json"
