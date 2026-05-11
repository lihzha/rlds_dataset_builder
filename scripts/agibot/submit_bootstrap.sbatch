#!/bin/bash
#SBATCH --job-name=agibot_bootstrap
#SBATCH --output=logs/agibot_bootstrap_%j.out
#SBATCH --error=logs/agibot_bootstrap_%j.err
#SBATCH -N 1
#SBATCH --gres=gpu:0
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --partition=all              # Or specify GPU partition if needed

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

export AGIBOT_BASE=/n/fs/robot-data/data/agibot

: "${AGIBOT_BASE:?Set AGIBOT_BASE to the shared scratch dir}"

mkdir -p logs

cd "$(dirname "$0")/../.."   # repo root
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

python /n/fs/robot-data/rlds_multithread/agibot_pipeline/bootstrap_hf_data.py \
    --base "$AGIBOT_BASE" \
    --workers 16

echo "Bootstrap complete. Manifest: $AGIBOT_BASE/tasks.json"
