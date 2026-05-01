#!/bin/bash
#SBATCH --job-name=agibot_proc
#SBATCH --output=logs/agibot_proc_%A_%a.out
#SBATCH --error=logs/agibot_proc_%A_%a.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-216%32
#
# Job array: one element per task in <BASE>/tasks.json.
# Indices 0..216 cover the 217 tasks. The %32 throttle caps concurrent runs;
# tune to your scratch / HF bandwidth budget.
#
# Resubmit the same sbatch to resume — process_task.py skips tars whose
# shard+manifest already exist in GCS, and skips tasks with the _DONE sentinel.
#
# Submit with:
#   sbatch scripts/agibot/submit_process_array.sbatch
# Or to retry a specific subset:
#   sbatch --array=12,57,103 scripts/agibot/submit_process_array.sbatch
#
# Required env:
#   AGIBOT_BASE                    shared scratch dir
#   AGIBOT_GCS_STAGING             gs://bucket/agibot_staging
#   GOOGLE_APPLICATION_CREDENTIALS path to service-account JSON

set -euo pipefail

: "${AGIBOT_BASE:?Set AGIBOT_BASE}"
: "${AGIBOT_GCS_STAGING:?Set AGIBOT_GCS_STAGING (gs://...)}"
: "${GOOGLE_APPLICATION_CREDENTIALS:?Set GOOGLE_APPLICATION_CREDENTIALS}"

mkdir -p logs

if [ -n "${CONDA_PREFIX:-}" ]; then
    echo "Using already-active conda env: $CONDA_PREFIX"
else
    source "${CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV:-rlds}"
fi

cd "$(dirname "$0")/../.."
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

# Map array index -> task_id from tasks.json.
TASK_ID=$(python -c "
import json, sys
with open('$AGIBOT_BASE/tasks.json') as f:
    tasks = json.load(f)
idx = int('$SLURM_ARRAY_TASK_ID')
if idx >= len(tasks):
    sys.exit(f'array idx {idx} >= {len(tasks)}')
print(tasks[idx]['task_id'])
")

# Per-job local scratch. SLURM_TMPDIR is node-local on most clusters; fall back
# to /tmp/<jobid> if it's unset (so we still get isolation between jobs).
WORK_BASE="${SLURM_TMPDIR:-/tmp}"
WORK_DIR="$WORK_BASE/agibot_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$WORK_DIR"
trap 'rm -rf "$WORK_DIR"' EXIT

echo "=================================================="
echo "Array $SLURM_ARRAY_JOB_ID idx $SLURM_ARRAY_TASK_ID -> task_id=$TASK_ID"
echo "Node: $(hostname)   Work dir: $WORK_DIR"
echo "Start: $(date)"
echo "=================================================="

python -m agibot_pipeline.process_task \
    --task-id "$TASK_ID" \
    --base "$AGIBOT_BASE" \
    --work-dir "$WORK_DIR" \
    --gcs-staging "$AGIBOT_GCS_STAGING"

echo "End: $(date)"
