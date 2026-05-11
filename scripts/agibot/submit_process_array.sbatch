#!/bin/bash
#SBATCH --job-name=agibot_proc
#SBATCH --output=logs/agibot_proc_%A_%a.out
#SBATCH --error=logs/agibot_proc_%A_%a.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=48:00:00
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
#   AGIBOT_BASE                    shared scratch dir (proprio_stats/, task_info/, tasks.json live here)
#   AGIBOT_WORK_BASE               per-job scratch parent (e.g. /n/fs/robot-data/agibot_work)
#   AGIBOT_GCS_STAGING             gs://bucket/agibot_staging
#
# GCS auth — any one of these is sufficient:
#   - GOOGLE_APPLICATION_CREDENTIALS pointing to a service-account JSON, OR
#   - Application Default Credentials at ~/.config/gcloud/application_default_credentials.json
#     (from `gcloud auth application-default login` on a login node — persists across jobs).
#   If `gsutil cp gs://your-bucket/...` works on a login node, ADC is already set up.

set -euo pipefail

# Fix SSL certificate path for libcurl/TensorFlow GCS access
export CURL_CA_BUNDLE=/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem
export SSL_CERT_FILE=/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem
export REQUESTS_CA_BUNDLE=/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem

export AGIBOT_BASE=/n/fs/robot-data/data/agibot
export AGIBOT_WORK_BASE=/n/fs/robot-data/agibot_work
export AGIBOT_GCS_STAGING=gs://pi0-cot/agibot_staging

: "${AGIBOT_BASE:?Set AGIBOT_BASE}"
: "${AGIBOT_WORK_BASE:?Set AGIBOT_WORK_BASE (e.g. /n/fs/robot-data/agibot_work)}"
: "${AGIBOT_GCS_STAGING:?Set AGIBOT_GCS_STAGING (gs://...)}"


mkdir -p logs

source /n/fs/robot-data/miniconda3/etc/profile.d/conda.sh
conda activate rlds_zarr3

# Add project root to PYTHONPATH so agibot_pipeline can be imported
export PYTHONPATH="/n/fs/robot-data/rlds_multithread:${PYTHONPATH:-}"

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

# Per-job scratch under the cluster-permitted path (e.g. /n/fs/robot-data/...).
# Each array element gets a uniquely-named subdir so concurrent jobs are isolated.
WORK_DIR="$AGIBOT_WORK_BASE/agibot_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$WORK_DIR"
trap 'rm -rf "$WORK_DIR"' EXIT

echo "=================================================="
echo "Array $SLURM_ARRAY_JOB_ID idx $SLURM_ARRAY_TASK_ID -> task_id=$TASK_ID"
echo "Node: $(hostname)   Work dir: $WORK_DIR"
echo "Start: $(date)"
echo "=================================================="

python /n/fs/robot-data/rlds_multithread/agibot_pipeline/process_task.py \
    --task-id "$TASK_ID" \
    --base "$AGIBOT_BASE" \
    --work-dir "$WORK_DIR" \
    --gcs-staging "$AGIBOT_GCS_STAGING"

echo "End: $(date)"
