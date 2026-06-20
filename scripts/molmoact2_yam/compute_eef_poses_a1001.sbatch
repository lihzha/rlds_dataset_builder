#!/usr/bin/env bash
#SBATCH --account=nvr_lpr_rvp
#SBATCH --partition=cpu_long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --job-name=molmo2_yam_eef
#SBATCH --output=/lustre/fsw/portfolios/nvr/users/lzha/slurm_logs/molmoact2_yam/%x_%j.out
#SBATCH --error=/lustre/fsw/portfolios/nvr/users/lzha/slurm_logs/molmoact2_yam/%x_%j.err

set -euo pipefail

NFS_ROOT="${NFS_ROOT:-/lustre/fsw/portfolios/nvr/users/lzha}"
CODE_DIR="${CODE_DIR:-$NFS_ROOT/src/rlds_dataset_builder-molmoact2-yam-eef}"
ENV_DIR="${ENV_DIR:-$NFS_ROOT/envs/rlds_molmoact2_yam}"
RAW_DIR="${RAW_DIR:-$NFS_ROOT/datasets/raw/molmoact2_yam_numeric}"
RAW_DIR_FOR_KEYS="${RAW_DIR_FOR_KEYS:-$NFS_ROOT/datasets/raw/molmoact2_yam}"
OUTPUT_DIR="${OUTPUT_DIR:-$NFS_ROOT/results/molmoact2_yam_eef}"
OUTPUT_NPZ="${OUTPUT_NPZ:-$OUTPUT_DIR/eef_poses.npz}"
GCS_DEST="${GCS_DEST:-gs://pi0-cot/OXE/molmoact2_yam_dataset/1.0.0/eef_poses.npz}"
TRUNCATION_LOG="${TRUNCATION_LOG:-$NFS_ROOT/slurm_logs/molmoact2_yam/molmo2_yam_tfds_29061545.out}"
MIN_FREE_TIB="${MIN_FREE_TIB:-2}"
export NFS_ROOT OUTPUT_NPZ MIN_FREE_TIB

mkdir -p "$NFS_ROOT/slurm_logs/molmoact2_yam" "$RAW_DIR" "$OUTPUT_DIR"
cd "$CODE_DIR"

echo "RUN_START=$(date --iso-8601=seconds)"
echo "HOST=$(hostname)"
echo "CWD=$(pwd)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-none}"
echo "RAW_DIR=$RAW_DIR"
echo "RAW_DIR_FOR_KEYS=$RAW_DIR_FOR_KEYS"
echo "OUTPUT_NPZ=$OUTPUT_NPZ"
echo "GCS_DEST=$GCS_DEST"
echo "TRUNCATION_LOG=$TRUNCATION_LOG"
echo "MIN_FREE_TIB=$MIN_FREE_TIB"
echo "HEAD=$(git rev-parse HEAD)"
git status --short --branch

"$ENV_DIR/bin/python" - <<'PY'
import os
import shutil

path = os.environ.get("NFS_ROOT", "/lustre/fsw/portfolios/nvr/users/lzha")
free_tib = shutil.disk_usage(path).free / (1024 ** 4)
min_free_tib = float(os.environ.get("MIN_FREE_TIB", "2"))
print(f"free_tib_before={free_tib:.3f}")
if free_tib < min_free_tib:
    raise SystemExit(f"free space {free_tib:.3f} TiB is below required {min_free_tib:.3f} TiB")
PY

export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_SNAPSHOT_MAX_WORKERS="${HF_SNAPSHOT_MAX_WORKERS:-1}"
export HF_DOWNLOAD_RETRIES="${HF_DOWNLOAD_RETRIES:-8}"
export HF_DOWNLOAD_RETRY_SLEEP="${HF_DOWNLOAD_RETRY_SLEEP:-30}"

"$ENV_DIR/bin/python" scripts/molmoact2_yam/download_hf_snapshot.py \
  --raw-dir "$RAW_DIR" \
  --numeric-only

"$ENV_DIR/bin/python" scripts/molmoact2_yam/compute_eef_poses_from_parquet.py \
  --raw-dir "$RAW_DIR" \
  --raw-dir-for-keys "$RAW_DIR_FOR_KEYS" \
  --truncation-log "$TRUNCATION_LOG" \
  --output "$OUTPUT_NPZ"

"$ENV_DIR/bin/gsutil" cp "$OUTPUT_NPZ" "$GCS_DEST"
"$ENV_DIR/bin/gsutil" ls -l "$GCS_DEST"

"$ENV_DIR/bin/python" - <<'PY'
import json
import os
from pathlib import Path
import numpy as np

path = Path(os.environ["OUTPUT_NPZ"])
with np.load(path, allow_pickle=False) as data:
    summary = {
        "output_npz": str(path),
        "bytes": path.stat().st_size,
        "episodes": int(len(data["episode_ids"])),
        "steps": int(len(data["left_eef_pose"])),
        "left_eef_pose_shape": list(data["left_eef_pose"].shape),
        "right_eef_pose_shape": list(data["right_eef_pose"].shape),
        "first_episode_id": str(data["episode_ids"][0]),
    }
summary_path = path.with_suffix(".summary.json")
summary_path.write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY

"$ENV_DIR/bin/python" - <<'PY'
import os
import shutil

path = os.environ.get("NFS_ROOT", "/lustre/fsw/portfolios/nvr/users/lzha")
free_tib = shutil.disk_usage(path).free / (1024 ** 4)
print(f"free_tib_after={free_tib:.3f}")
PY

echo "RUN_END=$(date --iso-8601=seconds)"
