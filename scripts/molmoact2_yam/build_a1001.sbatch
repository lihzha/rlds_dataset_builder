#!/usr/bin/env bash
#SBATCH --account=nvr_lpr_rvp
#SBATCH --partition=cpu_long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=7-00:00:00
#SBATCH --job-name=molmo2_yam_tfds
#SBATCH --output=/lustre/fsw/portfolios/nvr/users/lzha/slurm_logs/molmoact2_yam/%x_%j.out
#SBATCH --error=/lustre/fsw/portfolios/nvr/users/lzha/slurm_logs/molmoact2_yam/%x_%j.err

set -euo pipefail

NFS_ROOT="${NFS_ROOT:-/lustre/fsw/portfolios/nvr/users/lzha}"
CODE_DIR="${CODE_DIR:-$NFS_ROOT/src/rlds_dataset_builder-molmoact2-yam}"
ENV_DIR="${ENV_DIR:-$NFS_ROOT/envs/rlds_molmoact2_yam}"
RAW_DIR="${RAW_DIR:-$NFS_ROOT/datasets/raw/molmoact2_yam}"
TFDS_DATA_DIR="${TFDS_DATA_DIR:-$NFS_ROOT/tensorflow_datasets}"
MODE="${MODE:-smoke}"
MIN_FREE_TIB="${MIN_FREE_TIB:-2}"

mkdir -p "$NFS_ROOT/slurm_logs/molmoact2_yam" "$RAW_DIR" "$TFDS_DATA_DIR"
cd "$CODE_DIR"

echo "RUN_START=$(date --iso-8601=seconds)"
echo "HOST=$(hostname)"
echo "CWD=$(pwd)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-none}"
echo "MODE=$MODE"
echo "RAW_DIR=$RAW_DIR"
echo "TFDS_DATA_DIR=$TFDS_DATA_DIR"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
echo "HF_HUB_DISABLE_XET=$HF_HUB_DISABLE_XET"
echo "HEAD=$(git rev-parse HEAD)"
git status --short --branch

"$ENV_DIR/bin/python" - <<'PY'
import imageio_ffmpeg
print("ffmpeg", imageio_ffmpeg.get_ffmpeg_exe())
PY
"$ENV_DIR/bin/python" - <<'PY'
import shutil
path = "/lustre/fsw/portfolios/nvr/users/lzha"
usage = shutil.disk_usage(path)
free_tib = usage.free / (1024 ** 4)
print(f"free_tib={free_tib:.3f}")
PY

if [ "$MODE" = "full" ]; then
  "$ENV_DIR/bin/python" scripts/molmoact2_yam/download_hf_snapshot.py --raw-dir "$RAW_DIR" --full
else
  "$ENV_DIR/bin/python" scripts/molmoact2_yam/download_hf_snapshot.py \
    --raw-dir "$RAW_DIR" \
    --max-files "${MOLMOACT2_YAM_MAX_FILES:-1}" \
    --chunks "${MOLMOACT2_YAM_CHUNKS:-0}"
fi

export MOLMOACT2_YAM_RAW_DIR="$RAW_DIR"
export MOLMOACT2_YAM_N_WORKERS="${MOLMOACT2_YAM_N_WORKERS:-4}"
export MOLMOACT2_YAM_MAX_PATHS_IN_MEMORY="${MOLMOACT2_YAM_MAX_PATHS_IN_MEMORY:-8}"
if [ "$MODE" != "full" ]; then
  export MOLMOACT2_YAM_MAX_FILES="${MOLMOACT2_YAM_MAX_FILES:-1}"
  export MOLMOACT2_YAM_MAX_EPISODES="${MOLMOACT2_YAM_MAX_EPISODES:-4}"
fi

"$ENV_DIR/bin/python" -m pip install -e .
cd molmoact2_yam_dataset
CUDA_VISIBLE_DEVICES= "$ENV_DIR/bin/tfds" build --overwrite --data_dir="$TFDS_DATA_DIR"

echo "TFDS_OUTPUT=$TFDS_DATA_DIR/molmoact2_yam_dataset/1.0.0"
find "$TFDS_DATA_DIR/molmoact2_yam_dataset/1.0.0" -maxdepth 1 -type f -printf "%f %s bytes\n" | sort

"$ENV_DIR/bin/python" - <<'PY'
import json
import os
from pathlib import Path
import tensorflow_datasets as tfds

root = Path(os.environ["TFDS_DATA_DIR"]) / "molmoact2_yam_dataset" / "1.0.0"
info = json.loads((root / "dataset_info.json").read_text())
print("dataset_info_splits", info.get("splits"))
builder = tfds.builder("molmoact2_yam_dataset", data_dir=os.environ["TFDS_DATA_DIR"])
ds = builder.as_dataset(split="train[:1]", decoders={"steps": {"observation": {"top_image": tfds.decode.SkipDecoding(), "left_image": tfds.decode.SkipDecoding(), "right_image": tfds.decode.SkipDecoding()}}})
for ex in tfds.as_numpy(ds):
    steps = list(ex["steps"])
    print("sample_steps", len(steps))
    if steps:
        step = steps[0]
        print("action_shape", step["action"].shape, "state_shape", step["observation"]["state"].shape)
        print("language", step["language_instruction"][:200])
        print("image_bytes", len(step["observation"]["top_image"]), len(step["observation"]["left_image"]), len(step["observation"]["right_image"]))
PY

echo "RUN_END=$(date --iso-8601=seconds)"
