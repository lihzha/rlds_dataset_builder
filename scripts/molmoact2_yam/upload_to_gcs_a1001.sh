#!/usr/bin/env bash
set -euo pipefail

NFS_ROOT="${NFS_ROOT:-/lustre/fsw/portfolios/nvr/users/lzha}"
ENV_DIR="${ENV_DIR:-$NFS_ROOT/envs/rlds_molmoact2_yam}"
TFDS_DATA_DIR="${TFDS_DATA_DIR:-$NFS_ROOT/tensorflow_datasets}"
DATASET_DIR="${DATASET_DIR:-$TFDS_DATA_DIR/molmoact2_yam_dataset}"
GCS_DEST="${GCS_DEST:-gs://pi0-cot/OXE/molmoact2_yam_dataset}"

echo "DATASET_DIR=$DATASET_DIR"
echo "GCS_DEST=$GCS_DEST"
"$ENV_DIR/bin/gsutil" ls gs://pi0-cot/OXE >/dev/null
"$ENV_DIR/bin/gsutil" -m rsync -r "$DATASET_DIR" "$GCS_DEST"
"$ENV_DIR/bin/gsutil" ls "$GCS_DEST"
