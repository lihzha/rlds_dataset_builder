#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-lzha@10.49.167.111}"
REMOTE_DATASET_DIR="${REMOTE_DATASET_DIR:-/lustre/fsw/portfolios/nvr/users/lzha/tensorflow_datasets/molmoact2_yam_dataset}"
GCS_DEST="${GCS_DEST:-gs://pi0-cot/OXE/molmoact2_yam_dataset}"
GSUTIL="${GSUTIL:-gsutil}"
SSH_BIN="${SSH_BIN:-ssh}"
IDENTITY_FILE="${IDENTITY_FILE:-$HOME/.ssh/id_ed25519}"

GCS_DEST="${GCS_DEST%/}"
REMOTE_DATASET_DIR="${REMOTE_DATASET_DIR%/}"

SSH_OPTS=(
  -F none
  -o BatchMode=yes
  -o ConnectTimeout=8
  -o ConnectionAttempts=1
  -o ForwardAgent=no
  -o IdentityAgent=none
  -o IdentitiesOnly=yes
  -o ServerAliveInterval=20
  -o ServerAliveCountMax=3
  -i "$IDENTITY_FILE"
)

echo "REMOTE_HOST=$REMOTE_HOST"
echo "REMOTE_DATASET_DIR=$REMOTE_DATASET_DIR"
echo "GCS_DEST=$GCS_DEST"

mapfile -t files < <(
  "$SSH_BIN" "${SSH_OPTS[@]}" "$REMOTE_HOST" \
    "cd '$REMOTE_DATASET_DIR' && find . -type f -printf '%P\n' | sort"
)

if [ "${#files[@]}" -eq 0 ]; then
  echo "No files found under $REMOTE_DATASET_DIR" >&2
  exit 2
fi

for rel in "${files[@]}"; do
  src="$REMOTE_DATASET_DIR/$rel"
  dst="$GCS_DEST/$rel"
  echo "uploading $rel -> $dst"
  "$SSH_BIN" "${SSH_OPTS[@]}" "$REMOTE_HOST" "cat '$src'" | "$GSUTIL" cp - "$dst"
done

echo "uploaded ${#files[@]} files to $GCS_DEST"
