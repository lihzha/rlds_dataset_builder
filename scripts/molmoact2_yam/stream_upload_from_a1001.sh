#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-lzha@10.49.167.111}"
REMOTE_DATASET_DIR="${REMOTE_DATASET_DIR:-/lustre/fsw/portfolios/nvr/users/lzha/tensorflow_datasets/molmoact2_yam_dataset}"
GCS_DEST="${GCS_DEST:-gs://pi0-cot/OXE/molmoact2_yam_dataset}"
GSUTIL="${GSUTIL:-gsutil}"
SSH_BIN="${SSH_BIN:-ssh}"
IDENTITY_FILE="${IDENTITY_FILE:-$HOME/.ssh/id_ed25519}"
UPLOAD_JOBS="${UPLOAD_JOBS:-1}"
UPLOAD_RETRIES="${UPLOAD_RETRIES:-3}"

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
echo "UPLOAD_JOBS=$UPLOAD_JOBS"
echo "UPLOAD_RETRIES=$UPLOAD_RETRIES"

mapfile -t files < <(
  "$SSH_BIN" "${SSH_OPTS[@]}" "$REMOTE_HOST" \
    "cd '$REMOTE_DATASET_DIR' && find . -type f -printf '%P\t%s\n' | sort"
)

if [ "${#files[@]}" -eq 0 ]; then
  echo "No files found under $REMOTE_DATASET_DIR" >&2
  exit 2
fi

upload_one() {
  set -euo pipefail
  local entry="$1"
  local rel="${entry%%$'\t'*}"
  local remote_size="${entry##*$'\t'}"
  local src="$REMOTE_DATASET_DIR/$rel"
  local dst="$GCS_DEST/$rel"
  local attempt=1
  local ssh_opts=(
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
  local existing_size=""
  existing_size="$("$GSUTIL" ls -l "$dst" 2>/dev/null | awk 'NR == 1 {print $1}' || true)"
  if [ -n "$existing_size" ] && [ "$existing_size" = "$remote_size" ]; then
    echo "skipping $rel ($remote_size bytes already present)"
    return 0
  fi

  while [ "$attempt" -le "$UPLOAD_RETRIES" ]; do
    echo "uploading $rel -> $dst ($remote_size bytes, attempt $attempt/$UPLOAD_RETRIES)"
    if "$SSH_BIN" "${ssh_opts[@]}" "$REMOTE_HOST" "cat '$src'" | "$GSUTIL" -q cp - "$dst"; then
      existing_size="$("$GSUTIL" ls -l "$dst" 2>/dev/null | awk 'NR == 1 {print $1}' || true)"
      if [ "$existing_size" = "$remote_size" ]; then
        echo "uploaded $rel ($remote_size bytes)"
        return 0
      fi
      echo "Size mismatch after upload for $rel: remote=$remote_size gcs=${existing_size:-missing}" >&2
    fi
    attempt=$((attempt + 1))
    if [ "$attempt" -le "$UPLOAD_RETRIES" ]; then
      sleep $((attempt * 10))
    fi
  done
  echo "Failed to upload $rel after $UPLOAD_RETRIES attempts" >&2
  return 1
}
export REMOTE_HOST REMOTE_DATASET_DIR GCS_DEST GSUTIL SSH_BIN IDENTITY_FILE UPLOAD_RETRIES
export -f upload_one

if [ "$UPLOAD_JOBS" = "1" ]; then
  for entry in "${files[@]}"; do
    upload_one "$entry"
  done
else
  printf '%s\0' "${files[@]}" | xargs -0 -n1 -P "$UPLOAD_JOBS" bash -c 'upload_one "$1"' _
fi

echo "uploaded ${#files[@]} files to $GCS_DEST"
