#!/usr/bin/env bash
set -euo pipefail

NFS_ROOT="${NFS_ROOT:-/lustre/fsw/portfolios/nvr/users/lzha}"
ENV_DIR="${ENV_DIR:-$NFS_ROOT/envs/rlds_molmoact2_yam}"

if [ -n "${MOLMOACT2_YAM_PYTHON:-}" ]; then
  PYTHON_BIN="$MOLMOACT2_YAM_PYTHON"
else
  PYTHON_BIN=""
  for candidate in python3.11 python3.10 python3.9 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
      if "$candidate" - <<'PY'
import sys
raise SystemExit(0 if sys.version_info[:2] >= (3, 9) else 1)
PY
      then
        PYTHON_BIN="$candidate"
        break
      fi
    fi
  done
fi

if [ -z "$PYTHON_BIN" ]; then
  echo "No supported Python found. TensorFlow 2.15 requires Python >=3.9." >&2
  exit 2
fi

echo "PYTHON_BIN=$PYTHON_BIN"
"$PYTHON_BIN" -V

if [ -x "$ENV_DIR/bin/python" ]; then
  if ! "$ENV_DIR/bin/python" - <<'PY'
import sys
raise SystemExit(0 if sys.version_info[:2] >= (3, 9) else 1)
PY
  then
    echo "Removing unsupported existing venv: $ENV_DIR"
    rm -rf "$ENV_DIR"
  fi
fi

if [ ! -x "$ENV_DIR/bin/python" ]; then
  "$PYTHON_BIN" -m venv "$ENV_DIR"
fi

"$ENV_DIR/bin/python" -m pip install --upgrade pip wheel setuptools
"$ENV_DIR/bin/python" -m pip install \
  tensorflow==2.15.1 \
  tensorflow-datasets==4.9.4 \
  pyarrow==15.0.2 \
  huggingface_hub==0.33.0 \
  imageio-ffmpeg==0.5.1 \
  pillow==10.4.0 \
  tqdm==4.66.5 \
  gsutil==5.32

echo "ENV_DIR=$ENV_DIR"
