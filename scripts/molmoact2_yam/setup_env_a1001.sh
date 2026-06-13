#!/usr/bin/env bash
set -euo pipefail

NFS_ROOT="${NFS_ROOT:-/lustre/fsw/portfolios/nvr/users/lzha}"
ENV_DIR="${ENV_DIR:-$NFS_ROOT/envs/rlds_molmoact2_yam}"

python3 -m venv "$ENV_DIR"
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
