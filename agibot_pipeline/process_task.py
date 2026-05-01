"""Per-task processor: stream obs tars, parse, write tfrecord shards to GCS.

Idempotency model:
- Per-tar shard: gs://<staging>/shards/agibot-train.tfrecord-<task_id>-<tar_basename>
  Existence => skip the tar.
- Per-tar manifest: gs://<staging>/manifests/<task_id>-<tar_basename>.json
  Holds {num_examples, num_bytes, shard_path, task_id, tar}.
- Per-task sentinel: gs://<staging>/sentinels/_DONE_<task_id>
  Existence => skip the whole task (fast path; no HF listing needed).

Local disk model:
- One tar (~49 GB) on local scratch at a time. Extracted into <work>/observations/<task_id>/.
- Symlinks <work>/proprio_stats and <work>/task_info into shared scratch so the
  parser can find them with the existing layout.
- Tar + extracted dir are deleted after each tar's shard is uploaded.

Usage:
  python -m agibot_pipeline.process_task \
      --task-id 327 \
      --base /shared/scratch/agibot \
      --work-dir $SLURM_TMPDIR/agibot_$SLURM_JOB_ID \
      --gcs-staging gs://my-bucket/agibot_staging
"""

import argparse
import json
import os
import shutil
import sys
import tarfile
import time
import traceback
from pathlib import Path
from typing import List

import tensorflow as tf
import tensorflow_datasets as tfds
from huggingface_hub import hf_hub_download

from agibot_pipeline.features import (
    DATASET_NAME,
    SPLIT_NAME,
    build_features,
)
from agibot_pipeline.parser import parse_episode


HF_REPO = "agibot-world/AgiBotWorld-Beta"
HF_REPO_TYPE = "dataset"


def gcs_exists(uri: str) -> bool:
    return tf.io.gfile.exists(uri)


def gcs_write_text(uri: str, text: str) -> None:
    with tf.io.gfile.GFile(uri, "w") as f:
        f.write(text)


def shard_uri(gcs_staging: str, task_id: str, tar_basename: str) -> str:
    return f"{gcs_staging}/shards/{DATASET_NAME}-{SPLIT_NAME}.tfrecord-{task_id}-{tar_basename}"


def manifest_uri(gcs_staging: str, task_id: str, tar_basename: str) -> str:
    return f"{gcs_staging}/manifests/{task_id}-{tar_basename}.json"


def task_sentinel_uri(gcs_staging: str, task_id: str) -> str:
    return f"{gcs_staging}/sentinels/_DONE_{task_id}"


def setup_workdir(work_dir: Path, base: Path, task_id: str) -> Path:
    """Create <work>/observations/<task_id>/ and symlink shared assets.

    Returns the per-task base dir (the parent of observations/, proprio_stats/,
    and task_info/) which is what parser.parse_episode expects upstream.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "observations" / task_id).mkdir(parents=True, exist_ok=True)

    # Symlink shared dirs (overwrite if stale).
    for sub in ("proprio_stats", "task_info"):
        link = work_dir / sub
        target = (base / sub).resolve()
        if link.is_symlink() or link.exists():
            if link.is_symlink() and link.resolve() == target:
                continue
            if link.is_symlink():
                link.unlink()
            else:
                shutil.rmtree(link)
        link.symlink_to(target)

    return work_dir


def download_obs_tar(rfilename: str, dest_dir: Path) -> Path:
    """Download an obs tar from HF directly to dest_dir (no nested layout)."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    cached = hf_hub_download(
        repo_id=HF_REPO,
        filename=rfilename,
        repo_type=HF_REPO_TYPE,
        local_dir=str(dest_dir),
    )
    return Path(cached)


def extract_obs_tar(tar_path: Path, work_dir: Path, task_id: str) -> List[Path]:
    """Extract one obs tar under <work>/observations/<task_id>/.

    Tolerates two member-name layouts:
      A) "<task_id>/<episode_id>/..."   -> extract into <work>/observations/
      B) "<episode_id>/..."             -> extract into <work>/observations/<task_id>/
    """
    obs_root = work_dir / "observations"
    task_dir = obs_root / task_id
    task_dir.mkdir(parents=True, exist_ok=True)

    with tarfile.open(tar_path, "r") as tf_:
        members = tf_.getmembers()
        first = next((m for m in members if not m.isdir()), None)
        if first is None:
            return []
        head = first.name.split("/", 1)[0]
        dest = obs_root if head == task_id else task_dir
        tf_.extractall(dest)

    return sorted(
        p for p in task_dir.iterdir() if p.is_dir() and (p / "videos").exists()
    )


def cleanup_tar(tar_path: Path, work_dir: Path) -> None:
    """Delete the tar file and the extracted episode dirs (but keep the obs/<tid>/ dir
    itself so the next tar of the same task extracts cleanly)."""
    tar_path.unlink(missing_ok=True)
    obs_root = work_dir / "observations"
    if obs_root.exists():
        for tdir in obs_root.iterdir():
            for ep in tdir.iterdir():
                if ep.is_dir():
                    shutil.rmtree(ep, ignore_errors=True)


def process_one_tar(
    rfilename: str,
    task_id: str,
    work_dir: Path,
    base: Path,
    gcs_staging: str,
    features: tfds.features.FeaturesDict,
    serializer,
) -> dict:
    """Download → extract → parse → write shard → cleanup. Returns manifest dict."""
    tar_basename = Path(rfilename).stem  # e.g. "648642-685046"
    s_uri = shard_uri(gcs_staging, task_id, tar_basename)
    m_uri = manifest_uri(gcs_staging, task_id, tar_basename)

    if gcs_exists(s_uri) and gcs_exists(m_uri):
        with tf.io.gfile.GFile(m_uri, "r") as f:
            return json.load(f)

    print(f"  [tar {tar_basename}] downloading {rfilename}", flush=True)
    t0 = time.time()
    cache_dir = work_dir / ".hfcache"
    tar_path = download_obs_tar(rfilename, cache_dir)
    t_dl = time.time() - t0

    print(f"  [tar {tar_basename}] extracting ({tar_path.stat().st_size/1e9:.1f} GB)",
          flush=True)
    t0 = time.time()
    episode_dirs = extract_obs_tar(tar_path, work_dir, task_id)
    t_ex = time.time() - t0

    # Stream-write directly to GCS. tf.io.TFRecordWriter handles gs:// URIs.
    proprio_root = work_dir / "proprio_stats"
    task_info_dir = work_dir / "task_info"

    n_examples = 0
    n_skipped_eps = 0
    print(f"  [tar {tar_basename}] writing shard {s_uri}", flush=True)
    t0 = time.time()
    with tf.io.TFRecordWriter(s_uri) as writer:
        for ep_dir in episode_dirs:
            try:
                for _key, sample in parse_episode(ep_dir, task_info_dir, proprio_root):
                    encoded = features.encode_example(sample)
                    serialized = serializer.serialize_example(encoded)
                    writer.write(serialized)
                    n_examples += 1
            except Exception as e:
                n_skipped_eps += 1
                print(f"    WARN: skipping {ep_dir.name}: {type(e).__name__}: {e}",
                      flush=True)
    t_wr = time.time() - t0

    # Lookup byte size for the manifest.
    num_bytes = tf.io.gfile.stat(s_uri).length

    manifest = {
        "task_id": task_id,
        "tar": rfilename,
        "tar_basename": tar_basename,
        "shard_uri": s_uri,
        "num_examples": n_examples,
        "num_bytes": num_bytes,
        "skipped_episodes": n_skipped_eps,
        "timing_sec": {"download": t_dl, "extract": t_ex, "write": t_wr},
    }
    gcs_write_text(m_uri, json.dumps(manifest, indent=2))

    cleanup_tar(tar_path, work_dir)
    print(
        f"  [tar {tar_basename}] DONE: {n_examples} examples, "
        f"{num_bytes/1e6:.0f} MB, dl={t_dl:.0f}s ex={t_ex:.0f}s wr={t_wr:.0f}s",
        flush=True,
    )
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-id", required=True)
    ap.add_argument("--base", required=True, type=Path,
                    help="Shared scratch base (with proprio_stats/, task_info/, tasks.json).")
    ap.add_argument("--work-dir", required=True, type=Path,
                    help="Per-job local scratch dir (will be deleted on success).")
    ap.add_argument("--gcs-staging", required=True,
                    help="gs://bucket/path for staging shards/manifests/sentinels.")
    ap.add_argument("--cleanup-on-success", action="store_true", default=True)
    args = ap.parse_args()

    task_id = str(args.task_id)
    sentinel = task_sentinel_uri(args.gcs_staging, task_id)
    if gcs_exists(sentinel):
        print(f"Task {task_id}: sentinel exists at {sentinel}, skipping.")
        return 0

    # Load tasks.json to get the obs tar list for this task.
    with open(args.base / "tasks.json", "r") as f:
        all_tasks = json.load(f)
    entry = next((t for t in all_tasks if t["task_id"] == task_id), None)
    if entry is None:
        print(f"ERROR: task_id {task_id} not in tasks.json", file=sys.stderr)
        return 2

    obs_tars = entry["obs_tars"]
    print(
        f"Task {task_id}: {len(obs_tars)} obs tars, "
        f"{entry['size_bytes']/1e9:.1f} GB total."
    )

    work_dir = setup_workdir(args.work_dir, args.base, task_id)

    # Build features + serializer once.
    features = build_features()
    serialized_info = features.get_serialized_info()
    serializer = tfds.core.example_serializer.ExampleSerializer(serialized_info)

    manifests = []
    failures = []
    for i, rfilename in enumerate(obs_tars, 1):
        print(f"[{i}/{len(obs_tars)}] {rfilename}", flush=True)
        try:
            manifests.append(
                process_one_tar(
                    rfilename, task_id, work_dir, args.base, args.gcs_staging,
                    features, serializer,
                )
            )
        except Exception as e:
            failures.append({"tar": rfilename, "error": f"{type(e).__name__}: {e}"})
            traceback.print_exc()
            # Best-effort local cleanup so we don't leak disk on the next iteration.
            cleanup_tar(work_dir / ".hfcache" / rfilename, work_dir)

    if failures:
        # Don't write the per-task sentinel; the array job can be re-submitted to retry.
        print(f"Task {task_id}: {len(failures)} tar failure(s); not writing sentinel.",
              file=sys.stderr)
        for f_ in failures:
            print(f"  FAIL: {f_['tar']}: {f_['error']}", file=sys.stderr)
        return 1

    # Write per-task summary + sentinel.
    summary_uri = f"{args.gcs_staging}/sentinels/_summary_{task_id}.json"
    summary = {
        "task_id": task_id,
        "num_tars": len(obs_tars),
        "num_examples": sum(m["num_examples"] for m in manifests),
        "num_bytes": sum(m["num_bytes"] for m in manifests),
        "shard_uris": [m["shard_uri"] for m in manifests],
    }
    gcs_write_text(summary_uri, json.dumps(summary, indent=2))
    gcs_write_text(sentinel, "done\n")

    if args.cleanup_on_success:
        shutil.rmtree(args.work_dir, ignore_errors=True)

    print(
        f"Task {task_id}: COMPLETE — {summary['num_examples']} examples in "
        f"{summary['num_tars']} shards ({summary['num_bytes']/1e9:.1f} GB)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
