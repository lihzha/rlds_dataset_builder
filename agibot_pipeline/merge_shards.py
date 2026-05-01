"""Final merge: assemble per-task staging shards into one canonical TFDS dataset.

Reads:
  gs://<staging>/manifests/<task_id>-<tar_basename>.json   (per-tar manifests)
  gs://<staging>/shards/<DATASET>-<SPLIT>.tfrecord-<task_id>-<tar_basename>

Writes:
  gs://<final>/<DATASET>/<VERSION>/<DATASET>-<SPLIT>.tfrecord-XXXXX-of-YYYYY
  gs://<final>/<DATASET>/<VERSION>/dataset_info.json
  gs://<final>/<DATASET>/<VERSION>/features.json

By default does server-side rename (gcs.copy + delete source). Pass --copy to
keep the staging shards and copy instead.

Usage:
  python -m agibot_pipeline.merge_shards \
      --gcs-staging gs://my-bucket/agibot_staging \
      --gcs-final   gs://my-bucket/agibot_final \
      [--copy] [--dry-run]
"""

import argparse
import json
import sys
from datetime import datetime, timezone

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow_datasets.core import dataset_info as dataset_info_lib
from tensorflow_datasets.core import naming, splits as splits_lib

from agibot_pipeline.features import (
    DATASET_NAME,
    DATASET_VERSION,
    SPLIT_NAME,
    build_features,
)


def list_manifests(gcs_staging: str):
    pattern = f"{gcs_staging}/manifests/*.json"
    paths = sorted(tf.io.gfile.glob(pattern))
    out = []
    for p in paths:
        with tf.io.gfile.GFile(p, "r") as f:
            out.append(json.load(f))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gcs-staging", required=True)
    ap.add_argument("--gcs-final", required=True,
                    help="gs://bucket/path. Final dataset goes to "
                         "<final>/<dataset>/<version>/.")
    ap.add_argument("--copy", action="store_true",
                    help="Copy shards instead of server-side move (keeps staging).")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    final_dir = f"{args.gcs_final.rstrip('/')}/{DATASET_NAME}/{DATASET_VERSION}"
    print(f"Final dir: {final_dir}")

    manifests = list_manifests(args.gcs_staging)
    print(f"Loaded {len(manifests)} manifests.")
    if not manifests:
        print("No manifests found; aborting.", file=sys.stderr)
        return 1

    # Sort for deterministic shard ordering: by (task_id, tar_basename).
    manifests.sort(key=lambda m: (int(m["task_id"]), m["tar_basename"]))

    # Drop empties (a tar may legitimately have produced 0 examples if every
    # episode in it failed parsing — keep manifest, skip shard).
    manifests = [m for m in manifests if m["num_examples"] > 0]
    total = len(manifests)
    print(f"{total} non-empty shards will be merged.")
    if total == 0:
        print("No non-empty shards; aborting.", file=sys.stderr)
        return 1

    template = naming.ShardedFileTemplate(
        split=SPLIT_NAME,
        dataset_name=DATASET_NAME,
        data_dir=final_dir,
        filetype_suffix="tfrecord",
    )

    shard_lengths = []
    total_bytes = 0
    moves = []  # (src_uri, dst_uri)
    for idx, m in enumerate(manifests):
        dst = template.sharded_filepath(shard_index=idx, num_shards=total)
        moves.append((m["shard_uri"], str(dst)))
        shard_lengths.append(m["num_examples"])
        total_bytes += m["num_bytes"]

    print(f"Total examples: {sum(shard_lengths)}")
    print(f"Total bytes:    {total_bytes/1e9:.2f} GB")
    print(f"First mapping:  {moves[0][0]}\n             -> {moves[0][1]}")
    print(f"Last mapping:   {moves[-1][0]}\n             -> {moves[-1][1]}")

    if args.dry_run:
        print("Dry run — no GCS operations performed.")
        return 0

    if not tf.io.gfile.exists(final_dir):
        tf.io.gfile.makedirs(final_dir)

    # Move/copy shards. tf.io.gfile.rename across GCS is a server-side copy+delete.
    print(f"{'Copying' if args.copy else 'Renaming'} shards to final dir...")
    for i, (src, dst) in enumerate(moves):
        if tf.io.gfile.exists(dst):
            # Idempotency: previous merge run already placed this shard.
            continue
        if args.copy:
            tf.io.gfile.copy(src, dst, overwrite=False)
        else:
            tf.io.gfile.rename(src, dst, overwrite=False)
        if (i + 1) % 50 == 0 or (i + 1) == len(moves):
            print(f"  {i+1}/{len(moves)} done")

    # Build dataset metadata via a real (but data-free) builder, then write it.
    # Instantiating the builder against the final_dir gives DatasetInfo all the
    # plumbing it needs (name, version, data_dir, file_format) with no internal
    # API gymnastics.
    builder = _MergeBuilder(data_dir=args.gcs_final)
    split_info = splits_lib.SplitInfo(
        name=SPLIT_NAME,
        shard_lengths=shard_lengths,
        num_bytes=total_bytes,
        filename_template=template,
    )
    builder.info.set_splits(splits_lib.SplitDict([split_info]))
    builder.info.write_to_directory(final_dir)
    print(f"Wrote dataset_info.json + features.json under {final_dir}")

    # Optional: write a top-level merge log.
    log_uri = f"{final_dir}/_merge_log.json"
    with tf.io.gfile.GFile(log_uri, "w") as f:
        json.dump(
            {
                "merged_at_utc": datetime.now(timezone.utc).isoformat(),
                "num_shards": total,
                "num_examples": sum(shard_lengths),
                "num_bytes": total_bytes,
                "source_staging": args.gcs_staging,
                "moved_not_copied": not args.copy,
            },
            f,
            indent=2,
        )

    print("Merge complete.")
    return 0


class _MergeBuilder(tfds.core.GeneratorBasedBuilder):
    """Data-free builder used only to produce DatasetInfo for the merged shards.

    We never call download_and_prepare on this — we feed it the already-written
    GCS shard layout and just emit dataset_info.json + features.json.
    """

    name = DATASET_NAME
    VERSION = tfds.core.Version(DATASET_VERSION)
    RELEASE_NOTES = {DATASET_VERSION: "Initial release."}

    def _info(self) -> tfds.core.DatasetInfo:
        return self.dataset_info_from_configs(
            features=build_features(),
            description="AgiBotWorld-Beta as TFDS, subskill-segmented.",
        )

    def _split_generators(self, dl_manager):
        return {SPLIT_NAME: self._generate_examples()}

    def _generate_examples(self):
        return iter([])


if __name__ == "__main__":
    sys.exit(main())
