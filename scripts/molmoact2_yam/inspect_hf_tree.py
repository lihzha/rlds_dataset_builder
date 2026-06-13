#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from urllib.parse import urlparse
from urllib.parse import parse_qs

from huggingface_hub import HfApi


REPO_ID = "allenai/MolmoAct2-BimanualYAM-Dataset"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default=REPO_ID)
    args = parser.parse_args()

    api = HfApi()
    info = api.repo_info(args.repo_id, repo_type="dataset", files_metadata=True)
    files = [s for s in info.siblings if getattr(s, "rfilename", None)]
    total = sum((s.size or 0) for s in files)
    print(f"repo={args.repo_id}")
    print(f"files={len(files)}")
    print(f"bytes={total}")
    print(f"tib={total / (1024 ** 4):.3f}")

    groups = {}
    for sibling in files:
        name = sibling.rfilename
        if name.startswith("data/"):
            key = "data"
        elif name.startswith("videos/"):
            parts = name.split("/")
            key = "/".join(parts[:2])
        elif name.startswith("meta/"):
            key = "meta"
        else:
            key = "other"
        count, size = groups.get(key, (0, 0))
        groups[key] = (count + 1, size + (sibling.size or 0))

    for key in sorted(groups):
        count, size = groups[key]
        print(f"{key}: files={count} tib={size / (1024 ** 4):.3f}")


if __name__ == "__main__":
    main()
