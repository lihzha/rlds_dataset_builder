#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re

import numpy as np
import pyarrow.parquet as pq


_TRUNC_RE = re.compile(
    r"Truncating episode\s+(?P<episode>\d+)\s+in\s+.*?/data/chunk-(?P<chunk>\d+)/file-(?P<file>\d+)\.parquet;"
    r"\s+dropping\s+(?P<drop>\d+)/"
)


@dataclass
class RunningStats:
    dims: int

    def __post_init__(self) -> None:
        self.count = 0
        self.sum = np.zeros(self.dims, dtype=np.float64)
        self.sumsq = np.zeros(self.dims, dtype=np.float64)
        self.min = np.full(self.dims, np.inf, dtype=np.float64)
        self.max = np.full(self.dims, -np.inf, dtype=np.float64)

    def update(self, values: np.ndarray, weights: np.ndarray | None = None) -> None:
        if values.size == 0:
            return
        values = np.asarray(values, dtype=np.float64)
        finite = np.isfinite(values).all(axis=1)
        if weights is not None:
            weights = np.asarray(weights, dtype=np.float64)[finite]
        values = values[finite]
        if values.size == 0:
            return
        if weights is None:
            self.count += values.shape[0]
            self.sum += values.sum(axis=0)
            self.sumsq += np.square(values).sum(axis=0)
        else:
            self.count += int(weights.sum())
            self.sum += (values * weights[:, None]).sum(axis=0)
            self.sumsq += (np.square(values) * weights[:, None]).sum(axis=0)
        self.min = np.minimum(self.min, values.min(axis=0))
        self.max = np.maximum(self.max, values.max(axis=0))

    @property
    def mean(self) -> np.ndarray:
        return self.sum / max(self.count, 1)

    @property
    def std(self) -> np.ndarray:
        var = self.sumsq / max(self.count, 1) - np.square(self.mean)
        return np.sqrt(np.maximum(var, 0.0))


class HistogramQuantiles:
    def __init__(self, minimum: np.ndarray, maximum: np.ndarray, num_bins: int) -> None:
        self.minimum = np.asarray(minimum, dtype=np.float64)
        self.maximum = np.asarray(maximum, dtype=np.float64)
        self.num_bins = int(num_bins)
        self.hist = np.zeros((len(self.minimum), self.num_bins), dtype=np.float64)
        self.edges = [
            np.linspace(lo - 1e-12, hi + 1e-12, self.num_bins + 1)
            for lo, hi in zip(self.minimum, self.maximum)
        ]

    def update(self, values: np.ndarray, weights: np.ndarray | None = None) -> None:
        if values.size == 0:
            return
        values = np.asarray(values, dtype=np.float64)
        finite = np.isfinite(values).all(axis=1)
        values = values[finite]
        if weights is not None:
            weights = np.asarray(weights, dtype=np.float64)[finite]
        for dim, edges in enumerate(self.edges):
            if self.minimum[dim] == self.maximum[dim]:
                continue
            hist, _ = np.histogram(values[:, dim], bins=edges, weights=weights)
            self.hist[dim] += hist

    def quantile(self, q: float) -> np.ndarray:
        out = np.zeros(len(self.minimum), dtype=np.float32)
        for dim, counts in enumerate(self.hist):
            total = counts.sum()
            if total == 0 or self.minimum[dim] == self.maximum[dim]:
                out[dim] = self.minimum[dim]
                continue
            cdf = np.cumsum(counts)
            idx = int(np.searchsorted(cdf, q * total, side="left"))
            idx = min(idx, self.num_bins - 1)
            out[dim] = self.edges[dim][idx]
        return out


def parse_truncations(log_path: Path | None) -> dict[tuple[int, int, int], int]:
    truncations: dict[tuple[int, int, int], int] = {}
    if log_path is None:
        return truncations
    for line in log_path.read_text(errors="replace").splitlines():
        match = _TRUNC_RE.search(line)
        if not match:
            continue
        key = (int(match["chunk"]), int(match["file"]), int(match["episode"]))
        truncations[key] = int(match["drop"])
    return truncations


def parse_chunk_file(path: Path) -> tuple[int, int]:
    match = re.search(r"chunk-(\d+)/file-(\d+)\.parquet$", path.as_posix())
    if not match:
        raise ValueError(f"Could not parse chunk/file from {path}")
    return int(match.group(1)), int(match.group(2))


def action_window_weights(length: int, horizon: int) -> np.ndarray:
    idx = np.arange(length, dtype=np.int64)
    weights = np.minimum(horizon, idx + 1)
    tail = min(horizon - 1, length)
    if length > 0 and tail > 0:
        weights[-1] += tail * horizon - tail * (tail + 1) // 2
    return weights


def load_numeric_columns(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    table = pq.read_table(path, columns=["action", "observation.state", "episode_index"])
    actions = np.asarray(table.column("action").to_pylist(), dtype=np.float64)
    states = np.asarray(table.column("observation.state").to_pylist(), dtype=np.float64)
    episodes = np.asarray(table.column("episode_index").to_pylist(), dtype=np.int64)
    return actions, states, episodes


def iter_episode_arrays(
    parquet_files: list[Path],
    truncations: dict[tuple[int, int, int], int],
    *,
    state_dim: int,
):
    trajectories = 0
    transitions = 0
    for path_idx, path in enumerate(parquet_files, start=1):
        chunk_idx, file_idx = parse_chunk_file(path)
        actions, states, episodes = load_numeric_columns(path)
        for episode in np.unique(episodes):
            rows = np.flatnonzero(episodes == episode)
            drop = truncations.get((chunk_idx, file_idx, int(episode)), 0)
            if drop:
                rows = rows[:-drop]
            if rows.size == 0:
                continue
            ep_actions = actions[rows]
            ep_states = np.zeros((rows.size, state_dim), dtype=np.float64)
            ep_states[:, : states.shape[1]] = states[rows]
            trajectories += 1
            transitions += rows.size
            yield ep_actions, ep_states
        if path_idx % 100 == 0:
            print(f"loaded {path_idx}/{len(parquet_files)} parquet files; trajectories={trajectories} transitions={transitions}", flush=True)


def norm_entry(
    stats: RunningStats,
    q01: np.ndarray,
    q99: np.ndarray,
    *,
    num_trajectories: int,
) -> dict[str, object]:
    return {
        "mean": stats.mean.astype(np.float32).tolist(),
        "std": stats.std.astype(np.float32).tolist(),
        "q01": q01.astype(np.float32).tolist(),
        "q99": q99.astype(np.float32).tolist(),
        "num_transitions": int(stats.count),
        "num_trajectories": int(num_trajectories),
        "min": stats.min.astype(np.float32).tolist(),
        "max": stats.max.astype(np.float32).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--truncation-log", type=Path)
    parser.add_argument("--action-horizon", type=int, default=32)
    parser.add_argument("--state-dim", type=int, default=20)
    parser.add_argument("--bins", type=int, default=4096)
    args = parser.parse_args()

    parquet_files = sorted((args.raw_dir / "data").glob("chunk-*/file-*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files under {args.raw_dir / 'data'}")
    truncations = parse_truncations(args.truncation_log)
    print(f"parquet_files={len(parquet_files)} truncations={len(truncations)}")

    first_actions, first_states = next(iter_episode_arrays(parquet_files[:1], truncations, state_dim=args.state_dim))
    action_dim = first_actions.shape[1]
    action_stats = RunningStats(action_dim)
    state_stats = RunningStats(args.state_dim)

    trajectories = 0
    transitions = 0
    for actions, states in iter_episode_arrays(parquet_files, truncations, state_dim=args.state_dim):
        weights = action_window_weights(len(actions), args.action_horizon)
        action_stats.update(actions, weights=weights)
        state_stats.update(states)
        trajectories += 1
        transitions += len(actions)

    action_hist = HistogramQuantiles(action_stats.min, action_stats.max, args.bins)
    state_hist = HistogramQuantiles(state_stats.min, state_stats.max, args.bins)
    for actions, states in iter_episode_arrays(parquet_files, truncations, state_dim=args.state_dim):
        weights = action_window_weights(len(actions), args.action_horizon)
        action_hist.update(actions, weights=weights)
        state_hist.update(states)

    payload = {
        "norm_stats": {
            "state": norm_entry(
                state_stats,
                state_hist.quantile(0.01),
                state_hist.quantile(0.99),
                num_trajectories=trajectories,
            ),
            "actions": norm_entry(
                action_stats,
                action_hist.quantile(0.01),
                action_hist.quantile(0.99),
                num_trajectories=trajectories,
            ),
        }
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(f"wrote {args.output}")
    print(f"trajectories={trajectories} state_transitions={transitions} action_transitions={action_stats.count}")


if __name__ == "__main__":
    main()
