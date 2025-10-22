"""
Test script to visualize sub-episode extraction logic.

This script extracts and visualizes subskills from episodes, showing:
- Action text
- Skill name
- Video frames for each sub-episode
"""

import json
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def visualize_subskill(episode_path: str, max_frames_to_show: int = 8):
    """
    Visualize subskill extraction from a single episode.

    Args:
        episode_path: Path to episode directory
        max_frames_to_show: Maximum number of frames to display per subskill
    """
    episode_path = Path(episode_path)
    task_id = episode_path.parent.name
    episode_id = episode_path.name

    print(f"\n{'='*80}")
    print(f"Processing Episode: Task {task_id}, Episode {episode_id}")
    print(f"Path: {episode_path}")
    print(f"{'='*80}\n")

    # Get base path to AgiBotWorld-Beta
    base_path = episode_path.parent.parent.parent

    # Load task info for this specific episode
    task_json = base_path / f"task_info/task_{task_id}.json"
    with open(task_json, "r") as f:
        task_data = json.load(f)

    # Find the task_info entry for this specific episode_id
    episode_task_info = None
    for entry in task_data:
        if str(entry["episode_id"]) == episode_id:
            episode_task_info = entry
            break

    if episode_task_info is None:
        raise ValueError(f"Episode {episode_id} not found in task {task_id} JSON")

    task_name = episode_task_info["task_name"]
    action_configs = episode_task_info.get("label_info", {}).get("action_config", [])

    print(f"Task Name: {task_name}")
    print(f"Number of subskills: {len(action_configs)}\n")

    # Load video frames
    video_dir = episode_path / "videos"
    video_file = video_dir / "head_color.mp4"

    if not video_file.exists():
        raise FileNotFoundError(f"Video file not found: {video_file}")

    # Read video frames using OpenCV
    cap = cv2.VideoCapture(str(video_file))
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_file}")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)

    cap.release()
    total_frames = len(frames)
    print(f"Total frames in video: {total_frames}\n")

    # Process each subskill
    for idx, action_config in enumerate(action_configs):
        start_frame = action_config["start_frame"]
        end_frame = action_config["end_frame"]
        action_text = action_config.get("action_text", "").strip()
        skill_name = action_config.get("skill", "Unknown")

        print(f"{'─'*80}")
        print(f"Subskill {idx + 1}/{len(action_configs)}")
        print(f"{'─'*80}")
        print(f"  Skill Name:    {skill_name}")
        print(f"  Action Text:   {action_text}")
        print(f"  Frame Range:   {start_frame} → {end_frame}")
        print(f"  Duration:      {end_frame - start_frame} frames")

        # Validate frame range
        if start_frame < 0 or end_frame > total_frames:
            print(f"  ⚠️  WARNING: Frame range [{start_frame}, {end_frame}) is out of bounds [0, {total_frames})")
            continue

        if start_frame >= end_frame:
            print(f"  ⚠️  WARNING: Invalid frame range (start >= end)")
            continue

        # Extract frames for this subskill
        subskill_frames = frames[start_frame:end_frame]
        num_frames = len(subskill_frames)

        # Select frames to display (evenly spaced)
        if num_frames <= max_frames_to_show:
            frames_to_show = subskill_frames
            frame_indices = list(range(start_frame, end_frame))
        else:
            # Sample evenly across the subskill
            step = num_frames / max_frames_to_show
            indices = [int(i * step) for i in range(max_frames_to_show)]
            frames_to_show = [subskill_frames[i] for i in indices]
            frame_indices = [start_frame + i for i in indices]

        # Visualize frames
        n_cols = min(4, len(frames_to_show))
        n_rows = (len(frames_to_show) + n_cols - 1) // n_cols

        fig = plt.figure(figsize=(4 * n_cols, 4 * n_rows + 1.5))
        gs = GridSpec(n_rows + 1, n_cols, figure=fig, height_ratios=[0.3] + [1] * n_rows)

        # Add title with subskill info
        title_ax = fig.add_subplot(gs[0, :])
        title_ax.axis('off')
        title_text = (
            f"Subskill {idx + 1}/{len(action_configs)}: {skill_name}\n"
            f"Action: {action_text}\n"
            f"Frames: {start_frame} → {end_frame} ({num_frames} frames)"
        )
        title_ax.text(0.5, 0.5, title_text,
                     ha='center', va='center',
                     fontsize=14, fontweight='bold',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Plot frames
        for i, (frame, frame_idx) in enumerate(zip(frames_to_show, frame_indices)):
            row = i // n_cols + 1
            col = i % n_cols
            ax = fig.add_subplot(gs[row, col])
            ax.imshow(frame)
            ax.set_title(f"Frame {frame_idx}", fontsize=10)
            ax.axis('off')

        plt.tight_layout()

        # Save figure
        output_dir = Path("/n/fs/robot-data/rlds_multithread/subskill_visualizations")
        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / f"task_{task_id}_episode_{episode_id}_subskill_{idx:02d}.png"
        plt.savefig(output_file, dpi=100, bbox_inches='tight')
        print(f"  ✓ Saved visualization: {output_file}")
        plt.close()

    print(f"\n{'='*80}")
    print(f"Completed processing episode {episode_id}")
    print(f"{'='*80}\n")


def test_multiple_episodes(episode_paths: list, max_frames_to_show: int = 8):
    """
    Test subskill extraction on multiple episodes.

    Args:
        episode_paths: List of episode directory paths
        max_frames_to_show: Maximum number of frames to display per subskill
    """
    print(f"\nTesting subskill extraction on {len(episode_paths)} episodes")
    print(f"Maximum frames to show per subskill: {max_frames_to_show}\n")

    success_count = 0
    error_count = 0

    for episode_path in episode_paths:
        try:
            visualize_subskill(episode_path, max_frames_to_show)
            success_count += 1
        except Exception as e:
            error_count += 1
            print(f"\n❌ Error processing {episode_path}:")
            print(f"   {type(e).__name__}: {str(e)}\n")

    print(f"\n{'='*80}")
    print(f"Summary:")
    print(f"  ✓ Successfully processed: {success_count}")
    print(f"  ❌ Errors: {error_count}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    # Load tracking file to get some test episodes
    tracking_file = Path("/n/fs/robot-data/rlds_multithread/processed_episodes.json")

    if not tracking_file.exists():
        raise FileNotFoundError(
            f"Tracking file not found: {tracking_file}\n"
            "Please run scan_and_track_episodes.py first."
        )

    with open(tracking_file, "r") as f:
        tracking_data = json.load(f)

    base_path = Path("/n/fs/robot-data/data/AgiBotWorld-Beta-New")
    processed_episodes = tracking_data.get("processed_episodes", {})

    # Select a few episodes to test (first 3 tasks, first 2 episodes each)
    test_episodes = []
    for task_id in sorted(list(processed_episodes.keys()))[:3]:
        for episode_id in sorted(list(processed_episodes[task_id]))[:2]:
            episode_path = base_path / "observations" / task_id / episode_id
            if episode_path.exists():
                test_episodes.append(str(episode_path))

    if not test_episodes:
        print("No test episodes found!")
    else:
        print(f"Found {len(test_episodes)} episodes to test\n")
        test_multiple_episodes(test_episodes, max_frames_to_show=8)
