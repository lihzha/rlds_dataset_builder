#!/usr/bin/env python3
"""
Scan the AgiBotWorld-Beta-New dataset and track processable episodes.
This script identifies task IDs and episode IDs under observations directory,
limiting to maximum 10 episodes per task (or fewer if not available).
"""

import json
from pathlib import Path


def scan_and_track_episodes(
    base_path: Path, output_file: Path, max_episodes_per_task: int = 10
):
    """
    Scan the dataset directory and create a tracking file with processable episodes.

    Args:
        base_path: Path to AgiBotWorld-Beta-New directory
        output_file: Path to save the tracking JSON file
        max_episodes_per_task: Maximum number of episodes to process per task (default: 10)
    """
    observations_dir = base_path / "observations"

    if not observations_dir.exists():
        raise FileNotFoundError(f"Observations directory not found: {observations_dir}")

    print(f"Scanning dataset at: {base_path}")
    print(f"Maximum episodes per task: {max_episodes_per_task}")

    # Load corrupted episodes to skip them
    corrupted_episodes_file = Path(
        "/n/fs/robot-data/rlds_multithread/corrupted_episodes.json"
    )
    corrupted_episodes_set = set()

    if corrupted_episodes_file.exists():
        print(f"\nLoading corrupted episodes from: {corrupted_episodes_file}")
        with open(corrupted_episodes_file, "r") as f:
            corrupted_data = json.load(f)

        # Build a set of (task_id, episode_id) tuples for fast lookup
        for entry in corrupted_data.get("corrupted_episodes", []):
            task_id = str(entry["task_id"])
            episode_id = str(entry["episode_id"])
            corrupted_episodes_set.add((task_id, episode_id))

        print(f"Found {len(corrupted_episodes_set)} corrupted episodes to skip")
    else:
        print("\nNo corrupted episodes file found, processing all episodes")

    processed_episodes = {}
    episode_subskill_counts = {}  # Store subskill count for each episode
    skipped_tasks = []
    total_episodes = 0
    total_subskills = 0
    skipped_corrupted_count = 0

    # Iterate through all task directories
    for task_dir in sorted(observations_dir.iterdir()):
        if not task_dir.is_dir():
            continue

        task_id = task_dir.name
        print(f"\nProcessing task: {task_id}")

        # Load task info to count subskills
        task_json = base_path / f"task_info/task_{task_id}.json"
        task_data = {}
        if task_json.exists():
            with open(task_json, "r") as f:
                task_data = {str(entry["episode_id"]): entry for entry in json.load(f)}

        # Find all episode directories (ignore .tar files)
        episode_dirs = []
        for item in task_dir.iterdir():
            # Skip .tar files
            if item.is_file() and item.suffix == ".tar":
                continue

            # Only process directories
            if item.is_dir():
                episode_id = item.name

                # Skip corrupted episodes
                if (task_id, episode_id) in corrupted_episodes_set:
                    print(f"  Skipping episode {episode_id}: marked as corrupted")
                    skipped_corrupted_count += 1
                    continue

                # Verify the episode has required structure
                videos_dir = item / "videos"
                # depth_dir = item / "depth"

                if videos_dir.exists():
                    episode_dirs.append(episode_id)

                    # Count subskills for this episode
                    if episode_id in task_data:
                        action_configs = task_data[episode_id].get("label_info", {}).get("action_config", [])
                        subskill_count = len(action_configs) if action_configs else 1
                    else:
                        subskill_count = 1

                    # Store the count
                    episode_key = f"{task_id}_{episode_id}"
                    episode_subskill_counts[episode_key] = subskill_count
                else:
                    print(
                        f"  Skipping episode {episode_id}: missing videos or depth directory"
                    )

        # Sort episode IDs and take up to max_episodes_per_task
        episode_dirs.sort()
        selected_episodes = episode_dirs[:max_episodes_per_task]

        if selected_episodes:
            processed_episodes[task_id] = selected_episodes
            total_episodes += len(selected_episodes)

            # Calculate total subskills for selected episodes
            task_subskills = sum(
                episode_subskill_counts.get(f"{task_id}_{ep_id}", 1)
                for ep_id in selected_episodes
            )
            total_subskills += task_subskills

            print(
                f"  Found {len(episode_dirs)} episodes, selected {len(selected_episodes)} ({task_subskills} subskills)"
            )
        else:
            skipped_tasks.append(task_id)
            print("  No valid episodes found (only tar files or missing directories)")

    # Create tracking data structure
    tracking_data = {
        "processed_episodes": processed_episodes,
        "episode_subskill_counts": episode_subskill_counts,
        "skipped_tasks": skipped_tasks,
        "max_episodes_per_task": max_episodes_per_task,
        "total_tasks": len(processed_episodes),
        "total_episodes": total_episodes,
        "total_subskills": total_subskills,
        "skipped_corrupted_episodes": skipped_corrupted_count,
    }

    # Save to JSON file
    with open(output_file, "w") as f:
        json.dump(tracking_data, f, indent=2)

    print(f"\n{'=' * 60}")
    print("Scan complete!")
    print(f"Total tasks: {len(processed_episodes)}")
    print(f"Total episodes: {total_episodes}")
    print(f"Total subskills (examples): {total_subskills}")
    print(f"Skipped corrupted episodes: {skipped_corrupted_count}")
    print(f"Skipped tasks (no valid episodes): {len(skipped_tasks)}")
    print(f"Tracking file saved to: {output_file}")
    print(f"{'=' * 60}")

    return tracking_data


def main():
    # Configuration
    base_path = Path("/n/fs/robot-data/data/AgiBotWorld-Beta-New")
    output_file = Path("/n/fs/robot-data/rlds_multithread/processed_episodes.json")
    max_episodes_per_task = 10

    # Run the scan
    tracking_data = scan_and_track_episodes(
        base_path=base_path,
        output_file=output_file,
        max_episodes_per_task=max_episodes_per_task,
    )

    # Print summary of first few tasks
    print("\nSample of processed episodes:")
    for i, (task_id, episodes) in enumerate(
        tracking_data["processed_episodes"].items()
    ):
        if i >= 5:  # Only show first 5 tasks
            break
        print(f"  Task {task_id}: {len(episodes)} episodes - {episodes}")


if __name__ == "__main__":
    main()
