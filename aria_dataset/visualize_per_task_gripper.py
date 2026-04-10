#!/usr/bin/env python3
"""
Visualize gripper with PER-TASK adaptive thresholds.

Key improvement: Segments trajectory by timestamp gaps, then computes
separate thresholds for each task instance.

Usage:
    python visualize_gripper_per_task.py --zarr-path <path> --output per_task.mp4
"""

import argparse
from pathlib import Path
import cv2
import numpy as np
import zarr
from scipy.ndimage import uniform_filter1d
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

try:
    import simplejpeg
    HAS_SIMPLEJPEG = True
except ImportError:
    HAS_SIMPLEJPEG = False


# Keypoint indices
THUMB_TIP_IDX = 7
INDEX_TIP_IDX = 10

# Hand visualization
FINGER_EDGES = [
    (5, 6), (6, 7), (7, 0),  # thumb
    (5, 8), (8, 9), (9, 10), (9, 1),  # index
    (5, 11), (11, 12), (12, 13), (13, 2),  # middle
    (5, 14), (14, 15), (15, 16), (16, 3),  # ring
    (5, 17), (17, 18), (18, 19), (19, 4),  # pinky
]

FINGER_COLORS = {
    "thumb": (255, 100, 100),
    "index": (100, 255, 100),
    "middle": (100, 100, 255),
    "ring": (255, 255, 100),
    "pinky": (255, 100, 255),
}

FINGER_EDGE_RANGES = [
    ("thumb", 0, 3),
    ("index", 3, 7),
    ("middle", 7, 11),
    ("ring", 11, 15),
    ("pinky", 15, 19),
]


def decode_jpeg(data):
    """Decode JPEG data."""
    while isinstance(data, np.ndarray) and data.ndim == 0:
        data = data.item()

    if HAS_SIMPLEJPEG:
        if isinstance(data, bytes):
            return simplejpeg.decode_jpeg(data)
        elif isinstance(data, np.ndarray) and data.dtype == np.uint8:
            return simplejpeg.decode_jpeg(data.tobytes())

    if isinstance(data, bytes):
        arr = np.frombuffer(data, dtype=np.uint8)
    elif isinstance(data, np.ndarray) and data.dtype == np.uint8:
        arr = data
    else:
        raise ValueError(f"Unknown data type: {type(data)}")

    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode JPEG")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def compute_gripper_distance(keypoints_3d):
    """Compute distance between thumb and index tips."""
    thumb_tip = keypoints_3d[THUMB_TIP_IDX]
    index_tip = keypoints_3d[INDEX_TIP_IDX]
    return np.linalg.norm(thumb_tip - index_tip)


def detect_task_boundaries(timestamps, gap_threshold=2.5, fps=30):
    """
    Detect task boundaries from timestamp discontinuities.

    Returns:
        List of (start_frame, end_frame) for each task
    """
    time_deltas = np.diff(timestamps) / 1e9  # Convert to seconds
    gap_indices = np.where(time_deltas > gap_threshold)[0]

    tasks = []

    if len(gap_indices) == 0:
        # No gaps - entire file is one task
        tasks.append((0, len(timestamps) - 1))
    else:
        # First task
        if gap_indices[0] > 0:
            tasks.append((0, gap_indices[0]))

        # Middle tasks
        for i in range(len(gap_indices) - 1):
            start = gap_indices[i] + 1
            end = gap_indices[i + 1]
            if end - start >= 30:  # At least 1 second of data
                tasks.append((start, end))

        # Last task
        start = gap_indices[-1] + 1
        if len(timestamps) - start >= 30:
            tasks.append((start, len(timestamps) - 1))

    return tasks


def compute_task_threshold(distances, method='simple'):
    """
    Compute binary threshold for a single task.

    Args:
        distances: Array of gripper distances for one task
        method: 'simple' or 'hybrid'

    Returns:
        binary_threshold
    """
    if len(distances) < 10:
        # Too few samples, use median
        return np.median(distances)

    if method == 'simple':
        # Use percentiles
        closed = np.percentile(distances, 5)
        open_thresh = np.percentile(distances, 95)
        binary_thresh = (closed + open_thresh) / 2

    elif method == 'hybrid':
        # Detect sustained events
        smoothed = uniform_filter1d(distances, size=min(5, len(distances)))

        closed_thresh_detect = np.percentile(smoothed, 10)
        is_closed = smoothed < closed_thresh_detect

        # Find sustained close (>= 15 frames)
        min_sustained = float('inf')
        in_region = False
        start = 0

        for i, val in enumerate(is_closed):
            if val and not in_region:
                start = i
                in_region = True
            elif not val and in_region:
                if i - start >= 15:
                    region_min = smoothed[start:i].min()
                    min_sustained = min(min_sustained, region_min)
                in_region = False

        if in_region and len(is_closed) - start >= 15:
            region_min = smoothed[start:].min()
            min_sustained = min(min_sustained, region_min)

        # Combine with percentile
        percentile_closed = np.percentile(distances, 5)
        if min_sustained != float('inf'):
            closed = min(min_sustained, percentile_closed)
        else:
            closed = percentile_closed

        open_thresh = np.percentile(distances, 95)
        binary_thresh = (closed + open_thresh) / 2

    return binary_thresh


def classify_binary(distance, binary_thresh):
    """Classify as OPEN (True) or CLOSED (False)."""
    return distance > binary_thresh


def load_camera_intrinsics(zarr_path):
    """Load camera intrinsics."""
    store = zarr.open_group(str(zarr_path), mode='r')
    metadata = store.attrs.asdict()

    intrinsics = metadata.get('camera_intrinsics', {})
    if not intrinsics:
        intrinsics = {
            'fx': 320.0,
            'fy': 320.0,
            'cx': 320.0,
            'cy': 240.0,
            'width': 640,
            'height': 480,
        }

    return intrinsics


def project_3d_to_2d(points_3d, head_pose, intrinsics):
    """Project 3D points to 2D image coordinates."""
    from scipy.spatial.transform import Rotation as R

    cam_pos = head_pose[:3]
    cam_quat = head_pose[3:]

    quat_norm = np.linalg.norm(cam_quat)
    if quat_norm < 1e-6:
        invalid_pts = np.full((len(points_3d), 2), -1, dtype=np.float32)
        invalid_mask = np.zeros(len(points_3d), dtype=bool)
        return invalid_pts, invalid_mask

    try:
        R_world_to_cam = R.from_quat([cam_quat[1], cam_quat[2], cam_quat[3], cam_quat[0]]).as_matrix().T
    except (ValueError, RuntimeError):
        invalid_pts = np.full((len(points_3d), 2), -1, dtype=np.float32)
        invalid_mask = np.zeros(len(points_3d), dtype=bool)
        return invalid_pts, invalid_mask

    t_world_to_cam = -R_world_to_cam @ cam_pos
    points_cam = (R_world_to_cam @ points_3d.T).T + t_world_to_cam

    fx, fy = intrinsics['fx'], intrinsics['fy']
    cx, cy = intrinsics['cx'], intrinsics['cy']

    z = points_cam[:, 2]
    valid_mask = z > 0.01

    u = np.full(len(points_3d), -1, dtype=np.float32)
    v = np.full(len(points_3d), -1, dtype=np.float32)

    u[valid_mask] = fx * points_cam[valid_mask, 0] / z[valid_mask] + cx
    v[valid_mask] = fy * points_cam[valid_mask, 1] / z[valid_mask] + cy

    return np.stack([u, v], axis=1), valid_mask


def draw_hand_skeleton(image, keypoints_2d, keypoints_valid, is_open, is_in_task,
                       current_task_id, total_tasks):
    """Draw hand skeleton with gripper state indicator."""
    vis = image.copy()
    h, w = vis.shape[:2]

    # Draw skeleton
    for finger_name, start_idx, end_idx in FINGER_EDGE_RANGES:
        color = FINGER_COLORS[finger_name]
        for i in range(start_idx, end_idx):
            edge = FINGER_EDGES[i]
            if keypoints_valid[edge[0]] and keypoints_valid[edge[1]]:
                pt1 = tuple(keypoints_2d[edge[0]].astype(int))
                pt2 = tuple(keypoints_2d[edge[1]].astype(int))
                if (0 <= pt1[0] < w and 0 <= pt1[1] < h and
                    0 <= pt2[0] < w and 0 <= pt2[1] < h):
                    cv2.line(vis, pt1, pt2, color, 2)

    # Draw keypoints
    for i, (kp, valid) in enumerate(zip(keypoints_2d, keypoints_valid)):
        if valid:
            pt = tuple(kp.astype(int))
            if 0 <= pt[0] < w and 0 <= pt[1] < h:
                if i == THUMB_TIP_IDX or i == INDEX_TIP_IDX:
                    cv2.circle(vis, pt, 5, (255, 255, 0), -1)
                else:
                    cv2.circle(vis, pt, 3, (255, 165, 0), -1)

    # Draw gripper line with color based on state
    if keypoints_valid[THUMB_TIP_IDX] and keypoints_valid[INDEX_TIP_IDX]:
        thumb_pt = tuple(keypoints_2d[THUMB_TIP_IDX].astype(int))
        index_pt = tuple(keypoints_2d[INDEX_TIP_IDX].astype(int))
        if (0 <= thumb_pt[0] < w and 0 <= thumb_pt[1] < h and
            0 <= index_pt[0] < w and 0 <= index_pt[1] < h):
            if is_in_task:
                line_color = (0, 255, 0) if is_open else (255, 0, 0)
                cv2.line(vis, thumb_pt, index_pt, line_color, 3)
            else:
                # Gray for gap between tasks
                cv2.line(vis, thumb_pt, index_pt, (128, 128, 128), 2)

    # Draw state label
    if is_in_task:
        state_text = "OPEN" if is_open else "CLOSED"
        state_color = (0, 255, 0) if is_open else (255, 0, 0)
        task_info = f"Task {current_task_id}/{total_tasks}"
    else:
        state_text = "GAP"
        state_color = (128, 128, 128)
        task_info = f"Between tasks"

    # Background box for text
    text_size = cv2.getTextSize(state_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
    cv2.rectangle(vis, (10, 10), (20 + text_size[0], 50 + text_size[1]), (0, 0, 0), -1)
    cv2.putText(vis, state_text, (15, 40 + text_size[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, state_color, 3)

    # Task info
    cv2.putText(vis, task_info, (15, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return vis


def create_time_series_plot(distances, states, task_segments, current_frame,
                            per_task_thresholds):
    """Create time series plot with task segmentation."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))

    frames = np.arange(len(distances))

    # Plot 1: Raw distances with task-specific thresholds
    ax = axes[0]
    ax.plot(frames, distances * 1000, 'k-', linewidth=1, alpha=0.5, label='Distance')

    # Color background by task
    colors = plt.cm.Set3(np.linspace(0, 1, len(task_segments)))
    for i, (start, end) in enumerate(task_segments):
        ax.axvspan(start, end, alpha=0.2, color=colors[i])

        # Draw task-specific threshold
        if i < len(per_task_thresholds):
            thresh = per_task_thresholds[i]
            ax.plot([start, end], [thresh * 1000, thresh * 1000],
                   color=colors[i], linewidth=3, label=f'Task {i+1} threshold')

    ax.axvline(current_frame, color='orange', linestyle='-', linewidth=2,
               label='Current frame')

    ax.set_ylabel('Distance (mm)')
    ax.set_title('Raw Gripper Distance with Per-Task Thresholds')
    ax.legend(fontsize=7, loc='upper right', ncol=3)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, len(distances))

    # Plot 2: Binary states
    ax = axes[1]
    ax.fill_between(frames, 0, states, step='post', alpha=0.7, color='green',
                    label='OPEN')
    ax.fill_between(frames, 0, ~states, step='post', alpha=0.7, color='red',
                    label='CLOSED')

    # Mark task boundaries
    for start, end in task_segments:
        ax.axvline(start, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.axvline(end, color='black', linestyle='--', linewidth=1, alpha=0.5)

    ax.axvline(current_frame, color='orange', linestyle='-', linewidth=2)
    ax.set_xlabel('Frame')
    ax.set_ylabel('State')
    ax.set_title('Binary Gripper State (Per-Task Thresholds)')
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['CLOSED', 'OPEN'])
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3, axis='x')
    ax.set_xlim(0, len(distances))
    ax.set_ylim(-0.1, 1.1)

    plt.tight_layout()

    # Convert to image
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    plot_img = buf.reshape(canvas.get_width_height()[::-1] + (4,))
    plot_img = cv2.cvtColor(plot_img, cv2.COLOR_RGBA2RGB)

    plt.close(fig)

    return plot_img


def main():
    parser = argparse.ArgumentParser(
        description="Visualize gripper with per-task adaptive thresholds"
    )
    parser.add_argument("--zarr-path", required=True, help="Path to zarr episode")
    parser.add_argument("--output", default="per_task_gripper.mp4", help="Output video")
    parser.add_argument("--fps", type=int, default=30, help="Output FPS")
    parser.add_argument("--method", choices=['simple', 'hybrid'], default='simple',
                       help="Threshold computation method")
    parser.add_argument("--gap-threshold", type=float, default=2.5,
                       help="Gap threshold (seconds) for task segmentation")
    parser.add_argument("--start-frame", type=int, default=0, help="Start frame")
    parser.add_argument("--end-frame", type=int, default=-1, help="End frame (-1 for all)")
    args = parser.parse_args()

    zarr_path = Path(args.zarr_path)
    if not zarr_path.exists():
        raise ValueError(f"Zarr path does not exist: {zarr_path}")

    print(f"Loading data from {zarr_path}...")
    store = zarr.open_group(str(zarr_path), mode='r')

    # Load data
    images_encoded = store['images.front_1'][:]
    left_keypoints = store['left.obs_keypoints'][:]
    head_pose = store['obs_head_pose'][:]
    timestamps = store['obs_rgb_timestamps_ns'][:]

    metadata = store.attrs.asdict()
    fps = metadata.get('fps', 30)

    intrinsics = load_camera_intrinsics(zarr_path)

    T = len(images_encoded)
    if args.end_frame > 0:
        T = min(T, args.end_frame)

    print(f"Processing frames {args.start_frame} to {T}...")

    # Segment into tasks
    print(f"\nSegmenting by timestamps (gap_threshold={args.gap_threshold}s)...")
    task_segments = detect_task_boundaries(timestamps, gap_threshold=args.gap_threshold, fps=fps)
    print(f"Detected {len(task_segments)} task instances:")
    for i, (start, end) in enumerate(task_segments):
        duration = (end - start) / fps
        print(f"  Task {i+1}: frames {start:5d}-{end:5d} ({end-start:4d} frames, {duration:5.1f}s)")

    # Compute all distances
    print("\nComputing gripper distances...")
    all_distances = np.zeros(T, dtype=np.float32)
    for t in range(T):
        keypoints_3d = left_keypoints[t].reshape(21, 3)
        all_distances[t] = compute_gripper_distance(keypoints_3d)

    # Compute per-task thresholds
    print(f"\nComputing per-task thresholds (method={args.method})...")
    per_task_thresholds = []
    task_membership = np.full(T, -1, dtype=int)  # -1 = gap, >= 0 = task id

    for task_id, (start, end) in enumerate(task_segments):
        task_distances = all_distances[start:end+1]
        threshold = compute_task_threshold(task_distances, method=args.method)
        per_task_thresholds.append(threshold)
        task_membership[start:end+1] = task_id

        print(f"  Task {task_id+1}: threshold = {threshold*1000:.1f}mm")

    # Classify all frames
    print("\nClassifying frames...")
    states = np.zeros(T, dtype=bool)

    for t in range(T):
        task_id = task_membership[t]
        if task_id >= 0:
            threshold = per_task_thresholds[task_id]
            states[t] = classify_binary(all_distances[t], threshold)

    # Statistics per task
    print("\nPer-task statistics:")
    for task_id, (start, end) in enumerate(task_segments):
        task_states = states[start:end+1]
        pct_open = task_states.sum() / len(task_states) * 100
        print(f"  Task {task_id+1}: {pct_open:.1f}% OPEN, {100-pct_open:.1f}% CLOSED")

    # Overall statistics
    in_task_frames = task_membership >= 0
    gap_frames = task_membership < 0

    print(f"\nOverall statistics:")
    print(f"  Frames in tasks: {in_task_frames.sum()} ({in_task_frames.sum()/T*100:.1f}%)")
    print(f"  Frames in gaps:  {gap_frames.sum()} ({gap_frames.sum()/T*100:.1f}%)")

    # Decode first image
    if isinstance(images_encoded[0], bytes):
        first_img = decode_jpeg(images_encoded[0])
    else:
        first_img = images_encoded[0]

    img_h, img_w = first_img.shape[:2]

    # Output layout: [Video] [Plot]
    video_width = img_w
    plot_width = 1200
    plot_height = 600
    info_height = 80

    output_width = video_width + plot_width
    output_height = max(img_h, plot_height) + info_height

    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, args.fps, (output_width, output_height))

    is_encoded = isinstance(images_encoded[0], bytes)

    print(f"\nGenerating video...")
    for t in range(args.start_frame, T):
        if t % 30 == 0:
            print(f"  Frame {t}/{T}...")

        # Decode image
        if is_encoded:
            img = decode_jpeg(images_encoded[t])
        else:
            img = images_encoded[t]

        # Project keypoints
        keypoints_3d = left_keypoints[t].reshape(21, 3)
        kp_2d, kp_valid = project_3d_to_2d(keypoints_3d, head_pose[t], intrinsics)

        # Determine task info
        task_id = task_membership[t]
        is_in_task = task_id >= 0
        current_task_id = task_id + 1 if is_in_task else 0

        # Draw hand skeleton
        video_frame = draw_hand_skeleton(img, kp_2d, kp_valid, states[t], is_in_task,
                                        current_task_id, len(task_segments))

        # Create plot
        plot_img = create_time_series_plot(all_distances[:T], states[:T], task_segments,
                                          t, per_task_thresholds)
        plot_img = cv2.resize(plot_img, (plot_width, plot_height))

        # Create info panel
        info_panel = np.zeros((info_height, output_width, 3), dtype=np.uint8)

        distance = all_distances[t]

        cv2.putText(info_panel, f"Frame: {t}/{T}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(info_panel, f"Distance: {distance*1000:.1f}mm", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        if is_in_task:
            threshold = per_task_thresholds[task_id]
            state_text = "OPEN" if states[t] else "CLOSED"
            state_color = (0, 255, 0) if states[t] else (255, 0, 0)

            cv2.putText(info_panel, f"Task {current_task_id}: {state_text}", (300, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, state_color, 2)
            cv2.putText(info_panel, f"Threshold: {threshold*1000:.1f}mm", (600, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        else:
            cv2.putText(info_panel, "GAP (between tasks)", (300, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (128, 128, 128), 2)

        # Compose final frame
        final_frame = np.zeros((output_height, output_width, 3), dtype=np.uint8)
        final_frame[:img_h, :video_width] = video_frame
        final_frame[:plot_height, video_width:] = plot_img
        final_frame[max(img_h, plot_height):, :] = info_panel

        # Convert RGB to BGR
        final_frame_bgr = cv2.cvtColor(final_frame, cv2.COLOR_RGB2BGR)
        out.write(final_frame_bgr)

    out.release()
    print(f"\nVideo saved to: {args.output}")
    print(f"\nSummary: Computed {len(task_segments)} separate thresholds for {len(task_segments)} tasks")


if __name__ == "__main__":
    main()
