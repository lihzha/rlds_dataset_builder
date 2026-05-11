import io
import json
from pathlib import Path
from typing import Any, Iterator, Tuple

import cv2
import numpy as np
import tensorflow_datasets as tfds
import zarr
from PIL import Image
from scipy.ndimage import uniform_filter1d

from mecka_dataset.conversion_utils import MultiThreadedDatasetBuilder


# Keypoint indices for gripper distance computation
THUMB_TIP_IDX = 7
INDEX_TIP_IDX = 10


def decode_jpeg(data):
    """Decode JPEG data from zarr to numpy array."""
    # Unwrap nested numpy arrays
    while isinstance(data, np.ndarray) and data.ndim == 0:
        data = data.item()

    # Convert to bytes if needed
    if isinstance(data, bytes):
        arr = np.frombuffer(data, dtype=np.uint8)
    elif isinstance(data, np.ndarray) and data.dtype == np.uint8:
        arr = data
    else:
        raise ValueError(f"Unknown data type: {type(data)}")

    # Decode JPEG
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode JPEG")
    # Convert BGR to RGB
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def compute_gripper_distance(keypoints_3d):
    """Compute distance between thumb and index finger tips.

    Args:
        keypoints_3d: Array of shape (21, 3) containing 3D keypoint positions

    Returns:
        float: Distance in meters between thumb tip and index finger tip
    """
    thumb_tip = keypoints_3d[THUMB_TIP_IDX]
    index_tip = keypoints_3d[INDEX_TIP_IDX]
    return np.linalg.norm(thumb_tip - index_tip)


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


def load_annotations_from_zarr(store):
    """Load language annotations from zarr store.

    Args:
        store: Zarr group containing annotations

    Returns:
        List of annotation dictionaries with 'text', 'start_idx', 'end_idx'
    """
    if 'annotations' not in store:
        return []

    annotations_raw = store['annotations'][:]
    annotations = []

    for ann_data in annotations_raw:
        if isinstance(ann_data, bytes):
            ann_dict = json.loads(ann_data.decode('utf-8'))
            annotations.append(ann_dict)
        elif isinstance(ann_data, dict):
            annotations.append(ann_data)

    # Sort by start_idx to ensure chronological order
    annotations.sort(key=lambda x: x['start_idx'])
    return annotations


def convert_annotations_to_segments(annotations, total_frames):
    """Convert language annotations to task segments.

    Args:
        annotations: List of annotation dicts with 'text', 'start_idx', 'end_idx'
        total_frames: Total number of frames in the episode

    Returns:
        List of tuples: [(start_frame, end_frame, annotation_text), ...]
    """
    if not annotations:
        # No annotations - treat entire episode as one segment
        return [(0, total_frames - 1, "")]

    segments = []
    for ann in annotations:
        start = ann['start_idx']
        end = ann['end_idx']
        text = ann.get('text', '')

        # Ensure indices are within bounds
        start = max(0, min(start, total_frames - 1))
        end = max(0, min(end, total_frames - 1))

        if end >= start:
            segments.append((start, end, text))

    return segments


def _generate_examples(paths) -> Iterator[Tuple[str, Any]]:
    """Yields episodes for list of zarr directory paths."""

    # Debug: log what we received
    if not isinstance(paths, list):
        print(f"ERROR: Expected list of paths, got {type(paths)}: {paths}")
        return

    # Filter out any invalid paths
    valid_paths = [p for p in paths if p and isinstance(p, str) and len(p) > 0]
    if len(valid_paths) != len(paths):
        print(f"Warning: Filtered {len(paths) - len(valid_paths)} invalid paths from batch")

    def _parse_example(zarr_path):
        """Parse episodes from a single zarr directory.

        zarr_path format: /path/to/data/aria_fold_clothes/YYYY-MM-DD-HH-MM-SS-XXXXXX
        """
        # Validate path
        if not zarr_path or not isinstance(zarr_path, (str, Path)):
            print(f"Warning: Invalid path type: {type(zarr_path)}, value: {zarr_path}")
            return

        zarr_path = Path(zarr_path)

        # Check if path exists and is a directory
        if not zarr_path.exists():
            print(f"Warning: Path does not exist: {zarr_path}")
            return

        if not zarr_path.is_dir():
            print(f"Warning: Path is not a directory: {zarr_path}")
            return

        # Load zarr store with retry (zarr v3 can have multiprocessing issues)
        import time
        max_retries = 3
        store = None
        for attempt in range(max_retries):
            try:
                # Use zarr.open_group with explicit string path for better zarr v3 compatibility
                store = zarr.open_group(str(zarr_path), mode='r')
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(0.1 * (attempt + 1))  # Exponential backoff
                    continue
                else:
                    print(f"Warning: Failed to open zarr at {zarr_path} after {max_retries} attempts: {e}")
                    return

        if store is None:
            return

        # Load metadata
        try:
            metadata = store.attrs.asdict()
            task_name = metadata.get('task_description')
            fps = metadata.get('fps', 30)
        except Exception as e:
            print(f"Warning: Failed to load metadata from {zarr_path}: {e}")
            return

        # Load all observations
        try:
            images = store['images.front_1'][:]
            head_pose = store['obs_head_pose'][:]

            # Determine sequence length from images
            seq_len = images.shape[0]

            # Load hand data with availability tracking
            # Left hand data
            left_hand_available = True
            try:
                left_ee_pose = store['left.obs_ee_pose'][:]
                left_keypoints = store['left.obs_keypoints'][:]
                left_wrist_pose = store['left.obs_wrist_pose'][:]

                # Validate shapes
                if len(left_ee_pose) != seq_len or len(left_keypoints) != seq_len or len(left_wrist_pose) != seq_len:
                    print(f"Warning: Left hand data length mismatch in {zarr_path}, padding with zeros")
                    left_hand_available = False
            except (KeyError, Exception) as e:
                print(f"Info: Left hand data not available in {zarr_path}: {e}")
                left_hand_available = False

            # Pad left hand with zeros if missing
            if not left_hand_available:
                left_ee_pose = np.zeros((seq_len, 7), dtype=np.float32)
                left_keypoints = np.zeros((seq_len, 63), dtype=np.float32)
                left_wrist_pose = np.zeros((seq_len, 7), dtype=np.float32)

            # Right hand data
            right_hand_available = True
            try:
                right_ee_pose = store['right.obs_ee_pose'][:]
                right_keypoints = store['right.obs_keypoints'][:]
                right_wrist_pose = store['right.obs_wrist_pose'][:]

                # Validate shapes
                if len(right_ee_pose) != seq_len or len(right_keypoints) != seq_len or len(right_wrist_pose) != seq_len:
                    print(f"Warning: Right hand data length mismatch in {zarr_path}, padding with zeros")
                    right_hand_available = False
            except (KeyError, Exception) as e:
                print(f"Info: Right hand data not available in {zarr_path}: {e}")
                right_hand_available = False

            # Pad right hand with zeros if missing
            if not right_hand_available:
                right_ee_pose = np.zeros((seq_len, 7), dtype=np.float32)
                right_keypoints = np.zeros((seq_len, 63), dtype=np.float32)
                right_wrist_pose = np.zeros((seq_len, 7), dtype=np.float32)

            # Optionally load eye gaze if available
            # eye_gaze = None
            # if 'obs_eye_gaze' in store:
            #     eye_gaze = store['obs_eye_gaze'][:]
        except Exception as e:
            print(f"Warning: Failed to load data arrays from {zarr_path}: {e}")
            return

        # Load language annotations and convert to task segments
        # MECKA dataset uses dense language annotations for task segmentation
        annotations = load_annotations_from_zarr(store)
        task_segments = convert_annotations_to_segments(annotations, seq_len)

        # Check if images are encoded
        is_encoded = isinstance(images[0], (bytes, np.void)) or (
            isinstance(images[0], np.ndarray) and images[0].dtype == np.uint8 and images[0].ndim == 1
        )

        # Process each task segment
        for seg_idx, (start_frame, end_frame, annotation_text) in enumerate(task_segments):
            num_frames = end_frame - start_frame + 1

            # Extract segment data
            seg_images = images[start_frame:end_frame+1]
            seg_left_ee = left_ee_pose[start_frame:end_frame+1]
            seg_right_ee = right_ee_pose[start_frame:end_frame+1]
            seg_left_kp = left_keypoints[start_frame:end_frame+1]
            seg_right_kp = right_keypoints[start_frame:end_frame+1]
            seg_left_wrist = left_wrist_pose[start_frame:end_frame+1]
            seg_right_wrist = right_wrist_pose[start_frame:end_frame+1]
            seg_head = head_pose[start_frame:end_frame+1]

            # seg_eye_gaze = None
            # if eye_gaze is not None:
            #     seg_eye_gaze = eye_gaze[start_frame:end_frame+1]

            # Compute all gripper distances for this segment
            left_gripper_distances = np.zeros(num_frames, dtype=np.float32)
            right_gripper_distances = np.zeros(num_frames, dtype=np.float32)

            # Only compute gripper distances for available hands
            if left_hand_available:
                for i in range(num_frames):
                    left_kp_3d = seg_left_kp[i].reshape(21, 3)
                    left_gripper_distances[i] = compute_gripper_distance(left_kp_3d)

            if right_hand_available:
                for i in range(num_frames):
                    right_kp_3d = seg_right_kp[i].reshape(21, 3)
                    right_gripper_distances[i] = compute_gripper_distance(right_kp_3d)

            # Compute thresholds for both methods and both hands (only if hand is available)
            if left_hand_available:
                left_thresh_simple = compute_task_threshold(left_gripper_distances, method='simple')
                left_thresh_hybrid = compute_task_threshold(left_gripper_distances, method='hybrid')
            else:
                left_thresh_simple = 0.0
                left_thresh_hybrid = 0.0

            if right_hand_available:
                right_thresh_simple = compute_task_threshold(right_gripper_distances, method='simple')
                right_thresh_hybrid = compute_task_threshold(right_gripper_distances, method='hybrid')
            else:
                right_thresh_simple = 0.0
                right_thresh_hybrid = 0.0

            # Decode and resize images to 224x224
            decoded_images = []
            for i in range(num_frames):
                if is_encoded:
                    img = decode_jpeg(seg_images[i])
                else:
                    img = seg_images[i]

                # Resize to 224x224 using PIL
                img_pil = Image.fromarray(img)
                img_resized = img_pil.resize((224, 224), Image.BILINEAR)
                decoded_images.append(np.array(img_resized, dtype=np.uint8))

            # Build episode steps
            episode = []
            for i in range(num_frames):
                # Get precomputed gripper distances
                left_gripper_dist = left_gripper_distances[i]
                right_gripper_dist = right_gripper_distances[i]

                # Classify binary gripper states using both methods
                left_binary_simple = classify_binary(left_gripper_dist, left_thresh_simple)
                left_binary_hybrid = classify_binary(left_gripper_dist, left_thresh_hybrid)
                right_binary_simple = classify_binary(right_gripper_dist, right_thresh_simple)
                right_binary_hybrid = classify_binary(right_gripper_dist, right_thresh_hybrid)

                obs_dict = {
                    "image": decoded_images[i],
                    "hand_available": np.array([left_hand_available, right_hand_available], dtype=np.bool_),
                    "left_ee_pose": seg_left_ee[i].astype(np.float32),
                    "right_ee_pose": seg_right_ee[i].astype(np.float32),
                    "left_keypoints": seg_left_kp[i].astype(np.float32),
                    "right_keypoints": seg_right_kp[i].astype(np.float32),
                    "left_wrist_pose": seg_left_wrist[i].astype(np.float32),
                    "right_wrist_pose": seg_right_wrist[i].astype(np.float32),
                    "head_pose": seg_head[i].astype(np.float32),
                    "left_gripper_distance": np.float32(left_gripper_dist),
                    "right_gripper_distance": np.float32(right_gripper_dist),
                    "left_gripper_binary_simple": bool(left_binary_simple),
                    "left_gripper_binary_hybrid": bool(left_binary_hybrid),
                    "right_gripper_binary_simple": bool(right_binary_simple),
                    "right_gripper_binary_hybrid": bool(right_binary_hybrid),
                }

                # if seg_eye_gaze is not None:
                #     obs_dict["eye_gaze"] = seg_eye_gaze[i].astype(np.float32)

                episode.append(
                    {
                        "observation": obs_dict,
                        "action": np.zeros(14, dtype=np.float32),  # Placeholder for bimanual actions
                        "discount": 1.0,
                        "reward": float(i == (num_frames - 1)),
                        "is_first": i == 0,
                        "is_last": i == (num_frames - 1),
                        "is_terminal": i == (num_frames - 1),
                        "language_instruction": task_name if task_name else "",
                        "subtask": annotation_text if annotation_text else "",
                    }
                )

            # Create output data sample
            sample = {
                "steps": episode,
                "episode_metadata": {
                    "file_path": str(zarr_path),
                    "recording_name": zarr_path.name,
                    "segment_index": seg_idx,
                    "num_segments": len(task_segments),
                    "annotation_text": annotation_text,
                },
            }

            # Return with a unique key
            yield f"{zarr_path.name}_seg{seg_idx}", sample

    # Parse examples from valid paths
    for zarr_path in valid_paths:
        yield from _parse_example(zarr_path)


class MeckaDatasetBase(MultiThreadedDatasetBuilder):
    """Base class for Mecka bimanual manipulation dataset with shared logic."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
    }
    N_WORKERS = 15  # number of parallel workers (reduced to avoid OOM and zarr v3 multiprocessing issues)
    MAX_PATHS_IN_MEMORY = (
        15  # number of paths converted & stored in memory before writing to disk
    )
    PARSE_FCN = (
        _generate_examples  # handle to parse function from file paths to RLDS episodes
    )

    # Subclasses should override these to specify which subset to build
    SUBSET_START_IDX = None  # Starting index (inclusive)
    SUBSET_END_IDX = None    # Ending index (exclusive)

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            disable_shuffling=True,
            features=tfds.features.FeaturesDict(
                {
                    "steps": tfds.features.Dataset(
                        {
                            "observation": tfds.features.FeaturesDict(
                                {
                                    "image": tfds.features.Image(
                                        shape=(224, 224, 3),
                                        dtype=np.uint8,
                                        encoding_format="jpeg",
                                        doc="First-person RGB camera observation.",
                                    ),
                                    "hand_available": tfds.features.Tensor(
                                        shape=(2,),
                                        dtype=np.bool_,
                                        doc="Hand data availability [left_hand, right_hand]. True if hand data is available, False if padded with zeros.",
                                    ),
                                    "left_ee_pose": tfds.features.Tensor(
                                        shape=(7,),
                                        dtype=np.float32,
                                        doc="Left hand end-effector pose [x, y, z, qw, qx, qy, qz]. Position in meters, orientation as quaternion. Zero-padded if left hand not available.",
                                    ),
                                    "right_ee_pose": tfds.features.Tensor(
                                        shape=(7,),
                                        dtype=np.float32,
                                        doc="Right hand end-effector pose [x, y, z, qw, qx, qy, qz]. Position in meters, orientation as quaternion. Zero-padded if right hand not available.",
                                    ),
                                    "left_keypoints": tfds.features.Tensor(
                                        shape=(63,),
                                        dtype=np.float32,
                                        doc="Left hand keypoints (21 points × 3 coordinates). 3D positions in meters. Zero-padded if left hand not available.",
                                    ),
                                    "right_keypoints": tfds.features.Tensor(
                                        shape=(63,),
                                        dtype=np.float32,
                                        doc="Right hand keypoints (21 points × 3 coordinates). 3D positions in meters. Zero-padded if right hand not available.",
                                    ),
                                    "left_wrist_pose": tfds.features.Tensor(
                                        shape=(7,),
                                        dtype=np.float32,
                                        doc="Left wrist pose [x, y, z, qw, qx, qy, qz]. Position in meters, orientation as quaternion. Zero-padded if left hand not available.",
                                    ),
                                    "right_wrist_pose": tfds.features.Tensor(
                                        shape=(7,),
                                        dtype=np.float32,
                                        doc="Right wrist pose [x, y, z, qw, qx, qy, qz]. Position in meters, orientation as quaternion. Zero-padded if right hand not available.",
                                    ),
                                    "head_pose": tfds.features.Tensor(
                                        shape=(7,),
                                        dtype=np.float32,
                                        doc="Head/camera pose [x, y, z, qw, qx, qy, qz]. Position in meters, orientation as quaternion.",
                                    ),
                                    "left_gripper_distance": tfds.features.Scalar(
                                        dtype=np.float32,
                                        doc="Distance between left hand thumb tip and index finger tip in meters. Indicates gripper openness. Zero if left hand not available.",
                                    ),
                                    "right_gripper_distance": tfds.features.Scalar(
                                        dtype=np.float32,
                                        doc="Distance between right hand thumb tip and index finger tip in meters. Indicates gripper openness. Zero if right hand not available.",
                                    ),
                                    "left_gripper_binary_simple": tfds.features.Scalar(
                                        dtype=np.bool_,
                                        doc="Binary left gripper state using simple percentile method. True=OPEN, False=CLOSED. Threshold computed per task segment. False if left hand not available.",
                                    ),
                                    "left_gripper_binary_hybrid": tfds.features.Scalar(
                                        dtype=np.bool_,
                                        doc="Binary left gripper state using hybrid method with sustained event detection. True=OPEN, False=CLOSED. Threshold computed per task segment. False if left hand not available.",
                                    ),
                                    "right_gripper_binary_simple": tfds.features.Scalar(
                                        dtype=np.bool_,
                                        doc="Binary right gripper state using simple percentile method. True=OPEN, False=CLOSED. Threshold computed per task segment. False if right hand not available.",
                                    ),
                                    "right_gripper_binary_hybrid": tfds.features.Scalar(
                                        dtype=np.bool_,
                                        doc="Binary right gripper state using hybrid method with sustained event detection. True=OPEN, False=CLOSED. Threshold computed per task segment. False if right hand not available.",
                                    ),
                                    # "eye_gaze": tfds.features.Tensor(
                                    #     shape=(3,),
                                    #     dtype=np.float32,
                                    #     doc="Eye gaze direction vector [x, y, z]. 3D normalized direction vector.",
                                    # ),
                                }
                            ),
                            "action": tfds.features.Tensor(
                                shape=(14,),
                                dtype=np.float32,
                                doc="Placeholder for bimanual actions (2 arms × 7D). Currently zeros as this is teleoperation data.",
                            ),
                            "discount": tfds.features.Scalar(
                                dtype=np.float32,
                                doc="Discount if provided, default to 1.",
                            ),
                            "reward": tfds.features.Scalar(
                                dtype=np.float32,
                                doc="Reward if provided, 1 on final step for demos.",
                            ),
                            "is_first": tfds.features.Scalar(
                                dtype=np.bool_, doc="True on first step of the episode."
                            ),
                            "is_last": tfds.features.Scalar(
                                dtype=np.bool_, doc="True on last step of the episode."
                            ),
                            "is_terminal": tfds.features.Scalar(
                                dtype=np.bool_,
                                doc="True on last step of the episode if it is a terminal step, True for demos.",
                            ),
                            "language_instruction": tfds.features.Text(
                                doc="Global task description from metadata (e.g., 'fold clothes')."
                            ),
                            "subtask": tfds.features.Text(
                                doc="Dense annotation for this specific task segment (e.g., 'pick up the shirt'). Empty string if no dense annotations available."
                            ),
                        }
                    ),
                    "episode_metadata": tfds.features.FeaturesDict(
                        {
                            "file_path": tfds.features.Text(
                                doc="Path to the original zarr directory."
                            ),
                            "recording_name": tfds.features.Text(
                                doc="Name of the recording (timestamp)."
                            ),
                            "segment_index": tfds.features.Scalar(
                                dtype=np.int32, doc="Segment index within recording."
                            ),
                            "num_segments": tfds.features.Scalar(
                                dtype=np.int32, doc="Total number of segments in recording."
                            ),
                            "annotation_text": tfds.features.Text(
                                doc="Dense language annotation text for this task segment."
                            ),
                        }
                    ),
                }
            )
        )

    def _split_paths(self):
        """Define filepaths for data splits."""
        # Path to mecka zarr data
        base_dir = Path("/n/fs/robot-data/EgoVerse/data/mecka")

        if not base_dir.exists():
            print(f"Warning: Dataset path does not exist: {base_dir}")
            return {"train": []}

        # Find all zarr episode directories (timestamped directories)
        # Filter for directories that:
        # 1. Are directories
        # 2. Don't start with '.' (hidden files)
        # 3. Have the timestamp format (YYYY-MM-DD-HH-MM-SS-XXXXXX)
        # 4. Contain a zarr.json file (valid zarr directory)
        zarr_dirs = []
        for d in sorted(base_dir.iterdir()):
            if not d.is_dir():
                continue
            if d.name.startswith('.'):
                continue
            # Check if it has zarr.json (valid zarr directory)
            if not (d / 'zarr.json').exists():
                continue
            zarr_dirs.append(str(d))

        # Apply subset filtering if specified by subclass
        if self.SUBSET_START_IDX is not None:
            total_count = len(zarr_dirs)
            zarr_dirs = zarr_dirs[self.SUBSET_START_IDX:self.SUBSET_END_IDX]
            end_idx_display = self.SUBSET_END_IDX - 1 if self.SUBSET_END_IDX is not None else total_count - 1
            print(f"Subset filter: Using {len(zarr_dirs)} out of {total_count} recordings (indices {self.SUBSET_START_IDX}-{end_idx_display})")
        else:
            print(f"Found {len(zarr_dirs)} valid zarr recordings in {base_dir}")

        print(f"Each recording may contain multiple task instances (segmented by language annotations)")

        if len(zarr_dirs) == 0:
            print("Warning: No valid zarr directories found!")

        return {
            "train": zarr_dirs,
        }


class MeckaDataset(MeckaDatasetBase):
    """DatasetBuilder for Mecka bimanual manipulation dataset (full dataset)."""
    pass


class MeckaDatasetPart1(MeckaDatasetBase):
    """DatasetBuilder for Mecka dataset - Part 1 (~37K samples, indices 0-10402)."""
    SUBSET_START_IDX = 0
    SUBSET_END_IDX = 10403


class MeckaDatasetPart2(MeckaDatasetBase):
    """DatasetBuilder for Mecka dataset - Part 2 (~30K samples, indices 10403-15604)."""
    SUBSET_START_IDX = 10403
    SUBSET_END_IDX = 15605


class MeckaDatasetPart3(MeckaDatasetBase):
    """DatasetBuilder for Mecka dataset - Part 3 (~30K samples, indices 15605-20805)."""
    SUBSET_START_IDX = 15605
    SUBSET_END_IDX = 20806


class MeckaDatasetPart4(MeckaDatasetBase):
    """DatasetBuilder for Mecka dataset - Part 4 (~36K samples, indices 20806-26007)."""
    SUBSET_START_IDX = 20806
    SUBSET_END_IDX = 26008


class MeckaDatasetPart5(MeckaDatasetBase):
    """DatasetBuilder for Mecka dataset - Part 5 (~36K samples, indices 26008-31208)."""
    SUBSET_START_IDX = 26008
    SUBSET_END_IDX = 31209


class MeckaDatasetPart6(MeckaDatasetBase):
    """DatasetBuilder for Mecka dataset - Part 6 (~39K samples, indices 31209-36410)."""
    SUBSET_START_IDX = 31209
    SUBSET_END_IDX = 36411


class MeckaDatasetPart7(MeckaDatasetBase):
    """DatasetBuilder for Mecka dataset - Part 7 (~39K samples, indices 36411-41612)."""
    SUBSET_START_IDX = 36411
    SUBSET_END_IDX = None  # None means until the end
