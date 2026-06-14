# MolmoAct2 Bimanual YAM RLDS Builder

This builder converts `allenai/MolmoAct2-BimanualYAM-Dataset` from LeRobot v3
layout to RLDS/TFDS for ego-lap.

The builder expects a local Hugging Face snapshot directory:

```bash
export MOLMOACT2_YAM_RAW_DIR=/path/to/MolmoAct2-BimanualYAM-Dataset
cd molmoact2_yam_dataset
tfds build --overwrite --data_dir=/path/to/tensorflow_datasets
```

Smoke runs can limit the number of source parquet/video file triplets:

```bash
export MOLMOACT2_YAM_MAX_FILES=1
export MOLMOACT2_YAM_MAX_EPISODES=4
```

Some upstream MP4s are one frame shorter than the corresponding parquet
episode tail. The builder truncates only a small missing suffix by default:

```bash
export MOLMOACT2_YAM_MAX_TRAILING_MISSING_ROWS=30
export MOLMOACT2_YAM_MAX_TRAILING_MISSING_FRACTION=0.05
```
