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
