# export HDF5_FILE_PATH=/home/yixuan/prbench_dir/tidybot_planner/data/sim_demos_ground_random_10k/sim_10301.hdf5
# export OLD_DATASET_NAME=planning_dataset_with_base_trial1

# 1. Build dataset
conda activate rlds
cd ~/prbench_dir/tidybot_planner/data/rlds_dataset_builder/planning_dataset
tfds build --overwrite

#2. Rename existing dataset on GCS. Read new dataset name from env variable
gsutil mv gs://pi0-cot/OXE/planning_dataset gs://pi0-cot/OXE/${OLD_DATASET_NAME}

# 3. Transfer new dataset to GCS
gsutil -m cp -r ~/tensorflow_datasets/planning_dataset gs://pi0-cot/OXE/planning_dataset
