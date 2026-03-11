# RLDS Dataset Conversion

This repo demonstrates how to convert an existing dataset into RLDS format for X-embodiment experiment integration.
It provides an example for converting a dummy dataset to RLDS. To convert your own dataset, **fork** this repo and 
modify the example code for your dataset following the steps below.

## Installation

First create a conda environment using the provided environment.yml file (use `environment_ubuntu.yml` or `environment_macos.yml` depending on the operating system you're using):
```
conda env create -f environment_ubuntu.yml
```

Then activate the environment using:
```
conda activate rlds
```

If you want to manually create an environment, the key packages to install are `tensorflow`, 
`tensorflow_datasets`, `tensorflow_hub`, `matplotlib`, `plotly` and `wandb`.



## Converting your Own Dataset to RLDS

Before starting the processing, you need to install your 
dataset package by running `pip install -e`. For example, to use `planning_threedim_dataset`, you need to run `pip install -e planning_threedim_dataset` first.
Then, make sure that no GPUs are used during data processing (`export CUDA_VISIBLE_DEVICES=`) and inside the dataset directory, run:
```
tfds build --overwrite
```
The command line output should finish with a summary of the generated dataset (including size and number of samples). 
Please verify that this output looks as expected and that you can find the generated `tfrecord` files in `~/tensorflow_datasets/<name_of_your_dataset>`.


### Parallelizing Data Processing
By default, dataset conversion uses 10 parallel workers. If you are parsing a large dataset, you can increase the 
number of used workers by increasing `N_WORKERS` in the dataset class. Try to use slightly fewer workers than the 
number of cores in your machine (run `htop` in your command line if you don't know how many cores your machine has). 

The dataset value `MAX_PATHS_IN_MEMORY` controls how many filepaths will be processed in parallel before they get 
written to disk sequentially. As a rule of thumb, setting this value as high as possible will make dataset conversion
faster, but don't set it too high to not overflow the memory of your machine. Setting it to >10-20x the number of workers
is usually a good default. You can monitor `htop` during conversion and reduce the value in case your memory overflows.