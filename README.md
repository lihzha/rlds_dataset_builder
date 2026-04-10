# RLDS Dataset Conversion

This repo demonstrates how to convert an existing dataset into RLDS format for LAP training.

## Installation

First create a conda environment using the provided environment.yml file (use `environment_ubuntu.yml`):
```
conda env create -f environment_ubuntu.yml
```

Then activate the environment using:
```
conda activate rlds
```

## Converting your Own Dataset to RLDS

To convert your own dataset, you can follow an example dataset folder `aria_dataset`. Follow the steps below:

1. **Rename Dataset**: Change the name of the dataset folder from `aria_dataset` to the name of your dataset (e.g. egodex_dataset), 
also change the name of `aria_dataset_dataset_builder.py` by replacing `aria_dataset` with your dataset's name (e.g. egodex_dataset_dataset_builder.py)
and change the class name `AriaDataset` in the same file to match your dataset's name, using camel case instead of underlines (e.g. EgodexDataset).

2. **Modify Features**: Modify the `*_dataset_builder.py` to match your dataset's spec and loading.


That's it! You're all set to run dataset conversion. Before starting the processing, you need to install your 
dataset package by adding the name of your dataset in `setup.py` and running `pip install -e`.
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



## Upload Your Data

We provide a Google Cloud bucket that you can upload your data to. First, install `gsutil`, the Google cloud command 
line tool. You can follow the installation instructions [here](https://cloud.google.com/storage/docs/gsutil_install).

Next, authenticate your Google account with:
```
gcloud auth login
``` 
This will open a browser window that allows you to log into your Google account (if you're on a headless server, 
you can add the `--no-launch-browser` flag). Ideally, use the email address that
you used to communicate with Karl, since he will automatically grant permission to the bucket for this email address. 
If you want to upload data with a different email address / google account, please shoot Karl a quick email to ask 
to grant permissions to that Google account!

After logging in with a Google account that has access permissions, you can upload your data with the following 
command:
```
gsutil -m cp -r ~/tensorflow_datasets/<name_of_your_dataset> gs://pi0-cot/OXE
``` 
This will upload all data using multiple threads. If your internet connection gets interrupted anytime during the upload
you can just rerun the command and it will resume the upload where it was interrupted. You can verify that the upload
was successful by inspecting the bucket [here](https://console.cloud.google.com/storage/browser/pi0-cot).