# CS 578 Final Project

## Reinforced Meta Active Learning with Maximally Interfered Retrieval

### Members: Zach Lafeer, Rashmi Bhaskara, and Maurice Chiu

# How to run
## Dependencies
Create an environment:
```sh
conda create --name 578 python=3.10 -y
conda activate 578
```
Install the dependencies:
Make sure you are in the `PROJECT_ROOT_DIR/ocl` directory and do:
```sh
pip install -r requirements.txt
```

## Data preparation
- Mini-ImageNet: Download from https://www.kaggle.com/whitemoon/miniimagenet/download, and place it under `PROJECT_ROOT_DIR/ocl/datasets/mini_imagenet/`
- NonStationary-MiniImageNet will be generated on the fly

## Run the program
Make sure you are in the `PROJECT_ROOT_DIR/ocl` directory and do:
```sh
bash run_tests.sh
```
The output will be saved under `PROJECT_ROOT_DIR/logs`.
Each test takes about 2-8.5 hours to complete.