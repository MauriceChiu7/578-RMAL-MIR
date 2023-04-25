# CS 578 Final Project

## Reinforced Meta Active Learning with Maximally Interfered Retrieval

### Members: Zach Lafeer, Rashmi Bhaskara, and Maurice Chiu

# How to run
## Dependencies
```sh
conda create --name 578 python=3.10 -y
conda activate 578
```

```sh
cd ocl/
pip install -r requirements.txt
```

## Data preparation
- Mini-ImageNet: Download from https://www.kaggle.com/whitemoon/miniimagenet/download , and place it in `./ocl/datasets/mini_imagenet/`
- NonStationary-MiniImageNet will be generated on the fly

## Run the program
Each test takes about 8.5 hours to complete.
```sh
bash run_tests.sh
```