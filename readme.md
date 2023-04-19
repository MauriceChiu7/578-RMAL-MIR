# CS 578 Final Project

## Reinforced Meta Active Learning with Maximally Interfered Retrieval

### Zach Lafeer, Rashmi Bhaskara, and Maurice Chiu

# How to run:
## Dependencies:
```sh
conda create --name 578 python=3.10 -y
conda activate 578
```

```sh
pip install -r requirements.txt
```

## Data preparation
- CIFAR10 & CIFAR100 will be downloaded during the first run
- CORE50 download: `source fetch_data_setup.sh`
- Mini-ImageNet: Download from https://www.kaggle.com/whitemoon/miniimagenet/download , and place it in datasets/mini_imagenet/
- NonStationary-MiniImageNet will be generated on the fly

## Run the program:
```sh
python general_main.py --data mini_imagenet --cl_type ni --agent ER --retrieve MIR --update random --mem_size 5000
```