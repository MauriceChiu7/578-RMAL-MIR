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

## Run the program:
```sh
python general_main.py --data cifar100 --cl_type nc --agent ER --retrieve MIR --update random --mem_size 5000
```