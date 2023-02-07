#!/bin/bash
module load anaconda/2020.11
module load cuda/11.1
module load gcc/7.3

source activate mmclassification

export PYTHONUNBUFFERED=1

python tools/train.py configs/resnet/resnet_b16_cifar10.py --work-dir work_dir
