#!/bin/bash

##NECESSARY JOB SPECIFICATIONS

#SBATCH --job-name=soglcr_cifar10_1
#SBATCH --output=soglcr_cifar10_1.%j
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu
##SBATCH --gpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00              #Set the wall clock limit to 1hr and 30min
#SBATCH --mem=32GB                  #Request 2560MB (2.5GB) per node

python train_small.py --lr=.075 --batch-size=32 \
  --multiprocessing-distributed --world-size 1 --rank 0 --workers 32 \
  --dist-url 'tcp://localhost:51238' \
  --data_name CIFAR10 \
  --save_dir /scratch/user/sabuj.laskar/SogCLR/saved_models_cifar10/ \
  --print-freq 1
