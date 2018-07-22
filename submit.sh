#!/bin/bash

#SBATCH -J 'fmnist-fnn'
#SBATCH -N 1
#SBATCH --cpus-per-task=6
#SBATCH -t 20:00:00
#SBATCH --gres=gpu:1
#SBATCH -a 0-6

width=(512 1024 2048 4096 8192 16384 32768)
nepochs=(500 300 300 300 300 300 300)

python cifar10.py --nepochs ${nepochs[$SLURM_ARRAY_TASK_ID]}\
         --width ${width[$SLURM_ARRAY_TASK_ID]} \
            --lr 0.01 \

