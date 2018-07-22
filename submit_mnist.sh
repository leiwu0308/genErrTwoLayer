#!/bin/bash

#SBATCH -J 'mnist-fnn'
#SBATCH -N 1
#SBATCH --cpus-per-task=6
#SBATCH -t 20:00:00
#SBATCH --gres=gpu:1
#SBATCH -a 6-8

width=(50 100 500 1000 5000 10000 50000 100000 500000)
nepochs=(200 200 200 200 200 200 200 200 200)

python mnist.py --nepochs ${nepochs[$SLURM_ARRAY_TASK_ID]}\
         --width ${width[$SLURM_ARRAY_TASK_ID]} \
            --lr 0.1 --nsample 100 \
            --lmbd 5e-5 --batchsize 10 

