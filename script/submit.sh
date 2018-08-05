#!/bin/bash

#SBATCH -J 'mnist-fnn'
#SBATCH -N 1
#SBATCH --cpus-per-task=2
#SBATCH -t 20:00:00
#SBATCH --gres=gpu:1
#SBATCH -a 0-3


BZ=(50 50 50 50)
python mnist_train.py --nepochs 100 \
                --width 100000 \
                --lr 0.01 --nsamples 100 \
                --lmbd 0.0 --batch_size ${BZ[$SLURM_ARRAY_TASK_ID]} \
                --initialize_factor 1

