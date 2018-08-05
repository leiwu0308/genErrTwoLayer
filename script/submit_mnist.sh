#!/bin/bash

#SBATCH -J 'mnist-fnn'
#SBATCH -N 1
#SBATCH --cpus-per-task=6
#SBATCH -t 20:00:00
#SBATCH --gres=gpu:1
#SBATCH -a 0

#width=(50 100 500 1000 5000 10000 50000 100000 500000)
#nepochs=(200 200 200 200 200 200 200 200 200)
nsamples=(50 5000 10000 50000 60000)

python mnist.py --nepochs 100 \
                --width 10000 \
                --lr 0.001 --nsamples ${nsamples[$SLURM_ARRAY_TASK_ID]} \
                --lmbd 0.0 --batch_size 50

