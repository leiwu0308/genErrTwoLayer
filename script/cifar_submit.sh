#!/bin/bash


#SBATCH -J 'mnist_lmbd'
#SBATCH -N 1
#SBATCH --cpus-per-task=2
#SBATCH -t 20:00:00
#SBATCH --gres=gpu:1
#SBATCH -a 0-4
#SBATCH -o ./logs/mnist_%A_%a.out

source activate pytorch0.4
which python

nsamples=(10 100 500 1000 5000)
nepochs=(10000 10000 10000 10000 5000)

python cifar_train.py --nepochs ${nepochs[$SLURM_ARRAY_TASK_ID]} \
                --width 10000 --nsamples ${nsamples[$SLURM_ARRAY_TASK_ID]} \
                --lr 0.005 --init_fac 1 \
                --lmbd 1 \
                --weight_decay 0.00 \
                --batch_size 100

