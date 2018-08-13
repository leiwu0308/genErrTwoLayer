#!/bin/bash


#SBATCH -J 'mnist_lmbd'
#SBATCH -N 1
#SBATCH --cpus-per-task=2
#SBATCH -t 20:00:00
#SBATCH --gres=gpu:1
#SBATCH -a 0-3

source activate pytorch0.4
which python

#wd=(1e-4 5e-4 1e-3 5e-3 1e-2 5e-2 1e-1)
nsamples=(10 50 100 500)

python mnist_train.py --nepochs 6000 \
                --width 10000 --nsamples ${nsamples[$SLURM_ARRAY_TASK_ID]} \
                --lr 0.001 --initialize_factor 5 \
                --lmbd 0.01 \
                --weight_decay 0.00 \
                --batch_size 100


