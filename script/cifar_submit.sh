#!/bin/bash


#SBATCH -J 'cifar_lmbd'
#SBATCH -N 1
#SBATCH --cpus-per-task=2
#SBATCH -t 20:00:00
#SBATCH --gres=gpu:1
#SBATCH -a 0-4
#SBATCH -o ./logs/cifar_%A_%a.out

source activate pytorch0.4
which python

nsamples=(10 100 500 1000 5000)
nepochs=(10000 10000 10000 10000 10000)

python cifar_train.py --nepochs ${nepochs[$SLURM_ARRAY_TASK_ID]} \
                --width 10000 --nsamples ${nsamples[$SLURM_ARRAY_TASK_ID]} \
                --lr 0.001 --init_fac 1 \
                --lmbd 0.5 \
                --weight_decay 0.00 \
                --batch_size 100 \
                --dir checkpoints/cifar10_w10000_init1_lmbd0.5

