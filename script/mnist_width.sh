#!/bin/bash


#SBATCH -J 'mnist_lmbd'
#SBATCH -N 1
#SBATCH --cpus-per-task=2
#SBATCH -t 20:00:00
#SBATCH --gres=gpu:1
#SBATCH -a 0-6
#SBATCH -o ./logs/mnist_n100_init_%A_%a.out

source activate pytorch0.4
which python

width=(16 64 256 1024 4096 16384 65536)
nepochs=(5000 5000 1000 1000 500 500 500)

python mnist_train.py --nepochs ${nepochs[$SLURM_ARRAY_TASK_ID]} \
                --width ${width[$SLURM_ARRAY_TASK_ID]} --nsamples 10000 \
                --lr 0.001 --init_fac 1 \
                --lmbd 0 \
                --weight_decay 0.00 \
                --batch_size 100 \
                --dir checkpoints/mnist_n10000_lambd0_w \
                --ntries 3 \
