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

init=(0.01 0.1 1 5 10 30 50)
nepochs=(10000 10000 10000 10000 10000 10000 10000)

python mnist_train.py --nepochs ${nepochs[$SLURM_ARRAY_TASK_ID]} \
                --width 10000 --nsamples 100 \
                --lr 0.01 --init_fac 1 \
                --lmbd 1 \
                --weight_decay 0.00 \
                --batch_size 100 \
                --dir checkpoints/mnist_w10000_n100 \
                --ntries 3 \
