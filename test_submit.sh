#!/bin/bash

#SBATCH -J 'test'
#SBATCH -N 1
#SBATCH --cpus-per-task=6
#SBATCH -t 20:00:00
#SBATCH --gres=gpu:1
#SBATCH -a 4-9

width=(64 128 256 512 1024 2048 4096 8192 16384 32768)

python test.py --width ${width[$SLURM_ARRAY_TASK_ID]} 

