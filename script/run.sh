
#!/bin/bash

source activate pytorch0.4

python mnist_train.py --nepochs 2000 \
                --width 10000 \
                --lr 0.01 --nsamples 100 \
                --batch_size 25 \
                --init_fac 0.01 \
                --weight_decay 0.0 \
                --lmbd 1 \
                --device cuda:2 \
                --dir checkpoints
