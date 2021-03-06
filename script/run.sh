
#!/bin/bash

source activate pytorch0.4

python mnist_train.py --nepochs 2000 \
                --width 10000 \
                --lr 0.001 --nsamples 100 \
                --batch_size 10 \
                --init_fac 1 \
                --weight_decay 0.0 \
                --lmbd 0.1 \
                --device cuda:2 \
                --ntries 1 \
                --dir checkpoints
