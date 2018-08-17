
#!/bin/bash

source activate pytorch0.4

python cifar_train.py --nepochs 100 \
                --width 1000 \
                --lr 0.001 --nsamples 1000 \
                --batch_size 100 \
                --init_fac 10 \
                --weight_decay 0.0 \
                --lmbd 1

