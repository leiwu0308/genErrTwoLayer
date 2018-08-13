
#!/bin/bash

source activate pytorch0.4

python mnist_train.py --nepochs 5000 \
                --width 10000 \
                --lr 0.005 --nsamples 100 \
                --batch_size 100 \
                --initialize_factor 5 \
                --weight_decay 0.0 \
                --lmbd 0.2

