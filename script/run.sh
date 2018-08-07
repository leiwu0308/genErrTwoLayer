
#!/bin/bash

python mnist_train.py --nepochs 5000 \
                --width 10000 \
                --lr 0.001 --nsamples 100 \
                --batch_size 100 \
                --initialize_factor 5 \
                --weight_decay 0.01 \
                --lmbd 0.0

