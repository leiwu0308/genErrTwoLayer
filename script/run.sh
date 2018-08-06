
#!/bin/bash

python mnist_train.py --nepochs 4000 \
                --width 10000 \
                --lr 0.001 --nsamples 100 \
                --batch_size 100 \
                --initialize_factor 10 \
                --weight_decay 1e-2 \
                --lmbd 0.0 

