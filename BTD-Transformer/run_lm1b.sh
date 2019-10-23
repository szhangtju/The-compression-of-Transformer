#!/bin/bash

if [[ $1 == 'train' ]]; then
    echo 'Run training...'
    python train.py \
        --cuda \
        --data ../data/one-billion-words/ \
        --dataset lm1b \
        --adaptive \
        --n_layer 8 \
        --d_model 1024 \
        --div_val 4 \
        --n_head 1 \
        --d_head 40 \
        --d_inner 2100 \
        --dropout 0.1 \
        --dropatt 0.0 \
        --optim adam \
        --warmup_step 20000 \
        --max_step 500000 \
        --lr 0.00025 \
        --tgt_len 80 \
        --mem_len 0 \
        --eval_tgt_len 80 \
        --batch_size 60 \
        --gpu0_bsz 1 \
fi
