#!/bin/bash

if [[ $1 == 'train' ]]; then
    echo 'Run training...'
    python train_upload.py \
        --cuda \
        --data ../data/wiki-103/ \
        --dataset wt103 \
        --adaptive \
        --n_layer 6 \
        --d_model 256 \
        --n_head 1 \
        --d_head 40 \
        --d_inner 2100 \
        --dropout 0.1 \
        --dropatt 0.0 \
        --optim adam \
        --lr 0.00025 \
        --max_step 200000 \
        --tgt_len 80 \
        --mem_len 0 \
        --eval_tgt_len 80 \
        --batch_size 60 \
        --gpu0_bsz 1 \
fi