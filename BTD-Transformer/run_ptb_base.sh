#!/bin/bash

if [ $1 == 'train' ]; then
    echo 'Run training...'
    python train.py \
        --cuda \
        --data ../data/ptb/ \
        --dataset ptb \
        --n_layer 3 \
        --d_model 512 \
        --n_head 1 \
        --d_head 40 \
        --d_inner 2100 \
        --dropout 0.1 \
        --dropatt 0.0 \
        --optim adam \
        --lr 0.00025 \
        --warmup_step 0 \
        --max_step 20000 \
        --tgt_len 32 \
        --mem_len 32 \
        --eval_tgt_len 32 \
        --batch_size 120 \
        --gpu0_bsz 4 \
        ${@:2}
elif [ $1 == 'eval' ]; then
    echo 'Run evaluation...'
    python eval.py \sh
        --cuda \
        --data ../data/ptb/ \
        --dataset ptb \
        --tgt_len 32 \
        --mem_len 32 \
        --clamp_len 400 \
        --same_length \
        --split test \
        ${@:2}
else
    echo 'unknown argment 1'
fi
