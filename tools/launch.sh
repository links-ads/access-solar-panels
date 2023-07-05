#!/usr/bin/env bash

DATA_PATH=please/insert/here

# This script is used to launch the application.
CUDA_VISIBLE_DEVICES=$1 python run.py train-segmenter \
    --data-folder $DATA_PATH \
    --encoder resnet50 \
    --enc-lr 0.0005 \
    --enc-pretrained \
    --input-channels 4 \
    --loss combo \
    --optimizer adamw \
    --monitor f1 \
    --model unet \
    --scheduler cosine \
    --seed 42 \
    --trainer.batch-size 24 \
    --trainer.device cuda \
    --trainer.max-epochs 200 \
    --trainer.weight-decay 0.01 \
    --trainer.patience 20 \
    --trainer.num-workers 4 \
