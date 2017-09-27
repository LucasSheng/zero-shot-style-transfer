#!/usr/bin/env bash

MODEL_TYPE=$1
CUDA_ID=$2
LEARNING_RATE=$3

DATASET_DIR=/DATA/lsheng/lsheng_data/MSCOCO
CONFIG_DIR=/home/lsheng/lsheng_models/zero-shot-style-transfer/configs

MODEL_DIR=/DATA/lsheng/lsheng_model_checkpoints/zero-shot-style-transfer/${MODEL_TYPE}
TRAIN_DIR=${MODEL_DIR}/train

CUDA_VISIBLE_DEVICES=${CUDA_ID} \
    python train_autoencoder.py \
        --train_dir=${TRAIN_DIR} \
        --model_config=${CONFIG_DIR}/${MODEL_TYPE}_config.yml \
        --dataset_dir=${DATASET_DIR} \
        --dataset_name=MSCOCO \
        --dataset_split_name=train \
        --batch_size=4 \
        --max_number_of_step=120000 \
        --optimizer=adam \
        --learning_rate_decay_type=fixed \
        --learning_rate=${LEARNING_RATE}
