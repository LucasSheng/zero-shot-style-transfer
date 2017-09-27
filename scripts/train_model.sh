#!/usr/bin/env bash

MODEL_TYPE=$1
STYLE_DATASET_NAME=$2
CUDA_ID=$3
LEARNING_RATE=$4

# dataset paths
CONTENT_DATASET_DIR=/DATA/lsheng/lsheng_data/MSCOCO
STYLE_DATASET_DIR=/DATA/lsheng/lsheng_data/${STYLE_DATASET_NAME}

# configuration files
CONFIG_DIR=/home/lsheng/lsheng_models/zero-shot-style-transfer/configs

# model storage paths
MODEL_DIR=/DATA/lsheng/lsheng_model_checkpoints/zero-shot-style-transfer/${MODEL_TYPE}
TRAIN_DIR=${MODEL_DIR}/train

CUDA_VISIBLE_DEVICES=${CUDA_ID} \
    python train_model.py \
        --train_dir=${TRAIN_DIR} \
        --model_config=${CONFIG_DIR}/${MODEL_TYPE}_config.yml \
        --content_dataset_dir=${CONTENT_DATASET_DIR} \
        --content_dataset_name=MSCOCO \
        --content_dataset_split_name=train \
        --style_dataset_dir=${STYLE_DATASET_DIR} \
        --style_dataset_name=${STYLE_DATASET_NAME} \
        --style_dataset_split_name=train \
        --batch_size=4 \
        --max_number_of_steps=120000 \
        --optimizer=adam \
        --learning_rate_decay_type=fixed \
        --learning_rate=${LEARNING_RATE} \
