#!/usr/bin/env bash

MODEL_TYPE=$1
STYLE_DATASET_NAME=$2
CUDA_ID=$3

# dataset paths
CONTENT_DATASET_DIR=/DATA/lsheng/lsheng_data/contents_for_style_transfer
STYLE_DATASET_DIR=/DATA/lsheng/lsheng_data/${STYLE_DATASET_NAME}

# configuration files
CONFIG_DIR=/home/lsheng/lsheng_models/zero-shot-style-transfer/configs

# model storage files
MODEL_DIR=/DATA/lsheng/lsheng_model_checkpoints/zero-shot-style-transfer/${MODEL_TYPE}
TRAIN_DIR=${MODEL_DIR}/train
EVAL_DIR=${MODEL_DIR}/evaluation

CUDA_VISIBLE_DEVICES=${CUDA_ID} \
    python evaluate_model.py \
        --checkpoint_dir=${TRAIN_DIR} \
        --model_config_path=${CONFIG_DIR}/${MODEL_TYPE}_config.yml \
        --content_dataset_dir=${CONTENT_DATASET_DIR} \
        --style_dataset_dir=${STYLE_DATASET_DIR} \
        --eval_dir=${EVAL_DIR}
