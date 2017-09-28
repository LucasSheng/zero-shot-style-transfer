#!/usr/bin/env bash

MODEL_TYPE=$1
CUDA_ID=$2

# dataset paths, both content and style images are not trasferred to tfexample
CONTENT_DATASET_DIR=/DATA/lsheng/lsheng_data/contents_for_style_transfer
STYLE_DATASET_DIR=/DATA/lsheng/lsheng_data/source_dataset/simple

# configuration files
CONFIG_DIR=/home/lsheng/lsheng_models/zero-shot-style-transfer/configs

# model storage files
TRAIN_DIR=/DATA/lsheng/lsheng_model_checkpoints/zero-shot-style-transfer/AE/train
EVAL_DIR=/DATA/lsheng/lsheng_model_checkpoints/zero-shot-style-transfer/${MODEL_TYPE}/evaluation

CUDA_VISIBLE_DEVICES=${CUDA_ID} \
    python evaluate_model.py \
        --checkpoint_dir=${TRAIN_DIR} \
        --model_config_path=${CONFIG_DIR}/${MODEL_TYPE}_config.yml \
        --content_dataset_dir=${CONTENT_DATASET_DIR} \
        --style_dataset_dir=${STYLE_DATASET_DIR} \
        --eval_dir=${EVAL_DIR}
