#!/usr/bin/env bash

MODEL_TYPE=$1
CUDA_ID=$2

STYLE_DATASET_DIR=/DATA/lsheng/lsheng_data/DTD/dtd/images
CONFIG_DIR=/home/lsheng/lsheng_models/zero-shot-style-transfer/configs

MODEL_DIR=/DATA/lsheng/lsheng_model_checkpoints/zero_shot_style_transfer/${MODEL_TYPE}
TRAIN_DIR=${MODEL_DIR}/train
EVAL_DIR=${MODEL_DIR}/evaluation_texture

CUDA_VISIBLE_DEVICES=${CUDA_ID} \
    python evaluate_texture_synthesis.py \
        --checkpoint_dir=${TRAIN_DIR} \
        --model_config_path=${CONFIG_DIR}/${MODEL_TYPE}_config.yml \
        --style_dataset_dir=${STYLE_DATASET_DIR} \
        --eval_dir=${EVAL_DIR}
