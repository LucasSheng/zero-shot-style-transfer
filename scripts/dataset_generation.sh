#!/usr/bin/env bash

CUDA_ID=$1
DATASET_DIR=$2
DATASET_NAME=$3
OUTPUT_DIR=$4

CUDA_VISIBLE_DEVICES=${CUDA_ID} \
    python convert_datasets_to_tfexamples.py \
        --dataset_dir=${DATASET_DIR} \
        --dataset_name=${DATASET_NAME} \
        --output_dir=${OUTPUT_DIR}