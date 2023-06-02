#!/bin/bash

VICUNA_CKPT=$1
SQA_PATH=$2
PROJECTOR_PATH=$3
OUT_DIR=$4

python llava/train/train_mem.py \
    --model_name_or_path ${VICUNA_CKPT} \
    --data_path ${SQA_PATH}/llava_train_QCM-LEPA.json \
    --image_folder ${SQA_PATH}/images/train \
    --vision_tower google/vit-large-patch16-224-in21k \
    --image_patch_size 16 \
    --pretrain_mm_mlp_adapter ${PROJECTOR_PATH} \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end \
    --bf16 True \
    --output_dir ${OUT_DIR} \
    --num_train_epochs 12 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 5000 \
    --save_total_limit 3 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --report_to wandb