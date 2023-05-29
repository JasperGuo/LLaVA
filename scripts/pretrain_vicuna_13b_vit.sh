#!/bin/bash

LLM_DIR=$1
DATA_DIR=$2
OUT_DIR=$3

python llava/train/train_mem.py \
    --model_name_or_path ${LLM_DIR} \
    --data_path ${DATA_DIR}/chat.json \
    --image_folder ${DATA_DIR}/cc_subset_images \
    --vision_tower google/vit-large-patch16-224-in21k \
    --image_patch_size 16 \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end \
    --bf16 True \
    --output_dir ${OUT_DIR} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2400 \
    --save_total_limit 1 \
    --learning_rate 2e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --report_to wandb
