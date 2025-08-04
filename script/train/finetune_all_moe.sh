#!/bin/bash
export CUDA_LAUNCH_BLOCKING=1

MODEL_TYPE=mms-phi-3-moe

OUTPUT_DIR=$MODEL_TYPE-sft-full

mkdir -p ./checkpoints-$MODEL_TYPE/$OUTPUT_DIR

deepspeed bunny/train/train.py  \
    --deepspeed ./script/deepspeed/zero3.json \
    --model_name_or_path /path/to/base_llm_moe_model \
    --model_type $MODEL_TYPE \
    --moe_enable False \
    --version phi3 \
    --data_path ./data_zoo/data.json \
    --image_folder ./data/images \
    --vision_tower /path/to/siglip-so400m-patch14-384 \
    --use_s2 True \
    --mm_projector_type mlp2x_gelu \
    --image_aspect_ratio pad \
    --group_by_modality_length False \
    --bf16 True \
    --output_dir ./checkpoints-$MODEL_TYPE/$OUTPUT_DIR \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 0.2 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none | tee 2>&1 ./checkpoints-$MODEL_TYPE/$OUTPUT_DIR/log.txt