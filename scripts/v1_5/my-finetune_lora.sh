#!/bin/bash
# export CUDA_LAUNCH_BLOCKING=1

deepspeed  --include localhost:0 --master_port=29666 llava/train/train_xformers.py \
    --lora_enable True --lora_r 8 --lora_alpha 16 --mm_projector_lr 2e-5 \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path /mnt/data/home/lilanting/shenjie/code/Text4SegHub/checkpoints/text4seg-llava-7b-p24 \
    --version v1 \
    --data_path ./playground/data \
    --image_folder ./playground/data \
    --vision_tower /mnt/data/home/lilanting/shenjie/code/Text4SegHub/checkpoints/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio anyres \
    --group_by_modality_length True \
    --bf16 False \
    --fp16 True \
    --output_dir ./checkpoints/news/llava-v1.5-7b-lora-r64-p24 \
    --num_train_epochs 6 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate 0.0001 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length 3072 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to tensorboard