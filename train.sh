export HCCL_CONNECT_TIMEOUT=1800
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

accelerate launch --config_file accelerate_config/deepspeed_zero3.yaml \
    mwpo.py \
    --beta 0.01 \
    --bf16 \
    --model_name_or_path HuggingFaceH4/mistral-7b-sft-beta \
    --dataset_name HuggingFaceH4/ultrafeedback_binarized \
    --output_dir {YOUR_SAVE_PATH} \
    --dataset_train_split train_prefs \
    --do_train \
    --gradient_checkpointing \
    --gradient_accumulation_steps 8 \
    --learning_rate 0.0000005 \
    --len_lambda 0.01 \
    --logging_steps 10 \
    --loss_type sigmoid \
    --lr_scheduler_type cosine \
    --max_prompt_length 512 \
    --max_length 1024 \
    --max_steps -1 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --report_to tensorboard \
    --save_strategy epoch \
    --save_total_limit 1 \
    --save_only_model \
    --torch_dtype bfloat16 \
    --weighted \
    --weight_alpha 0.6 \
    --weight_beta 0.4 \
    --use_num_logits_to_keep \
    --warmup_ratio 0.1
