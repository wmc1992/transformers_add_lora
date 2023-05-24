# 常规训练配置
lr=1e-4
per_device_train_batch_size=8
per_device_eval_batch_size=8
gradient_accumulation_steps=4
training_steps=3000
max_seq_length=1024

# LoRA 训练配置
lora_rank=8
lora_alpha=32
target_modules="q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj"
lora_dropout=0.05

# 数据与预训练模型的路径配置
model_name_or_path=/the/pretrained/model/name/or/path
tokenizer_name=/the/tokenizer/name/or/path # 当使用 Chinese-LLaMA-Alpaca 时其扩充了词表，需要指定该值，其他模型一般不需要指定该值
train_file_path=/the/dataset/file/path
output_dir=/the/output/directory/to/save/model/and/state

# deepspeed配置
deepspeed_config_file=ds_zero2_no_offload.json

torchrun --nnodes 1 --nproc_per_node 2 run_clm_llama_with_lora.py \
    --deepspeed ${deepspeed_config_file} \
    --model_name_or_path ${model_name_or_path} \
    --tokenizer_name ${tokenizer_name} \
    --train_file ${train_file_path} \
    --validation_split_percentage 1 \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --per_device_eval_batch_size ${per_device_eval_batch_size} \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --max_steps ${training_steps} \
    --do_train \
    --do_eval \
    --seed $RANDOM \
    --fp16 \
    --lr_scheduler_type cosine \
    --learning_rate ${lr} \
    --warmup_ratio 0.03 \
    --weight_decay 0 \
    --logging_strategy steps \
    --logging_steps 10 \
    --save_strategy steps \
    --save_total_limit 3 \
    --evaluation_strategy steps \
    --eval_steps 250 \
    --save_steps 500 \
    --preprocessing_num_workers 8 \
    --block_size ${max_seq_length} \
    --output_dir ${output_dir} \
    --overwrite_output_dir \
    --ddp_timeout 30000 \
    --logging_first_step True \
    --use_peft True \
    --lora_rank ${lora_rank} \
    --lora_alpha ${lora_alpha} \
    --target_modules ${target_modules} \
    --lora_dropout ${lora_dropout} \
    --torch_dtype float16 \
    --gradient_checkpointing \
    --ddp_find_unused_parameters False
