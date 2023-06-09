# 常规训练配置
lr=5e-5
per_device_train_batch_size=8
per_device_eval_batch_size=8
gradient_accumulation_steps=4
training_steps=3000

# 数据配置
train_file_path=/the/train/dataset/file/path
validation_file=/the/validation/dataset/file/path
max_seq_length=1024

# 预训练模型的路径配置
model_name_or_path=/the/pretrained/model/name/or/path
output_dir=/the/output/directory/to/save/model/and/state

# deepspeed配置
deepspeed_config_file=../../ds_zero2_no_offload.json

# accelerate配置
accelerate_config_file=../../accelerate_config_two_process.yaml

# 直接使用 python 启动
# CUDA_VISIBLE_DEVICES=0 python3 run_mlm.py

# 使用 deepspeed 启动
# torchrun --nnodes 1 --nproc_per_node 2 run_mlm.py \
#     --deepspeed ${deepspeed_config_file} \

# 使用 accelerate 启动
CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
    --config_file ${accelerate_config_file} \
    run_mlm.py \
    --model_name_or_path ${model_name_or_path} \
    --train_file ${train_file_path} \
    --validation_file ${validation_file} \
    --validation_split_percentage 1 \
    --max_seq_length ${max_seq_length} \
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
    --output_dir ${output_dir} \
    --overwrite_output_dir \
    --ddp_timeout 30000 \
    --logging_first_step True \
    --gradient_checkpointing \
    --ddp_find_unused_parameters False
