CUDA_VISIBLE_DEVICES=0 python3 merge_model_with_lora.py \
    --base_model /path/for/base/model \
    --lora_model /path/for/lora/model \
    --output_dir /path/to/save/merged/model
