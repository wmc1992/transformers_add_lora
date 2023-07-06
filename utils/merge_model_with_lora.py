import os
import torch
import argparse

import peft
from peft import PeftModel
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


parser = argparse.ArgumentParser()
parser.add_argument("--base_model", default=None, required=True,
                    type=str, help="Please specify a base_model")
parser.add_argument("--lora_model", default=None, required=True,
                    type=str, help="Please specify LoRA models to be merged (ordered); use commas to separate multiple LoRA models.")
parser.add_argument("--output_dir", default=None, required=True, type=str)
args = parser.parse_args()


def transpose(weight, fan_in_fan_out):
    return weight.T if fan_in_fan_out else weight


base_model_path = args.base_model
lora_model_path = args.lora_model
output_dir = args.output_dir

# 载入 base model
print(f"Start Load Base Model: {base_model_path}")
config = AutoConfig.from_pretrained(base_model_path)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    load_in_8bit=False,
    torch_dtype=torch.float16,
    device_map={"": "cpu"},
)
print("Base Model Load Success.")

tokenizer = AutoTokenizer.from_pretrained(lora_model_path)
if base_model.get_input_embeddings().weight.size(0) != len(tokenizer):
    base_model.resize_token_embeddings(len(tokenizer))
    print(f"Extended vocabulary size to {len(tokenizer)}")



if hasattr(peft.LoraModel, "merge_and_unload") and config.model_type != "gpt2":
    # 载入 lora model
    print(f"Start Load LoRA Model: {lora_model_path}")
    lora_model = PeftModel.from_pretrained(
        base_model,
        lora_model_path,
        device_map={"": "cpu"},
        torch_dtype=torch.float16,
    )
    print("LoRA Model Load Success.")

    # 调用 LoraModel 的 merge_and_unload() 进行合并
    print("Merge LoRA Model With Peft merge_and_unload() Function")
    base_model = lora_model.merge_and_unload()
else:
    print(f"Start Load LoRA Model: {lora_model_path}")
    lora_model_sd = torch.load(os.path.join(lora_model_path, "adapter_model.bin"), map_location="cpu")
    print("LoRA Model Load Success.")

    base_model_sd = base_model.state_dict()

    lora_config = peft.LoraConfig.from_pretrained(lora_model_path)
    lora_scaling = lora_config.lora_alpha / lora_config.r
    fan_in_fan_out = lora_config.fan_in_fan_out
    lora_keys = [k for k in lora_model_sd if "lora_A" in k]
    non_lora_keys = [k for k in lora_model_sd if not "lora_" in k]

    for k in non_lora_keys:
        print(f"Merging No LoRA Layer: {k}")
        original_k = k.replace("base_model.model.", "")
        base_model_sd[original_k].copy_(lora_model_sd[k])

    for k in lora_keys:
        print(f"Merging LoRA Layer: {k}")
        original_key = k.replace(".lora_A", "").replace("base_model.model.", "")
        assert original_key in base_model_sd

        lora_a_key = k
        lora_b_key = k.replace("lora_A", "lora_B")

        x = torch.matmul(lora_model_sd[lora_b_key].float(), lora_model_sd[lora_a_key].float())
        x = transpose(x, fan_in_fan_out) * lora_scaling
        base_model_sd[original_key] += x

        assert base_model_sd[original_key].dtype == torch.float16

os.makedirs(output_dir, exist_ok=True)
torch.save(base_model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
print("Model Save Success:", os.path.join(output_dir, "pytorch_model.bin"))

base_model.config.to_json_file(os.path.join(output_dir, "config.json"))
print("Model Config Save Success:", os.path.join(output_dir, "config.json"))

tokenizer.save_pretrained(output_dir)
print("Tokenizer Save Success.")
