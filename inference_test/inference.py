import argparse

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

parser = argparse.ArgumentParser()
parser.add_argument("--model_name_or_path", default=None, required=True, type=str)
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)

s = "将下述英文文本翻译为"
inputs = tokenizer(s, return_tensors="pt")
outputs = model(**inputs)
outputs = torch.argmax(outputs["logits"], dim=-1)

print("输入文本：")
print(s)
print("输出文本：")
print(tokenizer.decode(outputs[0]))
