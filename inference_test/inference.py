import json
import argparse

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

parser = argparse.ArgumentParser()
parser.add_argument("--model_name_or_path", default=None, required=True, type=str)
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path).cuda()
model.eval()

s_list = [
    "德州市纪委立即成立",
    "请将下述中文文本",
    "请将下述中文",
    "请将下述",
    "请将",
]

for s in s_list:
    ids = tokenizer.encode(s)
    input_ids = torch.LongTensor([ids]).cuda()
    out = model.generate(
        input_ids=input_ids,
        max_length=150,
        do_sample=False,
        temperature=0,
    )
    out_text = tokenizer.decode(out[0])

    print("输入文本：")
    print(s)
    print("输出文本：")
    print(json.dumps(out_text, ensure_ascii=False, indent=2))
