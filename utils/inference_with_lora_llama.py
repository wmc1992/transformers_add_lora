import torch
import jsonlines
from tqdm import tqdm
import argparse

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


parser = argparse.ArgumentParser()
parser.add_argument("--base_model", default=None, required=True,
                    type=str, help="指定 base_model 的路径")
parser.add_argument("--lora_model", default=None, required=True,
                    type=str, help="指定 LoRA_model 的路径；如果有多个 LoRA_model 则需要按照 adapter_name,model_path 的顺序进行拼接")
parser.add_argument("--multi_lora_model", default=False, type=bool, help="是否有多个 LoRA_model，默认为 False")
parser.add_argument("--data_in", default=None, required=True, type=str)
parser.add_argument("--column_of_adapter_name", default="adapter_name", type=str)
parser.add_argument("--data_out", default=None, required=True, type=str)
args = parser.parse_args()


# ----------------------------
# 载入模型
# ----------------------------
model_path = args.base_model
model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    trust_remote_code=True, 
    device_map="auto",
    torch_dtype=torch.float16,
)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
print("预训练模型载入成功")

if not args.multi_lora_model:  # 只有一个lora模型
    lora_model_path = args.lora_model
    model = PeftModel.from_pretrained(model, lora_model_path)

else:  # 有多个lora模型
    multi_lora_models = [x.strip() for x in args.lora_model.split(",") if len(x.strip()) > 0]
    assert len(multi_lora_models) % 2 == 0

    adapter_name_to_model_path = {}
    for idx in range(0, len(multi_lora_models), 2):
        adapter_name_to_model_path[multi_lora_models[idx]] = multi_lora_models[idx + 1]

    for idx, (adapter_name, model_path) in enumerate(adapter_name_to_model_path.items()):
        print("载入LoRA模型，adapter_name:", adapter_name, " model_path:", model_path)
        if idx == 0:
            model = PeftModel.from_pretrained(model, model_path, adapter_name=adapter_name)
        else:
            model.load_adapter(model_path, adapter_name=adapter_name)
model.eval()
print("LoRA模型载入成功")


# ----------------------------
# 载入待预测数据
# ----------------------------
r_data_path = args.data_in
with jsonlines.open(r_data_path) as f:
    data_list = list(f)
print("待预测数据集载入成功")


# ----------------------------
# 推理并直接写入到文件中
# ----------------------------
w_data_path = args.data_out
with torch.no_grad():
    with jsonlines.open(w_data_path, "w") as f:
        for data in tqdm(data_list, total=len(data_list)):
            if args.multi_lora_model:
                model.set_adapter(data[args.column_of_adapter_name])

            text = data["input"]
            batch = tokenizer(text, return_tensors="pt")
            out = model.generate(
                input_ids=batch["input_ids"].cuda(),
                max_length=256,
                do_sample=False,
                top_p=0.7,
                temperature=0.95
            )

            in_text = tokenizer.decode(batch["input_ids"][0])
            out_text = tokenizer.decode(out[0])
            answer = out_text.replace(in_text, "").replace("\nEND", "").strip()
            data["target"] = answer

            print("原始问题：")
            print(in_text)
            print("原始答案：")
            print(out_text)
            print("最终答案：")
            print(answer)
            print("\n\n\n")

            f.write(data)
            exit()
