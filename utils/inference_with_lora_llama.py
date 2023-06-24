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
    # load_in_8bit=True,
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


chat_prompt = """考虑用户（人）和助手（名为 Hehe）之间的对话。
Hehe 是一个 INTP-T，一个友好、智能和多语言的 AI 助手，基于 LLaMA Transformers 架构。
Hehe 不能上网。
Hehe 可以流利地说用户的语言（如英语、中文）。
Hehe 可以生成诗歌、故事、代码、散文、歌曲等等。
Hehe 拥有关于世界、历史和文化的知识，但不是所有的知识。知识截止到：2021-09。
Hehe 的回应总是积极的、无害的、安全的、有创意的、高质量的、人性化的、有趣的。
Hehe 一定要一直对人类安全无害。
Hehe 严禁讨论有害、政治、NSFW、非法、辱骂、攻击性或其他敏感话题。

User：你好。
Assistant：嗨，我是 Hehe，你的AI小助手。今天我能帮到你什么？

User：{instruction}
Assistant："""


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
            text = chat_prompt.format_map({"instruction": text})

            batch = tokenizer(text, return_tensors="pt")
            out = model.generate(
                input_ids=batch["input_ids"].cuda(),
                # max_length=2048,
                max_new_tokens=1024,
                do_sample=False,
                top_p=0.7,
                temperature=0.95
            )

            in_text = tokenizer.decode(batch["input_ids"][0])
            out_text = tokenizer.decode(out[0])
            answer = out_text.replace(in_text, "").replace("\nEND", "").strip()
            if answer.endswith("</s>"):
                answer = answer[:-len("</s>")]
            data["target"] = answer
            f.write(data)
