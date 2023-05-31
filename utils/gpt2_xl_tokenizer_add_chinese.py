import re
import json

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel as PreByteLevel
from tokenizers.processors import ByteLevel as PostByteLevel
from tokenizers.decoders import ByteLevel as DecodeByteLevel
from tokenizers.trainers import BpeTrainer

# 功能：向 gpt2-xl 的 tokenizer 中增加中文词语，支持对中文语料做继续预训练

# --------------------------------------------------------------------------------
# 使用自己的语料训练一个新的中文 tokenizer
# --------------------------------------------------------------------------------

# 从头设置 tokenizer 的配置
# model = BPE()
# tokenizer = Tokenizer(model)
# tokenizer.pre_tokenizer = PreByteLevel(add_prefix_space=False)
# tokenizer.post_tokneizer = PostByteLevel()
# tokenizer.decoder = DecodeByteLevel()

# 直接加载 gpt2-xl 的配置
tokenizer = Tokenizer.from_file("./gpt2-xl/tokenizer.json")

trainer = BpeTrainer(special_tokens=["<|endoftext|>"])

# 使用语料进行训练
files = [
    "./data/data1.json",
    "./data/data2.json",
]
tokenizer.train(files, trainer)

# 保存新训练的 tokenizer.json 文件
tokenizer.save("./myself_new_tokenizer/tokenizer.json")


# --------------------------------------------------------------------------------
# 将新训练的中文 tokenizer 中的 vocab 添加到 gpt2-xl 的 tokenizer 中
# --------------------------------------------------------------------------------

with open("./gpt2-xl/tokenizer.json") as f:
    gpt2_tokenizer_config = json.load(f)
with open("./myself_new_tokenizer/tokenizer.json") as f:
    myself_tokenizer_config = json.load(f)

old_vocab_size = len(gpt2_tokenizer_config["models"]["vocab"])

# 合并 vocab 部分的内容，重复的丢掉
max_id = max(gpt2_tokenizer_config["models"]["vocab"].values()) + 1
for idx, token in enumerate(myself_tokenizer_config["model"]["vocab"].keys()):
    if token in gpt2_tokenizer_config["models"]["vocab"]:
        continue

    gpt2_tokenizer_config["models"]["vocab"][token] = max_id
    max_id += 1

# 合并 merges 部分的内容，直接把两个list加起来就行
for merge in myself_tokenizer_config["models"]["merges"]:
    if len(merge) - len(re.sub(" ", "", merge)) > 1:  # 做一下异常检查
        raise RuntimeError(merge)
    gpt2_tokenizer_config["models"]["merges"].append(merge)

print("原始gpt2词表数量:", old_vocab_size, 
      " 新生成的gpt2词表数量:", len(gpt2_tokenizer_config["models"]["vocab"]), 
      " 增加的词表数量:", len(gpt2_tokenizer_config["models"]["vocab"]) - old_vocab_size)

with open("./gpt2-xl-chinese/tokenizer.json", "w") as f:
    json.dump(gpt2_tokenizer_config, f, ensure_ascii=False, indent=2)


# --------------------------------------------------------------------------------
# 验证一下新的 tokenizer 的效果
# --------------------------------------------------------------------------------

tokenizer = Tokenizer.from_file("./gpt2-xl-chinese/tokenizer.json")
content = "你好中国，好好学习，天天向上"
output = tokenizer.encode(content)

print("原始文本:", content)
print("encode之后的token序列:", output.tokens)
print("encode之后的id序列:", output.ids)
print("新tokenizer的分词效果:", [tokenizer.decoder.decode([token]) for token in output.tokens])
