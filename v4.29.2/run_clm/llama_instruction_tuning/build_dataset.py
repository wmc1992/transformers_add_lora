import torch


IGNORE_INDEX = -100


def build_source_and_target_default(examples, prompt_column, response_column, history_column):
    """ 默认直接将 source 和 target 拼接到一起，并且忽略 history 字段 """

    sources = []
    targets = []
    for prompt, response in zip(examples[prompt_column], examples[response_column]):
        sources.append(prompt)
        targets.append(response)
    return sources, targets


chat_prompt = """Consider a conversation between User (a human) and Assistant (named Hehe).
Hehe is an INTP-T, a friendly, intelligent and multilingual AI assistant, based on LLaMA Transformers architecture.
Hehe cannot access the Internet.
Hehe can fluently speak the user's language (e.g. English, Chinese).
Hehe can generate poems, stories, code, essays, songs, and more.
Hehe possesses knowledge about the world, history, and culture, but not everything. Knowledge cutoff: 2021-09.
Hehe's responses are always positive, unharmful, safe, creative, high-quality, human-like, and interesting.
Hehe must always be safe and unharmful to humans.
Hehe strictly refuses to discuss harmful, political, NSFW, illegal, abusive, offensive, or other sensitive topics.

User: Hi.
Assistant: Hi, I'm Hehe, your AI assistant. How can I help you today?

User: {instruction}
Assistant: """


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


def build_source_and_target_with_chat_prompt(examples, prompt_column, response_column, history_column):
    sources = []
    targets = []
    for prompt, response in zip(examples[prompt_column], examples[response_column]):
        sources.append(chat_prompt.format_map({"instruction": prompt}))
        targets.append(response)
    return sources, targets


ziya_prompt = """<human>:{instruction}\n<bot>:"""


def build_source_and_target_with_ziya_prompt(examples, prompt_column, response_column, history_column):
    sources = []
    targets = []
    for prompt, response in zip(examples[prompt_column], examples[response_column]):
        sources.append(ziya_prompt.format_map({"instruction": prompt}))
        targets.append(response)
    return sources, targets


def build_source_and_target_with_alpaca_prompt(examples, prompt_column, response_column, history_column):
    """ 该函数中使用的是出自 standard alpaca 的 prompt，不过由于该 prompt 是英文的，实际测试：该指令会可能导致模型输出部分英文 """

    PROMPT_TEMPLATE = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response: "
    )

    sources = []
    targets = []
    for prompt, response in zip(examples[prompt_column], examples[response_column]):
        sources.append(PROMPT_TEMPLATE.format_map({"instruction": prompt}))
        targets.append(response)
    return sources, targets


prompt_type_to_func = {
    "default": build_source_and_target_default,
    "chat_prompt": build_source_and_target_with_chat_prompt,
    "ziya_prompt": build_source_and_target_with_ziya_prompt,
    "standard_alpaca": build_source_and_target_with_alpaca_prompt,
}


class DatasetUtil:

    def __init__(self, tokenizer, max_source_length, max_target_length,
                 prompt_column, response_column, history_column, prompt_type=None):
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.prompt_column = prompt_column
        self.response_column = response_column
        self.history_column = history_column
        self.prompt_type = prompt_type

        self.print_debug = True

    def tokenization(self, examples):
        f = prompt_type_to_func[self.prompt_type]
        sources, targets = f(examples, self.prompt_column, self.response_column, self.history_column)

        if self.print_debug:
            self.print_debug = False
            print("打印构造添加prompt之后的样例：\n")
            print(sources[0], "\n\n")
            print(targets[0])

        max_seq_len = self.max_source_length + self.max_target_length

        all_input_ids = []
        all_labels = []
        for source, target in zip(sources, targets):
            token_ids_0 = self.tokenizer.encode(text=source, add_special_tokens=False)
            token_ids_1 = self.tokenizer.encode(text=target, add_special_tokens=False)

            # 如果 source 和 target 的总长度没有超长，就不做截断
            if len(token_ids_0) + len(token_ids_1) > max_seq_len - 2:
                if len(token_ids_0) > self.max_source_length - 1:  # 留一个位置给 bos_token
                    token_ids_0 = token_ids_0[:self.max_source_length - 1]
                if len(token_ids_1) > self.max_target_length - 1:  # 留一个位置给 eos_token
                    token_ids_1 = token_ids_1[:self.max_target_length - 1]

            # 构造的模版：<s> A B </s>
            token_ids_0 = [self.tokenizer.bos_token_id] + token_ids_0
            token_ids_1 = token_ids_1 + [self.tokenizer.eos_token_id]
            token_ids = token_ids_0 + token_ids_1
            labels = [IGNORE_INDEX for _ in range(len(token_ids_0))] + token_ids_1
            assert len(token_ids) == len(labels)

            # PADDING
            if len(token_ids) < max_seq_len:
                token_ids = token_ids + [self.tokenizer.pad_token_id for _ in range(max_seq_len - len(token_ids))]
                labels = labels + [IGNORE_INDEX for _ in range(max_seq_len - len(labels))]

            assert len(token_ids) == max_seq_len
            assert len(labels) == max_seq_len

            all_input_ids.append(torch.LongTensor(token_ids))
            all_labels.append(torch.LongTensor(labels))

        results = {"input_ids": all_input_ids, "labels": all_labels}
        return results
