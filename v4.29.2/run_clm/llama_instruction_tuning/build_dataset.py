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

    def tokenization(self, examples):
        f = prompt_type_to_func[self.prompt_type]
        sources, targets = f(examples, self.prompt_column, self.response_column, self.history_column)

        max_seq_len = self.max_source_length + self.max_target_length

        all_input_ids = []
        all_labels = []
        for source, target in zip(sources, targets):
            token_ids_0 = self.tokenizer.encode(text=source, add_special_tokens=False)
            token_ids_1 = self.tokenizer.encode(text=target, add_special_tokens=False)

            if len(token_ids_0) > self.max_source_length - 2:  # 这两个位置留给 bos_token 和 eos_token
                token_ids_0 = token_ids_0[:self.max_source_length - 2]
            if len(token_ids_1) > self.max_target_length - 2:  # 这两个位置留给 bos_token 和 eos_token
                token_ids_1 = token_ids_1[:self.max_target_length - 2]

            # TODO 这里在构造labels时，写死了其第一部分的长度为 len(token_ids_0)+2 ，这种方法不是很好
            token_ids = self.tokenizer.build_inputs_with_special_tokens(token_ids_0, token_ids_1)
            labels = [IGNORE_INDEX for _ in range(len(token_ids_0) + 2)] + token_ids[len(token_ids_0) + 2:]
            assert len(token_ids) == len(labels)

            # PADDING
            if len(token_ids) < max_seq_len:
                token_ids = token_ids + [self.tokenizer.pad_token_id for _ in range(max_seq_len - len(token_ids))]
                labels = labels + [IGNORE_INDEX for _ in range(max_seq_len - len(token_ids))]
            assert len(token_ids) == max_seq_len

            all_input_ids.append(torch.LongTensor(token_ids))
            all_labels.append(torch.LongTensor(labels))

        results = {"input_ids": all_input_ids, "labels": all_labels}
        return results
