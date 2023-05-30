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
        f = prompt_type_to_func.get(self.prompt_type, "default")
        sources, targets = f(examples, self.prompt_column, self.response_column, self.history_column)

        tokenized_sources = self.tokenizer(
            sources,
            max_length=self.max_source_length - 1,
            padding="max_length",
            truncation=True,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        tokenized_targets = self.tokenizer(
            targets,
            max_length=self.max_target_length - 1,
            padding="max_length",
            truncation=True,
            return_attention_mask=False,
            return_token_type_ids=False,
        )

        all_input_ids = []
        all_labels = []
        for s,t in zip(tokenized_sources["input_ids"], tokenized_targets["input_ids"]):
            input_ids = torch.LongTensor(s + t)
            labels = torch.LongTensor([IGNORE_INDEX] * len(s) + t)
            assert len(input_ids) == len(labels)
            assert len(input_ids) <= self.max_source_length + self.max_target_length

            all_input_ids.append(input_ids)
            all_labels.append(labels)

        results = {"input_ids":all_input_ids, "labels": all_labels}
        return results
