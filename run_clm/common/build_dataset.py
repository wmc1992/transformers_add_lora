import torch


IGNORE_INDEX = -100

PROMPT_TEMPLATE = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response: "
    )


class DatasetUtil:

    def __init__(self, tokenizer, max_source_length, max_target_length,
                 prompt_column, response_column, history_column):
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.prompt_column = prompt_column
        self.response_column = response_column
        self.history_column = history_column

    def build_source_and_target_content(self, examples):
        """ 如果想设计不同的指令或者提示，可以修改这个函数 """

        sources = []
        targets = []

        # TODO 先实现一个简单版本，不添加额外指令，并且忽略 history 字段。
        for prompt, response in \
                zip(examples[self.prompt_column], examples[self.response_column]):

            sources.append(prompt)
            targets.append(response)

        return sources, targets

    def tokenization(self, examples):
        max_seq_length = self.max_source_length + self.max_target_length

        sources, targets = self.build_source_and_target_content(examples)

        tokenized_sources = self.tokenizer(sources, return_attention_mask=False)
        tokenized_targets = self.tokenizer(targets, return_attention_mask=False, add_special_tokens=False)

        all_input_ids = []
        all_labels = []
        for s,t in zip(tokenized_sources["input_ids"],tokenized_targets["input_ids"]):
            s = s[:self.max_source_length]
            t = t[:self.max_target_length]

            input_ids = s + t
            labels = [IGNORE_INDEX] * len(s) + t
            assert len(input_ids) == len(labels)

            pad_len = max_seq_length - len(input_ids)
            input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_len
            labels = labels + [self.tokenizer.pad_token_id] * pad_len
            assert len(input_ids) == len(labels)

            input_ids = torch.LongTensor(input_ids)
            labels = torch.LongTensor(labels)

            all_input_ids.append(input_ids)
            all_labels.append(labels)

        results = {"input_ids":all_input_ids, "labels": all_labels}
        return results
