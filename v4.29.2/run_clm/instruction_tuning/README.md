# 使用 LoRA 进行指令微调

## 如何使用



## 修改内容说明

以 [transformers](https://github.com/huggingface/transformers) 库中的脚本 `examples/pytorch/language-modeling/run_clm.py` 为基准，在其基础上进行修改，用于对 ChatGLM-6B 模型做 LoRA 微调。

基本原则是：保留 transformers 中所支持的所有功能；尽量少做修改，当基准脚本更新时可以方便的进行更新；

通用的需要修改的内容为：

* 增加对 LoRA 的支持；
* 对数据格式以及数据处理部分做修改，数据格式为两个字段：`prompt` 和 `response`；

ChatGLM 的专项修改的内容为：

* 加载config、tokenizer、model 时添加参数 `trust_remote_code=True`，且加载模型时使用 `AutoModel`，不能使用 `AutoModelForCausalLM`；
* 数据处理部分按照官方提供的代码进行处理，否则会有特殊符号报错问题；

## 数据处理部分

原始 `run_clm.py` 脚本是直接将数据中的文本作为输入，然后向后偏移一位之后作为 labels，而目前的应用场景更多是输入和输出是不同的，所以这里修改了数据格式，并且会按照新的数据格式添加相应的参数，以及修改相应的数据处理部分的代码。

#### 数据格式

单条数据的格式如下所示，主要是 `prompt`、`response`、`history` 这三个字段，其中 `history` 字段可以没有。

```
{"prompt": "世界上最高的山峰是什么？", "response": "珠穆朗玛峰", "history": [["你好", "您好，请问有什么可以帮您的？"], ["你可以做什么？", "我是智能语言大模型，您可以向我提问题。"]]}
```

#### 添加输入参数

将数据格式做了上述修改之后，添加上如下的输入参数，基本都是和上述的数据格式是一一对应的。

```python
@dataclass
class DataTrainingArguments:
    ... ...

+   prompt_type: Optional[str] = field(
+       default=None,
+       metadata={"help": f"The type of prompt to build input text, from list of {','.join(list(prompt_type_to_func.keys()))}"},
+   )
+   prompt_column: Optional[str] = field(
+       default=None,
+       metadata={"help": "The name of the column in the datasets containing the full texts (for summarization)."},
+   )
+   response_column: Optional[str] = field(
+       default=None,
+       metadata={"help": "The name of the column in the datasets containing the summaries (for summarization)."},
+   )
+   history_column: Optional[str] = field(
+       default=None,
+       metadata={"help": "The name of the column in the datasets containing the history of chat."},
+   )
+   max_source_length: Optional[int] = field(
+       default=1024,
+       metadata={
+           "help": (
+               "The maximum total input sequence length after tokenization. Sequences longer "
+               "than this will be truncated, sequences shorter will be padded."
+           )
+       },
+   )
+   max_target_length: Optional[int] = field(
+       default=128,
+       metadata={
+           "help": (
+               "The maximum total sequence length for target text after tokenization. Sequences longer "
+               "than this will be truncated, sequences shorter will be padded."
+           )
+       },
+   )
+   ignore_pad_token_for_loss: bool = field(
+       default=True,
+       metadata={
+           "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
+       },
+   )
```

#### 数据处理

下面部分的代码就是原始的 `run_clm.py` 中做数据处理的代码，这里会直接将这部分代码全部删掉，替换为新的数据处理的代码。这部分删掉的代码的功能简述如下：

* 下述代码中的函数 `tokenize_function` 是对文本做 tokenize，其中对象 raw_datasets 经过该函数处理成 tokenized_datasets；

* 获取每条文本的最大长度 `block_size`；（这里只有一个最大长度，修改后会有两个最大长度：`max_source_length` 和 `max_target_length`）

* 将所有的文本按照上述的最大长度进行分组，所谓分组就是：当某条数据超过了最大长度，那么超出的部分会拼接到下一条数据中；（这个逻辑修改后就被去除了）

* 使配置的参数 `max_train_samples` 和 `max_eval_samples` 生效，这两个参数的作用是训练集和验证集中分别使用多少条数据，多余的数据就不使用了；（这个逻辑修改后还是会保留的）

```python
-   text_column_name = "text" if "text" in column_names else column_names[0]
-
-   # since this will be pickled to avoid _LazyModule error in Hasher force logger loading before tokenize_function
-   tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")
-
-   def tokenize_function(examples):
-       with CaptureLogger(tok_logger) as cl:
-           output = tokenizer(examples[text_column_name])
-       # clm input could be much much longer than block_size
-       if "Token indices sequence length is longer than the" in cl.out:
-           tok_logger.warning(
-               "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits"
-               " before being passed to the model."
-           )
-       return output
-
-   with training_args.main_process_first(desc="dataset map tokenization"):
-       if not data_args.streaming:
-           tokenized_datasets = raw_datasets.map(
-               tokenize_function,
-               batched=True,
-               num_proc=data_args.preprocessing_num_workers,
-               remove_columns=column_names,
-               load_from_cache_file=not data_args.overwrite_cache,
-               desc="Running tokenizer on dataset",
-           )
-       else:
-           tokenized_datasets = raw_datasets.map(
-               tokenize_function,
-               batched=True,
-               remove_columns=column_names,
-           )
-
-   if data_args.block_size is None:
-       block_size = tokenizer.model_max_length
-       if block_size > 1024:
-           logger.warning(
-               "The chosen tokenizer supports a `model_max_length` that is longer than the default `block_size` value"
-               " of 1024. If you would like to use a longer `block_size` up to `tokenizer.model_max_length` you can"
-               " override this default with `--block_size xxx`."
-           )
-           block_size = 1024
-   else:
-       if data_args.block_size > tokenizer.model_max_length:
-           logger.warning(
-               f"The block_size passed ({data_args.block_size}) is larger than the maximum length for the model"
-               f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
-           )
-       block_size = min(data_args.block_size, tokenizer.model_max_length)
-
-   # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
-   def group_texts(examples):
-       # Concatenate all texts.
-       concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
-       total_length = len(concatenated_examples[list(examples.keys())[0]])
-       # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
-       # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
-       total_length = (total_length // block_size) * block_size
-       # Split by chunks of max_len.
-       result = {
-           k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
-           for k, t in concatenated_examples.items()
-       }
-       result["labels"] = result["input_ids"].copy()
-       return result
-
-   # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
-   # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
-   # to preprocess.
-   #
-   # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
-   # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map
-
-   with training_args.main_process_first(desc="grouping texts together"):
-       if not data_args.streaming:
-           lm_datasets = tokenized_datasets.map(
-               group_texts,
-               batched=True,
-               num_proc=data_args.preprocessing_num_workers,
-               load_from_cache_file=not data_args.overwrite_cache,
-               desc=f"Grouping texts in chunks of {block_size}",
-           )
-       else:
-           lm_datasets = tokenized_datasets.map(
-               group_texts,
-               batched=True,
-           )
-
-   if training_args.do_train:
-       if "train" not in tokenized_datasets:
-           raise ValueError("--do_train requires a train dataset")
-       train_dataset = lm_datasets["train"]
-       if data_args.max_train_samples is not None:
-           max_train_samples = min(len(train_dataset), data_args.max_train_samples)
-           train_dataset = train_dataset.select(range(max_train_samples))
-
-   if training_args.do_eval:
-       if "validation" not in tokenized_datasets:
-           raise ValueError("--do_eval requires a validation dataset")
-       eval_dataset = lm_datasets["validation"]
-       if data_args.max_eval_samples is not None:
-           max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
-           eval_dataset = eval_dataset.select(range(max_eval_samples))
-
-       def preprocess_logits_for_metrics(logits, labels):
-           if isinstance(logits, tuple):
-               # Depending on the model and config, logits may contain extra tensors,
-               # like past_key_values, but logits always come first
-               logits = logits[0]
-           return logits.argmax(dim=-1)
-
-       metric = evaluate.load("accuracy")
-
-       def compute_metrics(eval_preds):
-           preds, labels = eval_preds
-           # preds have the same shape as the labels, after the argmax(-1) has been calculated
-           # by preprocess_logits_for_metrics but we need to shift the labels
-           labels = labels[:, 1:].reshape(-1)
-           preds = preds[:, :-1].reshape(-1)
-           return metric.compute(predictions=preds, references=labels)
```

下面的代码是新的数据处理的代码，删除上述的代码之后，再把下述代码添加上即可。修改后的这部分代码的功能简述如下：

```python
+ from build_dataset import DatasetUtil


+   datasetutil = DatasetUtil(tokenizer, data_args.max_source_length, data_args.max_target_length,
+                             data_args.prompt_column, data_args.response_column, data_args.history_column)
+
+   logger.info("开始处理数据")
+   with training_args.main_process_first(desc="dataset map tokenization"):
+       if not data_args.streaming:
+           tokenized_datasets = raw_datasets.map(
+               datasetutil.tokenization,
+               batched=True,
+               num_proc=data_args.preprocessing_num_workers,
+               remove_columns=column_names,
+               load_from_cache_file=not data_args.overwrite_cache,
+               desc="Running tokenizer on dataset",
+           )
+       else:
+           tokenized_datasets = raw_datasets.map(
+               datasetutil.tokenization,
+               batched=True,
+               remove_columns=column_names,
+           )
```

不够可以很明显看出上述代码并不是核心的数据处理逻辑，核心的数据处理逻辑提出来放到了脚本 `build_dataset.py` 中，该脚本的代码如下：

```python
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

```
