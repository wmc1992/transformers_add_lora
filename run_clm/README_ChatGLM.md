# ChatGLM 训练脚本修改说明

以 [transformers](https://github.com/huggingface/transformers) 库中的脚本 `examples/pytorch/language-modeling/run_clm.py` 为基准，在其基础上进行修改，用于对 ChatGLM-6B 模型做 LoRA 微调。

基本原则是：保留 transformers 中所支持的所有功能；尽量少做修改，当基准脚本更新时可以方便的进行更新；

通用的需要修改的内容为：

* 增加对 LoRA 的支持；
* 对数据格式以及数据处理部分做修改，数据格式为两个字段：`input` 和 `target`；

ChatGLM 的专项修改的内容为：

* 加载config、tokenizer、model 时添加参数 `trust_remote_code=True`，且加载模型时使用 `AutoModel`，不能使用 `AutoModelForCausalLM`；
* 数据处理部分按照官方提供的代码进行处理，否则会有特殊符号报错问题；

## ChatGLM 专项修改的内容

### 加载和初始化

加载config、tokenizer、model 时添加参数 `trust_remote_code=True`，且加载模型时使用 `AutoModel`，不能使用 `AutoModelForCausalLM`。这部分比较简单，相应的代码修改如下所示。

加载 config：

```python
if model_args.config_name:
    config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
elif model_args.model_name_or_path:
-   config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
+   config = AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True, **config_kwargs)
```

加载 tokenizer：

```python
if model_args.tokenizer_name:
    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
elif model_args.model_name_or_path:
-   tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
+   tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True, **tokenizer_kwargs)
```

加载 model，使用 `AutoModel` 同时添加参数 `trust_remote_code=True`：

```python
if model_args.model_name_or_path:
    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )
-   model = AutoModelForCausalLM.from_pretrained(
+   model = AutoModel.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=model_args.low_cpu_mem_usage,
+       trust_remote_code=True,
    )
```

### 数据处理部分

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



## 通用的需要修改的内容


