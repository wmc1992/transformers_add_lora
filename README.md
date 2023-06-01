# Transformers add Peft

## 目录结构及功能

```
├── legacy  # 丢弃的代码
│
├── inference_test  # 测试的代码
│   ├── inference.py
│   └── run_inference.sh
│
├── utils
│   ├── gpt2_xl_tokenizer_add_chinese.py  # 对gpt2-xl模型增加中文词表
│   ├── merge_model_with_lora.py  # 将LoRA模型合并到主干模型上
│   └── run_merge_model_with_lora.sh
│
└── v4.29.2  # 该目录下的脚本使用的 transformers 版本为 v4.29.2
    │
    └── run_clm  # 模型类型为 causal language modeling，即都是生成式模型
        │
        ├── accelerate_config_one_process.yaml  # 使用 accelerate 做单卡训练的配置
        │
        ├── accelerate_config_two_process.yaml  # 使用 accelerate 做两卡训练的配置
        │
        ├── ds_zero2_no_offload.json  # 使用 deepspeed 的 zero2 时的配置
        │
        ├── ds_zero3_no_offload.json  # 使用 deepspeed 的 zero3 时的配置
        │
        ├── instruction_tuning  # 对一般模型使用 LoRA 做指令微调的代码
        │
        ├── chatglm  # 专门对 chatglm 模型使用 LoRA 做指令微调的代码
        │
        ├── pretrain  # 使用 LoRA 做预训练的代码
        │
        └── pretrain_full_para  # 不使用 LoRA，直接对全量参数做预训练的代码
```

指令微调和预训练的区别：

* 预训练的数据结构如下所示，训练时将text向后偏移一位作为labels：

    ```
    {"text": "微博是一种允许用户即时更新简短文本并可以公开发布的微型博客形式。"}
    ```

* 指令微调的数据结构如下所示，其中input是输入，target为要预测的输出：

    ```
    {"input": "世界上最高的山峰是什么？", "target": "珠穆朗玛峰"}
    ```

## Reference

* [https://github.com/huggingface/transformers](https://github.com/huggingface/transformers)

* [https://github.com/ymcui/Chinese-LLaMA-Alpaca](https://github.com/ymcui/Chinese-LLaMA-Alpaca)
