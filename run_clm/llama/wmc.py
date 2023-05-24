from itertools import chain

examples = {
    "input_ids": [
        [1,2,3],
        [4,5,6]
    ],
    "labels": [
        [7,8,9],
    ]
}

concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
print(concatenated_examples)
