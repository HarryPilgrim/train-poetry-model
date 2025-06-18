from datasets import load_dataset
from transformers import AutoTokenizer

def load_and_tokenize(txt_path: str, model_name: str = "distilgpt2"):
    dataset = load_dataset("text", data_files={"train": txt_path})
    tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(example):
        return tokenizer(example["text"], truncation=True, max_length=512)

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    # 🚨 Filter out sequences with 0 tokens
    tokenized_dataset = tokenized_dataset.filter(lambda x: len(x["input_ids"]) > 0)

    return tokenized_dataset, tokenizer


# from datasets import load_dataset
# from transformers import AutoTokenizer
#
# def load_and_tokenize(txt_path: str, model_name: str = "distilgpt2"):
#     dataset = load_dataset("text", data_files={"train": txt_path})
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#
#     def tokenize_function(example):
#         return tokenizer(example["text"], truncation=True, max_length=512)
#
#     tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
#     return tokenized_dataset, tokenizer