import argparse
import os

import datasets
from verl.utils.hdfs_io import copy, makedirs


def separate_prompt_and_response(row):
    # messages = row["messages"]
    # assert messages[-1]["role"] == "assistant"
    # row["prompt"] = messages[:-1]
    # row["response"] = messages[-1]["content"]
    # return row
    row["prompt"] = [{"role": "user", "content": row["messages"]}]
    return row


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="datasets/gsm8k")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--tokenizer", default="Qwen/Qwen2.5-7B-Instruct")

    args = parser.parse_args()

    dataset = datasets.load_dataset("openai/gsm8k", 'main', split='test').select(range(1000))
    dataset = dataset.rename_column("question", "messages")
    dataset = dataset.rename_column("answer", "response")
    dataset = dataset.map(separate_prompt_and_response)
    dataset = dataset.remove_columns(["messages"])

    # filter out too long samples
    if args.max_length is not None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

        def filter_by_length(row):
            prompt = row["prompt"]
            response = row["response"]
            messages = prompt + [
                {"role": "assistant", "content": response},
            ]
            tokens = tokenizer.apply_chat_template(messages, tokenize=True)
            return len(tokens) <= args.max_length

        dataset = dataset.filter(filter_by_length)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir
    dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
