from functools import partial
import glob
from re import sub
from typing import Any

from datasets import load_dataset
from transformers import AutoTokenizer, default_data_collator
import wandb
import re
import emoji
from soynlp.normalizer import repeat_normalize

emojis = ''.join(emoji.UNICODE_EMOJI.keys())
# pattern = re.compile(f'[^ .,?!/@$%~％·∼()\x00-\x7Fㄱ-ㅣ가-힣{emojis}]+')
pattern = re.compile(f'[^ .,?!/@$%~％·∼()\x00-\x7Fㄱ-ㅣ가-힣{emojis}]+')
url_pattern = re.compile(
    r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)')
repeat_pattern = re.compile('[ㅋㅎㅠㅜ]+')
emo = re.compile('[#@이모티콘#]+')
num = re.compile('[0-9]+')
eng = re.compile('[a-zA-Z]+')

def clean(x):
    x = pattern.sub(' ', x)
    x = url_pattern.sub('', x)
    x = repeat_pattern.sub('', x)
    # x = num.sub('', x)
    x = eng.sub('', x)
    x = x.strip()
    x = repeat_normalize(x, num_repeats=1)
    return x

def convert_to_features(tokenizer: AutoTokenizer, max_len: int, prefix, args, examples: Any):
    # print('Add prefix to dataset!!')
    # if prefix is not None:
    for i in range(len(examples["input"])):
        examples["input"][i] = clean(examples["input"][i])
        if prefix is not None:
            examples["input"][i] = prefix + examples["input"][i]

    model_inputs = tokenizer(
        examples["input"],
        add_special_tokens=True,
        padding="max_length",
        max_length=max_len,
        truncation=True,
    )

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            examples["output"],
            add_special_tokens=True,
            padding="max_length",
            max_length=args.max_target_len,
            truncation=True,
        )

    model_inputs["labels"] = labels["input_ids"]

    return model_inputs


def get_a_dataset(
    args, file_path: str, tokenizer: AutoTokenizer, max_len: int, prefix=None, split: str = "train"
):
    print(f'Subject of {file_path}')
    dataset = load_dataset(
        "csv", data_files=file_path, split=split
    )
    if args.mode == 'train' and args.cut and split == 'train':
        dataset = dataset.select(range(args.cut))
    convert = partial(convert_to_features, tokenizer, max_len, prefix, args)
    
    dataset = dataset.map(convert, batched=True, num_proc=4, load_from_cache_file=not args.overwrite_cache)

    cols_to_keep = [
        x
        for x in ["id", "input_ids", "attention_mask", "labels"]
        if x in dataset.features
    ]

    dataset.set_format(columns=cols_to_keep)
    return dataset

def get_prefix(subject: str) -> str:
    if subject == 'book':
        return '책 요약: '
    elif subject == 'dialouge':
        return '대화 요약: '
    elif subject == 'document':
        return '신문기사 요약: '
    elif subject == 'journal':
        return '특허 요약: '
    else:
        return '대화 요약: '
    
def get_subject_dataset(
    args, file_path: str, tokenizer: AutoTokenizer, max_len: int, split: str = "train"
):
    file_list = glob.glob(file_path + "/*.csv")

    dataset_d = {}
    for file in file_list:
        subject = file.split('/')[-1].split('_')[0]
        if args.prefix:
            # prefix = get_prefix(subject)
            prefix = args.prefix
            print(f'prefix is {args.prefix}')
        else:
            prefix = None
        print(f'get_subject_dataset: {file}, {subject}')
        dataset_d[subject] = get_a_dataset(args, file, tokenizer, max_len, prefix, split)

    return dataset_d
    


def get_dataset(
    args, file_paths, tokenizer: AutoTokenizer, max_len: int, split: str = "train"
):
    # dataset = load_dataset(
    #     "csv", data_files=glob.glob(file_path + "/*.csv"), split=split
    # )
    dataset = load_dataset(
        "csv", data_files=file_paths
    )
    if args.mode == 'train' and args.cut and split == 'train':
    # if args.cut:
        dataset = dataset.select(range(args.cut))
    convert = partial(convert_to_features, tokenizer, max_len, args)
    
    dataset = dataset.map(convert, batched=True, num_proc=4, load_from_cache_file=args.cache)

    cols_to_keep = [
        x
        for x in ["id", "input_ids", "attention_mask", "labels"]
        if x in dataset.features
    ]

    dataset.set_format(columns=cols_to_keep)

    return dataset


def add_id_collator(features: Any):
    batch = default_data_collator(features)
    if "id" in features[0].keys():
        batch["id"] = [f["id"] for f in features]
    return batch
