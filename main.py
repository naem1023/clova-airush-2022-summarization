import argparse
import os
import random
from typing import Any

import nsml
import numpy as np
import torch
from nsml import DATASET_PATH
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    BartForConditionalGeneration,
    T5ForConditionalGeneration, Seq2SeqTrainer,  Seq2SeqTrainingArguments
)
from data_utils import add_id_collator, get_dataset, get_subject_dataset, get_prefix

from train import predict, hf_train, hf_predict, validation
# from parallelformers import parallelize

import wandb
import copy, glob
from run_summarization import main
from tqdm.auto import tqdm
os.environ["WANDB_API_KEY"] = ""
token = ''

def list_files(startpath):
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print('{}{}/'.format(indent, os.path.basename(root)))
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print('{}{}'.format(subindent, f))

class ModelManage:
    def __init__(self) -> None:
        self.models = {}
        self.tokenizer = None


def bind_nsml(mg: Any, tokenizer: Any, args: Any = None):
    def save(dir_name, **kwargs):
        # os.makedirs(dir_name, exist_ok=True)
        # torch.save(model.state_dict(), os.path.join(dir_name, "model.pth"))
        pass

    def load(dir_name, **kwargs):
        print("Start loading model")
        print(dir_name)
        print(os.listdir(dir_name))
        list_files(dir_name)
        args.save_path = dir_name
        print(os.listdir(args.save_path))

        if 't5' in args.model.lower():
            all_model = T5ForConditionalGeneration.from_pretrained(os.path.join(dir_name, 'all'))
        else:    
            all_model = BartForConditionalGeneration.from_pretrained(os.path.join(dir_name, 'all'))
        for subject in os.listdir(dir_name):
            if subject == 'journal':
                model_path = os.path.join(dir_name, subject)
                if 't5' in args.model.lower():
                    model = T5ForConditionalGeneration.from_pretrained(model_path)
                else:
                    model = BartForConditionalGeneration.from_pretrained(model_path)
                mg.models[subject] = model
            else:
                mg.models[subject] = all_model
            # print('='*8, 'load model', model_path)
        if args.tokenizer:
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
        else:
            tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
        mg.tokenizer = tokenizer
        
    def infer(file_path, **kwargs):
        wandb.init(
            project="airush-summary", 
            entity="naem1023", 
            name=f'nsml-infer-{args.model}',
            settings=wandb.Settings(start_method="fork")
        )
        print("start inference")
        # print(mg.models)
        args.train_path = file_path
        

        all_results = []
        test_dataset = get_subject_dataset(args, args.train_path, tokenizer, args.max_len)
        print(test_dataset)
        for subject in test_dataset:
            args.max_len = get_max_len(subject)
            print('Get max length')
            if subject == 'journal':
                model = mg.models['journal']
            else:
                model = mg.models['all']
            print('Load model!')

            model.to(args.device)
            print('Send model to gpu')
            test_dataloader = DataLoader(
                test_dataset[subject],
                shuffle=False,
                batch_size=args.eval_batch,
                collate_fn=add_id_collator,
            )
            print('Load DataLoader')
            
            print(f'Start predicting {subject}')
            results = predict(model, tokenizer, test_dataloader, args)
            all_results.extend(results)
            print(results[:5])

        return all_results

    nsml.bind(save, load, infer)


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore

def get_max_len(subject):
    if subject == 'dialouge':
        return 63
    elif subject == 'note':
        return 82
    elif subject == 'journal':
        return 225
    elif subject == 'book':
        return 200
    elif subject == 'document':
        return 130

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--eval_batch", type=int, default=16)
    parser.add_argument("--max-len", type=int, default=1024)
    parser.add_argument("--max_target_len", type=int, default=256)
    # parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--model", type=str, default="MrBananaHuman/kobart-base-v2-summarization")
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--epochs", type=float, default=5)
    parser.add_argument("--pause", type=int, default=0)
    parser.add_argument("--train-path", type=str, default="train/train_data")
    parser.add_argument("--save_path", type=str, default="models")
    parser.add_argument("--gradient_accum", type=int, default=2)
    parser.add_argument("--warmup", type=float, default=0.0)
    parser.add_argument("--step", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cut", type=int, default=None)
    parser.add_argument("--fp16", type=bool, default=False)
    parser.add_argument("--overwrite_cache", type=bool, default=False)
    parser.add_argument("--t5", type=bool, default=False)
    parser.add_argument("--prefix", type=str, default="요약: ")
    parser.add_argument("--local", type=bool, default=False)
    parser.add_argument("--valid", type=bool, default=False)
    parser.add_argument("--single", type=bool, default=False)
    parser.add_argument("--ph", type=bool, default=False)
    parser.add_argument('--valid_targets', nargs='+', default=[])
    parser.add_argument('--load_session', type=str, default=None)
    
    
    args = parser.parse_args()
    seed_everything(args.seed)

    if args.pause:
        wandb_name = f"infer-{args.model}"
    else:
        wandb_name = f"{args.model}"

    wandb.init(
        project="airush-summary", 
        entity="naem1023", 
        name=wandb_name,
        settings=wandb.Settings(start_method="fork")
    )

    # initialize args
    args.train_path = os.path.join(DATASET_PATH, args.train_path,)
    print(args)
    
    if args.tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    
    mg = ModelManage()

    bind_nsml(mg, tokenizer, args=args)
    # test mode
    if args.pause:
        nsml.paused(scope=locals())


    if args.load_session:
        nsml.load(checkpoint='models', session=args.load_session)
        print(mg.models.keys())

        mg.models['all'].save_pretrained(os.path.join(args.save_path, 'all'))
        mg.models['journal'].save_pretrained(os.path.join(args.save_path, 'journal'))

        nsml.save_folder('models', args.save_path)
        # nsml.paused(scope=locals())



    if args.load_session is None:
        # train mode
        if args.mode == "train":
            root_path = copy.deepcopy(args.save_path)
            origin_model = args.model
            file_list = list(glob.glob(args.train_path + "/*.csv"))

            print('='*8, file_list)

            args.file_path = os.path.join(args.train_path, 'journal_text.csv')
            args.save_path = os.path.join(root_path, 'journal')
            hf_train(args, subject='journal', prefix=args.prefix)

            args.file_path = file_list
            args.save_path = os.path.join(root_path, 'all')
            hf_train(args, prefix=args.prefix)


            if not args.local:
                nsml.save_folder('models', root_path)