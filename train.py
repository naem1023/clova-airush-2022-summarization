from statistics import mean
from typing import Any

import nsml
import numpy as np
from konlpy.tag import Mecab
from transformers import (
    AdamW,
    AutoTokenizer,
    BartForConditionalGeneration,
    get_linear_schedule_with_warmup,
)

from rouge_metric import Rouge
from tqdm.auto import tqdm
from torch import nn
import torch
from torch.utils.data import DataLoader
import wandb
import evaluate
from run_summarization import main, ModelArguments, DataTrainingArguments

from transformers import (
    AutoTokenizer, AutoConfig, AutoModelForMaskedLM, Trainer, TrainingArguments, LineByLineTextDataset,
    PreTrainedTokenizer, PreTrainedTokenizerFast, DataCollatorWithPadding,
    EvalPrediction, TrainerCallback, Seq2SeqTrainer,  Seq2SeqTrainingArguments, DataCollatorForSeq2Seq,
)

access_token = ''

class RougeMetric:
    def __init__(self):
        self.rouge = Rouge(
            metrics=["rouge-n", "rouge-l"],
            max_n=2,
            limit_length=True,
            length_limit=1000,
            length_limit_type="words",
            apply_avg=True,
            apply_best=False,
            alpha=0.5,  # Default F1_score
            weight_factor=1.2,
            use_tokenizer=True,
        )

        self.mecab = Mecab()

    def evaluation(self, pred: str, label: str):
        generated_txt_norm = self.norm(pred)
        labels_txt_norm = self.norm(label)
        rouges = self.rouge.get_scores(generated_txt_norm, labels_txt_norm)
        return rouges

    def norm(self, sent: str):
        return " ".join(self.mecab.morphs(sent))


from torch.cuda.amp import autocast
def predict(
    model: nn.modules, tokenizer: AutoTokenizer, data_loader: DataLoader, args: Any
):
    print("start predict")
    model.eval()

    texts = []
    keys = []

    with autocast(dtype=torch.float16 if args.fp16 else torch.float32):
        for step, batch in enumerate(tqdm(data_loader)):
            keys.extend(batch["id"])
            batch = {
                key: item.to(args.device) for key, item in batch.items() if key != "id"
            }
            generated_ids = model.generate(
                input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                max_length=args.max_len
            )
            generated_texts = tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            generated_texts = [pred.strip() for pred in generated_texts]
            texts.extend(generated_texts)

        return [{"id": key, "output": text} for key, text in zip(keys, texts)]


def validation(
    model: nn.modules, tokenizer: AutoTokenizer, data_loader: DataLoader, args: Any
):
    print("start validation")
    model.eval()

    metric = RougeMetric()

    texts = []
    labels = []
    keys = []
    with autocast(dtype=torch.float16 if args.fp16 else torch.float32):
        for step, batch in enumerate(tqdm(data_loader)):
            keys.extend(batch["id"])
            batch = {
                key: item.to(args.device) for key, item in batch.items() if key != "id"
            }

            label_texts = tokenizer.batch_decode(batch["labels"], skip_special_tokens=True, clean_up_tokenization_spaces=True)
            label_texts = [str.strip(s) for s in label_texts]
            labels.extend(label_texts)
            generated_ids = model.generate(
                input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                max_length=args.max_len
            )
            generated_texts = tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            generated_texts = [pred.strip() for pred in generated_texts]
            texts.extend(generated_texts)

    r_1_f = []
    r_2_f = []
    r_l_f = []

    for text, label in zip(texts, labels):
        rouges = metric.evaluation(text, label)

        r_1_f.append(rouges["rouge-1"]["f"])
        r_2_f.append(rouges["rouge-2"]["f"])
        r_l_f.append(rouges["rouge-l"]["f"])

    result_metrics = {
        "rouge-1-f": mean(r_1_f),
        "rouge-2-f": mean(r_2_f),
        "rouge-l-f": mean(r_l_f),
    }
    return texts, result_metrics

def hf_predict(args):
    training_args = Seq2SeqTrainingArguments(
        do_train=False,
        do_predict=True,
        output_dir=args.save_path,
        overwrite_output_dir=True,
        per_device_train_batch_size=args.eval_batch,
        per_device_eval_batch_size=args.eval_batch,
        gradient_accumulation_steps=args.gradient_accum,
        eval_accumulation_steps=args.gradient_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        seed=args.seed,
        fp16=args.fp16,
        generation_max_length=args.max_len,
        predict_with_generate=True,
        # fp16_opt_level='O3',
        report_to="wandb"
    )

    model_args = ModelArguments(
        model_name_or_path=args.model,
        tokenizer_name=args.tokenizer if args.tokenizer else args.model
    )

    data_args = DataTrainingArguments(
        validation_file=args.file_path,
        test_file=args.file_path,
        text_column='input',
        summary_column='output',
        # source_prefix=prefix,
        overwrite_cache=args.cache,
        max_target_length=args.max_len,
        val_max_target_length=args.max_len,
        pad_to_max_length=True,
        ignore_pad_token_for_loss=False
    )
    results, labels = main(training_args, model_args, data_args)

    metric = RougeMetric()
    r_1_f = []
    r_2_f = []
    r_l_f = []

    for text, label in zip(results, labels):
        rouges = metric.evaluation(text, label)

        r_1_f.append(rouges["rouge-1"]["f"])
        r_2_f.append(rouges["rouge-2"]["f"])
        r_l_f.append(rouges["rouge-l"]["f"])

    result_metrics = {
        "rouge-1-f": mean(r_1_f),
        "rouge-2-f": mean(r_2_f),
        "rouge-l-f": mean(r_l_f),
    }

    return results, result_metrics

class NsmlCallback(TrainerCallback):
    """NSML Callback for Huggingface Trainer"""
    def __init__(self) -> None:
        super().__init__()
        self.count = 0

    def on_train_begin(self, args, state, control, **kwargs):
        print("Starting NSML callback")

    def on_log(self, args, state, control, **kwargs):
        print('On log!!!!!')
        self.count += 1
        # print(f'best metric={state.best_metric}')
        nsml.save(f'model-{state.global_step}')

# def hf_train(args, model, tokenizer, train_dataset):
def hf_train(args, prefix="", subject='all'):
    training_args = Seq2SeqTrainingArguments(
        do_train=True,
        do_eval=False,
        do_predict=False,
        output_dir=args.save_path,
        overwrite_output_dir=True,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.gradient_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        save_strategy='no',
        seed=args.seed,
        fp16=args.fp16,
        fp16_opt_level='O1',
        report_to="wandb",
    )

    model_args = ModelArguments(
        model_name_or_path=args.model,
        tokenizer_name=args.tokenizer if args.tokenizer else args.model,
    )

    print(f'run_summarizatoin: prefix is {prefix}')

    data_args = DataTrainingArguments(
        train_file=args.file_path,
        text_column='input',
        summary_column='output',
        max_source_length=args.max_len,
        max_target_length=args.max_target_len,
        source_prefix=prefix,
        preprocessing_num_workers=4,
        overwrite_cache=args.overwrite_cache,
        pad_to_max_length=True
    )

    main(training_args, model_args, data_args, subject, access_token, args.ph)