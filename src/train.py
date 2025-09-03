import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForQuestionAnswering,
    TrainingArguments, Trainer, default_data_collator
)
import evaluate

from preprocessing import prepare_train_features, prepare_validation_features
from utils import compute_metrics

def main():
    model_name = "xlm-roberta-large"
    dataset = load_dataset("taidng/UIT-ViQuAD2.0")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    tokenized_train = dataset["train"].map(
        lambda x: prepare_train_features(x, tokenizer),
        batched=True,
        remove_columns=dataset["train"].column_names,
    )
    tokenized_valid = dataset["validation"].map(
        lambda x: prepare_validation_features(x, tokenizer),
        batched=True,
        remove_columns=dataset["validation"].column_names,
    )
    
    bsz = 16
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    metric = evaluate.load("squad_v2")

    args = TrainingArguments(
        output_dir="xlmr-large-viquad",
        eval_strategy="steps",
        eval_steps=500,
        logging_steps=100,
        save_steps=1000,
        save_total_limit=2,
        learning_rate=2e-5,
        num_train_epochs=5,
        per_device_train_batch_size=bsz,
        per_device_eval_batch_size=bsz,
        gradient_accumulation_steps=1,
        weight_decay=0.01,
        fp16=True,
        bf16=False,
        report_to="none",
        dataloader_num_workers=4,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_valid,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
        compute_metrics=lambda eval_preds: compute_metrics(eval_preds, dataset, tokenized_valid, tokenizer, metric),
    )

    trainer.train()
    
    
    metrics = trainer.evaluate()
    print(metrics)

    trainer.save_model("models/xlmr-large-viquad-final")
    tokenizer.save_pretrained("models/xlmr-large-viquad-final")


if __name__ == "__main__":
    main()
