import torch
from transformers import (
    AutoTokenizer, AutoModelForQuestionAnswering,
    TrainingArguments, Trainer, default_data_collator
)
import evaluate

from .data_loader import get_dataset
from .preprocessing import prepare_train_features, prepare_validation_features
from .utils import compute_metrics
import os

def run_train(config: dict):
    """
    Orchestrates the complete training pipeline: pre-processing data, configuring TrainingArguments,
    initializing the Trainer, executing training, and saving the model/tokenizer.
    """
    model_name = config["model"]["name"]
    dataset = get_dataset(config)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    max_length = config["data"]["max_length"]
    doc_stride = config["data"]["doc_stride"]

    # Apply pre-processing on the training set
    tokenized_train = dataset["train"].map(
        lambda x: prepare_train_features(x, tokenizer, max_length, doc_stride),
        batched=True,
        remove_columns=dataset["train"].column_names,
    )
    
    # Apply pre-processing on the validation set
    tokenized_valid = dataset["validation"].map(
        lambda x: prepare_validation_features(x, tokenizer, max_length, doc_stride),
        batched=True,
        remove_columns=dataset["validation"].column_names,
    )
    
    bsz = config["training"]["bsz"]
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    metric = evaluate.load(config["training"].get("metric_name", "squad_v2"))

    # Define training hyperparameters
    args = TrainingArguments(
        output_dir=config["training"]["output_dir"],
        eval_strategy=config["training"]["eval_strategy"],
        eval_steps=config["training"]["eval_steps"],
        logging_steps=config["training"]["logging_steps"],
        save_steps=config["training"]["save_steps"],
        save_total_limit=config["training"]["save_total_limit"],
        learning_rate=float(config["training"]["learning_rate"]),
        num_train_epochs=config["training"]["num_train_epochs"],
        per_device_train_batch_size=bsz,
        per_device_eval_batch_size=bsz,
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        weight_decay=config["training"]["weight_decay"],
        fp16=config["training"]["fp16"],
        bf16=config["training"]["bf16"],
        report_to="none",  # Disable W&B or Tensorboard reporting to simplify dependencies
        dataloader_num_workers=config["training"]["dataloader_num_workers"],
    )

    # Initialize HuggingFace Trainer instance
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_valid,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
        compute_metrics=lambda eval_preds: compute_metrics(
            eval_preds, dataset, tokenized_valid, tokenizer, metric,
            n_best_size=config["data"].get("n_best_size", 20),
            max_answer_length=config["data"].get("max_answer_length", 30)
        ),
    )

    # Commence training
    trainer.train()
    
    # Evaluate model upon training completion
    metrics = trainer.evaluate()
    print("Evaluation metrics:", metrics)

    # Save final model checkpoint alongside its tokenizer
    os.makedirs(config["model"]["ckpt_path"], exist_ok=True)
    trainer.save_model(config["model"]["ckpt_path"])
    tokenizer.save_pretrained(config["model"]["ckpt_path"])
