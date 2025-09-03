import evaluate
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, Trainer
from datasets import load_dataset
from preprocessing import prepare_validation_features
from utils import compute_metrics

def main():
    dataset = load_dataset("taidng/UIT-ViQuAD2.0")
    ckpt_path = "models/xlmr-large-viquad-final"
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path, use_fast=True)
    model = AutoModelForQuestionAnswering.from_pretrained(ckpt_path)
    metric = evaluate.load("squad_v2")

    tokenized_valid = dataset["validation"].map(
        lambda x: prepare_validation_features(x, tokenizer),
        batched=True,
        remove_columns=dataset["validation"].column_names,
    )

    trainer = Trainer(model=model, tokenizer=tokenizer)
    raw_preds = trainer.predict(tokenized_valid)

    results = compute_metrics(raw_preds.predictions, dataset, tokenized_valid, tokenizer, metric)
    print(results)

if __name__ == "__main__":
    main()
