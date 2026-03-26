import evaluate
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, Trainer

from .data_loader import get_dataset
from .preprocessing import prepare_validation_features
from .utils import compute_metrics

def run_evaluate(config: dict):
    """
    Loads the model and runs the evaluation process on the validation dataset.
    Uses squad_v2 metric by default to compute scores like Exact Match (EM) and F1.
    """
    # Fetch dataset splits
    dataset = get_dataset(config)
    
    # Load tokenizer and trained model checkpoint
    ckpt_path = config["model"]["ckpt_path"]
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path, use_fast=True)
    model = AutoModelForQuestionAnswering.from_pretrained(ckpt_path)
    
    # Load metric pipeline from the evaluate library
    metric = evaluate.load(config["training"].get("metric_name", "squad_v2"))

    max_length = config["data"]["max_length"]
    doc_stride = config["data"]["doc_stride"]

    # Preprocess validation data features
    tokenized_valid = dataset["validation"].map(
        lambda x: prepare_validation_features(x, tokenizer, max_length, doc_stride),
        batched=True,
        remove_columns=dataset["validation"].column_names,
    )

    # Initialize a Trainer instance specifically for predictions
    trainer = Trainer(model=model, tokenizer=tokenizer)
    raw_preds = trainer.predict(tokenized_valid)

    # Compute final metrics using predictions
    results = compute_metrics(
        raw_preds.predictions, dataset, tokenized_valid, tokenizer, metric,
        n_best_size=config["data"].get("n_best_size", 20),
        max_answer_length=config["data"].get("max_answer_length", 30)
    )
    
    print("Evaluation Results:")
    print(results)
    return results
