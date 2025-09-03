import numpy as np
import collections

def postprocess_qa_predictions(examples, features, raw_predictions, tokenizer, n_best_size=20, max_answer_length=30):
    all_start_logits, all_end_logits = raw_predictions
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, f in enumerate(features):
        features_per_example[example_id_to_index[f["example_id"]]].append(i)

    predictions = {}
    for example_index, example in enumerate(examples):
        feature_indices = features_per_example[example_index]
        min_null_score = None
        valid_answers = []
        context = example["context"]

        for feature_index in feature_indices:
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            offsets = features[feature_index]["offset_mapping"]

            cls_index = features[feature_index]["input_ids"].index(tokenizer.cls_token_id)
            feature_null_score = start_logits[cls_index] + end_logits[cls_index]
            if min_null_score is None or min_null_score < feature_null_score:
                min_null_score = feature_null_score

            start_indexes = np.argsort(start_logits)[-1:-n_best_size-1:-1].tolist()
            end_indexes = np.argsort(end_logits)[-1:-n_best_size-1:-1].tolist()

            for start_index in start_indexes:
                for end_index in end_indexes:
                    if start_index >= len(offsets) or end_index >= len(offsets):
                        continue
                    if offsets[start_index] is None or offsets[end_index] is None:
                        continue
                    if end_index < start_index:
                        continue
                    if end_index - start_index + 1 > max_answer_length:
                        continue

                    start_char = offsets[start_index][0]
                    end_char = offsets[end_index][1]
                    text = context[start_char:end_char]
                    score = start_logits[start_index] + end_logits[end_index]
                    valid_answers.append({"score": float(score), "text": text})

        if len(valid_answers) > 0:
            best_non_null = max(valid_answers, key=lambda x: x["score"])
            if min_null_score is not None and min_null_score > best_non_null["score"]:
                predictions[example["id"]] = ""
            else:
                predictions[example["id"]] = best_non_null["text"]
        else:
            predictions[example["id"]] = ""

    return predictions


def compute_metrics(eval_preds, dataset, tokenized_valid, tokenizer, metric, n_best_size=20, max_answer_length=30):
    preds = postprocess_qa_predictions(
        examples=dataset["validation"],
        features=tokenized_valid,
        raw_predictions=eval_preds,
        tokenizer=tokenizer,
        n_best_size=n_best_size,
        max_answer_length=max_answer_length,
    )
    refs = [{"id": ex["id"], "answers": ex["answers"]} for ex in dataset["validation"]]
    
    return metric.compute(
        predictions=[{"id": k, "prediction_text": v} for k, v in preds.items()],
        references=refs,
    )
