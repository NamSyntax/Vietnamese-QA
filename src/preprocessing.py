from transformers import AutoTokenizer

def prepare_train_features(examples, tokenizer, max_length=384, doc_stride=128):
    """
    Pre-processes the training dataset:
    Tokenizes questions and contexts, truncates lengthy contexts using a sliding window,
    and maps the answer start/end positions to token indices.
    """
    tokenized_examples = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second", # Truncate only the context, preserve the full question
        max_length=max_length,
        stride=doc_stride,        # Stride for sliding window when context exceeds max_length
        return_overflowing_tokens=True,
        return_offsets_mapping=True, # Used to map tokens back to original string characters
        padding="max_length",
    )

    # Extract mapping from split chunks back to original examples
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples["offset_mapping"]

    start_positions, end_positions = [], []

    for i, offsets in enumerate(offset_mapping):
        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]
        sequence_ids = tokenized_examples.sequence_ids(i)

        # Handle unanswerable cases by mapping to the [CLS] token index
        if len(answers["answer_start"]) == 0:
            cls_index = tokenized_examples["input_ids"][i].index(tokenizer.cls_token_id)
            start_positions.append(cls_index)
            end_positions.append(cls_index)
            # Set question token offsets to None (ignored for context evaluation)
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == 1 else None) for k, o in enumerate(offsets)
            ]
            continue

        start_char = answers["answer_start"][0]
        end_char = start_char + len(answers["text"][0])

        # Find the start and end of the context within input_ids
        token_start_index = 0
        while sequence_ids[token_start_index] != 1:
            token_start_index += 1
        token_end_index = len(tokenized_examples["input_ids"][i]) - 1
        while sequence_ids[token_end_index] != 1:
            token_end_index -= 1

        # Check if the answer is completely contained within the current context chunk
        if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
            # If the answer is truncated, point to the [CLS] index
            cls_index = tokenized_examples["input_ids"][i].index(tokenizer.cls_token_id)
            start_positions.append(cls_index)
            end_positions.append(cls_index)
        else:
            # Otherwise, locate the exact start and end token indices
            while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                token_start_index += 1
            start_positions.append(token_start_index - 1)
            
            while offsets[token_end_index][1] >= end_char:
                token_end_index -= 1
            end_positions.append(token_end_index + 1)

        # Set question tokens offset to None
        tokenized_examples["offset_mapping"][i] = [
            (o if sequence_ids[k] == 1 else None) for k, o in enumerate(offsets)
        ]

    tokenized_examples["start_positions"] = start_positions
    tokenized_examples["end_positions"] = end_positions
    return tokenized_examples


def prepare_validation_features(examples, tokenizer, max_length=384, doc_stride=128):
    """
    Pre-processes the validation dataset:
    Only performs tokenization and records mapping details.
    Does NOT assign start/end positions, enabling accurate evaluation later.
    """
    tokenized_examples = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    tokenized_examples["example_id"] = []

    for i in range(len(tokenized_examples["input_ids"])):
        # Store the ID of the original example corresponding to each chunk
        sample_index = sample_mapping[i]
        tokenized_examples["example_id"].append(examples["id"][sample_index])

        # Exclude offsets that belong to the question
        sequence_ids = tokenized_examples.sequence_ids(i)
        offsets = tokenized_examples["offset_mapping"][i]
        tokenized_examples["offset_mapping"][i] = [
            (o if sequence_ids[k] == 1 else None) for k, o in enumerate(offsets)
        ]

    return tokenized_examples
