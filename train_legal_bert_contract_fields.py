#!/usr/bin/env python3
"""
Train LEGAL-BERT to identify contract parties and date-related spans.

Labels:

    O
    B-PARTIES / I-PARTIES
    B-EFFECTIVE_DATE / I-EFFECTIVE_DATE
    B-EXPIRATION_DATE / I-EXPIRATION_DATE

Unlike extractive QA, token classification can represent multiple answer spans
inside one paragraph. The script aggregates all matching QAs for the same
context before creating token labels.

Expected input: CUAD/AOK-style SQuAD JSON with data -> paragraphs -> qas.

Example:

    python train_legal_bert_contract_fields.py \
        --data_json /path/to/CUAD_v1.json \
        --output_dir ./legal-bert-contract-fields \
        --num_train_epochs 3 \
        --fp16
"""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
    set_seed,
)


LOGGER = logging.getLogger(__name__)

DEFAULT_MODEL = "nlpaueb/legal-bert-base-uncased"

CATEGORIES = [
    "PARTIES",
    "EFFECTIVE_DATE",
    "EXPIRATION_DATE",
]

LABELS = ["O"]
for category in CATEGORIES:
    LABELS.extend([f"B-{category}", f"I-{category}"])

LABEL2ID = {label: index for index, label in enumerate(LABELS)}
ID2LABEL = {index: label for label, index in LABEL2ID.items()}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict) or not isinstance(data.get("data"), list):
        raise ValueError("Dataset must contain a top-level 'data' list.")

    return data


def qa_category(qa: dict[str, Any]) -> str | None:
    """Identify the supported category from an ID or question string."""

    qa_id = str(qa.get("id", "")).casefold()
    question = str(qa.get("question", "")).casefold()
    combined = f"{qa_id} {question}"

    if "parties" in combined or "two or more parties" in combined:
        return "PARTIES"
    if "effective date" in combined:
        return "EFFECTIVE_DATE"
    if "expiration date" in combined or "expiry date" in combined:
        return "EXPIRATION_DATE"

    return None


def get_answers(context: str, qa: dict[str, Any]) -> list[dict[str, int]]:
    """Return every answer span that can be located in the context."""

    if qa.get("is_impossible", False):
        return []

    raw_answers = qa.get("answers") or []

    if isinstance(raw_answers, dict):
        raw_answers = [
            {"text": text, "answer_start": start}
            for text, start in zip(
                raw_answers.get("text", []),
                raw_answers.get("answer_start", []),
            )
        ]

    valid = []

    for answer in raw_answers:
        text = answer.get("text")
        start = answer.get("answer_start")

        if not isinstance(text, str) or not text:
            continue
        if not isinstance(start, int):
            continue

        if (
            start >= 0
            and context[start:start + len(text)] == text
        ):
            repaired_start = start
        else:
            repaired_start = context.find(text)

        if repaired_start < 0:
            LOGGER.warning(
                "Skipping answer %r for QA %s; text is not in context.",
                text,
                qa.get("id", "unknown"),
            )
            continue

        valid.append(
            {
                "start": repaired_start,
                "end": repaired_start + len(text),
            }
        )

    # Remove duplicate spans.
    return list({(item["start"], item["end"]): item for item in valid}.values())


def build_examples(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Aggregate all supported QA annotations for each context paragraph."""

    grouped: dict[tuple[str, int], dict[str, Any]] = {}

    for contract_index, contract in enumerate(payload["data"]):
        title = str(contract.get("title") or f"contract-{contract_index}")

        for paragraph_index, paragraph in enumerate(
            contract.get("paragraphs", [])
        ):
            context = paragraph.get("context", "")

            if not isinstance(context, str) or not context:
                continue

            key = (title, paragraph_index)
            grouped[key] = {
                "id": f"{contract_index}-{paragraph_index}",
                "title": title,
                "context": context,
                "spans": {
                    category: []
                    for category in CATEGORIES
                },
            }

            for qa in paragraph.get("qas", []):
                category = qa_category(qa)

                if category is None:
                    continue

                grouped[key]["spans"][category].extend(
                    get_answers(context, qa)
                )

            for category in CATEGORIES:
                unique = {
                    (span["start"], span["end"]): span
                    for span in grouped[key]["spans"][category]
                }
                grouped[key]["spans"][category] = list(unique.values())

    examples = list(grouped.values())

    if not examples:
        raise ValueError("No supported Parties/date records were found.")

    positive_counts = {
        category: sum(bool(item["spans"][category]) for item in examples)
        for category in CATEGORIES
    }

    LOGGER.info("Context records: %s", len(examples))
    LOGGER.info("Positive records by category: %s", positive_counts)

    return examples


def split_by_contract(
    examples: list[dict[str, Any]],
    eval_size: float,
    seed: int,
):
    grouped = defaultdict(list)

    for example in examples:
        grouped[example["title"]].append(example)

    titles = list(grouped)
    if len(titles) < 2:
        raise ValueError("At least two contracts are required for evaluation.")

    random.Random(seed).shuffle(titles)
    eval_count = min(max(1, round(len(titles) * eval_size)), len(titles) - 1)
    eval_titles = set(titles[:eval_count])

    train = [item for item in examples if item["title"] not in eval_titles]
    evaluation = [item for item in examples if item["title"] in eval_titles]
    return train, evaluation


def align_labels(
    attention_mask,
    offsets,
    spans_by_category,
):
    labels = [-100] * len(offsets)

    valid_positions = [
        index
        for index, (start, end) in enumerate(offsets)
        if attention_mask[index] == 1 and end > start
    ]

    for index in valid_positions:
        labels[index] = LABEL2ID["O"]

    # Label longer spans first so overlapping annotations remain stable.
    for category in CATEGORIES:
        spans = sorted(
            spans_by_category.get(category, []),
            key=lambda span: (span["start"], -span["end"]),
        )

        for span in spans:
            token_positions = [
                index
                for index in valid_positions
                if offsets[index][0] < span["end"]
                and offsets[index][1] > span["start"]
            ]

            if not token_positions:
                continue

            first = token_positions[0]
            first_label = LABEL2ID[f"B-{category}"]
            inside_label = LABEL2ID[f"I-{category}"]

            # Do not overwrite a different previously assigned entity.
            if labels[first] != LABEL2ID["O"]:
                continue

            labels[first] = first_label

            for index in token_positions[1:]:
                if labels[index] == LABEL2ID["O"]:
                    labels[index] = inside_label

    return labels


def tokenize_and_align(
    examples,
    tokenizer,
    max_length,
    doc_stride,
):
    tokenized = tokenizer(
        examples["context"],
        truncation=True,
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized.pop("offset_mapping")
    labels = []

    for feature_index, offsets in enumerate(offset_mapping):
        sample_index = sample_mapping[feature_index]
        labels.append(
            align_labels(
                attention_mask=tokenized["attention_mask"][feature_index],
                offsets=offsets,
                spans_by_category=examples["spans"][sample_index],
            )
        )

    tokenized["labels"] = labels
    return tokenized


def compute_metrics(eval_prediction):
    logits, labels = eval_prediction
    predictions = np.argmax(logits, axis=-1).reshape(-1)
    labels = labels.reshape(-1)
    mask = labels != -100

    predictions = predictions[mask]
    labels = labels[mask]

    entity_ids = np.array([
        LABEL2ID[label]
        for label in LABELS
        if label != "O"
    ])

    actual = np.isin(labels, entity_ids)
    predicted = np.isin(predictions, entity_ids)
    true_positive = np.logical_and(actual, predicted).sum()
    false_positive = np.logical_and(~actual, predicted).sum()
    false_negative = np.logical_and(actual, ~predicted).sum()

    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)

    return {
        "entity_token_precision": float(precision),
        "entity_token_recall": float(recall),
        "entity_token_f1": float(f1),
    }


def make_training_args(args, output_dir):
    parameters = inspect.signature(TrainingArguments.__init__).parameters
    kwargs = {}

    def add(name, value):
        if name in parameters:
            kwargs[name] = value

    add("output_dir", str(output_dir))
    add("overwrite_output_dir", args.overwrite_output_dir)
    add("num_train_epochs", args.num_train_epochs)
    add("per_device_train_batch_size", args.per_device_train_batch_size)
    add("per_device_eval_batch_size", args.per_device_eval_batch_size)
    add("gradient_accumulation_steps", args.gradient_accumulation_steps)
    add("learning_rate", args.learning_rate)
    add("weight_decay", args.weight_decay)
    add("logging_steps", args.logging_steps)
    add("seed", args.seed)
    add("data_seed", args.seed)
    add("fp16", args.fp16 and torch.cuda.is_available())
    add("gradient_checkpointing", args.gradient_checkpointing)
    add("report_to", "none")
    add("save_total_limit", 2)

    if "warmup_ratio" in parameters:
        add("warmup_ratio", args.warmup_ratio)
    elif "warmup_steps" in parameters:
        add("warmup_steps", 0)

    if "eval_strategy" in parameters:
        add("eval_strategy", "epoch")
        add("save_strategy", "epoch")
        add("load_best_model_at_end", True)
        add("metric_for_best_model", "eval_loss")
        add("greater_is_better", False)
    elif "evaluation_strategy" in parameters:
        add("evaluation_strategy", "epoch")
        add("save_strategy", "epoch")
        add("load_best_model_at_end", True)
        add("metric_for_best_model", "eval_loss")
        add("greater_is_better", False)

    if "optim" in parameters:
        add("optim", "adamw_torch")

    return TrainingArguments(**kwargs)


def make_trainer(model, tokenizer, training_args, train_dataset, eval_dataset):
    trainer_parameters = inspect.signature(Trainer.__init__).parameters
    kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": DataCollatorForTokenClassification(
            tokenizer=tokenizer,
        ),
        "compute_metrics": compute_metrics,
    }

    if "processing_class" in trainer_parameters:
        kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_parameters:
        kwargs["tokenizer"] = tokenizer

    return Trainer(**kwargs)


def save_json(path, value):
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, indent=2, ensure_ascii=False)
        file.write("\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train LEGAL-BERT on parties and contract dates."
    )
    parser.add_argument(
        "--data_json",
        "--cuad_json",
        dest="data_json",
        required=True,
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name", default=DEFAULT_MODEL)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--doc_stride", type=int, default=128)
    parser.add_argument("--eval_size", type=float, default=0.1)
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--overwrite_output_dir", action="store_true")
    parser.add_argument("--resume_from_checkpoint", default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )

    if args.doc_stride < 0 or args.doc_stride >= args.max_length:
        raise ValueError("doc_stride must be smaller than max_length.")
    if not 0 < args.eval_size < 1:
        raise ValueError("eval_size must be between 0 and 1.")

    output_dir = Path(args.output_dir)

    if (
        output_dir.exists()
        and any(output_dir.iterdir())
        and not args.overwrite_output_dir
        and not args.resume_from_checkpoint
    ):
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. "
            "Use --overwrite_output_dir or a new directory."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    payload = load_json(Path(args.data_json))
    all_examples = build_examples(payload)
    train_examples, eval_examples = split_by_contract(
        all_examples,
        args.eval_size,
        args.seed,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        use_fast=True,
    )

    train_dataset = Dataset.from_list(train_examples)
    eval_dataset = Dataset.from_list(eval_examples)

    train_dataset = train_dataset.map(
        lambda batch: tokenize_and_align(
            batch,
            tokenizer,
            args.max_length,
            args.doc_stride,
        ),
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing training contexts",
    )

    eval_dataset = eval_dataset.map(
        lambda batch: tokenize_and_align(
            batch,
            tokenizer,
            args.max_length,
            args.doc_stride,
        ),
        batched=True,
        remove_columns=eval_dataset.column_names,
        desc="Tokenizing evaluation contexts",
    )

    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name,
        num_labels=len(LABELS),
        label2id=LABEL2ID,
        id2label=ID2LABEL,
    )

    training_args = make_training_args(args, output_dir)
    trainer = make_trainer(
        model,
        tokenizer,
        training_args,
        train_dataset,
        eval_dataset,
    )

    if args.resume_from_checkpoint:
        train_result = trainer.train(
            resume_from_checkpoint=args.resume_from_checkpoint,
        )
    else:
        train_result = trainer.train()

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    eval_metrics = trainer.evaluate()

    save_json(
        output_dir / "label_mapping.json",
        {
            "labels": LABELS,
            "label2id": LABEL2ID,
            "id2label": ID2LABEL,
        },
    )

    save_json(
        output_dir / "training_config.json",
        {
            "task": "contract-fields-token-classification",
            "categories": CATEGORIES,
            "model_name": args.model_name,
            "data_json": args.data_json,
            "max_length": args.max_length,
            "doc_stride": args.doc_stride,
            "seed": args.seed,
            "train_records": len(train_examples),
            "eval_records": len(eval_examples),
            "train_features": len(train_dataset),
            "eval_features": len(eval_dataset),
        },
    )

    save_json(output_dir / "train_metrics.json", train_result.metrics)
    save_json(output_dir / "eval_metrics.json", eval_metrics)

    LOGGER.info("Training complete: %s", output_dir)


if __name__ == "__main__":
    main()
