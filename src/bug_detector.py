"""
Bug detection model — trains a classifier on buggy vs clean code snippets.

Uses a pre-trained code model as encoder with a classification head on top.
Trained on datasets like Defects4J or custom curated buggy code.

Usage:
    python bug_detector.py --config ../configs/training.yaml
"""

import argparse
import yaml
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    AutoModel,
)
from datasets import load_dataset, DatasetDict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def prepare_dataset(config):
    """Load and prepare buggy/clean code dataset."""
    dataset_name = config.get("dataset", "code_x_glue_tc_defect_detection")

    print(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, trust_remote_code=True)

    # the code_x_glue dataset has target=1 for buggy, target=0 for clean
    # field is usually "func" or "code"
    splits = DatasetDict()
    if "train" in dataset:
        splits["train"] = dataset["train"]
    if "validation" in dataset:
        splits["validation"] = dataset["validation"]
    elif "test" in dataset:
        splits["validation"] = dataset["test"]

    return splits


def compute_metrics(pred):
    """Compute accuracy, precision, recall, f1 for bug detection."""
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary"
    )
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def tokenize_dataset(dataset, tokenizer, text_field="func", max_length=512):
    def tokenize_fn(examples):
        return tokenizer(
            examples[text_field],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )

    tokenized = dataset.map(tokenize_fn, batched=True)

    # make sure target column is named "labels"
    if "target" in tokenized.column_names:
        tokenized = tokenized.rename_column("target", "labels")

    return tokenized


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Training config YAML")
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    train_cfg = config["training"]

    model_name = config.get("model_name", "microsoft/codebert-base")
    print(f"Using model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        problem_type="single_label_classification",
    )

    dataset = prepare_dataset(config)

    text_field = config.get("text_column", "func")
    tokenized_train = tokenize_dataset(
        dataset["train"], tokenizer, text_field=text_field,
        max_length=config.get("max_length", 512)
    )

    output_dir = config.get("output_dir", "./checkpoints/bug-detector")

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=train_cfg.get("epochs", 5),
        per_device_train_batch_size=train_cfg.get("batch_size", 16),
        per_device_eval_batch_size=train_cfg.get("eval_batch_size", 32),
        learning_rate=float(train_cfg.get("lr", 2e-5)),
        warmup_steps=train_cfg.get("warmup_steps", 200),
        logging_steps=train_cfg.get("logging_steps", 50),
        evaluation_strategy="steps" if "validation" in dataset else "no",
        eval_steps=train_cfg.get("eval_steps", 500) if "validation" in dataset else None,
        save_steps=train_cfg.get("save_steps", 500),
        fp16=config.get("fp16", True),
        load_best_model_at_end=True if "validation" in dataset else False,
        metric_for_best_model="f1" if "validation" in dataset else None,
        report_to="none",
        dataloader_pin_memory=False,
    )

    eval_dataset = None
    if "validation" in dataset:
        eval_dataset = tokenize_dataset(
            dataset["validation"], tokenizer, text_field=text_field,
            max_length=config.get("max_length", 512)
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    if args.eval_only:
        print("Running evaluation...")
        results = trainer.evaluate()
        print(results)
    else:
        print("Training bug detector...")
        trainer.train()

        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(f"Model saved to {output_dir}")

        if eval_dataset:
            results = trainer.evaluate()
            print(f"Final eval results: {results}")


if __name__ == "__main__":
    main()
