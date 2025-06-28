"""
Code summarization — generate docstrings/comments from function bodies.

Fine-tunes a small encoder-decoder model (CodeT5-small) on code-dataset pairs.
Works on ROCm with the usual caveats.

Usage:
    python code_summarizer.py --config ../configs/training.yaml
    python code_summarizer.py --generate --model ./checkpoints/summarizer --input "def add(a,b): return a+b"
"""

import argparse
import yaml
import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)
from datasets import load_dataset
import evaluate


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def prepare_dataset(tokenizer, config):
    """Load code/summary pairs and tokenize."""
    dataset_name = config.get("dataset", "code_x_glue_tc_nl_code_search_adv")
    print(f"Loading dataset: {dataset_name}")

    dataset = load_dataset(dataset_name, split="train[:10000]", trust_remote_code=True)

    # find the right columns
    code_col = None
    summary_col = None
    for col in dataset.column_names:
        if col in ("code", "code_tokens", "func_code"):
            code_col = col
        if col in ("docstring", "docstring_tokens", "summary", "nl"):
            summary_col = col

    if not code_col or not summary_col:
        print(f"Columns: {dataset.column_names}")
        raise ValueError("Can't find code/summary columns, check dataset format")

    print(f"Using columns: code={code_col}, summary={summary_col}")

    max_input = config.get("max_input_length", 256)
    max_target = config.get("max_target_length", 128)

    def tokenize_fn(examples):
        # if code_tokens is a list, join it
        code = examples[code_col]
        if isinstance(code, list):
            code = [" ".join(c) if isinstance(c, list) else c for c in code]

        summary = examples[summary_col]
        if isinstance(summary, list):
            summary = [" ".join(s) if isinstance(s, list) else s for s in summary]

        model_inputs = tokenizer(
            code, truncation=True, max_length=max_input, padding="max_length"
        )
        labels = tokenizer(
            summary, truncation=True, max_length=max_target, padding="max_length"
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names)
    return tokenized


def generate_summary(model, tokenizer, code):
    """Generate a docstring for a code snippet."""
    inputs = tokenizer(code, return_tensors="pt", truncation=True, max_length=256)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=128,
            num_beams=4,
            early_stopping=True,
        )

    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="../configs/training.yaml")
    parser.add_argument("--generate", action="store_true", help="Generate mode")
    parser.add_argument("--model", default=None)
    parser.add_argument("--input", default=None, help="Code to summarize")
    args = parser.parse_args()

    if args.generate:
        # inference mode
        model_path = args.model or "./checkpoints/summarizer"
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_path, torch_dtype=torch.float16, device_map="auto"
        )
        model.eval()

        if args.input:
            print(generate_summary(model, tokenizer, args.input))
        else:
            demos = [
                "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
                "def merge_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    mid = len(arr) // 2\n    left = merge_sort(arr[:mid])\n    right = merge_sort(arr[mid:])\n    return merge(left, right)",
            ]
            for code in demos:
                summary = generate_summary(model, tokenizer, code)
                print(f"Code:\n{code}\nSummary: {summary}\n")
        return

    # training mode
    config = load_config(args.config)
    train_cfg = config["training"]

    model_name = config.get("model_name", "Salesforce/codet5-small")
    print(f"Using model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    tokenized = prepare_dataset(tokenizer, config)

    output_dir = config.get("output_dir", "./checkpoints/summarizer")

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=train_cfg.get("epochs", 3),
        per_device_train_batch_size=train_cfg.get("batch_size", 8),
        learning_rate=float(train_cfg.get("lr", 5e-5)),
        warmup_steps=train_cfg.get("warmup_steps", 100),
        logging_steps=train_cfg.get("logging_steps", 50),
        save_steps=train_cfg.get("save_steps", 500),
        fp16=config.get("fp16", True),
        predict_with_generate=True,
        report_to="none",
        dataloader_pin_memory=False,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator,
    )

    print("Training summarizer...")
    trainer.train()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")


if __name__ == "__main__":
    main()
