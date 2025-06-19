"""
Code generation fine-tuning with LoRA on ROCm.
Supports CodeGen-350M and StarCoder-small.

Usage:
    python codegen_finetune.py --config ../configs/codegen_lora.yaml
"""

import os
import argparse
import yaml
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def tokenize_dataset(dataset, tokenizer, max_length=512):
    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )

    tokenized = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=dataset.column_names,
    )
    return tokenized


def setup_lora(model, config):
    """Apply LoRA adapters to the model."""
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.get("r", 16),
        lora_alpha=config.get("alpha", 32),
        lora_dropout=config.get("dropout", 0.1),
        target_modules=config.get("target_modules", ["q_proj", "v_proj"]),
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to LoRA config YAML")
    parser.add_argument("--resume", default=None, help="Resume from checkpoint")
    args = parser.parse_args()

    config = load_config(args.config)
    train_cfg = config["training"]
    lora_cfg = config.get("lora", {})

    print(f"Loading model: {config['model_name']}")
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])

    # some codegen models don't have a pad token, super annoying
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        config["model_name"],
        torch_dtype=torch.float16 if config.get("fp16", True) else torch.float32,
        device_map="auto",
    )

    model = setup_lora(model, lora_cfg)
    model.gradient_checkpointing_enable()

    # load dataset
    print(f"Loading dataset: {config['dataset']}")
    if config["dataset"] == "codeparrot/github-code":
        dataset = load_dataset(
            config["dataset"],
            split="train[:50000]",  # subset, full is huge
            trust_remote_code=True,
        )
    else:
        dataset = load_dataset(config["dataset"], split="train")

    # figure out which column has the code
    text_col = config.get("text_column", "content")
    if text_col not in dataset.column_names:
        # try common alternatives
        for col in ["code", "text", "content", "func_code"]:
            if col in dataset.column_names:
                text_col = col
                break
    dataset = dataset.rename_column(text_col, "text")

    tokenized = tokenize_dataset(
        dataset,
        tokenizer,
        max_length=config.get("max_length", 512),
    )

    output_dir = config.get("output_dir", "./checkpoints/codegen-lora")

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=train_cfg.get("epochs", 3),
        per_device_train_batch_size=train_cfg.get("batch_size", 4),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation", 4),
        learning_rate=float(train_cfg.get("lr", 2e-4)),
        warmup_steps=train_cfg.get("warmup_steps", 100),
        logging_steps=train_cfg.get("logging_steps", 25),
        save_steps=train_cfg.get("save_steps", 500),
        fp16=config.get("fp16", True),
        optim="adamw_torch",
        report_to="none",  # change to "wandb" if you want logging
        max_grad_norm=1.0,
        dataloader_pin_memory=False,  # ROCm doesn't love this
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator,
    )

    print("Starting training...")
    trainer.train(resume_from_checkpoint=args.resume)

    # save the final adapter
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")


if __name__ == "__main__":
    main()

