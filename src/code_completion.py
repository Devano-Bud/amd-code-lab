"""
Code completion inference pipeline.
Supports Python and JavaScript. Loads a fine-tuned model and generates completions.

Usage:
    python code_completion.py --model ./checkpoints/codegen-lora
    python code_completion.py --model ./checkpoints/codegen-lora --interactive
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def load_model(base_model, adapter_path=None):
    """Load base model, optionally with LoRA adapter."""
    tokenizer = AutoTokenizer.from_pretrained(
        adapter_path or base_model
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()  # merge for faster inference

    model.eval()
    return model, tokenizer


def generate_completion(model, tokenizer, prefix, max_new_tokens=128, temperature=0.2):
    """Generate a code completion given a prefix."""
    inputs = tokenizer(prefix, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            top_p=0.95,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # only return the new tokens
    completion = generated[len(prefix):]
    return completion


def interactive_mode(model, tokenizer):
    """Interactive code completion. Paste code and get completions."""
    print("Code completion mode. Type code and press Enter twice to get completion.")
    print("Type 'quit' to exit.\n")

    while True:
        lines = []
        while True:
            try:
                line = input()
            except EOFError:
                return
            if line.strip() == "quit":
                return
            if line == "" and lines:
                break
            lines.append(line)

        prefix = "\n".join(lines)
        if not prefix.strip():
            continue

        completion = generate_completion(model, tokenizer, prefix)
        # show only until first blank line or end of function
        result_lines = completion.split("\n")
        display = []
        for rl in result_lines:
            display.append(rl)
            # stop at logical breaks
            if len(display) > 1 and rl.strip() == "":
                break

        print("--- completion ---")
        print("\n".join(display))
        print("--- end ---\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Base model name or path")
    parser.add_argument("--adapter", default=None, help="LoRA adapter path")
    parser.add_argument("--prompt", default=None, help="Code prefix to complete")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.2)
    args = parser.parse_args()

    model, tokenizer = load_model(args.model, args.adapter)

    if args.interactive:
        interactive_mode(model, tokenizer)
    elif args.prompt:
        result = generate_completion(
            model, tokenizer, args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        print(f"\n{args.prompt}{result}")
    else:
        # demo
        demo_prompts = [
            "def fibonacci(n):\n    ",
            "class LinkedList:\n    def __init__(self):\n",
            "function quickSort(arr) {\n    ",
        ]
        for prompt in demo_prompts:
            result = generate_completion(model, tokenizer, prompt)
            print(f"=== {prompt[:40]}... ===")
            print(f"{result}\n")


if __name__ == "__main__":
    main()
