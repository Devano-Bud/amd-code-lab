"""
HumanEval benchmark evaluation for code generation models.

Loads a fine-tuned model and evaluates on OpenAI's HumanEval dataset.
Measures pass@k (functional correctness).

Usage:
    python eval_humaneval.py --model ./checkpoints/codegen-lora
    python eval_humaneval.py --model Salesforce/codegen-350M-multi --baseline
"""

import argparse
import json
import os
import re
import torch
import subprocess
import tempfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset


def load_model(base_model, adapter_path=None):
    tokenizer = AutoTokenizer.from_pretrained(adapter_path or base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.float16, device_map="auto"
    )
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
    model.eval()
    return model, tokenizer


def generate_completion(model, tokenizer, prompt, max_tokens=256, n=1, temperature=0.8):
    """Generate n completions for a prompt."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    completions = []

    for _ in range(n):
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                top_p=0.95,
                pad_token_id=tokenizer.pad_token_id,
            )
        generated = tokenizer.decode(output[0], skip_special_tokens=True)
        # remove the prompt from output
        completion = generated[len(prompt):]
        # cut at first "def " or class def to keep just the function
        completion = truncate_at_next_def(completion)
        completions.append(completion)

    return completions


def truncate_at_next_def(text):
    """Truncate generated text at next function/class definition."""
    lines = text.split("\n")
    result = []
    for i, line in enumerate(lines):
        # stop at next top-level def/class
        if i > 0 and (line.startswith("def ") or line.startswith("class ")):
            break
        result.append(line)
    return "\n".join(result)


def extract_function_name(prompt):
    """Extract function name from the prompt."""
    match = re.search(r"def\s+(\w+)\s*\(", prompt)
    if match:
        return match.group(1)
    return "candidate"


def run_test(generated_code, test_code, function_name, timeout=10):
    """Run generated code with test cases. Returns True if all tests pass."""
    full_code = generated_code + "\n\n" + test_code

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, dir="/tmp"
    ) as f:
        f.write(full_code)
        tmp_path = f.name

    try:
        result = subprocess.run(
            ["python3", tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False
    finally:
        os.unlink(tmp_path)


def pass_at_k(n, c, k):
    """Compute pass@k given n samples and c correct ones."""
    if n - c < k:
        return 1.0
    return 1.0 - reduce(lambda x, y: x * y, [(n - c - i) / (n - i) for i in range(k)])


from functools import reduce


def evaluate_humaneval(model, tokenizer, k_values=[1, 10, 100], n_samples=200):
    """Evaluate on HumanEval benchmark."""
    print("Loading HumanEval dataset...")
    dataset = load_dataset("openai_humaneval", split="test")

    results = []
    total = len(dataset)

    for i, problem in enumerate(tqdm(dataset, desc="Evaluating")):
        prompt = problem["prompt"]
        test_code = problem["test"]
        function_name = problem["entry_point"]

        completions = generate_completion(
            model, tokenizer, prompt, n=min(n_samples, 20)
        )

        correct = sum(
            1 for c in completions if run_test(c, test_code, function_name)
        )

        results.append({
            "task_id": problem["task_id"],
            "n_samples": len(completions),
            "correct": correct,
        })

    # compute pass@k
    total_correct = sum(r["correct"] for r in results)
    total_samples = sum(r["n_samples"] for r in results)

    print(f"\nResults: {total_correct}/{total_samples} correct")

    pass_at_k_results = {}
    for k in k_values:
        if k <= total_samples // total:
            # compute per-problem pass@k then average
            pk_scores = []
            for r in results:
                pk = pass_at_k(r["n_samples"], r["correct"], k)
                pk_scores.append(pk)
            pass_at_k_results[f"pass@{k}"] = sum(pk_scores) / len(pk_scores)
            print(f"pass@{k}: {pass_at_k_results[f'pass@{k}']:.4f}")

    return pass_at_k_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Base model name or path")
    parser.add_argument("--adapter", default=None, help="LoRA adapter path")
    parser.add_argument("--baseline", action="store_true", help="Eval base model without adapter")
    parser.add_argument("--n-samples", type=int, default=20, help="Samples per problem")
    parser.add_argument("--output", default="humaneval_results.json")
    args = parser.parse_args()

    model, tokenizer = load_model(args.model, args.adapter)

    results = evaluate_humaneval(model, tokenizer, n_samples=args.n_samples)

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
