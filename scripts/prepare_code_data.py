"""
Prepare code datasets for training.

Downloads and preprocesses code datasets, handles deduplication,
filters by language, and formats for fine-tuning.

Usage:
    python prepare_code_data.py --dataset codeparrot/github-code --lang python --output ./data/
    python prepare_code_data.py --format-codegen --input ./data/raw/ --output ./data/codegen/
"""

import os
import argparse
import hashlib
import json
from pathlib import Path
from datasets import load_dataset, Dataset
from tqdm import tqdm


LANG_EXTENSIONS = {
    "python": ".py",
    "javascript": ".js",
    "typescript": ".ts",
    "java": ".java",
    "go": ".go",
    "rust": ".rs",
    "cpp": ".cpp",
    "c": ".c",
}


def download_github_code(lang="python", max_files=100000, output_dir="./data/raw"):
    """Download code from codeparrot/github-code dataset."""
    print(f"Downloading {lang} code (up to {max_files} files)...")

    dataset = load_dataset(
        "codeparrot/github-code",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"github_{lang}.jsonl")

    count = 0
    seen_hashes = set()

    with open(output_path, "w") as f:
        for sample in tqdm(dataset, total=max_files, desc=f"Filtering {lang}"):
            if count >= max_files:
                break

            # filter by language
            if sample.get("language", "").lower() != lang:
                continue

            code = sample.get("content", "")
            if not code or len(code.strip()) < 50:
                continue

            # basic dedup by content hash
            code_hash = hashlib.md5(code.encode()).hexdigest()
            if code_hash in seen_hashes:
                continue
            seen_hashes.add(code_hash)

            # skip very short or very long files
            if len(code) > 50000:
                continue

            record = {
                "code": code,
                "language": lang,
                "repo": sample.get("repo_name", ""),
                "path": sample.get("path", ""),
                "size": len(code),
            }
            f.write(json.dumps(record) + "\n")
            count += 1

    print(f"Saved {count} {lang} files to {output_path}")
    return output_path


def format_for_codegen(input_path, output_path, max_length=512):
    """Format dataset for CodeGen fine-tuning (single text field)."""
    print(f"Formatting {input_path} for CodeGen...")

    with open(input_path, "r") as f:
        lines = f.readlines()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    records = []

    for line in tqdm(lines, desc="Formatting"):
        data = json.loads(line)
        code = data.get("code", "")

        # tokenize roughly by lines and truncate
        token_count = len(code) // 4  # rough estimate
        if token_count > max_length:
            # take first max_length tokens worth of code
            code = code[:max_length * 4]

        records.append({"text": code})

    dataset = Dataset.from_list(records)
    dataset.save_to_disk(output_path)
    print(f"Saved {len(records)} examples to {output_path}")
    return output_path


def prepare_from_local(input_dir, output_path, extensions=None):
    """Prepare training data from local code files."""
    if extensions is None:
        extensions = [".py", ".js"]

    files = []
    for ext in extensions:
        files.extend(Path(input_dir).rglob(f"*{ext}"))

    # skip junk
    files = [
        f for f in files
        if not any(skip in str(f) for skip in [".git", "node_modules", "__pycache__", ".venv", "venv"])
    ]

    print(f"Found {len(files)} files")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    records = []

    for filepath in tqdm(files, desc="Processing"):
        try:
            with open(filepath, "r") as f:
                code = f.read()

            if len(code.strip()) < 100:
                continue

            records.append({
                "text": code,
                "file": str(filepath),
                "language": filepath.suffix,
            })
        except (UnicodeDecodeError, PermissionError):
            continue

    with open(output_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print(f"Saved {len(records)} records to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare code datasets for training")
    parser.add_argument("--dataset", default="codeparrot/github-code", help="HuggingFace dataset")
    parser.add_argument("--lang", default="python", help="Language to filter")
    parser.add_argument("--max-files", type=int, default=100000)
    parser.add_argument("--input", default=None, help="Local input directory")
    parser.add_argument("--output", default="./data/", help="Output directory")
    parser.add_argument("--format-codegen", action="store_true", help="Format for CodeGen")
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()

    if args.format_codegen:
        input_path = args.input or f"./data/raw/github_{args.lang}.jsonl"
        output_path = os.path.join(args.output, "codegen_formatted")
        format_for_codegen(input_path, output_path, args.max_length)
    elif args.input:
        output_path = os.path.join(args.output, "local_data.jsonl")
        prepare_from_local(args.input, output_path)
    else:
        download_github_code(args.lang, args.max_files, os.path.join(args.output, "raw"))


if __name__ == "__main__":
    main()
