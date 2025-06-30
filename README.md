# amd-code-lab

Code generation and analysis experiments running on AMD GPUs (ROCm).

I got tired of everything being NVIDIA-only so I'm trying to get small code models
training on my Radeon VII. It's got 16GB HBM2 which is honestly decent for this
stuff if you're careful with batch sizes. ROCm support has gotten way better lately
but there's still some rough edges — check the [experiments notes](experiments/notes.md)
for the pain points.

## What's in here

- **Code generation** — LoRA fine-tuning CodeGen-350M and StarCoder-small on code datasets
- **Code completion** — inference pipeline for Python and JS autocomplete
- **Bug detection** — training a classifier on buggy vs clean code
- **Code summarization** — generating docstrings from function bodies
- **AST analysis** — tree-sitter based code structure analysis

## Setup

```bash
# you need ROCm installed, this is written for ROCm 5.6+
# tested on Radeon VII, should work on 6000 series too
pip install -r requirements.txt
```

## Training

```bash
# codegen lora fine-tune
python src/codegen_finetune.py --config configs/codegen_lora.yaml

# bug detector
python src/bug_detector.py --config configs/training.yaml
```

## Eval

```bash
python scripts/eval_humaneval.py --model ./checkpoints/codegen-lora
```

## Hardware notes

Radeon VII (gfx906) — works with ROCm 5.6+, some ops need fallback
RX 6800/6900 XT (gfx1030) — should be fine
RX 7900 XTX (gfx1100) — ROCm 6.0+ works well

16GB VRAM means you gotta be smart about memory. LoRA is basically required
for anything bigger than 350M params. Check configs for the batch sizes that
didn't OOM for me.

## Why AMD / ROCm

Code LLMs are increasingly important for developer productivity — autocompletion, code review, bug detection, and documentation generation. Running these models on AMD GPUs via ROCm is particularly compelling because:

- AMD GPUs offer competitive memory bandwidth (Radeon VII's HBM2 is excellent for transformer workloads)
- LoRA fine-tuning of code models is a practical, cost-sensitive use case that benefits from hardware diversity
- The 16GB VRAM on Radeon VII and 24GB on RX 7900 XTX are well-suited for small-to-medium code models
- ROCm 6.x has matured significantly, making code model training more reliable than a year ago
- Testing on AMD hardware exposes hidden CUDA assumptions in popular ML libraries

## AMD GPU Credit Use Plan

If granted AMD GPU access, I plan to:

1. Validate LoRA fine-tuning of CodeGen and StarCoder models on ROCm-compatible GPUs
2. Benchmark inference throughput (tokens/sec) on AMD vs CPU for code completion
3. Test fp16 and bf16 training — bf16 on gfx1100 hardware specifically
4. Measure VRAM usage during training to find optimal batch sizes for different GPU memory tiers
5. Document ROCm compatibility issues with PEFT, Transformers, and bitsandbytes
6. Publish benchmark data and ROCm setup guides back to this repository

## License

MIT, do whatever you want with it.
