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

## Hardware

Primary test target: AMD Radeon VII (ROCm 5.6+). RX 7900 XTX used for bf16 scaling tests.

## License

MIT, do whatever you want with it.
