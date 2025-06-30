# Dev Notes — AMD Code Lab

Personal log of experiments, issues, and findings while training code models on AMD GPUs.

## 2025-06-15

Started setting up the project. ROCm 5.6 installed on the Radeon VII (gfx906).
PyTorch with ROCm backend works out of the box now, which is a massive improvement
over a year ago when half the ops weren't supported.

First attempt: CodeGen-350M with batch size 8. Immediate OOM. Classic.

Dropped to batch size 4 with gradient accumulation of 4. Works fine, peaks at
about 12GB VRAM. The 16GB on the Radeon VII is tight but workable.

## 2025-06-17

Got LoRA working. Huge difference — memory usage dropped from 12GB to about 7GB
with the same config. Could probably bump batch size back up but let's not get
greedy.

Fun fact: `dataloader_pin_memory=True` causes weird hangs on ROCm. Set it to
False in all training scripts. Took me 2 hours to figure this out.

Also had an issue with the tokenizer — CodeGen's tokenizer doesn't set a pad
token by default. If you forget to set it, the data collator just silently
does the wrong thing and your loss looks crazy. Spent a whole evening debugging
that.

## 2025-06-19

Training is running! 3 epochs on the GitHub code dataset (Python subset, 50k files).
Each epoch takes about 45 minutes on the Radeon VII. Loss curve looks reasonable,
started at ~3.2 and is down to ~1.4 after 2 epochs.

LoRA config that works well:
- r=16, alpha=32, dropout=0.1
- targeting q_proj, v_proj, k_proj, out_proj
- lr=2e-4 with cosine schedule

Tried r=32 first but it didn't make a noticeable difference and used more memory.

## 2025-06-22

Code completion results are surprisingly decent. The fine-tuned CodeGen-350M can
complete simple functions pretty well. It's not Copilot but it's not garbage either.

Running HumanEval eval now. Takes forever because you have to run the generated code.
Need to add parallelism to the eval script.

Results so far (partial):
- pass@1: 0.087 (baseline 350M was 0.052)
- pass@10: 0.143

Not amazing but we're only using 50k training examples and a 350M param model.
Would love to try StarCoder-small next.

## 2025-06-24

OOM issues again. Tried to fine-tune StarCoder-small (1.3B) with LoRA.
Even with LoRA + fp16 + gradient checkpointing, OOMs at sequence length 512.
Dropped to max_length=256 and it fits, barely. Peaks at ~15.5GB.

The Radeon VII is really the bottleneck here. A 7900 XTX with 24GB would open
up way more possibilities. Might need to look into gradient accumulation with
CPU offloading for the optimizer states.

## 2025-06-26

Bug detector training started. Using CodeBERT as the base, training on
Defects4J / CodeXGLUE defect detection. This is a classification task so
memory isn't the issue — it's the data quality.

Accuracy after 2 epochs: 62%. Not great. The dataset has a lot of noisy labels.
Going to try cleaning the data and see if that helps.

Also started on the AST analysis module. Tree-sitter is really nice for this.
Python + JS support working. Extracted function signatures, nesting depth,
class hierarchies. Planning to use this for feature extraction for the bug
detector.

## 2025-06-28

Bug detector accuracy up to 67% after filtering out suspicious samples.
Added AST-derived features (nesting depth, function length) as additional
inputs. Small improvement, maybe 1-2%.

HumanEval pass@1 for CodeGen-350M LoRA: **0.112**. Not bad for a 350M model
trained on a Radeon VII. The full CodeGen-350M multi-paper result is ~0.12
with way more data, so we're in the right ballpark.

Next steps:
- Try StarCoder-small with max_length=256
- Code summarization with CodeT5
- Better data pipeline (current one is super janky)
- Look into QLoRA to save more VRAM
