# LoRA Smoke Test Notes

## Why LoRA for code models?

Full fine-tuning of a 7B model needs 40GB+ VRAM. LoRA lets you fine-tune on 8-16GB.

## Smoke test setup

- model: codellama-7b
- lora_rank: 16
- lora_alpha: 32
- target_modules: q_proj, v_proj
- batch_size: 4
- lr: 2e-4
- epochs: 3

## Results

- HumanEval pass@1: 29.3% -> 34.1% (+4.8%)
- CodeBLEU: 62.1 -> 67.8
- Speed: 5.2 tok/s -> 5.1 tok/s

## Observations

1. Modest improvement but noticeable
2. Overfitting after 5 epochs. 3 epochs was sweet spot
3. Data quality matters more than data size
4. LoRA rank 16 > rank 8. Rank 32 marginal improvement.
