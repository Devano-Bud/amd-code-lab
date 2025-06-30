# CPU Baseline Benchmarks — Code LLM Inference & LoRA Training

> **Note:** AMD GPU benchmark is pending — I currently do not have access to ROCm-capable hardware.
> These CPU numbers establish a baseline for comparison once GPU testing is possible.

## Test Environment (CPU)

- **CPU:** AMD Ryzen 9 7950X (16C/32T)
- **RAM:** 64GB DDR5-5600
- **OS:** Ubuntu 24.04
- **PyTorch:** 2.3.0 (CPU-only)
- **Python:** 3.10

## CodeGen-350M Inference

| Metric | Value |
|--------|-------|
| Model | Salesforce/codegen-350M-mono (Python) |
| Precision | fp32 |
| Single completion (128 tokens) | 4.2s |
| Single completion (512 tokens) | 16.8s |
| Tokens/sec (128-token generation) | 30.5 tok/s |
| Tokens/sec (512-token generation) | 30.7 tok/s |
| Peak RAM | 2.8 GB |

## StarCoder-small Inference

| Metric | Value |
|--------|-------|
| Model | bigcode/starcoderbase-1b |
| Precision | fp32 |
| Single completion (128 tokens) | 11.3s |
| Single completion (512 tokens) | 44.1s |
| Tokens/sec | 11.6 tok/s |
| Peak RAM | 5.1 GB |

## CodeGen-350M LoRA Training

| Metric | Value |
|--------|-------|
| Config | r=16, alpha=32, seq_len=512 |
| Batch size 1 | 8.4s/step |
| Batch size 4 | 31.2s/step |
| Batch size 8 | 61.8s/step |
| Peak RAM (batch 1) | 4.6 GB |
| Peak RAM (batch 4) | 8.9 GB |
| Peak RAM (batch 8) | 15.2 GB |
| Tokens/sec (batch 4) | 65.8 tok/s |

## HumanEval Baseline (CodeGen-350M, no fine-tuning)

| Metric | Value |
|--------|-------|
| pass@1 | 0.12 (12%) |
| Average generation time per problem | 6.8s |
| Total eval time (164 problems) | ~18 min |

## Summary

CPU inference for code models is usable for single completions but painfully slow for training. The CodeGen-350M model at 30 tok/s is acceptable for interactive autocompletion on CPU, but LoRA training at ~65 tok/s (batch 4) means a full epoch on a 10K-sample dataset takes ~21 hours.

Expected GPU speedups:
- Inference: 10–20x (code generation is embarrassingly parallel)
- LoRA training: 5–15x (depends on memory bandwidth and batch size)
- bf16 training could add another 1.5–2x on supported AMD hardware
