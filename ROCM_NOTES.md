# ROCm Notes — Code LLM Inference & LoRA Fine-Tuning

## Target Environment

- **ROCm version:** 6.0 / 6.1.2
- **PyTorch:** torch 2.3+ ROCm build
- **GPU target:** Radeon VII (gfx906, 16GB HBM2), RX 7900 XTX (gfx1100, 24GB GDDR6) for scaling tests
- **Precision:** fp32 baseline, fp16 and bf16 for LoRA training comparisons

## Current Status

### What Works
- CodeGen-350M LoRA fine-tuning runs on Radeon VII with ROCm 5.6+ and 6.x
- StarCoder-small inference is stable for single-request code completion
- Bug detector classifier trains without issues (small model, no special ops)
- AST analysis (tree-sitter) is CPU-only — no ROCm dependency

### Known Blockers
- `bitsandbytes` for 4-bit quantization does NOT support ROCm natively — QLoRA is not viable without custom HIP builds of the CUDA kernels
- `xformers` memory-efficient attention requires ROCm-specific build; the PyPI wheel is CUDA-only
- Gradient checkpointing has edge-case failures on `gfx906` with torch 2.2 — upgrade to 2.3+ recommended
- `flash-attn` ROCm builds exist but only for gfx90a (MI210/MI250); Radeon VII (gfx906) must use standard attention

### Memory Considerations for LoRA
- CodeGen-350M + LoRA (r=16, alpha=32): ~5.2 GB VRAM at batch size 4, seq_len 512
- Same config at batch size 8: ~8.1 GB — fits comfortably on 16GB
- StarCoder-small (1.3B) + LoRA: ~12.4 GB at batch size 2, seq_len 512 — tight on 16GB, need gradient accumulation
- bf16 training would reduce VRAM by ~30% but `gfx906` does NOT support native bf16 — must use fp16 on Radeon VII
- RX 7900 XTX (gfx1100) supports bf16 natively — that's where bf16 tests will happen

## Planned Benchmarks

| Test | Metric | Hardware Target |
|------|--------|----------------|
| CodeGen-350M LoRA train (fp32) | tokens/sec | Radeon VII |
| CodeGen-350M LoRA train (fp16) | tokens/sec | Radeon VII |
| CodeGen-350M LoRA train (bf16) | tokens/sec | RX 7900 XTX |
| Code completion inference | latency/token | Both GPUs |
| StarCoder-small inference | tokens/sec | RX 7900 XTX |
| Peak VRAM during LoRA training | GB | Both GPUs |
| HumanEval pass@1 | accuracy | Before/after LoRA |

## ROCm-Specific Technical Notes

- The Radeon VII (gfx906) is still in ROCm's "extended support" — some newer libraries mark it as deprecated. If `torch.cuda.is_available()` returns False, check that `HSA_OVERRIDE_GFX_VERSION=9.0.0` is NOT set (it's a gfx906, not gfx900).
- For LoRA specifically, `peft` library works fine on ROCm since it uses standard PyTorch ops. No changes needed.
- `transformers` `Trainer` class works with `--fp16` on ROCm, but `--bf16` will fail on gfx906 since the hardware doesn't support it. Use `--fp16` for Radeon VII.
- `torch.compile` on code models with dynamic shapes (variable-length code snippets) can trigger recompilation storms. Use `torch._dynamo.config.suppress_errors = True` and consider disabling compile for variable-length inputs.

## Validation Checklist

- [ ] Train CodeGen-350M LoRA for 1 epoch, verify loss curve matches CPU
- [ ] Run HumanEval eval, confirm pass@1 is within noise of baseline
- [ ] Measure fp32 vs fp16 VRAM usage, confirm ~30-40% savings
- [ ] Test bf16 on RX 7900 XTX only (not supported on Radeon VII)
- [ ] Profile code completion latency — target <50ms/token
- [ ] Check for silent CPU fallbacks during attention computation
