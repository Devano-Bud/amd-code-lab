# Coder Model Notes

## Models tested

### CodeLlama 7B
- Best overall quality for code tasks
- Good at explaining code, suggesting fixes
- Slow on my hardware (~5 tokens/sec)

### StarCoder2 3B
- Good balance of quality and speed
- Better at Python than other languages
- Sometimes produces outdated patterns

### Qwen2.5-Coder 1.5B
- Surprisingly good for its size
- Fast (~25 tokens/sec)
- Struggles with complex logic

### Phi-3 Mini (3.8B)
- Not a coder model, but decent at code
- Better at explaining than generating

## What I've learned

1. Prompt format matters a lot
2. Context window: small models struggle with long code
3. Temperature: 0.1-0.3 for code generation, 0.7 for explanations
4. Stop sequences: include triple-quote chars to prevent docstring generation

## Practical limits

- Complex refactoring: too hard for < 7B models
- Multi-file changes: not supported
- Understanding project context: impossible without RAG
