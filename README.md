# amd-code-lab

This repo is a small playground for code-generation models. I'm mostly testing practical tasks: explaining a function, suggesting a bugfix, generating unit tests, and comparing how small coder models behave on the same prompt.

## Why I built this

I wanted to see if small code models (1B-3B parameters) are useful enough for daily coding tasks. Not building Copilot -- just testing if they can handle simple things like:
- Explain what a function does
- Suggest a fix for a bug
- Generate basic unit tests
- Write docstrings

## What's in here

- Prompt experiments with different formats for code tasks
- Model comparisons: same prompt across different models
- Output logs: raw model outputs for analysis

## Models tested

- CodeLlama 7B
- StarCoder2 3B
- Qwen2.5-Coder 1.5B
- Phi-3 Mini (not a coder model, but interesting comparison)

## Quick start

```bash
pip install -r requirements.txt
python code_lab.py explain examples/sample_function.py
python code_lab.py bugfix examples/buggy_code.py
python code_lab.py tests examples/sample_function.py
```

## Philosophy

See `docs/coder_model_notes.md` for what I've learned about small code models.
