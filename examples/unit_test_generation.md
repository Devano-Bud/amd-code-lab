# Unit Test Generation Example

## Input function

```python
def merge_dicts(dicts):
    result = {}
    for d in dicts:
        result.update(d)
    return result
```

## Model output (CodeLlama 7B)

Generated tests for: empty input, single dict, no overlap, with overlap, multiple dicts, override order.

Coverage: good. Missing: edge case with None values.
