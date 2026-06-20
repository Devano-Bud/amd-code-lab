# Bugfix Prompt Example

## Input

```python
def read_csv_columns(filepath, columns):
    import csv
    with open(filepath) as f:
        reader = csv.DictReader(f)
        return [{col: row[col] for col in columns} for row in reader]
```

**Bug report**: KeyError when column name doesn't exist in CSV.

## Model outputs

### CodeLlama 7B
Handles missing columns with warning. Clean code.

### StarCoder2 3B
Checks missing columns upfront, uses .get() for safety.

### Qwen2.5-Coder 1.5B
Uses try/except, returns empty list on error. Not ideal.
