# Code Review Output Example

## Input

```python
def get_user_age(users, name):
    for user in users:
        if user['name'] == name:
            return user['age']
    return None
```

## Model output (CodeLlama 7B)

- No input validation
- KeyError if user dict missing keys
- Linear search O(n) - consider dict lookup
- Returns None for both 'not found' and 'age is None' - ambiguous

Suggested improvement with .get() and validation.
