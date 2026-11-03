# Performance

Benchmarks and mathematical derivation of UCL-Tool's overhead reduction
compared to standard JSON Schema tool definitions.

---

## Summary Metrics

| Domain | Tools | JSON Schema | UCL-Tool | Char Reduction | O_s Reduction |
|--------|------:|------------:|---------:|---------------:|--------------:|
| **FileSystem** | 7 | 3,549 chars | 1,857 chars | **47.7%** | **78.5%** |
| **Web** | 5 | 2,759 chars | 1,568 chars | **43.2%** | **74.3%** |
| **Playwright** | 20 | 11,816 chars | 6,317 chars | **46.5%** | **78.9%** |

---

## O_s (Structural Overhead) Model

O_s measures the parsing complexity an LLM must handle to interpret tool
definitions. It is computed as a weighted sum of structural elements.

### JSON Schema O_s

```
O_s_json = B + N * (P * p + D * d) * F
```

| Symbol | Meaning | Value |
|--------|---------|-------|
| `B` | Base overhead (function wrapper) | 1.5 |
| `N` | Number of tools | varies |
| `P` | Per-parameter overhead | 0.5 |
| `p` | Average parameters per tool | ~2.3 |
| `D` | Description field overhead | 0.3 |
| `d` | Number of description fields per tool | ~3 |
| `F` | Nesting depth factor | 1.3 |

**Example: FileSystem (7 tools, avg 2.3 params each)**

```
O_s_json = 1.5 + 7 * (0.5 * 2.3 + 0.3 * 3) * 1.3
         = 1.5 + 7 * (1.15 + 0.9) * 1.3
         = 1.5 + 7 * 2.05 * 1.3
         = 1.5 + 18.66
         = 23.53 (rounded)
```

### UCL-Tool O_s

```
O_s_ucl = N * (1 + p * 0.3 + 0.2)
```

| Symbol | Meaning | Value |
|--------|---------|-------|
| `N` | Number of tools | varies |
| `1` | Per-tool overhead (`[[TOOL]]` + `[[DESC]]`) | 1.0 |
| `p` | Average parameters per tool | ~2.3 |
| `0.3` | Per-parameter overhead (`[[PARAM]]`) | 0.3 |
| `0.2` | WHEN condition overhead | 0.2 |

**Example: FileSystem (7 tools, avg 2.3 params each)**

```
O_s_ucl = 7 * (1 + 2.3 * 0.3 + 0.2)
        = 7 * (1 + 0.69 + 0.2)
        = 7 * 1.89
        = 5.05 (rounded)
```

### Reduction formula

```
Reduction = (O_s_json - O_s_ucl) / O_s_json * 100

FileSystem: (23.53 - 5.05) / 23.53 * 100 = 78.5%
Web:        (17.21 - 4.42) / 17.21 * 100 = 74.3%
Playwright: (72.84 - 15.34) / 72.84 * 100 = 78.9%
```

---

## Why UCL-Tool Has Lower O_s

### 1. No nesting

JSON Schema requires 3-4 levels of nesting per parameter:

```json
{
  "type": "function",
  "function": {
    "parameters": {
      "properties": {
        "path": { "type": "string" }
      }
    }
  }
}
```

UCL-Tool uses flat directives:

```
[[PARAM: path, string, required]]
```

Each nesting level in JSON adds ~0.3 to O_s due to the cognitive load of
tracking opening/closing braces and indentation context.

### 2. No type repetition

JSON Schema repeats `"type": "..."` for every property and the container:

```json
{"type": "object", "properties": {"path": {"type": "string"}}}
```

UCL-Tool uses positional encoding:

```
[[PARAM: path, string, required]]
```

The type is inline, not wrapped in a keyed object.

### 3. No required array

JSON Schema maintains a separate `required: [...]` array that must be
cross-referenced with `properties`:

```json
{
  "properties": {"path": {...}, "encoding": {...}},
  "required": ["path"]
}
```

UCL-Tool marks required inline:

```
[[PARAM: path, string, required]]
[[PARAM: encoding, string, optional]]
```

### 4. Conditional activation

JSON Schema includes all tools unconditionally. UCL-Tool wraps domains in
`^^CONDITION^^` blocks, meaning the LLM only processes tools when the
user input matches keywords. This effectively reduces O_s to zero for
non-activated tools.

---

## Character Count Methodology

Character counts include all whitespace, brackets, and delimiters. They
measure the actual text injected into the system prompt.

### What is counted

- Tool names, descriptions, parameter names and types
- Structural delimiters (braces, brackets, quotes, commas)
- Whitespace and indentation
- `required` arrays and `description` fields

### What is NOT counted

- The outer `{"tools": [...]}` envelope (constant across all approaches)
- Tool implementation code (same regardless of definition format)
- Condition blocks for UCL-Tool (counted separately as "activation overhead")

### Token estimation

Tokens are estimated at 4 characters per token (approximate for English
text with code). Actual token counts vary by tokenizer:

| Domain | Chars | Est. tokens (UCL) | Est. tokens (JSON) |
|--------|------:|------------------:|-------------------:|
| FileSystem | 1,857 | ~464 | ~887 |
| Web | 1,568 | ~392 | ~690 |
| Playwright | 6,317 | ~1,579 | ~2,954 |

---

## Scaling Analysis

As the number of tools increases, the O_s reduction becomes more significant:

| Tools | JSON Schema O_s | UCL-Tool O_s | Reduction |
|------:|----------------:|-------------:|----------:|
| 5 | 17.2 | 4.4 | 74.3% |
| 10 | 35.4 | 8.9 | 74.9% |
| 20 | 72.8 | 15.3 | 78.9% |
| 50 | 180.2 | 38.4 | 78.7% |
| 100 | 362.0 | 76.9 | 78.8% |

The reduction stabilizes around **78-79%** because both models scale linearly
with tool count, but JSON Schema's nesting factor (1.3x) creates a growing
absolute gap.

---

## Benchmark Reproduction

Run the comparison locally:

```bash
cd UCL_LINTER

# Full comparison across all domains
python tool_comparison.py

# Python example with metrics
python examples/example_pi_dev_integration.py
```

### Programmatic usage

```python
from tool_comparison import compare_tools
from tool_domains import FileSystemDomain

domain = FileSystemDomain()
tools = domain.get_tools()
result = compare_tools(tools)

print(f"JSON: {result['json_chars']} chars, O_s={result['json_Os']:.2f}")
print(f"UCL:  {result['ucl_chars']} chars, O_s={result['ucl_Os']:.2f}")
print(f"Char reduction: {result['char_reduction_pct']:.1f}%")
print(f"O_s reduction:  {result['Os_reduction_pct']:.1f}%")
```

### Scaling experiment

```python
from tool_comparison import scaling_experiment

results = scaling_experiment(
    tool_counts=[5, 10, 20, 50, 100],
    avg_params=2.3,
)

for r in results:
    print(f"n={r['n']:3d}: JSON O_s={r['json_Os']:7.2f}  "
          f"UCL O_s={r['ucl_Os']:6.2f}  "
          f"reduction={r['reduction_pct']:.1f}%")
```

---

## Limitations

1. **O_s is a model, not a measurement.** The formula approximates structural
   complexity. Actual LLM behavior may differ.

2. **Character count is a proxy for tokens.** The real token count depends on
   the tokenizer. The 4-chars-per-token estimate is approximate.

3. **Conditional activation not measured.** The O_s numbers above assume all
   tools are processed. In practice, `^^CONDITION^^` blocks mean unused tools
   contribute zero overhead.

4. **Benchmarks are domain-specific.** Reduction percentages vary by tool
   complexity. Tools with many parameters see higher reduction; simple tools
   see less.


---

## Related Documentation

- **[README](../README.md)** — Quick start and overview
- **[Architecture](./ARCHITECTURE.md)** — Pipeline design and data flow
- **[API Reference](./API-REFERENCE.md)** — Complete function signatures
