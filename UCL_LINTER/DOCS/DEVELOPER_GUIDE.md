# UCL Developer Guide

This guide explains how to programmatically use the UCL Builder and Analyzer tools to create and score prompts.

## 🛠️ Using the Builder API (`builder.py`)

The `UCLPromptBuilder` uses a fluent (chainable) interface to ensure best practices are applied by default.

### Basic Usage Example

```python
from builder import UCLPromptBuilder, JSONOutput
from domains import MathDomain

# Build an optimized math prompt
prompt = (UCLPromptBuilder()
    .set_identity("MathematicalThinker")
    .add_critical() # Adds the +9.3% bonus directive
    .set_task("Resolve complex integrals")
    .add_domain(MathDomain()) # Automatically adds keyword conditions
    .set_output(JSONOutput(fields=["integral", "solution", "steps"]))
    .build())

# Access the results
print(prompt.text)
print(f"Quality Score: {prompt.score.Q_final:.1%}")
print(f"Grade: {prompt.score.grade}")
```

### Key Methods

- `.add_critical(content: str)`: Manually specify or use the template for the CRITICAL directive.
- `.add_domain(domain_obj)`: Injects domain-specific keywords and logic.
- `.set_output(spec_obj)`: Configures the output format (JSON, Code, etc.) and adds necessary directives.
- `.preview()`: Returns the string without performing a full scoring calculation.

---

## 🔍 Using the Analyzer/Linter (`prompt_analyzer.py`)

The Analyzer is used to score prompts that were *not* necessarily built with the UCL fluent API.

### Scoring a Raw String

```python
from prompt_analyzer import ucl_score, analyze_prompt

raw_text = """
Write a story about a robot. 
[[CRITICAL: Use ONLY JSON]]
"""

# Simple score
score = ucl_score(raw_text)
print(f"Quality Prediction: {score:.1%}")

# Detailed analysis
report = analyze_prompt(raw_text)
report.print_report() # Prints a formatted text report
```

### Analyzing Files via CLI

You can run the analyzer directly from your terminal:

```bash
python prompt_analyzer.py my_prompt.txt -o result.json
```

---

## 🧩 Creating Custom Domains

To extend the system, subclass the `Domain` base class in `domains.py`.

```python
from domains import Domain
from core import Directive, DirectiveType

class MyCustomDomain(Domain):
    def __init__(self):
        super().__init__(
            name="MyDomain",
            keywords=["keyword1", "keyword2"],
            directives=[
                Directive(DirectiveType.APPLY, "Special Domain Rule")
            ]
        )
```

Adding this domain to a builder will automatically create a `^^CONDITION: content CONTAINS "keyword1" OR "keyword2"^^` block containing your domain directives.

---

## ⚙️ Understanding the Score Object (`UCLScore`)

The `build()` and `analyze_prompt()` methods return a score object with detailed metrics:

| Attribute | Description |
|-----------|-------------|
| `Os` | Total Structural Overhead calculated for the prompt. |
| `specification_level` | The $S$ value (0.0 to 1.0). |
| `Q_final` | Predicted success rate of the prompt. |
| `grade` | Letter grade (A-F) based on `Q_final`. |
| `warnings` | List of `Warning` objects flagging anti-patterns. |
| `architecture` | Detected style: `KEYWORD`, `SWITCH`, `MIXED`, or `MINIMAL`. |
