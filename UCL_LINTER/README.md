# UCL Linter & Framework

> **Mathematical prompt optimization based on the Universal Conditional Logic (UCL) framework**

[![arXiv](https://img.shields.io/badge/arXiv-2601.00880-b31b1b.svg)](https://arxiv.org/abs/2601.00880)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](../LICENSE)

## What Is This?

The UCL Linter is a **4-tier prompt optimization system** that turns prompt engineering from guesswork into measurable science. It implements the mathematical framework from the [UCL research paper](https://arxiv.org/abs/2601.00880) (N=305, p < 2.89 x 10^-13).

**Core insight**: Quality follows an inverted-U curve. Too little specification = vague output. Too much = the model gets overwhelmed. There's a mathematical sweet spot at S* = 0.509.

```
Q(S) = 1.96 x S              for S <= 0.509  (linear growth)
Q(S) = 1.0 - 4.0(S-0.509)^2  for S >  0.509  (quadratic degradation)
```

## Quick Start

### Install

```bash
cd UCL_LINTER
pip install -e .
```

### Analyze Any Prompt

```bash
# CLI
ucl-analyze my_prompt.txt

# Python
from prompt_analyzer import ucl_score
score = ucl_score("Your prompt here")
print(f"Quality: {score:.1%}")
```

### Build Optimized Prompts

```python
from builder import UCLPromptBuilder, JSONOutput
from domains import LinearAlgebraDomain

prompt = (UCLPromptBuilder()
    .set_identity("Assistant")
    .add_critical()           # +9.3% quality!
    .add_domain(LinearAlgebraDomain())
    .set_output(JSONOutput(fields=["result"]))
    .build()
)

print(f"Quality: {prompt.score.Q_final:.1%}")
```

## Architecture

```
UCL_LINTER/
├── core.py              # Mathematical engine (Q(S), O_s, constants)
├── prompt_analyzer.py   # Linter: scores ANY prompt text
├── builder.py           # Fluent API for building prompts
├── domains.py           # 12 pre-built domains + custom domains
├── tool_domains.py      # 6 tool domains (31 pre-built tools)
├── tool_comparison.py   # O_s validation: UCL-Tool vs JSON Schema
├── tool_export.py       # Bidirectional JSON Schema interop
├── ucl_demo.py          # Comprehensive demonstration
├── ucl-builder/         # Web app (HTML/CSS/JS + OpenRouter)
├── DOCS/                # Deep documentation
│   ├── ARCHITECTURE.md
│   ├── UCL_TOOL_ARCHITECTURE.md
│   ├── LINTER_RULES.md
│   ├── SCORING_MATHEMATICS.md
│   ├── DEVELOPER_GUIDE.md
│   └── FUTURE_ROADMAP.md
├── tests/               # Validation suite (8 test files, 50+ tests)
├── examples/            # Usage examples
└── pyproject.toml       # Package configuration
```

## Key Findings (from the paper)

### The [[CRITICAL:]] Directive

Placing `[[CRITICAL:]]` at the TOP of a prompt blocks the perceived-sophistication penalty (P_perceived), yielding a **+9.3% quality boost**. This single change took V4 (90.7%) to V4.1 (100%).

### Keyword CONDITIONS > SWITCH Statements

| Architecture | O_s Cost | Quality |
|---|---|---|
| SWITCH (8 cases) | 8.47 | 72.7% |
| Keyword CONDITIONS | 2.0 | 100% |

SWITCH/CASE creates high structural overhead (gamma x ln(cases)). Keyword-based conditions are 4-16x more efficient.

### Three Failure Modes

1. **Role Confusion** - Prompts that make the model question its identity
2. **Complexity Overwhelm** - Too much structure (S > S*)
3. **Perceived Sophistication** - Model tries to explain instead of execute

## Domains

12 pre-built domains for instant optimization:

| Category | Domains |
|---|---|
| Math | BasicMath, LinearAlgebra, VectorCalculus, Integrals, DifferentialEquations, Statistics |
| Science | Physics |
| Code | CodeGeneration, Python, JavaScript |
| Analysis | DataAnalysis, TechnicalWriting |

```python
# Use built-in domains
from domains import DomainRegistry
domain = DomainRegistry.get("linear_algebra")

# Create custom domains
from domains import CustomDomain
medical = CustomDomain(
    name="medical",
    keywords=["diagnosis", "symptom", "treatment"],
    transformations=[("dosage", "amount unit frequency")],
    requirements=["HIPAA compliance"],
)
```

## UCL-Tool: Keyword-Conditional Tool Encoding

UCL-Tool extends the framework with 5 new primitives for encoding tool definitions in system prompts. Instead of verbose JSON Schema, tools are activated by keyword conditions — matching Anthropic's Tool Search Tool approach.

### New Primitives

| Primitive | Syntax | Purpose |
|---|---|---|
| `[[TOOL: name]]` | Tool declaration | Defines a named tool |
| `[[PARAM: name, type, req]]` | Parameter | Defines a tool parameter (required/optional) |
| `[[WHEN: keywords]]` | Activation condition | Keywords that trigger tool selection |
| `[[DESC: text]]` | Description | Tool or parameter description |
| `[[EXAMPLE: input -> output]]` | Usage example | Shows expected invocation pattern |

### Building Tools

```python
from builder import UCLPromptBuilder
from core import ToolDefinition, ToolParameter, ToolExample

tool = ToolDefinition(
    name="read_file",
    description="Read file content from disk",
    parameters=[
        ToolParameter("path", "string", required=True, description="File path"),
    ],
    keywords=["read", "file"],
    examples=[
        ToolExample("Read config", "read config.json", "{'debug': true}"),
    ],
)

prompt = (UCLPromptBuilder()
    .set_identity("Agent")
    .add_critical()
    .add_tool(tool)
    .build()
)
```

### Pre-built Tool Domains

9 domains with 71 ready-to-use tools:

| Domain | Tools | Examples |
|---|---|---|
| FileSystem | 7 | read_file, write_file, list_dir, delete_file, move_file, copy_file, file_info |
| Web | 5 | http_get, http_post, web_search, fetch_url, download_file |
| Database | 6 | query_sql, insert_row, update_row, delete_row, create_table, list_tables |
| API | 4 | rest_call, graphql_query, webhook_send, auth_token |
| MCP | 5 | mcp_call, mcp_list_tools, mcp_list_resources, mcp_read_resource, mcp_list_prompts |
| Shell | 4 | run_command, run_script, get_env, set_env |
| **Playwright** | **20** | browser_navigate, browser_click, browser_type, browser_snapshot, browser_take_screenshot, browser_fill_form, ... |
| **Chroma** | **10** | chroma_create_collection, chroma_add_documents, chroma_query_documents, chroma_get_documents, ... |
| **ClearThought** | **10** | sequential_thinking, debugging_approach, decision_framework, mental_model, scientific_method, ... |

Plus `MCPServerTemplate` for creating custom MCP server domains with your own tools.

```python
from builder import UCLPromptBuilder
from tool_domains import FileSystemDomain, PlaywrightDomain, ChromaDomain

prompt = (UCLPromptBuilder()
    .set_identity("Agent")
    .add_critical()
    .add_tool_domain(FileSystemDomain())    # 7 file tools
    .add_tool_domain(PlaywrightDomain())    # 20 browser tools
    .add_tool_domain(ChromaDomain())        # 10 vector DB tools
    .build()
)

# Or create a custom MCP server domain:
from tool_domains import MCPServerTemplate
from core import ToolDefinition, ToolParameter

my_server = MCPServerTemplate(
    server_name="my_api",
    tools=[
        ToolDefinition("get_user", "Get user by ID",
            [ToolParameter("user_id", "string", required=True)],
            keywords=["user", "get"]),
    ],
)
prompt = (UCLPromptBuilder()
    .set_identity("Agent")
    .add_tool_domain(my_server)
    .build()
)
```

### JSON Schema Interop & Migration

Convert existing JSON Schema tool definitions to UCL-Tool format:

```python
from tool_export import migrate_prompt, keyword_heuristic

# Migrate a prompt with JSON Schema tools to UCL-Tool
old_prompt = '''You are a helpful assistant.
{"type": "function", "function": {"name": "search", "description": "Search the web", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}}
'''

new_text, tools = migrate_prompt(old_prompt)
# new_text now contains [[TOOL: search]] instead of JSON Schema
# tools[0].keywords == ['search', 'web']  (auto-inferred)

# Auto-generate keywords from tool name/description
kw = keyword_heuristic("read_file", "Read file content from disk")
# kw == ['read', 'file', 'content', 'disk']
```

### O_s Reduction vs JSON Schema

| Tools | JSON Schema O_s | UCL-Tool O_s | Reduction |
|---|---|---|---|
| 1 | 2.73 | 1.13 | 58.6% |
| 5 | 13.65 | 2.63 | 80.7% |
| 10 | 27.30 | 3.76 | 86.2% |
| 20 | 54.60 | 5.92 | 89.2% |
| 31 | 84.63 | 7.61 | 91.0% |

JSON Schema O_s grows **linearly** (3.64/tool). UCL-Tool O_s grows **sub-linearly** (0.68/tool) — a **5.4x efficiency gain** at scale.

## CLI Commands

```bash
# Analyze a prompt file
ucl-analyze my_prompt.txt

# Analyze with JSON output
ucl-analyze my_prompt.txt -o report.json

# Run the full demo
ucl-demo
```

## Web App

The `ucl-builder/` directory contains a standalone web application with OpenRouter integration:

```bash
cd ucl-builder
start index.html  # Windows
open index.html   # macOS
```

Features: dark mode UI, 20+ LLM models, click-to-insert syntax reference, auto-save, keyboard shortcuts.

> **Security note**: The web app stores your OpenRouter API key in browser localStorage. Never use it on shared machines.

## UCL Syntax Quick Reference

| Element | Syntax | Purpose |
|---|---|---|
| Concept | `{{concept:name:attr}}` | Variable reference |
| Directive | `[[TYPE: content]]` | Core instruction |
| Condition | `^^CONDITION: expr^^ ... ^^/CONDITION^^` | Conditional logic |
| Section | `<name>...</name>` | Content grouping |
| Transform | `[[TRANSFORM: src TO tgt]]` | Conversion rule |
| Tool | `[[TOOL: name]]` | Tool declaration |
| Param | `[[PARAM: name, type, req]]` | Tool parameter |
| When | `[[WHEN: keywords]]` | Tool activation condition |

**Directive types**: CRITICAL, REQUIRE, ENFORCE, VALIDATE, APPLY, TRANSFORM, LLM, EMPHASIZE, and more.

## Running Tests

```bash
cd UCL_LINTER

# Core framework tests
python tests/test_imports.py
python tests/test_analyzer.py
python tests/test_tool_primitives.py

# UCL-Tool tests
python tests/test_tool_domains.py
python tests/test_tool_comparison.py
python tests/test_tool_export.py

# Integration tests
python tests/test_builder_tools.py
python tests/test_analyzer_tools.py
```

## Research Foundation

Based on the paper:

> Mikinka, A. (2025). Universal Conditional Logic: A formal language for prompt engineering. *arXiv preprint arXiv:2601.00880*. https://doi.org/10.48550/arXiv.2601.00880

- **N** = 305 observations
- **Models** = 11 different LLMs
- **p-value** < 2.89 x 10^-13
- **Cohen's d** = 1.15 (large effect)
- **Token reduction** = 30.9%

## License

MIT
