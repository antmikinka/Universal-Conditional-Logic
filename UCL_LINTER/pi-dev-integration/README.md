# UCL-Tool x pi.dev Integration

**Cut tool-definition overhead by 78% in LLM system prompts.**

UCL-Tool replaces verbose JSON Schema tool blocks with compact Universal
Conditional Logic directives. This package bridges the Python UCL-Tool
framework with the [pi.dev](https://github.com/earendil-works/pi) TypeScript
system-prompt builder.

```
Python (tool authoring)          TypeScript (prompt assembly)         LLM
┌────────────────────┐          ┌──────────────────────┐          ┌─────────┐
│  tool_domains.py   │          │  ucl-tool-types.ts   │          │         │
│  tool_export.py    │──JSON───>│  ucl-tool-formatter.ts│──text───>│ System  │
│  builder.py        │          │  system-prompt-with-  │          │ Prompt  │
│                    │          │  ucl.ts              │          │         │
└────────────────────┘          └──────────────────────┘          └─────────┘
  export_for_typescript()         formatUCLToolsForPrompt()
  -> ucl-tools-export.json        -> ^^CONDITION^^ / [[TOOL]] directives
```

---

## Performance at a Glance

| Domain | Tools | JSON Schema | UCL-Tool | Char Reduction | O_s Reduction |
|--------|------:|------------:|---------:|---------------:|--------------:|
| **FileSystem** | 7 | 3,549 chars | 1,857 chars | **47.7%** | **78.5%** |
| **Web** | 5 | 2,759 chars | 1,568 chars | **43.2%** | **74.3%** |
| **Playwright** | 20 | 11,816 chars | 6,317 chars | **46.5%** | **78.9%** |

> **O_s** = structural overhead metric. See [docs/PERFORMANCE.md](docs/PERFORMANCE.md) for the full derivation.

---

## Quick Start

### 1. Export from Python

```python
from tool_export import export_for_typescript

config = export_for_typescript(
    domains=["filesystem", "web"],
    identity="CodeAssistant",
    output_path="ucl-tools-export.json",
)

print(f"Exported {config['metadata']['totalTools']} tools")
# -> Exported 12 tools
```

### 2. Import in TypeScript

```typescript
import uclConfig from "./ucl-tools-export.json";
import { formatUCLToolsForPrompt } from "./src/ucl-tool-formatter.ts";

const promptBlock = formatUCLToolsForPrompt({
  ...uclConfig,
  format: "ucl",
  groupTools: true,
});

console.log(promptBlock);
```

### 3. Use with pi.dev's `buildSystemPrompt`

```typescript
import { buildSystemPrompt } from "./src/system-prompt-with-ucl.ts";
import uclConfig from "./ucl-tools-export.json";

const prompt = buildSystemPrompt({
  cwd: process.cwd(),
  selectedTools: ["read", "bash", "edit", "write"],
  toolSnippets: {
    read: "Read file contents",
    bash: "Execute shell commands",
    edit: "Edit existing files",
    write: "Write new files",
  },
  uclTools: uclConfig,
  enableUCLPlaceholders: true,
});
```

### 4. Placeholder injection in custom prompts

```typescript
const template = `You are a helpful assistant.

{{UCL_TOOLS}}

Guidelines:
- Be concise
`;

const prompt = buildSystemPrompt({
  cwd: process.cwd(),
  customPrompt: template,
  uclTools: uclConfig,
  enableUCLPlaceholders: true, // replaces {{UCL_TOOLS}}
});
```

---

## Configuration Reference

All fields belong to the `UCLToolConfig` interface.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `domains` | `string[]` | `[]` | Pre-built domain names: `"filesystem"`, `"web"`, `"playwright"` |
| `customTools` | `UCLTool[]` | `[]` | Additional tool definitions to include |
| `format` | `"ucl" \| "json" \| "both"` | `"ucl"` | Output format (UCL directives, JSON Schema, or both) |
| `groupTools` | `boolean` | `false` | Group tools under shared `^^CONDITION^^` to reduce O_s |
| `maxTools` | `number` | _(none)_ | Cap on number of tools included (prevents prompt bloat) |
| `identity` | `string` | _(none)_ | Agent role emitted as `<system_role>` block |

### Output format comparison

| `format` value | Output | Best for |
|----------------|--------|----------|
| `"ucl"` | `^^CONDITION^^` / `[[TOOL]]` directives | Minimal token count, maximum O_s reduction |
| `"json"` | Standard `{"tools": [...]}` JSON Schema | OpenAI-compatible APIs, existing tool-calling code |
| `"both"` | UCL block + JSON block | Migration / side-by-side comparison |

---

## File Structure

```
pi-dev-integration/
├── README.md                          <- You are here
├── ucl-tools-export.json              <- Sample exported config (12 tools)
├── src/
│   ├── ucl-tool-types.ts              <- TypeScript interfaces
│   ├── ucl-tool-formatter.ts          <- formatUCLToolsForPrompt(), replaceUCLPlaceholder()
│   └── system-prompt-with-ucl.ts      <- Modified pi.dev buildSystemPrompt()
├── docs/
│   ├── ARCHITECTURE.md                <- Full pipeline design and data flow
│   ├── API-REFERENCE.md               <- Complete function signatures and types
│   ├── MIGRATION-GUIDE.md             <- Before/after comparisons
│   └── PERFORMANCE.md                 <- Benchmarks and O_s mathematics
└── examples/
    ├── README.md                      <- Examples overview
    ├── python-export-example.py       <- Python export workflow
    └── typescript-usage-example.ts    <- TypeScript usage patterns
```

---

## How It Works

### The problem

Standard JSON Schema tool definitions are verbose. Every tool requires nested
`type`, `function`, `parameters`, `properties`, and `required` wrappers. For a
7-tool filesystem domain this produces 3,549 characters and an O_s (structural
overhead) of 23.53.

### The solution

UCL-Tool encodes the same information as flat directives:

```
<filesystem>
    [[TOOL: read_file]]
    [[DESC: Read file content from disk]]
    [[PARAM: path, string, required]]
    [[PARAM: encoding, string, optional]]
    [[WHEN: read, file, open, load]]
</filesystem>
```

The same 7 tools fit in 1,857 characters with O_s = 5.05. That is a
**78.5% reduction in structural overhead** and a **47.7% reduction in raw
character count**.

### Activation via conditions

UCL-Tool uses `^^CONDITION^^` blocks for keyword-based activation. The LLM
only "sees" tool definitions when the user input matches relevant keywords:

```
^^CONDITION: {{concept:user_input}} CONTAINS "file" OR "read" OR "write"^^
<filesystem>
    ...tool definitions...
</filesystem>
^^/CONDITION^^
```

This conditional activation means tools are present in the prompt but gated
behind semantic triggers, reducing effective overhead even further.

---

## Python API

### `export_for_typescript()`

```python
def export_for_typescript(
    domains: List[str] = None,          # e.g. ["filesystem", "web"]
    custom_tools: List[ToolDefinition] = None,
    identity: str = None,               # e.g. "CodeAssistant"
    output_path: str = None,            # save JSON to file
) -> Dict[str, Any]
```

Returns a dict matching `UCLToolExport` (also valid JSON). Pass `output_path`
to write the file directly.

### `migrate_prompt()`

```python
def migrate_prompt(
    prompt_text: str,                   # prompt with JSON Schema tools
    auto_keywords: bool = True,         # infer keywords from name/desc
    group: bool = True,                 # group under shared condition
) -> Tuple[str, List[ToolDefinition]]
```

In-place conversion of JSON Schema tool blocks inside a prompt to UCL-Tool
directives. Useful for migrating existing system prompts.

---

## TypeScript API

### `formatUCLToolsForPrompt(config: UCLToolConfig): string`

Converts a UCL-Tool configuration into a formatted text block ready for
system prompt injection. Returns UCL directives, JSON Schema, or both
depending on `config.format`.

### `replaceUCLPlaceholder(template: string, config: UCLToolConfig): string`

Replaces `{{UCL_TOOLS}}` in a prompt template with the formatted UCL-Tool
block. Returns the template unchanged if no placeholder is found.

### `buildSystemPrompt(options: BuildSystemPromptOptions): string`

Drop-in replacement for pi.dev's `buildSystemPrompt`. Adds two new fields:
`uclTools` and `enableUCLPlaceholders`.

See [docs/API-REFERENCE.md](docs/API-REFERENCE.md) for full type definitions.

---

## Examples

| File | What it shows |
|------|---------------|
| [`examples/python-export-example.py`](examples/python-export-example.py) | Exporting domains, saving JSON, building prompts in Python |
| [`examples/typescript-usage-example.ts`](examples/typescript-usage-example.ts) | Importing JSON, formatting, placeholder injection in TypeScript |

Run the Python example:

```bash
cd UCL_LINTER/pi-dev-integration
python examples/python-export-example.py
```

---

## Deep-Dive Documentation

| Document | Contents |
|----------|----------|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Full pipeline design, data flow diagrams, module responsibilities |
| [docs/API-REFERENCE.md](docs/API-REFERENCE.md) | Complete function signatures, type definitions, return values |
| [docs/MIGRATION-GUIDE.md](docs/MIGRATION-GUIDE.md) | Before/after comparisons, step-by-step migration checklist |
| [docs/PERFORMANCE.md](docs/PERFORMANCE.md) | Benchmark methodology, O_s derivation, scaling analysis |

---

## Related Resources

- [UCL-Tool Architecture](../DOCS/UCL_TOOL_ARCHITECTURE.md) -- core UCL primitives
- [Tool Implementation Spec](../DOCS/UCL_TOOL_IMPLEMENTATION_SPEC.md) -- implementation details
- [Scoring Mathematics](../DOCS/SCORING_MATHEMATICS.md) -- O_s formula derivation
- [pi.dev System Prompt](https://github.com/earendil-works/pi/blob/main/packages/coding-agent/src/core/system-prompt.ts) -- original pi.dev source

---

## Troubleshooting

### Tools not appearing in prompt

1. Verify `uclTools` is passed to `buildSystemPrompt()`.
2. Check domain names are lowercase and match available domains.
3. Ensure `format` is set to `"ucl"`, `"json"`, or `"both"`.

### `{{UCL_TOOLS}}` not replaced

1. Set `enableUCLPlaceholders: true` in options.
2. Verify the placeholder is exactly `{{UCL_TOOLS}}` (case-sensitive).
3. Confirm `uclTools` config is provided and non-empty.

### JSON import errors in TypeScript

1. Validate the JSON: `npx jsonlint ucl-tools-export.json`.
2. Ensure the JSON matches the `UCLToolExport` interface.
3. Regenerate with `export_for_typescript(output_path=...)` if needed.

### Prompt is too long

1. Use `groupTools: true` to collapse tools under shared conditions.
2. Set `maxTools` to limit the number of tools included.
3. Use `format: "ucl"` (not `"both"`) for minimal output.
4. Select fewer domains -- each domain adds its full tool set.
