# API Reference

Complete function signatures and type definitions for the UCL-Tool x pi.dev
integration.

---

## TypeScript Types

### `UCLToolParameter`

Describes a single parameter of a tool.

```typescript
interface UCLToolParameter {
  name: string;           // Parameter name (e.g., "path", "url")
  type: string;           // Type: "string", "number", "boolean", "object", "array"
  required: boolean;      // Whether the parameter is required
  description?: string;   // Human-readable description
  default?: any;          // Default value if omitted
  constraints?: Record<string, any>;  // Validation constraints (future use)
}
```

### `UCLToolExample`

A few-shot input/output pair for a tool.

```typescript
interface UCLToolExample {
  input: string;   // Example invocation (e.g., "read_file('/etc/hosts')")
  output: string;  // Example result (e.g., "127.0.0.1 localhost")
}
```

### `UCLTool`

A single tool definition.

```typescript
interface UCLTool {
  name: string;                  // Unique tool name (e.g., "read_file")
  description: string;           // What the tool does
  parameters: UCLToolParameter[]; // Parameter list
  keywords?: string[];           // Activation keywords for WHEN condition
  when?: string[];               // Alternative: explicit WHEN expressions
  examples?: UCLToolExample[];   // Few-shot examples
  returns?: string;              // Description of return value
}
```

### `UCLToolDomain`

A named group of related tools sharing activation keywords.

```typescript
interface UCLToolDomain {
  name: string;           // Domain identifier (e.g., "filesystem")
  description: string;    // What this domain covers
  category: string;       // Category tag (e.g., "core", "web", "testing")
  keywords: string[];     // Domain-level activation keywords
  tools: UCLTool[];       // Tools in this domain
}
```

### `UCLToolConfig`

Configuration for prompt injection. This is the primary input type.

```typescript
interface UCLToolConfig {
  domains?: string[];       // Pre-built domain names to include
  customTools?: UCLTool[];  // Additional custom tool definitions
  format?: "ucl" | "json" | "both";  // Output format (default: "ucl")
  groupTools?: boolean;     // Group tools under shared condition (default: false)
  maxTools?: number;        // Maximum number of tools to include
  identity?: string;        // Agent identity/role for <system_role> block
}
```

### `UCLToolExport`

Schema for the JSON file produced by Python's `export_for_typescript()`.

```typescript
interface UCLToolExport {
  version: string;          // Schema version (e.g., "2.1")
  generatedAt: string;      // ISO 8601 timestamp
  domains: UCLToolDomain[]; // Exported domains
  customTools: UCLTool[];   // Exported custom tools
  metadata: {
    totalTools: number;         // Total tool count across all domains
    estimatedTokens: number;    // Estimated token count for the formatted output
    overheadReduction: number;  // Percentage O_s reduction vs JSON Schema
  };
}
```

---

## TypeScript Functions

### `formatUCLToolsForPrompt(config: UCLToolConfig): string`

Converts a UCL-Tool configuration into a formatted text block for system
prompt injection.

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `config` | `UCLToolConfig` | Tool configuration with domains, format, and options |

**Returns:** `string` -- Formatted UCL directives, JSON Schema block, or both.

**Behavior:**

1. If `config.identity` is set, emits a `<system_role>` block.
2. For each domain in `config.domains`, formats tools according to `config.format`.
3. If `config.format` is `"json"` or `"both"`, wraps tools in `<ucl_tools_json>`.
4. If `config.groupTools` is `true`, collapses all tools under a single `^^CONDITION^^` using the union of all tool keywords (capped at 10).
5. Applies `config.maxTools` limit to the combined tool list.

**Example:**

```typescript
const config: UCLToolConfig = {
  domains: ["filesystem"],
  format: "ucl",
  groupTools: true,
  identity: "CodeAssistant",
};

const result = formatUCLToolsForPrompt(config);
// <system_role>
//     {{concept:ai_identity}} = "CodeAssistant"
// </system_role>
//
// ^^CONDITION: {{concept:user_input}} CONTAINS "file" OR "path" OR ...^^
// <filesystem>
// [[DESC: Local file system operations]]
//     [[TOOL: read_file]]
//     [[DESC: Read file content from disk]]
//     [[PARAM: path, string, required]]
//     [[WHEN: read, file, open, load]]
// ...
// </filesystem>
// ^^/CONDITION^^
```

---

### `replaceUCLPlaceholder(template: string, config: UCLToolConfig): string`

Replaces `{{UCL_TOOLS}}` in a prompt template with the formatted UCL-Tool
block.

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `template` | `string` | Prompt template containing `{{UCL_TOOLS}}` placeholder |
| `config` | `UCLToolConfig` | UCL-Tool configuration |

**Returns:** `string` -- Template with placeholder replaced, or original
template if no placeholder found.

**Example:**

```typescript
const template = "You are an assistant.\n\n{{UCL_TOOLS}}\n\nBe helpful.";
const config: UCLToolConfig = { domains: ["filesystem"], identity: "Helper" };
const result = replaceUCLPlaceholder(template, config);
// "You are an assistant.\n\n<system_role>...\n\nBe helpful."
```

---

### `buildSystemPrompt(options: BuildSystemPromptOptions): string`

Drop-in replacement for pi.dev's `buildSystemPrompt` with UCL-Tool support.

**New fields in `BuildSystemPromptOptions`:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `uclTools` | `UCLToolConfig` | _(undefined)_ | UCL-Tool configuration to inject |
| `enableUCLPlaceholders` | `boolean` | `true` | Enable `{{UCL_TOOLS}}` replacement in `customPrompt` |

**Behavior changes from original:**

- If `uclTools` is provided and using default prompt: appends formatted UCL
  block after the "Available tools" section.
- If `uclTools` is provided and using `customPrompt`: replaces `{{UCL_TOOLS}}`
  placeholder (when `enableUCLPlaceholders` is `true`).
- Adds UCL-specific guidelines when tools are present.

---

## Python Functions

### `export_for_typescript(domains, custom_tools, identity, output_path)`

```python
def export_for_typescript(
    domains: List[str] = None,
    custom_tools: List[ToolDefinition] = None,
    identity: str = None,
    output_path: str = None,
) -> Dict[str, Any]
```

Exports UCL-Tool configuration as a JSON-serializable dictionary compatible
with the TypeScript `UCLToolExport` interface.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `domains` | `List[str]` | `None` | Domain names to export (e.g., `["filesystem", "web"]`) |
| `custom_tools` | `List[ToolDefinition]` | `None` | Additional custom tool definitions |
| `identity` | `str` | `None` | Agent identity for the system prompt |
| `output_path` | `str` | `None` | File path to write JSON (if `None`, returns dict only) |

**Returns:** `Dict[str, Any]` matching the `UCLToolExport` schema:

```python
{
    "version": "2.1",
    "generatedAt": "2026-06-26T10:32:11.824430",
    "domains": [...],          # List of UCLToolDomain dicts
    "customTools": [...],      # List of UCLTool dicts
    "metadata": {
        "totalTools": 12,
        "estimatedTokens": 1200,
        "overheadReduction": 78
    },
    "identity": "CodeAssistant"
}
```

**Example:**

```python
from tool_export import export_for_typescript

# Export to dict
config = export_for_typescript(domains=["filesystem", "web"])

# Export to file
export_for_typescript(
    domains=["filesystem"],
    identity="FileHelper",
    output_path="ucl-tools.json",
)
```

---

### `migrate_prompt(prompt_text, auto_keywords, group)`

```python
def migrate_prompt(
    prompt_text: str,
    auto_keywords: bool = True,
    group: bool = True,
) -> Tuple[str, List[ToolDefinition]]
```

Converts JSON Schema tool definitions found in a prompt string to UCL-Tool
directives.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `prompt_text` | `str` | _(required)_ | Original prompt text containing JSON Schema tools |
| `auto_keywords` | `bool` | `True` | Auto-generate keywords from tool name/description |
| `group` | `bool` | `True` | Group converted tools under shared `^^CONDITION^^` |

**Returns:** `Tuple[str, List[ToolDefinition]]` -- The migrated prompt text
and the list of extracted tool definitions.

**Example:**

```python
from tool_export import migrate_prompt

old_prompt = '''You are an assistant.
{"tools": [{"type": "function", "function": {"name": "read_file", ...}}]}
'''

new_prompt, tools = migrate_prompt(old_prompt)
# new_prompt now contains ^^CONDITION^^ / [[TOOL]] directives
```

---

### `keyword_heuristic(name, description)`

```python
def keyword_heuristic(name: str, description: str) -> List[str]
```

Infers activation keywords from a tool's name and description by extracting
content words and filtering stopwords.

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `name` | `str` | Tool name (e.g., `"read_file"`) |
| `description` | `str` | Tool description (e.g., `"Read file content from disk"`) |

**Returns:** `List[str]` -- Inferred keywords (e.g., `["read", "file", "content", "disk"]`).

---

## Error Handling

### TypeScript

- `formatUCLToolsForPrompt` returns an empty string if no domains or custom
  tools are provided.
- `replaceUCLPlaceholder` returns the original template if `{{UCL_TOOLS}}`
  is not found (no error).
- Invalid domain names are silently skipped (only valid domains produce output).

### Python

- `export_for_typescript` raises `ValueError` if a domain name is not found
  in the registry.
- `migrate_prompt` returns the original text unchanged if no JSON Schema
  blocks are detected.
- `keyword_heuristic` always returns at least one keyword (falls back to the
  tool name itself).


---

## Related Documentation

- **[README](../README.md)** — Quick start and overview
- **[Architecture](./ARCHITECTURE.md)** — Pipeline design and data flow
- **[Performance](./PERFORMANCE.md)** — Benchmark data and O_s analysis
