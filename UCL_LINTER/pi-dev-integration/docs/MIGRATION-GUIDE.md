# Migration Guide

Step-by-step guide for migrating from JSON Schema tool definitions to
UCL-Tool directives, and for moving from the old `typescript/` layout to
the new `pi-dev-integration/` structure.

---

## Migrating JSON Schema Tools to UCL-Tool

### Before: JSON Schema in your prompt

```python
system_prompt = """You are a helpful assistant with access to tools.

Tools:
{
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "read_file",
        "description": "Read file content from disk",
        "parameters": {
          "type": "object",
          "properties": {
            "path": {
              "type": "string",
              "description": "Path to the file"
            },
            "encoding": {
              "type": "string",
              "description": "File encoding"
            }
          },
          "required": ["path"]
        }
      }
    },
    {
      "type": "function",
      "function": {
        "name": "write_file",
        "description": "Write content to a file",
        "parameters": {
          "type": "object",
          "properties": {
            "path": { "type": "string" },
            "content": { "type": "string" }
          },
          "required": ["path", "content"]
        }
      }
    }
  ]
}
"""
```

**Character count:** ~780 chars for 2 tools
**O_s:** ~7.5

### After: UCL-Tool directives

```python
from tool_export import migrate_prompt

new_prompt, tools = migrate_prompt(system_prompt)
```

**Result:**

```
You are a helpful assistant with access to tools.

^^CONDITION: {{concept:user_input}} CONTAINS "read" OR "file" OR "write"^^
<tools>
    [[TOOL: read_file]]
    [[DESC: Read file content from disk]]
    [[PARAM: path, string, required]]
    [[PARAM: encoding, string, optional]]
    [[WHEN: read, file, open, load]]

    [[TOOL: write_file]]
    [[DESC: Write content to a file]]
    [[PARAM: path, string, required]]
    [[PARAM: content, string, required]]
    [[WHEN: write, save, create]]
</tools>
^^/CONDITION^^
```

**Character count:** ~420 chars for 2 tools
**O_s:** ~3.1
**Reduction:** ~46% characters, ~59% O_s

---

## Migrating from `typescript/` to `pi-dev-integration/`

The old directory layout:

```
UCL_LINTER/
├── typescript/
│   ├── ucl-tool-types.ts
│   ├── ucl-tool-formatter.ts
│   ├── system-prompt-with-ucl.ts
│   └── INTEGRATION.md
├── tool_export.py
└── examples/
    └── example_pi_dev_integration.py
```

The new layout:

```
UCL_LINTER/
├── pi-dev-integration/         <- NEW: self-contained integration package
│   ├── README.md
│   ├── ucl-tools-export.json
│   ├── src/
│   │   ├── ucl-tool-types.ts
│   │   ├── ucl-tool-formatter.ts
│   │   └── system-prompt-with-ucl.ts
│   ├── docs/
│   │   ├── ARCHITECTURE.md
│   │   ├── API-REFERENCE.md
│   │   ├── MIGRATION-GUIDE.md
│   │   └── PERFORMANCE.md
│   └── examples/
│       ├── README.md
│       ├── python-export-example.py
│       └── typescript-usage-example.ts
├── typescript/                  <- KEPT: redirect notice only
│   └── INTEGRATION.md           <- Points to pi-dev-integration/
├── tool_export.py               <- UNCHANGED: still the source of truth
└── examples/
    └── example_pi_dev_integration.py  <- UNCHANGED
```

### Import path changes

```diff
- import { UCLToolConfig } from "../../typescript/ucl-tool-types.ts";
- import { formatUCLToolsForPrompt } from "../../typescript/ucl-tool-formatter.ts";
+ import { UCLToolConfig } from "../../pi-dev-integration/src/ucl-tool-types.ts";
+ import { formatUCLToolsForPrompt } from "../../pi-dev-integration/src/ucl-tool-formatter.ts";
```

```diff
- import { buildSystemPrompt } from "../../typescript/system-prompt-with-ucl.ts";
+ import { buildSystemPrompt } from "../../pi-dev-integration/src/system-prompt-with-ucl.ts";
```

---

## Migration Checklist

### From JSON Schema to UCL-Tool

- [ ] Identify all JSON Schema tool blocks in your system prompts
- [ ] Run `migrate_prompt()` on each prompt to auto-convert
- [ ] Review generated keywords (add/remove as needed)
- [ ] Test that tool activation triggers correctly
- [ ] Measure character count and O_s before/after
- [ ] Update CI to validate UCL syntax in prompts

### From `typescript/` to `pi-dev-integration/`

- [ ] Update import paths in all TypeScript files
- [ ] Replace `ucl-tools.json` references with `pi-dev-integration/ucl-tools-export.json`
- [ ] Run TypeScript type-check to verify interfaces match
- [ ] Update any documentation links pointing to `typescript/INTEGRATION.md`
- [ ] Delete `typescript/` directory after confirming all references updated

### From Python-only to Python + TypeScript bridge

- [ ] Install Python UCL-Tool package (`pip install -e UCL_LINTER/`)
- [ ] Run `export_for_typescript()` to generate JSON config
- [ ] Copy `ucl-tools-export.json` into your TypeScript project
- [ ] Install TypeScript types (copy `src/ucl-tool-types.ts`)
- [ ] Use `formatUCLToolsForPrompt()` in your prompt assembly code
- [ ] Test end-to-end: Python export -> TypeScript format -> LLM prompt

---

## Side-by-Side Comparison

### Character counts by domain

| Domain | Tools | JSON Schema chars | UCL-Tool chars | Reduction |
|--------|------:|------------------:|---------------:|----------:|
| FileSystem | 7 | 3,549 | 1,857 | **47.7%** |
| Web | 5 | 2,759 | 1,568 | **43.2%** |
| Playwright | 20 | 11,816 | 6,317 | **46.5%** |

### O_s (structural overhead) comparison

| Domain | Tools | JSON Schema O_s | UCL-Tool O_s | Reduction |
|--------|------:|----------------:|-------------:|----------:|
| FileSystem | 7 | 23.53 | 5.05 | **78.5%** |
| Web | 5 | 17.21 | 4.42 | **74.3%** |
| Playwright | 20 | 72.84 | 15.34 | **78.9%** |

### Prompt structure comparison

**JSON Schema approach:**
```
[unconditional] All tool definitions always present in context
[flat structure] No grouping or conditional activation
[high overhead] Nested type/properties/required wrappers
```

**UCL-Tool approach:**
```
[conditional] ^^CONDITION^^ gates tools behind keyword triggers
[grouped] Related tools share domain-level conditions
[low overhead] Flat [[TOOL]] / [[PARAM]] / [[WHEN]] directives
```

---

## Common Migration Issues

### Keywords are too broad

If tools activate on unrelated input, narrow the keyword list:

```python
# Before: too broad
keywords=["get", "fetch", "request", "data", "info"]

# After: more specific
keywords=["http_get", "fetch_url", "api_request"]
```

### Keywords are too narrow

If tools fail to activate when expected, add synonyms:

```python
# Before: misses common phrasing
keywords=["directory_listing"]

# After: covers common usage
keywords=["list", "directory", "folder", "ls", "dir", "contents"]
```

### JSON Schema still needed for API calls

Use `format: "both"` to emit both UCL (for the prompt) and JSON Schema
(for the tool-calling API):

```typescript
const config: UCLToolConfig = {
  domains: ["filesystem"],
  format: "both",
};
// Output contains both <ucl_tools_json> and ^^CONDITION^^ blocks
```

### Import path not resolving

If TypeScript can't resolve `.ts` imports, ensure your `tsconfig.json` has:

```json
{
  "compilerOptions": {
    "moduleResolution": "bundler",
    "allowImportingTsExtensions": true
  }
}
```
