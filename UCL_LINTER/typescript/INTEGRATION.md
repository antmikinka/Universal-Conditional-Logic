# UCL-Tool Integration with pi.dev System Prompt Builder

> **MOVED:** This documentation has been reorganized into a self-contained
> package at [`../pi-dev-integration/`](../pi-dev-integration/).
>
> - **Quick start:** [`../pi-dev-integration/README.md`](../pi-dev-integration/README.md)
> - **Architecture:** [`../pi-dev-integration/docs/ARCHITECTURE.md`](../pi-dev-integration/docs/ARCHITECTURE.md)
> - **API reference:** [`../pi-dev-integration/docs/API-REFERENCE.md`](../pi-dev-integration/docs/API-REFERENCE.md)
> - **Migration guide:** [`../pi-dev-integration/docs/MIGRATION-GUIDE.md`](../pi-dev-integration/docs/MIGRATION-GUIDE.md)
> - **Performance data:** [`../pi-dev-integration/docs/PERFORMANCE.md`](../pi-dev-integration/docs/PERFORMANCE.md)
> - **Source files:** [`../pi-dev-integration/src/`](../pi-dev-integration/src/)
> - **Examples:** [`../pi-dev-integration/examples/`](../pi-dev-integration/examples/)
>
> The files in this `typescript/` directory are kept for reference but are no
> longer maintained. Use the copies in `pi-dev-integration/src/` instead.

---

This guide shows how to integrate UCL-Tool into the pi.dev build system for dynamic tool injection into system prompts.

## Architecture Overview

```
Python (UCL_LINTER)              TypeScript (pi.dev)
┌─────────────────────┐         ┌──────────────────────┐
│  tool_domains.py    │         │  ucl-tool-types.ts   │
│  - FileSystemDomain │         │  - UCLToolConfig     │
│  - WebDomain        │         │  - UCLTool           │
│  - PlaywrightDomain │         │                      │
└──────────┬──────────┘         └──────────┬───────────┘
           │                               │
           │  export_for_typescript()      │  formatUCLToolsForPrompt()
           │  ─────────────────────────►   │  ─────────────────────────►
           │                               │
           │     JSON (ucl-tools.json)     │     Formatted prompt block
           │                               │
           └───────────────────────────────┴──────────────────────────►
                                                        System Prompt
```

## Quick Start

### Step 1: Export UCL-Tool Config from Python

```python
# In your Python project
from tool_export import export_for_typescript

# Export filesystem and web domains
config = export_for_typescript(
    domains=["filesystem", "web"],
    identity="CodeAssistant",
    output_path="ucl-tools.json"
)

print(f"Exported {config['metadata']['totalTools']} tools")
```

### Step 2: Import JSON in TypeScript

```typescript
// In your pi.dev project
import uclConfig from './ucl-tools.json';
import { buildSystemPrompt } from './system-prompt-with-ucl.ts';

// Build system prompt with UCL-Tool integration
const prompt = buildSystemPrompt({
  cwd: process.cwd(),
  selectedTools: ['read', 'bash', 'edit', 'write'],
  toolSnippets: {
    read: 'Read file contents',
    bash: 'Execute shell commands',
    edit: 'Edit existing files',
    write: 'Write new files',
  },
  uclTools: uclConfig,  // Inject UCL-Tool configuration
});

console.log(prompt);
```

### Step 3: Use Placeholder in Custom Prompts

```typescript
// Custom prompt with {{UCL_TOOLS}} placeholder
const customPrompt = `You are a helpful coding assistant.

{{UCL_TOOLS}}

Guidelines:
- Be concise
- Show file paths clearly
`;

const prompt = buildSystemPrompt({
  cwd: process.cwd(),
  customPrompt: customPrompt,
  uclTools: uclConfig,
  enableUCLPlaceholders: true,  // Enable {{UCL_TOOLS}} replacement
});
```

## Advanced Usage

### Dynamic Tool Selection

```typescript
import { formatUCLToolsForPrompt } from './ucl-tool-formatter.ts';

// Dynamically select tools based on user request
function getToolsForTask(task: string): UCLToolConfig {
  const config: UCLToolConfig = {
    domains: [],
    customTools: [],
    format: 'ucl',
    groupTools: true,
    identity: 'TaskAssistant',
  };

  if (task.includes('file')) {
    config.domains.push('filesystem');
  }
  if (task.includes('web') || task.includes('api')) {
    config.domains.push('web');
  }
  if (task.includes('browser') || task.includes('test')) {
    config.domains.push('playwright');
  }

  return config;
}

// Use it
const task = 'Read files and test web UI';
const tools = getToolsForTask(task);
const formatted = formatUCLToolsForPrompt(tools);
```

### Custom Tool Definitions

```typescript
import type { UCLTool, UCLToolConfig } from './ucl-tool-types.ts';

// Define custom tools
const calculateBMI: UCLTool = {
  name: 'calculate_bmi',
  description: 'Calculate Body Mass Index',
  parameters: [
    {
      name: 'weight_kg',
      type: 'number',
      required: true,
      description: 'Weight in kilograms',
    },
    {
      name: 'height_m',
      type: 'number',
      required: true,
      description: 'Height in meters',
    },
  ],
  keywords: ['bmi', 'body', 'mass', 'index', 'weight', 'height'],
};

// Use in config
const config: UCLToolConfig = {
  customTools: [calculateBMI],
  format: 'ucl',
  identity: 'HealthAssistant',
};
```

### JSON Format (for OpenAI-compatible APIs)

```typescript
const config: UCLToolConfig = {
  domains: ['filesystem'],
  format: 'json',  // Output as JSON Schema
  identity: 'CodeAssistant',
};

const formatted = formatUCLToolsForPrompt(config);
// Returns: <ucl_tools_json>{"tools": [...]}</ucl_tools_json>
```

### Both Formats (UCL + JSON)

```typescript
const config: UCLToolConfig = {
  domains: ['filesystem'],
  format: 'both',  // Output both UCL and JSON
  identity: 'CodeAssistant',
};

const formatted = formatUCLToolsForPrompt(config);
// Returns: UCL directives + JSON Schema block
```

## Integration with pi.dev Build System

### Modified system-prompt.ts

The modified `system-prompt.ts` adds:

1. **UCLToolConfig interface** - TypeScript types for UCL-Tool configuration
2. **formatUCLToolsForPrompt()** - Converts UCL-Tool config to formatted text
3. **{{UCL_TOOLS}} placeholder** - Dynamic injection in custom prompts
4. **uclTools field** - New option in BuildSystemPromptOptions

### Usage in pi.dev

```typescript
// In packages/coding-agent/src/core/system-prompt.ts
import { buildSystemPrompt } from './system-prompt-with-ucl.ts';
import uclConfig from './ucl-tools.json';

export function createAgentPrompt(options: AgentOptions): string {
  return buildSystemPrompt({
    cwd: options.cwd,
    selectedTools: options.tools,
    toolSnippets: getToolSnippets(options.tools),
    uclTools: uclConfig,  // Inject UCL-Tool
    enableUCLPlaceholders: true,
  });
}
```

## Performance Metrics

### Overhead Reduction

| Domain | Tools | JSON Schema | UCL-Tool | Reduction |
|--------|-------|-------------|----------|-----------|
| FileSystem | 7 | 3,549 chars | 1,857 chars | **47.7%** |
| Web | 5 | 2,759 chars | 1,568 chars | **43.2%** |
| Playwright | 20 | 11,816 chars | 6,317 chars | **46.5%** |

### Structural Overhead (O_s)

- **JSON Schema**: O_s = 23.53 (7 tools)
- **UCL-Tool**: O_s = 5.05 (7 tools)
- **Reduction**: **78.5% lower overhead**

## File Structure

```
UCL_LINTER/
├── typescript/
│   ├── ucl-tool-types.ts          # TypeScript interfaces
│   ├── ucl-tool-formatter.ts       # Formatting functions
│   ├── system-prompt-with-ucl.ts   # Modified system prompt builder
│   └── INTEGRATION.md              # This file
├── tool_export.py                  # Python export function
├── tool_domains.py                 # Pre-built tool domains
└── core.py                         # UCL-Tool primitives
```

## Migration Guide

### From JSON Schema to UCL-Tool

```python
# Before (JSON Schema in prompt)
old_prompt = """
You are an assistant with tools:
{"tools": [{"type": "function", "function": {"name": "read_file", ...}}]}
"""

# After (UCL-Tool)
from tool_export import migrate_prompt

new_prompt, tools = migrate_prompt(old_prompt)
# Converts JSON Schema to UCL directives automatically
```

### From Python to TypeScript

```python
# Python: Export config
from tool_export import export_for_typescript

config = export_for_typescript(
    domains=["filesystem", "web"],
    output_path="ucl-tools.json"
)
```

```typescript
// TypeScript: Import and use
import config from './ucl-tools.json';
import { formatUCLToolsForPrompt } from './ucl-tool-formatter.ts';

const formatted = formatUCLToolsForPrompt(config);
console.log(formatted);
```

## API Reference

### Python: export_for_typescript()

```python
def export_for_typescript(
    domains: List[str] = None,        # Domain names to export
    custom_tools: List[ToolDefinition] = None,  # Custom tools
    identity: str = None,             # Agent identity
    output_path: str = None           # Save to file
) -> Dict[str, Any]
```

### TypeScript: formatUCLToolsForPrompt()

```typescript
function formatUCLToolsForPrompt(config: UCLToolConfig): string
```

### TypeScript: replaceUCLPlaceholder()

```typescript
function replaceUCLPlaceholder(
  template: string,        // Prompt with {{UCL_TOOLS}}
  config: UCLToolConfig    // UCL-Tool configuration
): string
```

## Troubleshooting

### Tools Not Appearing in Prompt

1. Check that `uclTools` is provided to `buildSystemPrompt()`
2. Verify domain names match available domains (case-insensitive)
3. Ensure `format` is set to "ucl", "json", or "both"

### Placeholder Not Replaced

1. Set `enableUCLPlaceholders: true` in options
2. Verify placeholder is exactly `{{UCL_TOOLS}}`
3. Check that `uclTools` config is provided

### JSON Import Errors

1. Ensure JSON file is valid (use `jsonlint`)
2. Check that TypeScript interfaces match JSON structure
3. Verify `output_path` in Python export is accessible

## Next Steps

1. **Test the integration**: Run examples in both Python and TypeScript
2. **Customize domains**: Create custom tool domains for your project
3. **Optimize performance**: Use `groupTools: true` and `maxTools` to limit overhead
4. **Extend the bridge**: Add more export formats as needed

## Resources

- [UCL-Tool Architecture](../DOCS/UCL_TOOL_ARCHITECTURE.md)
- [Tool Implementation Spec](../DOCS/UCL_TOOL_IMPLEMENTATION_SPEC.md)
- [pi.dev System Prompt](https://github.com/earendil-works/pi/blob/main/packages/coding-agent/src/core/system-prompt.ts)
