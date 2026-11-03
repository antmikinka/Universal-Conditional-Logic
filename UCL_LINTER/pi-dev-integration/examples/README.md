# Examples

Working code examples for the UCL-Tool x pi.dev integration.

## Files

| File | Language | What it shows |
|------|----------|---------------|
| `python-export-example.py` | Python | Exporting domains, saving JSON, building prompts, keyword heuristics, custom tools |
| `typescript-usage-example.ts` | TypeScript | Importing JSON, formatting, placeholder injection, dynamic tool selection |

## Running the Python example

```bash
cd UCL_LINTER
python pi-dev-integration/examples/python-export-example.py
```

This produces output for each example:
- Example 1: Basic export with metadata
- Example 2: Saving JSON to `ucl-tools-export.json`
- Example 3: Building a UCL-optimized prompt in Python
- Example 4: Auto-generating keywords from tool names
- Example 5: Exporting with custom tool definitions

## Running the TypeScript example

```bash
# From the UCL_LINTER directory
npx tsx pi-dev-integration/examples/typescript-usage-example.ts

# Or if you have tsx installed globally
tsx pi-dev-integration/examples/typescript-usage-example.ts
```

This produces output for each example:
- Example 1: Basic UCL formatting
- Example 2: Multiple domains with grouping
- Example 3: JSON Schema format output
- Example 4: Custom tool definitions
- Example 5: `{{UCL_TOOLS}}` placeholder injection
- Example 6: Dynamic tool selection based on task
- Example 7: Importing from exported JSON file

## End-to-End Workflow

The complete workflow spans both languages:

```
1. Python: Define or select tool domains
2. Python: export_for_typescript() -> ucl-tools-export.json
3. TypeScript: import ucl-tools-export.json
4. TypeScript: formatUCLToolsForPrompt(config) -> formatted UCL text
5. TypeScript: buildSystemPrompt({ uclTools: config }) -> full system prompt
```

Run the Python example first to generate the JSON, then run the TypeScript
example to see it consumed.
