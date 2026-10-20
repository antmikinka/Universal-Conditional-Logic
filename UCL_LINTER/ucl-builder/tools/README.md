# UCL Tool Domains

Tool definitions are loaded dynamically from JSON files in this folder.

## Structure

```
tools/
├── index.json              # Registry of all available domains
├── tool-loader.js          # Dynamic loader with caching
├── filesystem.json         # File system operations (7 tools)
├── web.json                # HTTP, search, downloads (5 tools)
├── database.json           # SQL queries and tables (6 tools)
├── api.json                # REST, GraphQL, webhooks (4 tools)
├── mcp.json                # Model Context Protocol (5 tools)
├── shell.json              # Commands and environment (4 tools)
├── playwright.json         # Browser automation (20 tools)
├── chroma.json             # Vector database (10 tools)
└── clear_thought.json      # Reasoning and analysis (10 tools)
```

**Total: 71 tools across 9 domains**

## Adding a New Domain

1. Create a new JSON file (e.g., `custom.json`)
2. Use this template:

```json
{
  "name": "custom",
  "description": "Custom tool domain",
  "category": "original",
  "icon": "🔧",
  "keywords": ["custom", "tool"],
  "tools": {
    "my_tool": {
      "name": "my_tool",
      "description": "Does something useful",
      "parameters": [
        {
          "name": "input",
          "type": "string",
          "required": true,
          "description": "The input value"
        }
      ],
      "keywords": ["my", "tool", "useful"],
      "return_type": "string"
    }
  }
}
```

3. Add the domain name to `index.json`
4. Reload the page

## Domain Categories

- **original**: Core domains (filesystem, web, database, api, mcp, shell)
- **mcp_server**: MCP server integrations (playwright, chroma, clear_thought)

## Tool Schema

Each tool has:
- `name`: Unique identifier
- `description`: What the tool does
- `parameters`: Array of parameter objects
  - `name`: Parameter name
  - `type`: string, integer, boolean, object, array, enum, float
  - `required`: true/false
  - `description`: What the parameter does
  - `default`: (optional) Default value
  - `constraints`: (optional) Array of allowed values for enums
- `keywords`: Array of activation keywords
- `return_type`: Expected return type
- `examples`: (optional) Array of {input, output} pairs

## Dynamic Loading

Tools are loaded via `ToolLoader` (tool-loader.js):

```javascript
// Load all domains
await ToolLoader.loadAll();

// Get all domains
const domains = ToolLoader.getAllDomains();

// Get specific domain
const fs = ToolLoader.getDomain('filesystem');

// Check if loaded
if (ToolLoader.isLoaded()) {
    console.log(`Loaded ${ToolLoader.getTotalToolCount()} tools`);
}
```

## Fallback

If dynamic loading fails, the system falls back to `tools-data.js` which contains the same data as hardcoded JavaScript.

## Import from JSON or an MCP server

The Tools tab in the builder has an Import form. Start the page with
`python serve_nocache.py` and open `http://localhost:8765`. Paste JSON or
name an MCP command, then Insert UCL to drop the block into the prompt.
Check "Save into tools/" if the domain should survive a reload.

The page loads whatever `index.json` lists. The same converter runs from
`UCL_LINTER` without the browser:

```bash
python tool_importer.py --json my_tools.json --name my_tools --tools-dir ucl-builder/tools
python tool_importer.py --mcp npx --mcp-arg -y --mcp-arg @playwright/mcp --name playwright_live --tools-dir ucl-builder/tools
```

Accepted JSON is an OpenAI `{"tools":[...]}` envelope, a bare list, an MCP
`tools/list` result (`inputSchema`), or a domain file already in this folder.

`--framing newline` is the MCP spec. Pass `--framing content-length` for a
peer that speaks LSP-style frames (Loom's stdio client is one of them).

The same run can also write the three artifacts without editing this catalog:

```bash
python tool_importer.py --json my_tools.json --name my_tools --out-dir ./out --print none
```

That writes `my_tools.ucl.txt` (grouped `^^CONDITION^^` block), `my_tools.domain.json`
(this folder's shape), and `my_tools.openai.json` (the envelope Loom already
passes to native function calling).
