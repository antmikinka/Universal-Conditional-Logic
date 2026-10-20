// ============================================================================
// UCL TOOL DOMAINS DATA
// Complete reference of all 71 pre-built tools across 9 domains
// Mirrors tool_domains.py — single source of truth for the web UI
// ============================================================================

const UCL_TOOL_DOMAINS = {
    filesystem: {
        name: "filesystem",
        description: "Local file system operations",
        category: "original",
        icon: "📁",
        keywords: ["file", "path", "directory", "read", "write", "delete", "move", "copy"],
        tools: {
            read_file: {
                name: "read_file",
                description: "Read file content from disk",
                parameters: [
                    { name: "path", type: "string", required: true, description: "Path to the file" },
                    { name: "encoding", type: "string", required: false, description: "File encoding", default: "utf-8" }
                ],
                keywords: ["read", "file", "open", "load"],
                return_type: "string",
                examples: [{ input: "read_file('/etc/hosts')", output: "127.0.0.1 localhost" }]
            },
            write_file: {
                name: "write_file",
                description: "Write content to a file",
                parameters: [
                    { name: "path", type: "string", required: true, description: "Path to the file" },
                    { name: "content", type: "string", required: true, description: "Content to write" }
                ],
                keywords: ["write", "save", "create"],
                return_type: "boolean"
            },
            list_dir: {
                name: "list_dir",
                description: "List directory contents",
                parameters: [
                    { name: "path", type: "string", required: true, description: "Directory path" }
                ],
                keywords: ["list", "directory", "folder", "ls"],
                return_type: "array"
            },
            delete_file: {
                name: "delete_file",
                description: "Delete a file from disk",
                parameters: [
                    { name: "path", type: "string", required: true, description: "Path to the file" }
                ],
                keywords: ["delete", "remove"],
                return_type: "boolean"
            },
            move_file: {
                name: "move_file",
                description: "Move or rename a file",
                parameters: [
                    { name: "source", type: "string", required: true, description: "Source path" },
                    { name: "dest", type: "string", required: true, description: "Destination path" }
                ],
                keywords: ["move", "rename"],
                return_type: "boolean"
            },
            copy_file: {
                name: "copy_file",
                description: "Copy a file to a new location",
                parameters: [
                    { name: "source", type: "string", required: true, description: "Source path" },
                    { name: "dest", type: "string", required: true, description: "Destination path" }
                ],
                keywords: ["copy", "duplicate"],
                return_type: "boolean"
            },
            file_info: {
                name: "file_info",
                description: "Get file metadata (size, modified time, permissions)",
                parameters: [
                    { name: "path", type: "string", required: true, description: "Path to the file" }
                ],
                keywords: ["info", "stat", "metadata", "size"],
                return_type: "object"
            }
        }
    },

    web: {
        name: "web",
        description: "Web operations: HTTP requests, search, and downloads",
        category: "original",
        icon: "🌐",
        keywords: ["http", "web", "url", "request", "fetch", "search", "download"],
        tools: {
            http_get: {
                name: "http_get",
                description: "Send an HTTP GET request",
                parameters: [
                    { name: "url", type: "string", required: true, description: "Request URL" },
                    { name: "headers", type: "object", required: false, description: "HTTP headers" }
                ],
                keywords: ["get", "fetch", "request"],
                return_type: "object",
                examples: [{ input: "http_get('https://api.example.com/data')", output: '{"status": 200, "body": ...}' }]
            },
            http_post: {
                name: "http_post",
                description: "Send an HTTP POST request",
                parameters: [
                    { name: "url", type: "string", required: true, description: "Request URL" },
                    { name: "body", type: "string", required: true, description: "Request body" },
                    { name: "headers", type: "object", required: false, description: "HTTP headers" }
                ],
                keywords: ["post", "send", "submit"],
                return_type: "object"
            },
            web_search: {
                name: "web_search",
                description: "Search the web for information",
                parameters: [
                    { name: "query", type: "string", required: true, description: "Search query" },
                    { name: "max_results", type: "integer", required: false, description: "Maximum results", default: "10" }
                ],
                keywords: ["search", "find", "query"],
                return_type: "array",
                examples: [{ input: "web_search('UCL paper')", output: "arXiv:2601.00880 - Universal Conditional Logic..." }]
            },
            fetch_url: {
                name: "fetch_url",
                description: "Fetch content from a URL",
                parameters: [
                    { name: "url", type: "string", required: true, description: "URL to fetch" }
                ],
                keywords: ["fetch", "download", "retrieve"],
                return_type: "string"
            },
            download_file: {
                name: "download_file",
                description: "Download a file from a URL to local disk",
                parameters: [
                    { name: "url", type: "string", required: true, description: "URL of the file" },
                    { name: "dest", type: "string", required: false, description: "Local destination path" }
                ],
                keywords: ["download", "save"],
                return_type: "string"
            }
        }
    },

    database: {
        name: "database",
        description: "Database operations: SQL queries and table management",
        category: "original",
        icon: "🗄️",
        keywords: ["database", "sql", "table", "query", "row", "db"],
        tools: {
            query_sql: {
                name: "query_sql",
                description: "Execute a SQL query and return results",
                parameters: [
                    { name: "query", type: "string", required: true, description: "SQL query string" },
                    { name: "params", type: "array", required: false, description: "Query parameters" }
                ],
                keywords: ["query", "select", "sql", "search"],
                return_type: "array",
                examples: [{ input: "query_sql('SELECT * FROM users WHERE id = ?', [1])", output: "[{id: 1, name: 'Alice'}]" }]
            },
            insert_row: {
                name: "insert_row",
                description: "Insert a row into a table",
                parameters: [
                    { name: "table", type: "string", required: true, description: "Table name" },
                    { name: "data", type: "object", required: true, description: "Column-value pairs" }
                ],
                keywords: ["insert", "create", "add"],
                return_type: "object"
            },
            update_row: {
                name: "update_row",
                description: "Update rows in a table matching a condition",
                parameters: [
                    { name: "table", type: "string", required: true, description: "Table name" },
                    { name: "data", type: "object", required: true, description: "Column-value pairs to update" },
                    { name: "condition", type: "string", required: true, description: "WHERE condition" }
                ],
                keywords: ["update", "modify"],
                return_type: "integer"
            },
            delete_row: {
                name: "delete_row",
                description: "Delete rows from a table matching a condition",
                parameters: [
                    { name: "table", type: "string", required: true, description: "Table name" },
                    { name: "condition", type: "string", required: true, description: "WHERE condition" }
                ],
                keywords: ["delete", "remove"],
                return_type: "integer"
            },
            create_table: {
                name: "create_table",
                description: "Create a new table with a schema",
                parameters: [
                    { name: "name", type: "string", required: true, description: "Table name" },
                    { name: "schema", type: "object", required: true, description: "Column definitions" }
                ],
                keywords: ["create", "table", "schema"],
                return_type: "boolean"
            },
            list_tables: {
                name: "list_tables",
                description: "List all tables in the database",
                parameters: [],
                keywords: ["list", "tables", "schema"],
                return_type: "array"
            }
        }
    },

    api: {
        name: "api",
        description: "API operations: REST, GraphQL, webhooks, and authentication",
        category: "original",
        icon: "🔌",
        keywords: ["api", "rest", "graphql", "webhook", "endpoint", "request"],
        tools: {
            rest_call: {
                name: "rest_call",
                description: "Make a REST API call",
                parameters: [
                    { name: "method", type: "enum", required: true, description: "HTTP method", constraints: ["GET", "POST", "PUT", "DELETE", "PATCH"] },
                    { name: "url", type: "string", required: true, description: "API endpoint URL" },
                    { name: "body", type: "object", required: false, description: "Request body" },
                    { name: "headers", type: "object", required: false, description: "HTTP headers" }
                ],
                keywords: ["api", "rest", "call", "endpoint"],
                return_type: "object",
                examples: [{ input: "rest_call('GET', 'https://api.example.com/users')", output: '{"users": [...]}' }]
            },
            graphql_query: {
                name: "graphql_query",
                description: "Execute a GraphQL query",
                parameters: [
                    { name: "query", type: "string", required: true, description: "GraphQL query string" },
                    { name: "variables", type: "object", required: false, description: "Query variables" }
                ],
                keywords: ["graphql", "query", "mutation"],
                return_type: "object"
            },
            webhook_send: {
                name: "webhook_send",
                description: "Send a webhook notification",
                parameters: [
                    { name: "url", type: "string", required: true, description: "Webhook URL" },
                    { name: "payload", type: "object", required: true, description: "Webhook payload" }
                ],
                keywords: ["webhook", "notify", "callback"],
                return_type: "boolean"
            },
            auth_token: {
                name: "auth_token",
                description: "Obtain an authentication token",
                parameters: [
                    { name: "provider", type: "string", required: true, description: "Auth provider name" },
                    { name: "scopes", type: "array", required: false, description: "Requested scopes" }
                ],
                keywords: ["auth", "token", "login", "credential"],
                return_type: "string"
            }
        }
    },

    mcp: {
        name: "mcp",
        description: "MCP (Model Context Protocol) operations",
        category: "original",
        icon: "🔗",
        keywords: ["mcp", "server", "protocol", "tool", "resource"],
        tools: {
            mcp_call: {
                name: "mcp_call",
                description: "Call a tool on an MCP server (maps to tools/call)",
                parameters: [
                    { name: "server", type: "string", required: true, description: "MCP server name" },
                    { name: "tool", type: "string", required: true, description: "Tool name to invoke" },
                    { name: "arguments", type: "object", required: false, description: "Tool arguments" }
                ],
                keywords: ["mcp", "call", "invoke", "tool"],
                return_type: "object",
                examples: [{ input: "mcp_call('filesystem', 'read_file', {'path': '/tmp/x'})", output: "file contents..." }]
            },
            mcp_list_tools: {
                name: "mcp_list_tools",
                description: "List available tools on an MCP server (maps to tools/list)",
                parameters: [
                    { name: "server", type: "string", required: false, description: "MCP server name (omit for all)" }
                ],
                keywords: ["mcp", "tools", "list", "discover"],
                return_type: "array"
            },
            mcp_list_resources: {
                name: "mcp_list_resources",
                description: "List available resources on an MCP server (maps to resources/list)",
                parameters: [
                    { name: "server", type: "string", required: false, description: "MCP server name (omit for all)" }
                ],
                keywords: ["mcp", "resources", "list"],
                return_type: "array"
            },
            mcp_read_resource: {
                name: "mcp_read_resource",
                description: "Read a resource from an MCP server (maps to resources/read)",
                parameters: [
                    { name: "server", type: "string", required: true, description: "MCP server name" },
                    { name: "uri", type: "string", required: true, description: "Resource URI" }
                ],
                keywords: ["mcp", "resource", "read", "fetch"],
                return_type: "object"
            },
            mcp_list_prompts: {
                name: "mcp_list_prompts",
                description: "List available prompts on an MCP server (maps to prompts/list)",
                parameters: [
                    { name: "server", type: "string", required: false, description: "MCP server name (omit for all)" }
                ],
                keywords: ["mcp", "prompts", "list"],
                return_type: "array"
            }
        }
    },

    shell: {
        name: "shell",
        description: "Shell operations: commands, scripts, and environment",
        category: "original",
        icon: "💻",
        keywords: ["shell", "command", "execute", "script", "env", "terminal"],
        tools: {
            run_command: {
                name: "run_command",
                description: "Execute a shell command",
                parameters: [
                    { name: "command", type: "string", required: true, description: "Command to execute" },
                    { name: "cwd", type: "string", required: false, description: "Working directory" },
                    { name: "timeout", type: "integer", required: false, description: "Timeout in seconds", default: "30" }
                ],
                keywords: ["command", "execute", "run", "shell"],
                return_type: "object",
                examples: [{ input: "run_command('ls -la /tmp')", output: "total 8\ndrwxrwxrwt..." }]
            },
            run_script: {
                name: "run_script",
                description: "Execute a script file",
                parameters: [
                    { name: "path", type: "string", required: true, description: "Path to the script" },
                    { name: "args", type: "array", required: false, description: "Script arguments" }
                ],
                keywords: ["script", "execute", "run"],
                return_type: "object"
            },
            get_env: {
                name: "get_env",
                description: "Get an environment variable value",
                parameters: [
                    { name: "name", type: "string", required: true, description: "Variable name" }
                ],
                keywords: ["env", "environment", "variable", "get"],
                return_type: "string"
            },
            set_env: {
                name: "set_env",
                description: "Set an environment variable",
                parameters: [
                    { name: "name", type: "string", required: true, description: "Variable name" },
                    { name: "value", type: "string", required: true, description: "Variable value" }
                ],
                keywords: ["env", "set", "environment"],
                return_type: "boolean"
            }
        }
    },

    playwright: {
        name: "playwright",
        description: "Browser automation via Playwright MCP",
        category: "mcp_server",
        icon: "🎭",
        keywords: ["browser", "playwright", "web", "page", "click", "navigate", "screenshot"],
        tools: {
            browser_navigate: {
                name: "browser_navigate",
                description: "Navigate to a URL in the browser",
                parameters: [
                    { name: "url", type: "string", required: true, description: "URL to navigate to" }
                ],
                keywords: ["navigate", "url", "goto", "page"]
            },
            browser_click: {
                name: "browser_click",
                description: "Click on an element on the page",
                parameters: [
                    { name: "target", type: "string", required: true, description: "Element selector or reference" },
                    { name: "button", type: "enum", required: false, description: "Mouse button", constraints: ["left", "right", "middle"] },
                    { name: "doubleClick", type: "boolean", required: false, description: "Perform double click" }
                ],
                keywords: ["click", "button", "element", "tap"]
            },
            browser_type: {
                name: "browser_type",
                description: "Type text into an editable element",
                parameters: [
                    { name: "target", type: "string", required: true, description: "Element to type into" },
                    { name: "text", type: "string", required: true, description: "Text to type" },
                    { name: "slowly", type: "boolean", required: false, description: "Type one character at a time" },
                    { name: "submit", type: "boolean", required: false, description: "Press Enter after typing" }
                ],
                keywords: ["type", "input", "text", "enter", "keyboard"]
            },
            browser_snapshot: {
                name: "browser_snapshot",
                description: "Capture accessibility snapshot of the current page",
                parameters: [
                    { name: "target", type: "string", required: false, description: "Specific element to snapshot" },
                    { name: "filename", type: "string", required: false, description: "Save snapshot to file" }
                ],
                keywords: ["snapshot", "accessibility", "capture", "page"]
            },
            browser_take_screenshot: {
                name: "browser_take_screenshot",
                description: "Take a screenshot of the page or element",
                parameters: [
                    { name: "type", type: "enum", required: true, description: "Image format", constraints: ["png", "jpeg"] },
                    { name: "filename", type: "string", required: false, description: "Save screenshot to file" },
                    { name: "fullPage", type: "boolean", required: false, description: "Screenshot full scrollable page" },
                    { name: "target", type: "string", required: false, description: "Specific element to screenshot" }
                ],
                keywords: ["screenshot", "image", "capture", "photo"]
            },
            browser_hover: {
                name: "browser_hover",
                description: "Hover over an element on the page",
                parameters: [
                    { name: "target", type: "string", required: true, description: "Element to hover over" }
                ],
                keywords: ["hover", "mouseover", "element"]
            },
            browser_press_key: {
                name: "browser_press_key",
                description: "Press a keyboard key",
                parameters: [
                    { name: "key", type: "string", required: true, description: "Key name (e.g. ArrowLeft, Enter, a)" }
                ],
                keywords: ["key", "keyboard", "press", "shortcut"]
            },
            browser_fill_form: {
                name: "browser_fill_form",
                description: "Fill multiple form fields at once",
                parameters: [
                    { name: "fields", type: "array", required: true, description: "List of field objects with target, name, type, value" }
                ],
                keywords: ["form", "fill", "input", "field"]
            },
            browser_select_option: {
                name: "browser_select_option",
                description: "Select an option in a dropdown",
                parameters: [
                    { name: "target", type: "string", required: true, description: "Select element" },
                    { name: "values", type: "array", required: true, description: "Values to select" }
                ],
                keywords: ["select", "dropdown", "option", "choose"]
            },
            browser_drag: {
                name: "browser_drag",
                description: "Perform drag and drop between elements",
                parameters: [
                    { name: "startTarget", type: "string", required: true, description: "Source element" },
                    { name: "endTarget", type: "string", required: true, description: "Target element" }
                ],
                keywords: ["drag", "drop", "move", "element"]
            },
            browser_file_upload: {
                name: "browser_file_upload",
                description: "Upload one or multiple files",
                parameters: [
                    { name: "paths", type: "array", required: true, description: "Absolute file paths to upload" }
                ],
                keywords: ["upload", "file", "attach"]
            },
            browser_handle_dialog: {
                name: "browser_handle_dialog",
                description: "Handle an alert, confirm, or prompt dialog",
                parameters: [
                    { name: "accept", type: "boolean", required: true, description: "Accept or dismiss the dialog" },
                    { name: "promptText", type: "string", required: false, description: "Text for prompt dialogs" }
                ],
                keywords: ["dialog", "alert", "confirm", "prompt"]
            },
            browser_tabs: {
                name: "browser_tabs",
                description: "List, create, close, or select browser tabs",
                parameters: [
                    { name: "action", type: "enum", required: true, description: "Operation", constraints: ["list", "new", "close", "select"] },
                    { name: "index", type: "integer", required: false, description: "Tab index for close/select" },
                    { name: "url", type: "string", required: false, description: "URL for new tab" }
                ],
                keywords: ["tab", "window", "switch"]
            },
            browser_navigate_back: {
                name: "browser_navigate_back",
                description: "Go back to the previous page",
                parameters: [],
                keywords: ["back", "history", "previous"]
            },
            browser_wait_for: {
                name: "browser_wait_for",
                description: "Wait for text to appear or a time duration",
                parameters: [
                    { name: "text", type: "string", required: false, description: "Text to wait for" },
                    { name: "textGone", type: "string", required: false, description: "Text to wait for disappearance" },
                    { name: "time", type: "float", required: false, description: "Seconds to wait" }
                ],
                keywords: ["wait", "delay", "timeout", "pause"]
            },
            browser_evaluate: {
                name: "browser_evaluate",
                description: "Evaluate JavaScript expression on page",
                parameters: [
                    { name: "function", type: "string", required: true, description: "JavaScript function to evaluate" },
                    { name: "target", type: "string", required: false, description: "Element to evaluate on" },
                    { name: "filename", type: "string", required: false, description: "Save result to file" }
                ],
                keywords: ["evaluate", "javascript", "js", "script", "execute"]
            },
            browser_console_messages: {
                name: "browser_console_messages",
                description: "Return console messages from the page",
                parameters: [
                    { name: "level", type: "enum", required: false, description: "Log level", constraints: ["info", "warning", "error", "debug"] },
                    { name: "all", type: "boolean", required: false, description: "Return all messages since session start" }
                ],
                keywords: ["console", "log", "debug", "error", "message"]
            },
            browser_network_requests: {
                name: "browser_network_requests",
                description: "Return list of network requests since page load",
                parameters: [
                    { name: "static", type: "boolean", required: false, description: "Include static resources" },
                    { name: "filter", type: "string", required: false, description: "URL regexp filter" }
                ],
                keywords: ["network", "request", "http", "fetch", "api"]
            },
            browser_close: {
                name: "browser_close",
                description: "Close the browser page",
                parameters: [],
                keywords: ["close", "quit", "exit", "browser"]
            },
            browser_resize: {
                name: "browser_resize",
                description: "Resize the browser window",
                parameters: [
                    { name: "width", type: "integer", required: true, description: "Window width in pixels" },
                    { name: "height", type: "integer", required: true, description: "Window height in pixels" }
                ],
                keywords: ["resize", "window", "viewport", "size"]
            }
        }
    },

    chroma: {
        name: "chroma",
        description: "Chroma vector DB: collections, documents, similarity search",
        category: "mcp_server",
        icon: "🧬",
        keywords: ["chroma", "vector", "embedding", "collection", "document", "similarity"],
        tools: {
            chroma_create_collection: {
                name: "chroma_create_collection",
                description: "Create a new Chroma collection",
                parameters: [
                    { name: "collection_name", type: "string", required: true, description: "Name of the collection" },
                    { name: "embedding_function_name", type: "string", required: false, description: "Embedding function: default, cohere, openai, jina" }
                ],
                keywords: ["create", "collection", "new", "vector"]
            },
            chroma_delete_collection: {
                name: "chroma_delete_collection",
                description: "Delete a Chroma collection",
                parameters: [
                    { name: "collection_name", type: "string", required: true, description: "Name of the collection to delete" }
                ],
                keywords: ["delete", "collection", "remove", "drop"]
            },
            chroma_list_collections: {
                name: "chroma_list_collections",
                description: "List all collection names in the database",
                parameters: [
                    { name: "limit", type: "integer", required: false, description: "Maximum collections to return" },
                    { name: "offset", type: "integer", required: false, description: "Offset for pagination" }
                ],
                keywords: ["list", "collections", "all", "show"]
            },
            chroma_add_documents: {
                name: "chroma_add_documents",
                description: "Add documents to a Chroma collection",
                parameters: [
                    { name: "collection_name", type: "string", required: true, description: "Collection to add to" },
                    { name: "documents", type: "array", required: true, description: "List of text documents" },
                    { name: "ids", type: "array", required: true, description: "List of document IDs" },
                    { name: "metadatas", type: "array", required: false, description: "List of metadata dicts" }
                ],
                keywords: ["add", "document", "insert", "store", "ingest"]
            },
            chroma_get_documents: {
                name: "chroma_get_documents",
                description: "Get documents from a collection by ID or filter",
                parameters: [
                    { name: "collection_name", type: "string", required: true, description: "Collection to query" },
                    { name: "ids", type: "array", required: false, description: "Document IDs to retrieve" },
                    { name: "where", type: "object", required: false, description: "Metadata filter" },
                    { name: "limit", type: "integer", required: false, description: "Maximum documents" }
                ],
                keywords: ["get", "document", "retrieve", "fetch", "find"]
            },
            chroma_query_documents: {
                name: "chroma_query_documents",
                description: "Query documents by similarity search",
                parameters: [
                    { name: "collection_name", type: "string", required: true, description: "Collection to search" },
                    { name: "query_texts", type: "array", required: true, description: "Query strings for similarity search" },
                    { name: "n_results", type: "integer", required: false, description: "Number of results per query" },
                    { name: "where", type: "object", required: false, description: "Metadata filter" }
                ],
                keywords: ["query", "search", "similarity", "find", "vector"]
            },
            chroma_update_documents: {
                name: "chroma_update_documents",
                description: "Update documents in a collection",
                parameters: [
                    { name: "collection_name", type: "string", required: true, description: "Collection to update" },
                    { name: "ids", type: "array", required: true, description: "Document IDs to update" },
                    { name: "documents", type: "array", required: false, description: "New document text" },
                    { name: "metadatas", type: "array", required: false, description: "New metadata" }
                ],
                keywords: ["update", "modify", "change", "edit"]
            },
            chroma_delete_documents: {
                name: "chroma_delete_documents",
                description: "Delete documents from a collection",
                parameters: [
                    { name: "collection_name", type: "string", required: true, description: "Collection to delete from" },
                    { name: "ids", type: "array", required: true, description: "Document IDs to delete" }
                ],
                keywords: ["delete", "remove", "document"]
            },
            chroma_peek_collection: {
                name: "chroma_peek_collection",
                description: "Peek at documents in a collection",
                parameters: [
                    { name: "collection_name", type: "string", required: true, description: "Collection to peek at" },
                    { name: "limit", type: "integer", required: false, description: "Number of documents to peek" }
                ],
                keywords: ["peek", "preview", "sample", "view"]
            },
            chroma_get_collection_info: {
                name: "chroma_get_collection_info",
                description: "Get information about a collection",
                parameters: [
                    { name: "collection_name", type: "string", required: true, description: "Collection name" }
                ],
                keywords: ["info", "collection", "metadata", "describe"]
            }
        }
    },

    clear_thought: {
        name: "clear_thought",
        description: "Structured reasoning: thinking, analysis, decisions, patterns",
        category: "mcp_server",
        icon: "🧠",
        keywords: ["think", "reason", "analyze", "decide", "debug", "pattern", "mental_model"],
        tools: {
            sequential_thinking: {
                name: "sequential_thinking",
                description: "Dynamic problem-solving through sequential thought steps",
                parameters: [
                    { name: "thought", type: "string", required: true, description: "Current thinking step" },
                    { name: "thoughtNumber", type: "integer", required: true, description: "Current thought number" },
                    { name: "totalThoughts", type: "integer", required: true, description: "Estimated total thoughts needed" },
                    { name: "nextThoughtNeeded", type: "boolean", required: true, description: "Whether another step is needed" }
                ],
                keywords: ["think", "reason", "step", "analyze", "problem"]
            },
            collaborative_reasoning: {
                name: "collaborative_reasoning",
                description: "Simulate expert collaboration with diverse perspectives",
                parameters: [
                    { name: "topic", type: "string", required: true, description: "Topic for collaboration" },
                    { name: "personas", type: "array", required: true, description: "List of expert personas" },
                    { name: "stage", type: "enum", required: true, description: "Stage", constraints: ["problem-definition", "ideation", "critique", "integration", "decision", "reflection"] }
                ],
                keywords: ["collaborate", "perspective", "expert", "team"]
            },
            debugging_approach: {
                name: "debugging_approach",
                description: "Apply systematic debugging approaches to issues",
                parameters: [
                    { name: "approachName", type: "enum", required: true, description: "Method", constraints: ["binary_search", "reverse_engineering", "divide_conquer", "backtracking", "cause_elimination", "program_slicing"] },
                    { name: "issue", type: "string", required: true, description: "Description of the issue" },
                    { name: "steps", type: "array", required: false, description: "Debugging steps" }
                ],
                keywords: ["debug", "fix", "error", "issue", "troubleshoot"]
            },
            decision_framework: {
                name: "decision_framework",
                description: "Structured decision analysis and rational choice",
                parameters: [
                    { name: "decisionStatement", type: "string", required: true, description: "What decision needs to be made" },
                    { name: "options", type: "array", required: true, description: "Available options with descriptions" },
                    { name: "analysisType", type: "enum", required: true, description: "Type", constraints: ["pros-cons", "weighted-criteria", "decision-tree", "expected-value", "scenario-analysis"] },
                    { name: "stage", type: "enum", required: true, description: "Stage", constraints: ["problem-definition", "options-generation", "criteria-definition", "evaluation", "sensitivity-analysis", "decision"] }
                ],
                keywords: ["decide", "choice", "option", "evaluate", "compare"]
            },
            design_pattern: {
                name: "design_pattern",
                description: "Apply software design patterns to architecture",
                parameters: [
                    { name: "patternName", type: "enum", required: true, description: "Pattern", constraints: ["modular_architecture", "api_integration", "state_management", "async_processing", "scalability", "security", "agentic_design"] },
                    { name: "context", type: "string", required: true, description: "Problem context" },
                    { name: "implementation", type: "array", required: false, description: "Implementation steps" }
                ],
                keywords: ["pattern", "design", "architecture", "structure"]
            },
            mental_model: {
                name: "mental_model",
                description: "Apply structured mental models to problem-solving",
                parameters: [
                    { name: "modelName", type: "enum", required: true, description: "Model", constraints: ["first_principles", "opportunity_cost", "error_propagation", "rubber_duck", "pareto_principle", "occams_razor"] },
                    { name: "problem", type: "string", required: true, description: "Problem to analyze" },
                    { name: "steps", type: "array", required: false, description: "Analysis steps" }
                ],
                keywords: ["model", "framework", "think", "perspective"]
            },
            metacognitive_monitoring: {
                name: "metacognitive_monitoring",
                description: "Systematic self-monitoring of knowledge and reasoning quality",
                parameters: [
                    { name: "task", type: "string", required: true, description: "Task being monitored" },
                    { name: "stage", type: "enum", required: true, description: "Stage", constraints: ["knowledge-assessment", "planning", "execution", "monitoring", "evaluation", "reflection"] },
                    { name: "overallConfidence", type: "float", required: true, description: "Overall confidence 0-1" },
                    { name: "uncertaintyAreas", type: "array", required: false, description: "Areas of uncertainty" }
                ],
                keywords: ["monitor", "confidence", "uncertainty", "self-assess", "quality"]
            },
            programming_paradigm: {
                name: "programming_paradigm",
                description: "Apply different programming paradigms to problems",
                parameters: [
                    { name: "paradigmName", type: "enum", required: true, description: "Paradigm", constraints: ["imperative", "procedural", "object_oriented", "functional", "declarative", "logic", "event_driven", "concurrent", "reactive"] },
                    { name: "problem", type: "string", required: true, description: "Problem to solve" },
                    { name: "approach", type: "array", required: false, description: "Approach steps" }
                ],
                keywords: ["paradigm", "programming", "approach", "style"]
            },
            scientific_method: {
                name: "scientific_method",
                description: "Apply formal scientific reasoning and hypothesis testing",
                parameters: [
                    { name: "stage", type: "enum", required: true, description: "Stage", constraints: ["observation", "question", "hypothesis", "experiment", "analysis", "conclusion", "iteration"] },
                    { name: "inquiryId", type: "string", required: true, description: "Unique inquiry identifier" },
                    { name: "iteration", type: "integer", required: true, description: "Current iteration number" },
                    { name: "nextStageNeeded", type: "boolean", required: true, description: "Whether another stage is needed" }
                ],
                keywords: ["science", "hypothesis", "experiment", "test", "evidence"]
            },
            structured_argumentation: {
                name: "structured_argumentation",
                description: "Systematic dialectical reasoning and argument analysis",
                parameters: [
                    { name: "claim", type: "string", required: true, description: "The argument claim" },
                    { name: "premises", type: "array", required: true, description: "Supporting premises" },
                    { name: "conclusion", type: "string", required: true, description: "Derived conclusion" },
                    { name: "argumentType", type: "enum", required: true, description: "Type", constraints: ["thesis", "antithesis", "synthesis", "objection", "rebuttal"] },
                    { name: "confidence", type: "float", required: true, description: "Confidence 0-1" }
                ],
                keywords: ["argument", "claim", "premise", "logic", "debate"]
            }
        }
    }
};

// ============================================================================
// DERIVED METRICS & HELPERS
// ============================================================================

const UCL_TOOLS_METADATA = {
    totalTools: 71,
    totalDomains: 9,
    overheadReduction: 78,  // percentage
    characterReduction: 47, // percentage
    categories: {
        original: { label: "Core Domains", count: 6 },
        mcp_server: { label: "MCP Server Domains", count: 3 }
    }
};

// Calculate tool count for a specific domain
function getDomainToolCount(domainKey) {
    const domain = UCL_TOOL_DOMAINS[domainKey];
    return domain ? Object.keys(domain.tools).length : 0;
}

// Get all domain keys
function getAllDomainKeys() {
    return Object.keys(UCL_TOOL_DOMAINS);
}

// Get total tools across selected domains
function getSelectedToolCount(selectedDomains) {
    return selectedDomains.reduce((sum, key) => sum + getDomainToolCount(key), 0);
}

// Export for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { UCL_TOOL_DOMAINS, UCL_TOOLS_METADATA, getDomainToolCount, getAllDomainKeys, getSelectedToolCount };
}
