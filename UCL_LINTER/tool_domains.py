#!/usr/bin/env python3
"""
UCL Tool Domains
================

Pre-built tool domain templates for common tool categories.

Each ToolDomain provides a curated set of ToolDefinitions bundled
into a ToolGroup for optimal O_s (keyword-conditional activation
under a shared ^^CONDITION^^ block).

Usage:
    from tool_domains import FileSystemDomain, WebDomain, PlaywrightDomain

    builder = (UCLPromptBuilder()
        .add_tool_domain(FileSystemDomain())   # 7 file tools
        .add_tool_domain(WebDomain())          # 5 web tools
        .add_tool_domain(PlaywrightDomain())   # 20 browser tools
        .add_tool_domain(ChromaDomain())       # 10 vector DB tools
        .build()
    )

    # Or by name from registry:
    from tool_domains import ToolDomainRegistry
    builder.add_tool_domain(ToolDomainRegistry.get("playwright"))

    # Custom MCP server:
    from tool_domains import MCPServerTemplate
    my_server = MCPServerTemplate(
        server_name="my_api",
        tools=[...],
    )
    builder.add_tool_domain(my_server)

Original Domains (31 tools):
    - FileSystemDomain  (7 tools): file read/write/list/delete/move/copy/info
    - WebDomain         (5 tools): http_get/post, search, fetch, download
    - DatabaseDomain    (6 tools): SQL query/insert/update/delete, tables
    - APIDomain         (4 tools): REST, GraphQL, webhooks, auth
    - MCPDomain         (5 tools): MCP protocol tools/resources/prompts
    - ShellDomain       (4 tools): commands, scripts, environment

MCP Server Domains (40 tools):
    - PlaywrightDomain  (20 tools): browser automation — navigate, click, type, screenshot
    - ChromaDomain      (10 tools): vector DB — collections, documents, similarity search
    - ClearThoughtDomain(10 tools): reasoning — thinking, analysis, decisions, patterns

Plus:
    - MCPServerTemplate: user-extensible base for custom MCP servers

Total: 71+ pre-built tools across 9 domains.
"""

from __future__ import annotations
from typing import List, Dict, Type, Optional
from dataclasses import dataclass, field

try:
    from .core import (
        ToolParameter, ToolExample, ToolDefinition, ToolGroup,
        UCLToolConstants
    )
except ImportError:
    from core import (
        ToolParameter, ToolExample, ToolDefinition, ToolGroup,
        UCLToolConstants
    )


# =============================================================================
# BASE CLASS
# =============================================================================

class ToolDomain:
    """
    Base class for pre-built tool domain templates.

    Subclasses define:
    - name: Unique identifier
    - description: What this domain provides
    - category: Grouping category
    - keywords: Shared activation keywords for the tool group

    And implement:
    - get_tools() → List[ToolDefinition]
    """

    name: str = ""
    description: str = ""
    category: str = ""
    keywords: List[str] = []

    def get_tools(self) -> List[ToolDefinition]:
        """Return the list of tool definitions for this domain."""
        raise NotImplementedError

    def create_tool_group(self) -> ToolGroup:
        """Create a ToolGroup bundling all domain tools under shared keywords."""
        return ToolGroup(
            name=self.name,
            keywords=self.keywords,
            tools=self.get_tools(),
            description=self.description,
        )

    def get_tool_count(self) -> int:
        """Return the number of tools in this domain."""
        return len(self.get_tools())

    def get_individual_Os(self) -> float:
        """Calculate O_s if all tools were standalone (no grouping)."""
        return sum(t.calculate_Os() for t in self.get_tools())

    def get_grouped_Os(self) -> float:
        """Calculate O_s with tools grouped under shared condition."""
        return self.create_tool_group().calculate_Os()

    def get_Os_savings(self) -> float:
        """Return O_s savings percentage from grouping."""
        individual = self.get_individual_Os()
        grouped = self.get_grouped_Os()
        if individual == 0:
            return 0.0
        return ((individual - grouped) / individual) * 100


# =============================================================================
# REGISTRY
# =============================================================================

class ToolDomainRegistry:
    """Registry for tool domains with string-based lookup."""

    _registry: Dict[str, Type[ToolDomain]] = {}

    @classmethod
    def register(cls, name: str = None):
        """
        Decorator to register a ToolDomain class.

        Usage:
            @ToolDomainRegistry.register()
            class MyDomain(ToolDomain):
                ...

            @ToolDomainRegistry.register("custom_name")
            class MyDomain(ToolDomain):
                ...
        """
        def decorator(domain_cls):
            reg_name = name or domain_cls.__name__.lower().replace('domain', '')
            cls._registry[reg_name] = domain_cls
            return domain_cls
        return decorator

    @classmethod
    def get(cls, name: str) -> ToolDomain:
        """Get a tool domain instance by name."""
        if name not in cls._registry:
            available = ", ".join(sorted(cls._registry.keys()))
            raise ValueError(
                f"Unknown tool domain: '{name}'. Available: {available}"
            )
        return cls._registry[name]()

    @classmethod
    def list_domains(cls) -> List[str]:
        """List all registered domain names."""
        return sorted(cls._registry.keys())

    @classmethod
    def get_all(cls) -> List[ToolDomain]:
        """Get instances of all registered domains."""
        return [domain_cls() for domain_cls in cls._registry.values()]

    @classmethod
    def get_total_tools(cls) -> int:
        """Get total number of tools across all domains."""
        return sum(d.get_tool_count() for d in cls.get_all())


# =============================================================================
# FILE SYSTEM DOMAIN
# =============================================================================

@ToolDomainRegistry.register()
class FileSystemDomain(ToolDomain):
    """
    File system operations: read, write, list, delete, move, copy, info.

    7 tools under shared condition: file, path, directory, read, write, delete
    """

    name = "filesystem"
    description = "Local file system operations"
    category = "filesystem"
    keywords = ["file", "path", "directory", "read", "write", "delete", "move", "copy"]

    def get_tools(self) -> List[ToolDefinition]:
        return [
            ToolDefinition(
                name="read_file",
                description="Read file content from disk",
                parameters=[
                    ToolParameter("path", "string", required=True,
                                 description="Path to the file"),
                    ToolParameter("encoding", "string", required=False,
                                 description="File encoding", default="utf-8"),
                ],
                keywords=["read", "file", "open", "load"],
                examples=[
                    ToolExample("read_file('/etc/hosts')", "127.0.0.1 localhost"),
                ],
                return_type="string",
            ),
            ToolDefinition(
                name="write_file",
                description="Write content to a file",
                parameters=[
                    ToolParameter("path", "string", required=True,
                                 description="Path to the file"),
                    ToolParameter("content", "string", required=True,
                                 description="Content to write"),
                ],
                keywords=["write", "save", "create"],
                return_type="boolean",
            ),
            ToolDefinition(
                name="list_dir",
                description="List directory contents",
                parameters=[
                    ToolParameter("path", "string", required=True,
                                 description="Directory path"),
                ],
                keywords=["list", "directory", "folder", "ls"],
                return_type="array",
            ),
            ToolDefinition(
                name="delete_file",
                description="Delete a file from disk",
                parameters=[
                    ToolParameter("path", "string", required=True,
                                 description="Path to the file"),
                ],
                keywords=["delete", "remove"],
                return_type="boolean",
            ),
            ToolDefinition(
                name="move_file",
                description="Move or rename a file",
                parameters=[
                    ToolParameter("source", "string", required=True,
                                 description="Source path"),
                    ToolParameter("dest", "string", required=True,
                                 description="Destination path"),
                ],
                keywords=["move", "rename"],
                return_type="boolean",
            ),
            ToolDefinition(
                name="copy_file",
                description="Copy a file to a new location",
                parameters=[
                    ToolParameter("source", "string", required=True,
                                 description="Source path"),
                    ToolParameter("dest", "string", required=True,
                                 description="Destination path"),
                ],
                keywords=["copy", "duplicate"],
                return_type="boolean",
            ),
            ToolDefinition(
                name="file_info",
                description="Get file metadata (size, modified time, permissions)",
                parameters=[
                    ToolParameter("path", "string", required=True,
                                 description="Path to the file"),
                ],
                keywords=["info", "stat", "metadata", "size"],
                return_type="object",
            ),
        ]


# =============================================================================
# WEB DOMAIN
# =============================================================================

@ToolDomainRegistry.register()
class WebDomain(ToolDomain):
    """
    Web operations: HTTP requests, search, fetch, download.

    5 tools under shared condition: http, web, url, request, fetch, search
    """

    name = "web"
    description = "Web operations: HTTP requests, search, and downloads"
    category = "web"
    keywords = ["http", "web", "url", "request", "fetch", "search", "download"]

    def get_tools(self) -> List[ToolDefinition]:
        return [
            ToolDefinition(
                name="http_get",
                description="Send an HTTP GET request",
                parameters=[
                    ToolParameter("url", "string", required=True,
                                 description="Request URL"),
                    ToolParameter("headers", "object", required=False,
                                 description="HTTP headers"),
                ],
                keywords=["get", "fetch", "request"],
                examples=[
                    ToolExample("http_get('https://api.example.com/data')",
                               '{"status": 200, "body": ...}'),
                ],
                return_type="object",
            ),
            ToolDefinition(
                name="http_post",
                description="Send an HTTP POST request",
                parameters=[
                    ToolParameter("url", "string", required=True,
                                 description="Request URL"),
                    ToolParameter("body", "string", required=True,
                                 description="Request body"),
                    ToolParameter("headers", "object", required=False,
                                 description="HTTP headers"),
                ],
                keywords=["post", "send", "submit"],
                return_type="object",
            ),
            ToolDefinition(
                name="web_search",
                description="Search the web for information",
                parameters=[
                    ToolParameter("query", "string", required=True,
                                 description="Search query"),
                    ToolParameter("max_results", "integer", required=False,
                                 description="Maximum results", default="10"),
                ],
                keywords=["search", "find", "query"],
                examples=[
                    ToolExample("web_search('UCL paper')",
                               "arXiv:2601.00880 - Universal Conditional Logic..."),
                ],
                return_type="array",
            ),
            ToolDefinition(
                name="fetch_url",
                description="Fetch content from a URL",
                parameters=[
                    ToolParameter("url", "string", required=True,
                                 description="URL to fetch"),
                ],
                keywords=["fetch", "download", "retrieve"],
                return_type="string",
            ),
            ToolDefinition(
                name="download_file",
                description="Download a file from a URL to local disk",
                parameters=[
                    ToolParameter("url", "string", required=True,
                                 description="URL of the file"),
                    ToolParameter("dest", "string", required=False,
                                 description="Local destination path"),
                ],
                keywords=["download", "save"],
                return_type="string",
            ),
        ]


# =============================================================================
# DATABASE DOMAIN
# =============================================================================

@ToolDomainRegistry.register()
class DatabaseDomain(ToolDomain):
    """
    Database operations: SQL queries, row manipulation, table management.

    6 tools under shared condition: database, sql, table, query, row
    """

    name = "database"
    description = "Database operations: SQL queries and table management"
    category = "database"
    keywords = ["database", "sql", "table", "query", "row", "db"]

    def get_tools(self) -> List[ToolDefinition]:
        return [
            ToolDefinition(
                name="query_sql",
                description="Execute a SQL query and return results",
                parameters=[
                    ToolParameter("query", "string", required=True,
                                 description="SQL query string"),
                    ToolParameter("params", "array", required=False,
                                 description="Query parameters"),
                ],
                keywords=["query", "select", "sql", "search"],
                examples=[
                    ToolExample("query_sql('SELECT * FROM users WHERE id = ?', [1])",
                               "[{id: 1, name: 'Alice'}]"),
                ],
                return_type="array",
            ),
            ToolDefinition(
                name="insert_row",
                description="Insert a row into a table",
                parameters=[
                    ToolParameter("table", "string", required=True,
                                 description="Table name"),
                    ToolParameter("data", "object", required=True,
                                 description="Column-value pairs"),
                ],
                keywords=["insert", "create", "add"],
                return_type="object",
            ),
            ToolDefinition(
                name="update_row",
                description="Update rows in a table matching a condition",
                parameters=[
                    ToolParameter("table", "string", required=True,
                                 description="Table name"),
                    ToolParameter("data", "object", required=True,
                                 description="Column-value pairs to update"),
                    ToolParameter("condition", "string", required=True,
                                 description="WHERE condition"),
                ],
                keywords=["update", "modify"],
                return_type="integer",
            ),
            ToolDefinition(
                name="delete_row",
                description="Delete rows from a table matching a condition",
                parameters=[
                    ToolParameter("table", "string", required=True,
                                 description="Table name"),
                    ToolParameter("condition", "string", required=True,
                                 description="WHERE condition"),
                ],
                keywords=["delete", "remove"],
                return_type="integer",
            ),
            ToolDefinition(
                name="create_table",
                description="Create a new table with a schema",
                parameters=[
                    ToolParameter("name", "string", required=True,
                                 description="Table name"),
                    ToolParameter("schema", "object", required=True,
                                 description="Column definitions"),
                ],
                keywords=["create", "table", "schema"],
                return_type="boolean",
            ),
            ToolDefinition(
                name="list_tables",
                description="List all tables in the database",
                parameters=[],
                keywords=["list", "tables", "schema"],
                return_type="array",
            ),
        ]


# =============================================================================
# API DOMAIN
# =============================================================================

@ToolDomainRegistry.register()
class APIDomain(ToolDomain):
    """
    API operations: REST calls, GraphQL, webhooks, authentication.

    4 tools under shared condition: api, rest, graphql, webhook, endpoint
    """

    name = "api"
    description = "API operations: REST, GraphQL, webhooks, and authentication"
    category = "api"
    keywords = ["api", "rest", "graphql", "webhook", "endpoint", "request"]

    def get_tools(self) -> List[ToolDefinition]:
        return [
            ToolDefinition(
                name="rest_call",
                description="Make a REST API call",
                parameters=[
                    ToolParameter("method", "enum", required=True,
                                 description="HTTP method",
                                 constraints={"values": ["GET", "POST", "PUT", "DELETE", "PATCH"]}),
                    ToolParameter("url", "string", required=True,
                                 description="API endpoint URL"),
                    ToolParameter("body", "object", required=False,
                                 description="Request body"),
                    ToolParameter("headers", "object", required=False,
                                 description="HTTP headers"),
                ],
                keywords=["api", "rest", "call", "endpoint"],
                examples=[
                    ToolExample("rest_call('GET', 'https://api.example.com/users')",
                               '{"users": [...]}'),
                ],
                return_type="object",
            ),
            ToolDefinition(
                name="graphql_query",
                description="Execute a GraphQL query",
                parameters=[
                    ToolParameter("query", "string", required=True,
                                 description="GraphQL query string"),
                    ToolParameter("variables", "object", required=False,
                                 description="Query variables"),
                ],
                keywords=["graphql", "query", "mutation"],
                return_type="object",
            ),
            ToolDefinition(
                name="webhook_send",
                description="Send a webhook notification",
                parameters=[
                    ToolParameter("url", "string", required=True,
                                 description="Webhook URL"),
                    ToolParameter("payload", "object", required=True,
                                 description="Webhook payload"),
                ],
                keywords=["webhook", "notify", "callback"],
                return_type="boolean",
            ),
            ToolDefinition(
                name="auth_token",
                description="Obtain an authentication token",
                parameters=[
                    ToolParameter("provider", "string", required=True,
                                 description="Auth provider name"),
                    ToolParameter("scopes", "array", required=False,
                                 description="Requested scopes"),
                ],
                keywords=["auth", "token", "login", "credential"],
                return_type="string",
            ),
        ]


# =============================================================================
# MCP DOMAIN
# =============================================================================

@ToolDomainRegistry.register()
class MCPDomain(ToolDomain):
    """
    MCP (Model Context Protocol) operations: tool calls, resource access, prompts.

    5 tools mapping to MCP protocol operations:
    - mcp_call → tools/call
    - mcp_list_tools → tools/list
    - mcp_list_resources → resources/list
    - mcp_read_resource → resources/read
    - mcp_list_prompts → prompts/list

    Under shared condition: mcp, server, protocol
    """

    name = "mcp"
    description = "MCP (Model Context Protocol) operations"
    category = "mcp"
    keywords = ["mcp", "server", "protocol", "tool", "resource"]

    def get_tools(self) -> List[ToolDefinition]:
        return [
            ToolDefinition(
                name="mcp_call",
                description="Call a tool on an MCP server (maps to tools/call)",
                parameters=[
                    ToolParameter("server", "string", required=True,
                                 description="MCP server name"),
                    ToolParameter("tool", "string", required=True,
                                 description="Tool name to invoke"),
                    ToolParameter("arguments", "object", required=False,
                                 description="Tool arguments"),
                ],
                keywords=["mcp", "call", "invoke", "tool"],
                examples=[
                    ToolExample("mcp_call('filesystem', 'read_file', {'path': '/tmp/x'})",
                               "file contents..."),
                ],
                return_type="object",
            ),
            ToolDefinition(
                name="mcp_list_tools",
                description="List available tools on an MCP server (maps to tools/list)",
                parameters=[
                    ToolParameter("server", "string", required=False,
                                 description="MCP server name (omit for all)"),
                ],
                keywords=["mcp", "tools", "list", "discover"],
                return_type="array",
            ),
            ToolDefinition(
                name="mcp_list_resources",
                description="List available resources on an MCP server (maps to resources/list)",
                parameters=[
                    ToolParameter("server", "string", required=False,
                                 description="MCP server name (omit for all)"),
                ],
                keywords=["mcp", "resources", "list"],
                return_type="array",
            ),
            ToolDefinition(
                name="mcp_read_resource",
                description="Read a resource from an MCP server (maps to resources/read)",
                parameters=[
                    ToolParameter("server", "string", required=True,
                                 description="MCP server name"),
                    ToolParameter("uri", "string", required=True,
                                 description="Resource URI"),
                ],
                keywords=["mcp", "resource", "read", "fetch"],
                return_type="object",
            ),
            ToolDefinition(
                name="mcp_list_prompts",
                description="List available prompts on an MCP server (maps to prompts/list)",
                parameters=[
                    ToolParameter("server", "string", required=False,
                                 description="MCP server name (omit for all)"),
                ],
                keywords=["mcp", "prompts", "list"],
                return_type="array",
            ),
        ]


# =============================================================================
# SHELL DOMAIN
# =============================================================================

@ToolDomainRegistry.register()
class ShellDomain(ToolDomain):
    """
    Shell operations: commands, scripts, environment variables.

    4 tools under shared condition: shell, command, execute, script, env
    """

    name = "shell"
    description = "Shell operations: commands, scripts, and environment"
    category = "shell"
    keywords = ["shell", "command", "execute", "script", "env", "terminal"]

    def get_tools(self) -> List[ToolDefinition]:
        return [
            ToolDefinition(
                name="run_command",
                description="Execute a shell command",
                parameters=[
                    ToolParameter("command", "string", required=True,
                                 description="Command to execute"),
                    ToolParameter("cwd", "string", required=False,
                                 description="Working directory"),
                    ToolParameter("timeout", "integer", required=False,
                                 description="Timeout in seconds", default="30"),
                ],
                keywords=["command", "execute", "run", "shell"],
                examples=[
                    ToolExample("run_command('ls -la /tmp')", "total 8\ndrwxrwxrwt..."),
                ],
                return_type="object",
            ),
            ToolDefinition(
                name="run_script",
                description="Execute a script file",
                parameters=[
                    ToolParameter("path", "string", required=True,
                                 description="Path to the script"),
                    ToolParameter("args", "array", required=False,
                                 description="Script arguments"),
                ],
                keywords=["script", "execute", "run"],
                return_type="object",
            ),
            ToolDefinition(
                name="get_env",
                description="Get an environment variable value",
                parameters=[
                    ToolParameter("name", "string", required=True,
                                 description="Variable name"),
                ],
                keywords=["env", "environment", "variable", "get"],
                return_type="string",
            ),
            ToolDefinition(
                name="set_env",
                description="Set an environment variable",
                parameters=[
                    ToolParameter("name", "string", required=True,
                                 description="Variable name"),
                    ToolParameter("value", "string", required=True,
                                 description="Variable value"),
                ],
                keywords=["env", "set", "environment"],
                return_type="boolean",
            ),
        ]


# =============================================================================
# MCP SERVER DOMAINS
# =============================================================================

@ToolDomainRegistry.register()
class PlaywrightDomain(ToolDomain):
    """
    Browser automation via Playwright MCP server.

    20 tools for web interaction: navigation, clicking, typing,
    screenshots, form filling, network monitoring, and more.

    Maps to the @playwright/mcp server tools.
    """
    name = "playwright"
    description = "Browser automation via Playwright MCP — navigate, click, type, screenshot, fill forms"
    category = "mcp_server"
    keywords = ["browser", "playwright", "web", "page", "click", "navigate", "screenshot"]

    def get_tools(self) -> List[ToolDefinition]:
        return [
            ToolDefinition(
                name="browser_navigate",
                description="Navigate to a URL in the browser",
                parameters=[
                    ToolParameter("url", "string", required=True,
                                 description="URL to navigate to"),
                ],
                keywords=["navigate", "url", "goto", "page"],
                examples=[
                    ToolExample("Go to Google", 'browser_navigate(url="https://google.com")', "Page loaded"),
                ],
            ),
            ToolDefinition(
                name="browser_click",
                description="Click on an element on the page",
                parameters=[
                    ToolParameter("target", "string", required=True,
                                 description="Element selector or reference"),
                    ToolParameter("button", "enum", required=False,
                                 description="Mouse button: left, right, middle",
                                 constraints={"values": ["left", "right", "middle"]}),
                    ToolParameter("doubleClick", "boolean", required=False,
                                 description="Perform double click"),
                ],
                keywords=["click", "button", "element", "tap"],
            ),
            ToolDefinition(
                name="browser_type",
                description="Type text into an editable element",
                parameters=[
                    ToolParameter("target", "string", required=True,
                                 description="Element to type into"),
                    ToolParameter("text", "string", required=True,
                                 description="Text to type"),
                    ToolParameter("slowly", "boolean", required=False,
                                 description="Type one character at a time"),
                    ToolParameter("submit", "boolean", required=False,
                                 description="Press Enter after typing"),
                ],
                keywords=["type", "input", "text", "enter", "keyboard"],
            ),
            ToolDefinition(
                name="browser_snapshot",
                description="Capture accessibility snapshot of the current page",
                parameters=[
                    ToolParameter("target", "string", required=False,
                                 description="Specific element to snapshot"),
                    ToolParameter("filename", "string", required=False,
                                 description="Save snapshot to file"),
                ],
                keywords=["snapshot", "accessibility", "capture", "page"],
            ),
            ToolDefinition(
                name="browser_take_screenshot",
                description="Take a screenshot of the page or element",
                parameters=[
                    ToolParameter("type", "enum", required=True,
                                 description="Image format: png or jpeg",
                                 constraints={"values": ["png", "jpeg"]}),
                    ToolParameter("filename", "string", required=False,
                                 description="Save screenshot to file"),
                    ToolParameter("fullPage", "boolean", required=False,
                                 description="Screenshot full scrollable page"),
                    ToolParameter("target", "string", required=False,
                                 description="Specific element to screenshot"),
                ],
                keywords=["screenshot", "image", "capture", "photo"],
            ),
            ToolDefinition(
                name="browser_hover",
                description="Hover over an element on the page",
                parameters=[
                    ToolParameter("target", "string", required=True,
                                 description="Element to hover over"),
                ],
                keywords=["hover", "mouseover", "element"],
            ),
            ToolDefinition(
                name="browser_press_key",
                description="Press a keyboard key",
                parameters=[
                    ToolParameter("key", "string", required=True,
                                 description="Key name (e.g. ArrowLeft, Enter, a)"),
                ],
                keywords=["key", "keyboard", "press", "shortcut"],
            ),
            ToolDefinition(
                name="browser_fill_form",
                description="Fill multiple form fields at once",
                parameters=[
                    ToolParameter("fields", "array", required=True,
                                 description="List of field objects with target, name, type, value"),
                ],
                keywords=["form", "fill", "input", "field"],
            ),
            ToolDefinition(
                name="browser_select_option",
                description="Select an option in a dropdown",
                parameters=[
                    ToolParameter("target", "string", required=True,
                                 description="Select element"),
                    ToolParameter("values", "array", required=True,
                                 description="Values to select"),
                ],
                keywords=["select", "dropdown", "option", "choose"],
            ),
            ToolDefinition(
                name="browser_drag",
                description="Perform drag and drop between elements",
                parameters=[
                    ToolParameter("startTarget", "string", required=True,
                                 description="Source element"),
                    ToolParameter("endTarget", "string", required=True,
                                 description="Target element"),
                ],
                keywords=["drag", "drop", "move", "element"],
            ),
            ToolDefinition(
                name="browser_file_upload",
                description="Upload one or multiple files",
                parameters=[
                    ToolParameter("paths", "array", required=True,
                                 description="Absolute file paths to upload"),
                ],
                keywords=["upload", "file", "attach"],
            ),
            ToolDefinition(
                name="browser_handle_dialog",
                description="Handle an alert, confirm, or prompt dialog",
                parameters=[
                    ToolParameter("accept", "boolean", required=True,
                                 description="Accept or dismiss the dialog"),
                    ToolParameter("promptText", "string", required=False,
                                 description="Text for prompt dialogs"),
                ],
                keywords=["dialog", "alert", "confirm", "prompt"],
            ),
            ToolDefinition(
                name="browser_tabs",
                description="List, create, close, or select browser tabs",
                parameters=[
                    ToolParameter("action", "enum", required=True,
                                 description="Operation: list, new, close, select",
                                 constraints={"values": ["list", "new", "close", "select"]}),
                    ToolParameter("index", "integer", required=False,
                                 description="Tab index for close/select"),
                    ToolParameter("url", "string", required=False,
                                 description="URL for new tab"),
                ],
                keywords=["tab", "window", "switch"],
            ),
            ToolDefinition(
                name="browser_navigate_back",
                description="Go back to the previous page",
                parameters=[],
                keywords=["back", "history", "previous"],
            ),
            ToolDefinition(
                name="browser_wait_for",
                description="Wait for text to appear or a time duration",
                parameters=[
                    ToolParameter("text", "string", required=False,
                                 description="Text to wait for"),
                    ToolParameter("textGone", "string", required=False,
                                 description="Text to wait for disappearance"),
                    ToolParameter("time", "float", required=False,
                                 description="Seconds to wait"),
                ],
                keywords=["wait", "delay", "timeout", "pause"],
            ),
            ToolDefinition(
                name="browser_evaluate",
                description="Evaluate JavaScript expression on page",
                parameters=[
                    ToolParameter("function", "string", required=True,
                                 description="JavaScript function to evaluate"),
                    ToolParameter("target", "string", required=False,
                                 description="Element to evaluate on"),
                    ToolParameter("filename", "string", required=False,
                                 description="Save result to file"),
                ],
                keywords=["evaluate", "javascript", "js", "script", "execute"],
            ),
            ToolDefinition(
                name="browser_console_messages",
                description="Return console messages from the page",
                parameters=[
                    ToolParameter("level", "enum", required=False,
                                 description="Log level: info, warning, error, debug",
                                 constraints={"values": ["info", "warning", "error", "debug"]}),
                    ToolParameter("all", "boolean", required=False,
                                 description="Return all messages since session start"),
                ],
                keywords=["console", "log", "debug", "error", "message"],
            ),
            ToolDefinition(
                name="browser_network_requests",
                description="Return list of network requests since page load",
                parameters=[
                    ToolParameter("static", "boolean", required=False,
                                 description="Include static resources"),
                    ToolParameter("filter", "string", required=False,
                                 description="URL regexp filter"),
                ],
                keywords=["network", "request", "http", "fetch", "api"],
            ),
            ToolDefinition(
                name="browser_close",
                description="Close the browser page",
                parameters=[],
                keywords=["close", "quit", "exit", "browser"],
            ),
            ToolDefinition(
                name="browser_resize",
                description="Resize the browser window",
                parameters=[
                    ToolParameter("width", "integer", required=True,
                                 description="Window width in pixels"),
                    ToolParameter("height", "integer", required=True,
                                 description="Window height in pixels"),
                ],
                keywords=["resize", "window", "viewport", "size"],
            ),
        ]


@ToolDomainRegistry.register()
class ChromaDomain(ToolDomain):
    """
    Chroma vector database operations via MCP server.

    10 tools for vector storage: create collections, add/get/query/update
    documents, manage metadata, and perform similarity search.
    """
    name = "chroma"
    description = "Chroma vector DB — collections, documents, similarity search, embeddings"
    category = "mcp_server"
    keywords = ["chroma", "vector", "embedding", "collection", "document", "similarity"]

    def get_tools(self) -> List[ToolDefinition]:
        return [
            ToolDefinition(
                name="chroma_create_collection",
                description="Create a new Chroma collection",
                parameters=[
                    ToolParameter("collection_name", "string", required=True,
                                 description="Name of the collection"),
                    ToolParameter("embedding_function_name", "string", required=False,
                                 description="Embedding function: default, cohere, openai, jina"),
                ],
                keywords=["create", "collection", "new", "vector"],
            ),
            ToolDefinition(
                name="chroma_delete_collection",
                description="Delete a Chroma collection",
                parameters=[
                    ToolParameter("collection_name", "string", required=True,
                                 description="Name of the collection to delete"),
                ],
                keywords=["delete", "collection", "remove", "drop"],
            ),
            ToolDefinition(
                name="chroma_list_collections",
                description="List all collection names in the database",
                parameters=[
                    ToolParameter("limit", "integer", required=False,
                                 description="Maximum collections to return"),
                    ToolParameter("offset", "integer", required=False,
                                 description="Offset for pagination"),
                ],
                keywords=["list", "collections", "all", "show"],
            ),
            ToolDefinition(
                name="chroma_add_documents",
                description="Add documents to a Chroma collection",
                parameters=[
                    ToolParameter("collection_name", "string", required=True,
                                 description="Collection to add to"),
                    ToolParameter("documents", "array", required=True,
                                 description="List of text documents"),
                    ToolParameter("ids", "array", required=True,
                                 description="List of document IDs"),
                    ToolParameter("metadatas", "array", required=False,
                                 description="List of metadata dicts"),
                ],
                keywords=["add", "document", "insert", "store", "ingest"],
            ),
            ToolDefinition(
                name="chroma_get_documents",
                description="Get documents from a collection by ID or filter",
                parameters=[
                    ToolParameter("collection_name", "string", required=True,
                                 description="Collection to query"),
                    ToolParameter("ids", "array", required=False,
                                 description="Document IDs to retrieve"),
                    ToolParameter("where", "object", required=False,
                                 description="Metadata filter"),
                    ToolParameter("limit", "integer", required=False,
                                 description="Maximum documents"),
                ],
                keywords=["get", "document", "retrieve", "fetch", "find"],
            ),
            ToolDefinition(
                name="chroma_query_documents",
                description="Query documents by similarity search",
                parameters=[
                    ToolParameter("collection_name", "string", required=True,
                                 description="Collection to search"),
                    ToolParameter("query_texts", "array", required=True,
                                 description="Query strings for similarity search"),
                    ToolParameter("n_results", "integer", required=False,
                                 description="Number of results per query"),
                    ToolParameter("where", "object", required=False,
                                 description="Metadata filter"),
                ],
                keywords=["query", "search", "similarity", "find", "vector"],
            ),
            ToolDefinition(
                name="chroma_update_documents",
                description="Update documents in a collection",
                parameters=[
                    ToolParameter("collection_name", "string", required=True,
                                 description="Collection to update"),
                    ToolParameter("ids", "array", required=True,
                                 description="Document IDs to update"),
                    ToolParameter("documents", "array", required=False,
                                 description="New document text"),
                    ToolParameter("metadatas", "array", required=False,
                                 description="New metadata"),
                ],
                keywords=["update", "modify", "change", "edit"],
            ),
            ToolDefinition(
                name="chroma_delete_documents",
                description="Delete documents from a collection",
                parameters=[
                    ToolParameter("collection_name", "string", required=True,
                                 description="Collection to delete from"),
                    ToolParameter("ids", "array", required=True,
                                 description="Document IDs to delete"),
                ],
                keywords=["delete", "remove", "document"],
            ),
            ToolDefinition(
                name="chroma_peek_collection",
                description="Peek at documents in a collection",
                parameters=[
                    ToolParameter("collection_name", "string", required=True,
                                 description="Collection to peek at"),
                    ToolParameter("limit", "integer", required=False,
                                 description="Number of documents to peek"),
                ],
                keywords=["peek", "preview", "sample", "view"],
            ),
            ToolDefinition(
                name="chroma_get_collection_info",
                description="Get information about a collection",
                parameters=[
                    ToolParameter("collection_name", "string", required=True,
                                 description="Collection name"),
                ],
                keywords=["info", "collection", "metadata", "describe"],
            ),
        ]


@ToolDomainRegistry.register("clear_thought")
class ClearThoughtDomain(ToolDomain):
    """
    Reasoning and analysis tools via Clear Thought MCP server.

    10 tools for structured thinking: sequential thinking, collaborative
    reasoning, decision frameworks, design patterns, mental models, and more.
    """
    name = "clear_thought"
    description = "Structured reasoning — thinking, analysis, decisions, patterns, debugging"
    category = "mcp_server"
    keywords = ["think", "reason", "analyze", "decide", "debug", "pattern", "mental_model"]

    def get_tools(self) -> List[ToolDefinition]:
        return [
            ToolDefinition(
                name="sequential_thinking",
                description="Dynamic problem-solving through sequential thought steps",
                parameters=[
                    ToolParameter("thought", "string", required=True,
                                 description="Current thinking step"),
                    ToolParameter("thoughtNumber", "integer", required=True,
                                 description="Current thought number"),
                    ToolParameter("totalThoughts", "integer", required=True,
                                 description="Estimated total thoughts needed"),
                    ToolParameter("nextThoughtNeeded", "boolean", required=True,
                                 description="Whether another step is needed"),
                ],
                keywords=["think", "reason", "step", "analyze", "problem"],
                examples=[
                    ToolExample(
                        "Start analysis",
                        'sequential_thinking(thought="Let me break down...", thoughtNumber=1, totalThoughts=3, nextThoughtNeeded=true)',
                        "Thinking step recorded"
                    ),
                ],
            ),
            ToolDefinition(
                name="collaborative_reasoning",
                description="Simulate expert collaboration with diverse perspectives",
                parameters=[
                    ToolParameter("topic", "string", required=True,
                                 description="Topic for collaboration"),
                    ToolParameter("personas", "array", required=True,
                                 description="List of expert personas"),
                    ToolParameter("stage", "enum", required=True,
                                 description="Stage: problem-definition, ideation, critique, integration, decision, reflection",
                                 constraints={"values": ["problem-definition", "ideation", "critique", "integration", "decision", "reflection"]}),
                ],
                keywords=["collaborate", "perspective", "expert", "team"],
            ),
            ToolDefinition(
                name="debugging_approach",
                description="Apply systematic debugging approaches to issues",
                parameters=[
                    ToolParameter("approachName", "enum", required=True,
                                 description="Method: binary_search, reverse_engineering, divide_conquer, backtracking, cause_elimination, program_slicing",
                                 constraints={"values": ["binary_search", "reverse_engineering", "divide_conquer", "backtracking", "cause_elimination", "program_slicing"]}),
                    ToolParameter("issue", "string", required=True,
                                 description="Description of the issue"),
                    ToolParameter("steps", "array", required=False,
                                 description="Debugging steps"),
                ],
                keywords=["debug", "fix", "error", "issue", "troubleshoot"],
            ),
            ToolDefinition(
                name="decision_framework",
                description="Structured decision analysis and rational choice",
                parameters=[
                    ToolParameter("decisionStatement", "string", required=True,
                                 description="What decision needs to be made"),
                    ToolParameter("options", "array", required=True,
                                 description="Available options with descriptions"),
                    ToolParameter("analysisType", "enum", required=True,
                                 description="Type: pros-cons, weighted-criteria, decision-tree, expected-value, scenario-analysis",
                                 constraints={"values": ["pros-cons", "weighted-criteria", "decision-tree", "expected-value", "scenario-analysis"]}),
                    ToolParameter("stage", "enum", required=True,
                                 description="Stage: problem-definition, options-generation, criteria-definition, evaluation, sensitivity-analysis, decision",
                                 constraints={"values": ["problem-definition", "options-generation", "criteria-definition", "evaluation", "sensitivity-analysis", "decision"]}),
                ],
                keywords=["decide", "choice", "option", "evaluate", "compare"],
            ),
            ToolDefinition(
                name="design_pattern",
                description="Apply software design patterns to architecture",
                parameters=[
                    ToolParameter("patternName", "enum", required=True,
                                 description="Pattern: modular_architecture, api_integration, state_management, async_processing, scalability, security, agentic_design",
                                 constraints={"values": ["modular_architecture", "api_integration", "state_management", "async_processing", "scalability", "security", "agentic_design"]}),
                    ToolParameter("context", "string", required=True,
                                 description="Problem context"),
                    ToolParameter("implementation", "array", required=False,
                                 description="Implementation steps"),
                ],
                keywords=["pattern", "design", "architecture", "structure"],
            ),
            ToolDefinition(
                name="mental_model",
                description="Apply structured mental models to problem-solving",
                parameters=[
                    ToolParameter("modelName", "enum", required=True,
                                 description="Model: first_principles, opportunity_cost, error_propagation, rubber_duck, pareto_principle, occams_razor",
                                 constraints={"values": ["first_principles", "opportunity_cost", "error_propagation", "rubber_duck", "pareto_principle", "occams_razor"]}),
                    ToolParameter("problem", "string", required=True,
                                 description="Problem to analyze"),
                    ToolParameter("steps", "array", required=False,
                                 description="Analysis steps"),
                ],
                keywords=["model", "framework", "think", "perspective"],
            ),
            ToolDefinition(
                name="metacognitive_monitoring",
                description="Systematic self-monitoring of knowledge and reasoning quality",
                parameters=[
                    ToolParameter("task", "string", required=True,
                                 description="Task being monitored"),
                    ToolParameter("stage", "enum", required=True,
                                 description="Stage: knowledge-assessment, planning, execution, monitoring, evaluation, reflection",
                                 constraints={"values": ["knowledge-assessment", "planning", "execution", "monitoring", "evaluation", "reflection"]}),
                    ToolParameter("overallConfidence", "float", required=True,
                                 description="Overall confidence 0-1"),
                    ToolParameter("uncertaintyAreas", "array", required=False,
                                 description="Areas of uncertainty"),
                ],
                keywords=["monitor", "confidence", "uncertainty", "self-assess", "quality"],
            ),
            ToolDefinition(
                name="programming_paradigm",
                description="Apply different programming paradigms to problems",
                parameters=[
                    ToolParameter("paradigmName", "enum", required=True,
                                 description="Paradigm: imperative, procedural, object_oriented, functional, declarative, logic, event_driven, concurrent, reactive",
                                 constraints={"values": ["imperative", "procedural", "object_oriented", "functional", "declarative", "logic", "event_driven", "concurrent", "reactive"]}),
                    ToolParameter("problem", "string", required=True,
                                 description="Problem to solve"),
                    ToolParameter("approach", "array", required=False,
                                 description="Approach steps"),
                ],
                keywords=["paradigm", "programming", "approach", "style"],
            ),
            ToolDefinition(
                name="scientific_method",
                description="Apply formal scientific reasoning and hypothesis testing",
                parameters=[
                    ToolParameter("stage", "enum", required=True,
                                 description="Stage: observation, question, hypothesis, experiment, analysis, conclusion, iteration",
                                 constraints={"values": ["observation", "question", "hypothesis", "experiment", "analysis", "conclusion", "iteration"]}),
                    ToolParameter("inquiryId", "string", required=True,
                                 description="Unique inquiry identifier"),
                    ToolParameter("iteration", "integer", required=True,
                                 description="Current iteration number"),
                    ToolParameter("nextStageNeeded", "boolean", required=True,
                                 description="Whether another stage is needed"),
                ],
                keywords=["science", "hypothesis", "experiment", "test", "evidence"],
            ),
            ToolDefinition(
                name="structured_argumentation",
                description="Systematic dialectical reasoning and argument analysis",
                parameters=[
                    ToolParameter("claim", "string", required=True,
                                 description="The argument claim"),
                    ToolParameter("premises", "array", required=True,
                                 description="Supporting premises"),
                    ToolParameter("conclusion", "string", required=True,
                                 description="Derived conclusion"),
                    ToolParameter("argumentType", "enum", required=True,
                                 description="Type: thesis, antithesis, synthesis, objection, rebuttal",
                                 constraints={"values": ["thesis", "antithesis", "synthesis", "objection", "rebuttal"]}),
                    ToolParameter("confidence", "float", required=True,
                                 description="Confidence 0-1"),
                ],
                keywords=["argument", "claim", "premise", "logic", "debate"],
            ),
        ]


# =============================================================================
# MCP SERVER TEMPLATE (User-Extensible)
# =============================================================================

class MCPServerTemplate(ToolDomain):
    """
    Template for creating custom MCP server tool domains.

    Allows users to define their own MCP server tool sets dynamically.

    Usage:
        from tool_domains import MCPServerTemplate
        from core import ToolDefinition, ToolParameter

        my_server = MCPServerTemplate(
            server_name="my_api",
            description="My custom API MCP server",
            tools=[
                ToolDefinition(
                    name="get_user",
                    description="Get user by ID",
                    parameters=[ToolParameter("user_id", "string", required=True)],
                    keywords=["user", "get", "profile"],
                ),
                ToolDefinition(
                    name="create_user",
                    description="Create a new user",
                    parameters=[
                        ToolParameter("name", "string", required=True),
                        ToolParameter("email", "string", required=True),
                    ],
                    keywords=["user", "create", "new"],
                ),
            ],
            keywords=["api", "user", "my_server"],
        )

        builder.add_tool_domain(my_server)
    """

    def __init__(self,
                 server_name: str,
                 description: str = "",
                 tools: List[ToolDefinition] = None,
                 keywords: List[str] = None,
                 category: str = "mcp_server"):
        self.name = server_name
        self.description = description or f"MCP server: {server_name}"
        self.category = category
        self.keywords = keywords or [server_name]
        self._tools = tools or []

    def get_tools(self) -> List[ToolDefinition]:
        return list(self._tools)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Base
    'ToolDomain', 'ToolDomainRegistry',

    # Original domains
    'FileSystemDomain', 'WebDomain', 'DatabaseDomain',
    'APIDomain', 'MCPDomain', 'ShellDomain',

    # MCP server domains
    'PlaywrightDomain', 'ChromaDomain', 'ClearThoughtDomain',
    'MCPServerTemplate',
]
