#!/usr/bin/env python3
"""
UCL-Tool Auto-Importer
======================

Universal tool import from ANY source — files, MCP servers, prompts, dicts.

Provides:
1. import_from_json_file() — load tools from JSON file
2. import_from_prompt_file() — migrate JSON Schema in prompt files
3. import_from_dict() — convert any dict/list to ToolDefinitions
4. auto_domain() — wrap tools into a ready-to-use domain
5. discover_mcp_tools() — connect to MCP server, discover tools
6. import_mcp_server() — auto-import an entire MCP server as a domain
7. import_mcp_servers() — batch import multiple MCP servers
8. auto_build() — full pipeline: import + build prompt

Usage:
    from tool_importer import (
        import_from_json_file,
        import_mcp_server,
        auto_build,
    )

    # Import tools from a JSON file
    tools = import_from_json_file("my_tools.json")

    # Auto-discover tools from an MCP server
    domain = import_mcp_server("npx", args=["@playwright/mcp"])

    # Full pipeline: migrate prompt + add MCP servers
    built = auto_build(
        identity="Agent",
        prompt_path="my_prompt.txt",
        mcp_servers={"playwright": {"command": "npx", "args": ["@playwright/mcp"]}},
    )
"""

from __future__ import annotations
import json
import subprocess
import sys
import os
from typing import List, Dict, Any, Tuple, Optional

try:
    from .core import (
        ToolParameter, ToolExample, ToolDefinition, ToolGroup,
        UCLToolConstants
    )
    from .tool_export import (
        keyword_heuristic, from_json_schema_batch, migrate_prompt,
    )
    from .tool_domains import MCPServerTemplate
except ImportError:
    from core import (
        ToolParameter, ToolExample, ToolDefinition, ToolGroup,
        UCLToolConstants
    )
    from tool_export import (
        keyword_heuristic, from_json_schema_batch, migrate_prompt,
    )
    from tool_domains import MCPServerTemplate


# =============================================================================
# FILE IMPORTS
# =============================================================================

def import_from_json_file(path: str,
                          auto_keywords: bool = True) -> List[ToolDefinition]:
    """
    Load tool definitions from a JSON file.

    Supports formats:
    - {"tools": [...]} — OpenAI/Anthropic envelope
    - [{"name": ..., ...}] — list of tool schemas
    - {"type": "function", "function": {...}} — single tool

    Args:
        path: Path to JSON file
        auto_keywords: Auto-generate keywords from name/description

    Returns:
        List of ToolDefinition objects

    Example:
        tools = import_from_json_file("tools.json")
    """
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return from_json_schema_batch(data, auto_keywords=auto_keywords)


def import_from_prompt_file(path: str,
                            auto_keywords: bool = True,
                            group: bool = True,
                            output_path: str = None) -> Tuple[str, List[ToolDefinition]]:
    """
    Load a prompt file and migrate any JSON Schema tools to UCL-Tool.

    Args:
        path: Path to prompt text file
        auto_keywords: Auto-generate keywords
        group: Group migrated tools under shared condition
        output_path: If provided, write migrated prompt to this file

    Returns:
        Tuple of (migrated_text, list_of_tools)

    Example:
        new_text, tools = import_from_prompt_file("prompt.txt")
        # Optionally save: import_from_prompt_file("prompt.txt", output_path="prompt_ucl.txt")
    """
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()

    new_text, tools = migrate_prompt(text, auto_keywords=auto_keywords, group=group)

    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(new_text)

    return new_text, tools


def import_from_dict(data: Any,
                     auto_keywords: bool = True) -> List[ToolDefinition]:
    """
    Convert any dict/list of tool schemas to ToolDefinitions.

    Accepts:
    - {"tools": [...]} — envelope
    - [{"name": ..., ...}] — list
    - {"type": "function", ...} — single tool
    - {"name": ..., "parameters": ...} — bare tool schema

    Args:
        data: Dict, list, or tool schema
        auto_keywords: Auto-generate keywords

    Returns:
        List of ToolDefinition objects
    """
    return from_json_schema_batch(data, auto_keywords=auto_keywords)


# =============================================================================
# AUTO-DOMAIN CREATION
# =============================================================================

def auto_domain(tools: List[ToolDefinition],
                name: str = "custom",
                description: str = "",
                keywords: List[str] = None) -> MCPServerTemplate:
    """
    Wrap any list of ToolDefinitions into a ready-to-use domain.

    Auto-derives keywords from tools if not provided.

    Args:
        tools: List of ToolDefinition objects
        name: Domain name
        description: Domain description
        keywords: Shared keywords (auto-derived if None)

    Returns:
        MCPServerTemplate ready for builder.add_tool_domain()

    Example:
        tools = import_from_json_file("tools.json")
        domain = auto_domain(tools, name="my_tools")
        builder.add_tool_domain(domain)
    """
    if keywords is None:
        # Auto-derive from all tool keywords
        all_kw = set()
        for t in tools:
            all_kw.update(t.keywords)
        keywords = sorted(all_kw)

    if not description:
        description = f"Auto-imported domain: {name} ({len(tools)} tools)"

    return MCPServerTemplate(
        server_name=name,
        description=description,
        tools=tools,
        keywords=keywords,
    )


# =============================================================================
# MCP SERVER AUTO-DISCOVERY
# =============================================================================

def _mcp_jsonrpc(proc, method: str, params: dict = None,
                 id: int = 1, timeout: float = 10) -> dict:
    """
    Send a JSON-RPC request to an MCP server subprocess and read response.

    Args:
        proc: subprocess.Popen object
        method: JSON-RPC method name
        params: Method parameters
        id: Request ID
        timeout: Seconds to wait for response

    Returns:
        Parsed JSON response dict
    """
    request = {
        "jsonrpc": "2.0",
        "method": method,
        "id": id,
    }
    if params is not None:
        request["params"] = params

    # Send request
    request_line = json.dumps(request) + "\n"
    proc.stdin.write(request_line)
    proc.stdin.flush()

    # Read response (line by line until we get a complete JSON object with matching id)
    import select
    response_lines = []
    start_time = __import__('time').time()

    while True:
        if __import__('time').time() - start_time > timeout:
            raise TimeoutError(
                f"MCP server did not respond to '{method}' within {timeout}s"
            )

        line = proc.stdout.readline()
        if not line:
            if proc.poll() is not None:
                raise RuntimeError(
                    f"MCP server exited with code {proc.returncode}"
                )
            continue

        line = line.strip()
        if not line:
            continue

        try:
            response = json.loads(line)
            if response.get("id") == id:
                return response
            # Notification or different id — skip
        except json.JSONDecodeError:
            continue


def _mcp_notification(proc, method: str, params: dict = None):
    """Send a JSON-RPC notification (no response expected)."""
    notification = {
        "jsonrpc": "2.0",
        "method": method,
    }
    if params is not None:
        notification["params"] = params

    line = json.dumps(notification) + "\n"
    proc.stdin.write(line)
    proc.stdin.flush()


def discover_mcp_tools(command: str,
                       args: List[str] = None,
                       timeout: float = 15,
                       env: Dict[str, str] = None) -> List[ToolDefinition]:
    """
    Connect to an MCP server and discover its tools.

    Launches the MCP server as a subprocess, performs the JSON-RPC
    handshake, calls tools/list, and converts discovered tools to
    ToolDefinition objects with auto-generated keywords.

    Args:
        command: Command to launch MCP server (e.g. "npx", "python", "node")
        args: Command arguments (e.g. ["@playwright/mcp"])
        timeout: Seconds to wait for server response
        env: Additional environment variables

    Returns:
        List of ToolDefinition objects

    Example:
        tools = discover_mcp_tools("npx", args=["@playwright/mcp"])
        # → 20 ToolDefinitions for Playwright browser tools
    """
    args = args or []

    # Build environment
    full_env = dict(os.environ)
    if env:
        full_env.update(env)

    # Launch MCP server
    try:
        proc = subprocess.Popen(
            [command] + args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            env=full_env,
        )
    except FileNotFoundError:
        raise RuntimeError(
            f"Cannot launch MCP server: '{command}' not found. "
            f"Make sure it's installed and in PATH."
        )

    try:
        # Step 1: Initialize
        init_response = _mcp_jsonrpc(proc, "initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "ucl-tool-importer", "version": "1.0.0"},
        }, id=1, timeout=timeout)

        if "error" in init_response:
            raise RuntimeError(
                f"MCP initialize failed: {init_response['error']}"
            )

        # Step 2: Send initialized notification
        _mcp_notification(proc, "notifications/initialized")

        # Step 3: List tools
        tools_response = _mcp_jsonrpc(proc, "tools/list", {}, id=2, timeout=timeout)

        if "error" in tools_response:
            raise RuntimeError(
                f"MCP tools/list failed: {tools_response['error']}"
            )

        result = tools_response.get("result", {})
        mcp_tools = result.get("tools", [])

        # Step 4: Convert to ToolDefinitions
        tool_defs = []
        for tool in mcp_tools:
            name = tool.get("name", "")
            description = tool.get("description", "")
            input_schema = tool.get("inputSchema", {})

            # Convert MCP format to OpenAI format for from_json_schema_batch
            openai_format = {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": input_schema,
                }
            }

            # Auto-generate keywords
            keywords = keyword_heuristic(name, description)

            # Create ToolDefinition
            tool_def = ToolDefinition.from_json_schema_tool(
                openai_format, keywords=keywords
            )
            tool_defs.append(tool_def)

        return tool_defs

    finally:
        # Clean up subprocess
        try:
            proc.stdin.close()
        except Exception:
            pass
        try:
            proc.terminate()
            proc.wait(timeout=3)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass


def import_mcp_server(command: str,
                      args: List[str] = None,
                      server_name: str = None,
                      timeout: float = 15,
                      env: Dict[str, str] = None) -> MCPServerTemplate:
    """
    Auto-import an entire MCP server as a UCL-Tool domain.

    Connects to the MCP server, discovers all tools, converts them
    to UCL-Tool format, and wraps them in a ready-to-use domain.

    Args:
        command: Command to launch MCP server
        args: Command arguments
        server_name: Name for the domain (derived from command if None)
        timeout: Seconds to wait for server response
        env: Additional environment variables

    Returns:
        MCPServerTemplate ready for builder.add_tool_domain()

    Example:
        playwright = import_mcp_server("npx", args=["@playwright/mcp"])
        builder.add_tool_domain(playwright)
    """
    if server_name is None:
        # Derive name from command + first arg
        parts = [command] + (args or [])
        server_name = "_".join(
            p.replace("@", "").replace("/", "_").replace(".", "_")
            for p in parts[:2]
        ).lower()

    tools = discover_mcp_tools(command, args, timeout=timeout, env=env)

    # Auto-derive keywords from all tools
    all_kw = set()
    for t in tools:
        all_kw.update(t.keywords)

    return MCPServerTemplate(
        server_name=server_name,
        description=f"MCP server: {command} {' '.join(args or [])} ({len(tools)} tools)",
        tools=tools,
        keywords=sorted(all_kw) if all_kw else [server_name],
    )


def import_mcp_servers(servers: Dict[str, Dict[str, Any]],
                       timeout: float = 15) -> List[MCPServerTemplate]:
    """
    Batch import multiple MCP servers.

    Args:
        servers: Dict mapping names to server configs:
            {
                "playwright": {"command": "npx", "args": ["@playwright/mcp"]},
                "chroma": {"command": "python", "args": ["-m", "chroma_mcp"]},
            }
        timeout: Seconds to wait per server

    Returns:
        List of MCPServerTemplate domains

    Example:
        domains = import_mcp_servers({
            "playwright": {"command": "npx", "args": ["@playwright/mcp"]},
            "chroma": {"command": "python", "args": ["-m", "chroma_mcp"]},
        })
        for d in domains:
            builder.add_tool_domain(d)
    """
    results = []
    for name, config in servers.items():
        command = config["command"]
        args = config.get("args", [])
        env = config.get("env", None)
        server_timeout = config.get("timeout", timeout)

        domain = import_mcp_server(
            command=command,
            args=args,
            server_name=name,
            timeout=server_timeout,
            env=env,
        )
        results.append(domain)

    return results


# =============================================================================
# FULL PIPELINE
# =============================================================================

def auto_build(identity: str,
               prompt_path: str = None,
               prompt_text: str = None,
               mcp_servers: Dict[str, Dict[str, Any]] = None,
               tool_files: List[str] = None,
               tool_domains: List[str] = None,
               output_path: str = None) -> Any:
    """
    Full auto-import pipeline: import tools + build optimized prompt.

    Combines:
    - Prompt migration (JSON Schema → UCL-Tool)
    - MCP server auto-discovery
    - Tool file imports
    - Pre-built domain imports
    - Builder assembly

    Args:
        identity: Agent identity/role
        prompt_path: Path to prompt file (migrated if contains JSON Schema)
        prompt_text: Prompt text (alternative to prompt_path)
        mcp_servers: Dict of MCP server configs to auto-import
        tool_files: List of JSON tool file paths to import
        tool_domains: List of pre-built domain names (e.g. ["filesystem", "web"])
        output_path: If provided, write the built prompt to this file

    Returns:
        BuiltPrompt object with score, text, tools, etc.

    Example:
        built = auto_build(
            identity="Research Agent",
            prompt_path="system_prompt.txt",
            mcp_servers={
                "playwright": {"command": "npx", "args": ["@playwright/mcp"]},
                "chroma": {"command": "python", "args": ["-m", "chroma_mcp"]},
            },
            tool_domains=["filesystem"],
        )
        print(f"Quality: {built.score.Q_final:.1%}")
        print(f"Tools: {len(built.tools)}")
    """
    try:
        from .builder import UCLPromptBuilder
        from .tool_domains import ToolDomainRegistry
    except ImportError:
        from builder import UCLPromptBuilder
        from tool_domains import ToolDomainRegistry

    builder = UCLPromptBuilder()
    builder.set_identity(identity)
    builder.add_critical()

    all_tools = []

    # 1. Migrate prompt file/text
    base_text = ""
    if prompt_path:
        with open(prompt_path, 'r', encoding='utf-8') as f:
            prompt_text = f.read()

    if prompt_text:
        migrated_text, migrated_tools = migrate_prompt(prompt_text)
        base_text = migrated_text
        all_tools.extend(migrated_tools)

    # 2. Import tool files
    if tool_files:
        for path in tool_files:
            tools = import_from_json_file(path)
            all_tools.extend(tools)

    # 3. Import MCP servers
    if mcp_servers:
        domains = import_mcp_servers(mcp_servers)
        for domain in domains:
            builder.add_tool_domain(domain)

    # 4. Import pre-built domains
    if tool_domains:
        for name in tool_domains:
            domain = ToolDomainRegistry.get(name)
            builder.add_tool_domain(domain)

    # 5. Add remaining tools as auto-domain
    if all_tools:
        domain = auto_domain(all_tools, name="imported_tools")
        builder.add_tool_domain(domain)

    # 6. Set task from migrated text or default
    if base_text:
        builder.set_task(base_text)

    # 7. Build
    built = builder.build()

    # 8. Optionally save
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(built.text)

    return built


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # File imports
    'import_from_json_file',
    'import_from_prompt_file',
    'import_from_dict',

    # Auto-domain
    'auto_domain',

    # MCP server auto-discovery
    'discover_mcp_tools',
    'import_mcp_server',
    'import_mcp_servers',

    # Full pipeline
    'auto_build',
]
