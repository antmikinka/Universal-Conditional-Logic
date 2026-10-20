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
9. convert_tools() — one list in, three artifacts out:
   canonical UCL text, website domain JSON, OpenAI tool envelope

Usage:
    from tool_importer import (
        import_from_json_file,
        import_mcp_server,
        convert_tools,
        write_website_domain,
        auto_build,
    )

    # Import tools from a JSON file
    tools = import_from_json_file("my_tools.json")

    # Auto-discover tools from an MCP server.
    # framing="newline" is the MCP spec. Use "content-length" for peers
    # that speak LSP-style frames (Loom's StdioMCPClient is one of them).
    domain = import_mcp_server("npx", args=["@playwright/mcp"])

    # Three artifacts from a JSON file, a list, or a live server
    converted = convert_tools(name="playwright", json_path="tools.json")
    write_website_domain(converted, "ucl-builder/tools")

    # Full pipeline: migrate prompt + add MCP servers
    built = auto_build(
        identity="Agent",
        prompt_path="my_prompt.txt",
        mcp_servers={"playwright": {"command": "npx", "args": ["@playwright/mcp"]}},
    )
"""

from __future__ import annotations
import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

try:
    from .core import (
        ToolParameter, ToolExample, ToolDefinition, ToolGroup,
        UCLToolConstants
    )
    from .tool_export import (
        keyword_heuristic, from_json_schema_batch, migrate_prompt,
        choose_domain_keywords, to_website_domain, render_tools_ucl,
        to_json_schema_batch,
    )
    from .tool_domains import MCPServerTemplate
    from .mcp_stdio import MCPStdioSession, FRAMINGS
except ImportError:
    from core import (
        ToolParameter, ToolExample, ToolDefinition, ToolGroup,
        UCLToolConstants
    )
    from tool_export import (
        keyword_heuristic, from_json_schema_batch, migrate_prompt,
        choose_domain_keywords, to_website_domain, render_tools_ucl,
        to_json_schema_batch,
    )
    from tool_domains import MCPServerTemplate
    from mcp_stdio import MCPStdioSession, FRAMINGS


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

def _list_mcp_raw(command: str,
                  args: List[str] = None,
                  timeout: float = 15,
                  env: Dict[str, str] = None,
                  framing: str = "newline") -> List[Dict[str, Any]]:
    """Handshake with an MCP stdio server and return the raw tools/list array.

    ``framing="newline"`` matches the MCP spec (2026-07-28). Pass
    ``framing="content-length"`` for peers that use LSP-style frames,
    including Loom's StdioMCPClient.
    """
    if framing not in FRAMINGS:
        raise ValueError(f"framing must be one of {FRAMINGS}, got {framing!r}")

    with MCPStdioSession(
        command,
        args=args or [],
        env=env,
        framing=framing,
        timeout=timeout,
    ) as session:
        init_response = session.request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "ucl-tool-importer", "version": "1.0.0"},
        })
        if "error" in init_response:
            raise RuntimeError(f"MCP initialize failed: {init_response['error']}")

        session.notify("notifications/initialized")

        tools_response = session.request("tools/list", {})
        if "error" in tools_response:
            raise RuntimeError(f"MCP tools/list failed: {tools_response['error']}")

        result = tools_response.get("result") or {}
        mcp_tools = result.get("tools") or []
        if not isinstance(mcp_tools, list):
            raise RuntimeError("MCP tools/list result did not contain a tools array")
        return mcp_tools


def discover_mcp_tools(command: str,
                       args: List[str] = None,
                       timeout: float = 15,
                       env: Dict[str, str] = None,
                       framing: str = "newline") -> List[ToolDefinition]:
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
        framing: "newline" (MCP spec) or "content-length" (Loom / LSP-style)

    Returns:
        List of ToolDefinition objects

    Example:
        tools = discover_mcp_tools("npx", args=["@playwright/mcp"])
        # → ToolDefinitions for that server's tools
    """
    raw_tools = _list_mcp_raw(
        command, args=args, timeout=timeout, env=env, framing=framing
    )
    tools, _meta, _warnings = definitions_from_payload({"tools": raw_tools})
    return tools


def import_mcp_server(command: str,
                      args: List[str] = None,
                      server_name: str = None,
                      timeout: float = 15,
                      env: Dict[str, str] = None,
                      framing: str = "newline") -> MCPServerTemplate:
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

    tools = discover_mcp_tools(
        command, args, timeout=timeout, env=env, framing=framing
    )

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
        framing = config.get("framing", "newline")

        domain = import_mcp_server(
            command=command,
            args=args,
            server_name=name,
            timeout=server_timeout,
            env=env,
            framing=framing,
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

# =============================================================================
# ONE LIST, THREE ARTIFACTS
# =============================================================================

_DOMAIN_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def sanitize_tool_name(name: str) -> str:
    """Make an MCP or JSON tool name legal for ToolDefinition."""
    text = re.sub(r"[^A-Za-z0-9_]+", "_", (name or "").strip())
    text = re.sub(r"_+", "_", text).strip("_")
    if text and text[0].isdigit():
        text = "tool_" + text
    return text


def _split_payload(data: Any) -> Tuple[List[Any], Dict[str, Any]]:
    """Return (tool items, domain metadata) for the shapes convert_tools accepts."""
    if isinstance(data, list):
        return data, {}
    if not isinstance(data, dict):
        raise ValueError(
            f"Tool payload must be a dict or list, got {type(data).__name__}"
        )
    if "tools" not in data:
        return [data], {}
    tools = data.get("tools")
    meta = {
        key: data[key]
        for key in ("name", "description", "keywords", "category", "icon")
        if key in data
    }
    if isinstance(tools, dict):
        return list(tools.values()), meta
    if isinstance(tools, list):
        return tools, meta
    raise ValueError("The 'tools' field must be a list or a website domain object")


def _website_parameters_to_schema(parameters: List[Dict[str, Any]]) -> Dict[str, Any]:
    properties: Dict[str, Any] = {}
    required: List[str] = []
    for param in parameters:
        if not isinstance(param, dict) or not param.get("name"):
            continue
        pname = str(param["name"])
        raw_type = str(param.get("type") or "string")
        prop: Dict[str, Any] = {}
        if raw_type == "enum":
            values = list(param.get("constraints") or [])
            prop["type"] = "string"
            if values:
                prop["enum"] = values
        elif raw_type == "float":
            prop["type"] = "number"
        elif raw_type == "file":
            prop["type"] = "string"
        else:
            prop["type"] = raw_type
        if param.get("description"):
            prop["description"] = param["description"]
        if "default" in param:
            prop["default"] = param["default"]
        if "minimum" in param:
            prop["minimum"] = param["minimum"]
        if "maximum" in param:
            prop["maximum"] = param["maximum"]
        if "pattern" in param:
            prop["pattern"] = param["pattern"]
        properties[pname] = prop
        if param.get("required"):
            required.append(pname)
    return {"type": "object", "properties": properties, "required": required}


def _to_openai_tool(item: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    """Normalize one record to an OpenAI function tool plus catalog extras."""
    extras: Dict[str, Any] = {
        "keywords": [],
        "return_type": "",
        "examples": [],
        "param_types": {},
    }
    if isinstance(item.get("function"), dict):
        func = item["function"]
        extras["keywords"] = item.get("keywords") or func.get("keywords") or []
        extras["return_type"] = item.get("return_type") or func.get("return_type") or ""
        extras["examples"] = item.get("examples") or func.get("examples") or []
        return item, extras

    parameters = item.get("parameters")
    if isinstance(parameters, list):
        for param in parameters:
            if isinstance(param, dict) and param.get("type") == "file" and param.get("name"):
                extras["param_types"][str(param["name"])] = "file"
        openai = {
            "type": "function",
            "function": {
                "name": item.get("name") or "",
                "description": item.get("description") or "",
                "parameters": _website_parameters_to_schema(parameters),
            },
        }
        extras["keywords"] = item.get("keywords") or []
        extras["return_type"] = item.get("return_type") or ""
        extras["examples"] = item.get("examples") or []
        return openai, extras

    schema = item.get("inputSchema", item.get("input_schema"))
    if schema is None and isinstance(parameters, dict):
        schema = parameters
    if schema is not None or "name" in item:
        if not isinstance(schema, dict):
            schema = {"type": "object", "properties": {}}
        openai = {
            "type": "function",
            "function": {
                "name": item.get("name") or "",
                "description": item.get("description") or "",
                "parameters": schema,
            },
        }
        extras["keywords"] = item.get("keywords") or []
        extras["return_type"] = item.get("return_type") or ""
        extras["examples"] = item.get("examples") or []
        return openai, extras

    return None, extras


def definitions_from_payload(
    data: Any,
    auto_keywords: bool = True,
) -> Tuple[List[ToolDefinition], Dict[str, Any], List[str]]:
    """Parse a JSON file, MCP tools/list result, or website domain into tools."""
    items, meta = _split_payload(data)
    warnings: List[str] = []
    tools: List[ToolDefinition] = []
    used = set()

    for item in items:
        if not isinstance(item, dict):
            warnings.append("skipped a tool entry that was not an object")
            continue
        openai, extras = _to_openai_tool(item)
        if openai is None:
            warnings.append("skipped an unrecognized tool entry")
            continue
        func = openai.setdefault("function", {})
        raw_name = str(func.get("name") or "")
        clean = sanitize_tool_name(raw_name)
        if not clean:
            warnings.append(f"skipped a tool with an empty name ({raw_name!r})")
            continue
        if clean != raw_name:
            warnings.append(f"renamed tool {raw_name!r} to {clean!r}")
        base = clean
        suffix = 2
        while clean in used:
            clean = f"{base}_{suffix}"
            suffix += 1
        if clean != base:
            warnings.append(f"renamed duplicate {base!r} to {clean!r}")
        used.add(clean)
        func["name"] = clean

        explicit = [str(kw) for kw in (extras.get("keywords") or []) if str(kw).strip()]
        if explicit:
            keywords = explicit
        elif auto_keywords:
            keywords = keyword_heuristic(clean, func.get("description") or "")
        else:
            keywords = []

        try:
            tool = ToolDefinition.from_json_schema_tool(openai, keywords=keywords)
        except (ValueError, KeyError, TypeError) as exc:
            warnings.append(f"skipped {clean!r}: {exc}")
            used.discard(clean)
            continue

        return_type = extras.get("return_type") or ""
        if return_type:
            tool.return_type = str(return_type)
        examples = []
        for example in extras.get("examples") or []:
            if isinstance(example, dict) and example.get("input"):
                examples.append(ToolExample(
                    input_text=str(example.get("input")),
                    output_text=str(example.get("output") or ""),
                ))
        tool.examples = examples
        for param in tool.parameters:
            if extras.get("param_types", {}).get(param.name) == "file":
                param.param_type = "file"
        tools.append(tool)

    return tools, meta, warnings


@dataclass
class ToolConversion:
    """One tool list rendered for the prompt builder, the website, and Loom."""

    name: str
    tools: List[ToolDefinition]
    ucl_text: str
    website_domain: Dict[str, Any]
    openai_envelope: Dict[str, Any]
    source: str
    warnings: List[str] = field(default_factory=list)


def _check_domain_name(name: str) -> str:
    if not name or not _DOMAIN_NAME.match(name):
        raise ValueError(
            "Domain name must start with a letter or underscore and contain "
            f"only letters, digits, and underscores. Got {name!r}."
        )
    return name


def convert_tools(
    name: str,
    data: Any = None,
    json_path: str = None,
    mcp_command: str = None,
    mcp_args: List[str] = None,
    description: str = "",
    keywords: List[str] = None,
    category: str = None,
    icon: str = None,
    framing: str = "newline",
    timeout: float = 15,
    env: Dict[str, str] = None,
    auto_keywords: bool = True,
) -> ToolConversion:
    """Turn a JSON file, an in-memory list, or a live MCP server into three artifacts.

    Exactly one of ``data``, ``json_path``, or ``mcp_command`` is required.
    ``website_domain`` matches ``ucl-builder/tools/*.json``.
    ``openai_envelope`` is what Loom's tool prompt and function-calling loop consume.
    ``ucl_text`` is the canonical grouped ``^^CONDITION^^`` / ``[[TOOL:]]`` block.
    """
    sources = [data is not None, bool(json_path), bool(mcp_command)]
    if sum(sources) != 1:
        raise ValueError("Pass exactly one of data, json_path, or mcp_command")
    _check_domain_name(name)

    meta: Dict[str, Any] = {}
    if mcp_command:
        raw_tools = _list_mcp_raw(
            mcp_command,
            args=mcp_args,
            timeout=timeout,
            env=env,
            framing=framing,
        )
        tools, meta, warnings = definitions_from_payload(
            {"tools": raw_tools}, auto_keywords=auto_keywords
        )
        source = "mcp"
    else:
        if json_path:
            with open(json_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            source = "json"
        else:
            source = "list"
        tools, meta, warnings = definitions_from_payload(
            data, auto_keywords=auto_keywords
        )

    if not tools:
        detail = "; ".join(warnings) if warnings else "the payload contained no tools"
        raise ValueError(f"No tools to convert: {detail}")

    if not description:
        description = str(meta.get("description") or "")
    if keywords is None and meta.get("keywords"):
        keywords = list(meta["keywords"])
    if category is None:
        category = str(meta.get("category") or ("mcp_server" if source == "mcp" else "original"))
    if icon is None:
        icon = str(meta.get("icon") or ("🔌" if source == "mcp" else "📥"))

    domain_keywords = choose_domain_keywords(tools, name, explicit=keywords)
    return ToolConversion(
        name=name,
        tools=tools,
        ucl_text=render_tools_ucl(
            tools, name, description=description, keywords=domain_keywords
        ),
        website_domain=to_website_domain(
            tools,
            name,
            description=description,
            keywords=domain_keywords,
            category=category,
            icon=icon,
        ),
        openai_envelope=to_json_schema_batch(tools),
        source=source,
        warnings=warnings,
    )


def write_website_domain(
    conversion: ToolConversion,
    tools_dir: str,
    update_index: bool = True,
) -> Path:
    """Write ``<name>.json`` into the website tools folder and refresh index.json."""
    directory = Path(tools_dir)
    directory.mkdir(parents=True, exist_ok=True)
    domain_path = directory / f"{conversion.name}.json"
    _write_json(domain_path, conversion.website_domain)
    if update_index:
        _update_tool_index(directory, conversion.name, conversion.website_domain.get("category", "original"))
    return domain_path


def write_artifact_dir(conversion: ToolConversion, out_dir: str) -> Dict[str, Path]:
    """Write the three artifacts side by side, without touching the website index."""
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    ucl_path = directory / f"{conversion.name}.ucl.txt"
    domain_path = directory / f"{conversion.name}.domain.json"
    openai_path = directory / f"{conversion.name}.openai.json"
    ucl_path.write_text(conversion.ucl_text, encoding="utf-8")
    _write_json(domain_path, conversion.website_domain)
    _write_json(openai_path, conversion.openai_envelope)
    return {"ucl": ucl_path, "domain": domain_path, "openai": openai_path}


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    path.write_text(text, encoding="utf-8")


def _update_tool_index(tools_dir: Path, name: str, category: str) -> None:
    index_path = tools_dir / "index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
    else:
        index = {}
    index.setdefault("version", "1.0")
    index.setdefault("description", "UCL Tool Domains Registry")
    domains = index.setdefault("domains", [])
    if name not in domains:
        domains.append(name)

    meta = index.setdefault("metadata", {})
    categories = meta.setdefault("categories", {})
    categories.setdefault("original", {"label": "Core Domains", "domains": []})
    categories.setdefault("mcp_server", {"label": "MCP Server Domains", "domains": []})
    bucket = "mcp_server" if category == "mcp_server" else "original"
    for key, entry in categories.items():
        if not isinstance(entry, dict):
            continue
        names = [item for item in entry.get("domains", []) if item != name]
        if key == bucket:
            names.append(name)
        entry["domains"] = names

    total = 0
    for domain_name in domains:
        path = tools_dir / f"{domain_name}.json"
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        tool_map = payload.get("tools") or {}
        if isinstance(tool_map, dict):
            total += len(tool_map)
    meta["totalDomains"] = len(domains)
    meta["totalTools"] = total
    _write_json(index_path, index)


def main(argv: List[str] = None) -> int:
    """CLI: convert a JSON file or MCP server and optionally update the website catalog."""
    parser = argparse.ArgumentParser(
        description=(
            "Convert an MCP server, a JSON file, or a tool list into UCL text, "
            "a ucl-builder domain file, and an OpenAI tool envelope."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--json", dest="json_path", help="Tool JSON file")
    source.add_argument("--mcp", dest="mcp_command", help="MCP stdio server executable")
    parser.add_argument("--mcp-arg", action="append", default=[], help="Argument for the MCP server. Repeatable.")
    parser.add_argument("--name", required=True, help="Domain name, used as the website filename")
    parser.add_argument("--description", default="", help="Domain description")
    parser.add_argument("--keyword", action="append", default=None, help="Shared activation keyword. Repeatable.")
    parser.add_argument("--category", default=None, help="original or mcp_server. Default follows the source.")
    parser.add_argument(
        "--framing",
        choices=list(FRAMINGS),
        default="newline",
        help="newline is the MCP spec. content-length matches Loom's stdio client.",
    )
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--tools-dir", help="Website tools directory. Writes <name>.json and updates index.json.")
    parser.add_argument("--out-dir", help="Directory for <name>.ucl.txt, .domain.json, and .openai.json.")
    parser.add_argument(
        "--print",
        dest="print_which",
        choices=("ucl", "domain", "openai", "all", "none"),
        default="ucl",
        help="What to write to stdout.",
    )
    args = parser.parse_args(argv)

    try:
        conversion = convert_tools(
            name=args.name,
            json_path=args.json_path,
            mcp_command=args.mcp_command,
            mcp_args=args.mcp_arg,
            description=args.description,
            keywords=args.keyword,
            category=args.category,
            framing=args.framing,
            timeout=args.timeout,
        )
    except (OSError, ValueError, RuntimeError, TimeoutError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    for warning in conversion.warnings:
        print(f"warning: {warning}", file=sys.stderr)

    if args.tools_dir:
        path = write_website_domain(conversion, args.tools_dir)
        print(f"wrote {path}", file=sys.stderr)
        print(f"updated {Path(args.tools_dir) / 'index.json'}", file=sys.stderr)
    if args.out_dir:
        written = write_artifact_dir(conversion, args.out_dir)
        for label, path in written.items():
            print(f"wrote {label} {path}", file=sys.stderr)
    print(f"tools: {len(conversion.tools)}", file=sys.stderr)

    def emit(text: str) -> None:
        try:
            print(text)
        except UnicodeEncodeError:
            sys.stdout.buffer.write(text.encode("utf-8", errors="replace"))
            sys.stdout.buffer.write(b"\n")

    if args.print_which == "ucl":
        emit(conversion.ucl_text)
    elif args.print_which == "domain":
        emit(json.dumps(conversion.website_domain, indent=2, ensure_ascii=False))
    elif args.print_which == "openai":
        emit(json.dumps(conversion.openai_envelope, indent=2, ensure_ascii=False))
    elif args.print_which == "all":
        emit(conversion.ucl_text)
        emit("--- domain ---")
        emit(json.dumps(conversion.website_domain, indent=2, ensure_ascii=False))
        emit("--- openai ---")
        emit(json.dumps(conversion.openai_envelope, indent=2, ensure_ascii=False))
    return 0


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

    # Three-artifact converter
    'convert_tools',
    'write_website_domain',
    'write_artifact_dir',
    'definitions_from_payload',
    'sanitize_tool_name',
    'ToolConversion',
]


if __name__ == "__main__":
    sys.exit(main())
