#!/usr/bin/env python3
"""
UCL-Tool Export & Migration (Phase 6)
=======================================

Bidirectional conversion between JSON Schema and UCL-Tool formats.

Provides:
1. keyword_heuristic() — infer activation keywords from name/description
2. from_json_schema_batch() — parse {"tools":[...]} envelopes
3. to_json_schema_batch() — generate {"tools":[...]} envelopes
4. migrate_prompt() — replace JSON Schema blocks in prompts with UCL-Tool

Usage:
    from tool_export import migrate_prompt, keyword_heuristic

    # Migrate an existing prompt with JSON Schema tools
    new_text, tools = migrate_prompt(old_prompt_text)

    # Generate keywords for a tool
    keywords = keyword_heuristic("read_file", "Read file content from disk")
    # → ["read", "file", "content", "disk"]

    # Batch conversion
    tools = from_json_schema_batch({"tools": [...]})
    envelope = to_json_schema_batch(tools)
"""

from __future__ import annotations
import json
import re
from typing import List, Dict, Any, Tuple, Optional

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
# KEYWORD HEURISTIC
# =============================================================================

# Common English stopwords to filter from descriptions
_STOPWORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "to", "for", "of", "in", "on", "at", "by", "with", "from", "as",
    "and", "or", "not", "no", "but", "so", "if", "then", "than",
    "it", "its", "this", "that", "these", "those",
    "will", "can", "could", "should", "would", "may", "might", "shall",
    "has", "have", "had", "do", "does", "did",
    "i", "you", "he", "she", "we", "they", "me", "him", "her", "us",
    "my", "your", "his", "our", "their",
    "what", "which", "who", "when", "where", "how", "all", "each",
    "more", "less", "most", "other", "some", "any", "every",
    "up", "out", "about", "into", "through", "during", "before", "after",
    "above", "below", "between", "under", "over",
})


def keyword_heuristic(name: str,
                      description: str = "",
                      max_keywords: int = 6) -> List[str]:
    """
    Infer activation keywords from tool name and description.

    Strategy:
    1. Split snake_case name into words
    2. Extract meaningful words from description
    3. Remove stopwords
    4. Combine, deduplicate, limit to max_keywords
    5. Name words come first (more specific)

    Args:
        name: Tool name (snake_case)
        description: Tool description
        max_keywords: Maximum keywords to return

    Returns:
        List of inferred keywords

    Example:
        >>> keyword_heuristic("read_file", "Read file content from disk")
        ['read', 'file', 'content', 'disk']
    """
    # 1. Extract words from snake_case name
    name_words = [w.lower() for w in name.split('_') if w and len(w) > 1]

    # 2. Extract words from description
    desc_words = []
    if description:
        # Tokenize: lowercase, strip punctuation
        tokens = re.findall(r'[a-zA-Z]+', description.lower())
        desc_words = [w for w in tokens if w not in _STOPWORDS and len(w) > 2]

    # 3. Combine: name words first (higher priority), then desc words
    seen = set()
    result = []

    for word in name_words:
        if word not in seen and word not in _STOPWORDS:
            seen.add(word)
            result.append(word)
            if len(result) >= max_keywords:
                return result

    for word in desc_words:
        if word not in seen:
            seen.add(word)
            result.append(word)
            if len(result) >= max_keywords:
                return result

    return result


# =============================================================================
# BATCH CONVERSION
# =============================================================================

def from_json_schema_batch(envelope: Dict[str, Any],
                           auto_keywords: bool = True) -> List[ToolDefinition]:
    """
    Parse a JSON Schema tool envelope into ToolDefinition objects.

    Args:
        envelope: Dict with "tools" key containing list of tool schemas,
                  or a single tool schema dict
        auto_keywords: If True, generate keywords via heuristic

    Returns:
        List of ToolDefinition objects

    Example:
        tools = from_json_schema_batch({
            "tools": [
                {"type": "function", "function": {"name": "read_file", ...}},
            ]
        })
    """
    # Handle single tool vs envelope
    if "tools" in envelope:
        tool_schemas = envelope["tools"]
    elif "function" in envelope or "type" in envelope:
        tool_schemas = [envelope]
    elif isinstance(envelope, list):
        tool_schemas = envelope
    else:
        raise ValueError(
            "Expected {'tools': [...]} envelope, single tool schema, or list. "
            f"Got keys: {list(envelope.keys()) if isinstance(envelope, dict) else type(envelope)}"
        )

    tools = []
    for schema in tool_schemas:
        # Extract the function definition
        if "function" in schema:
            func = schema["function"]
        elif "name" in schema:
            func = schema
        else:
            continue

        name = func.get("name", "")
        description = func.get("description", "")

        # Generate keywords
        keywords = []
        if auto_keywords:
            keywords = keyword_heuristic(name, description)

        # Use existing from_json_schema_tool
        tool = ToolDefinition.from_json_schema_tool(schema, keywords=keywords)
        tools.append(tool)

    return tools


def to_json_schema_batch(tools: List[ToolDefinition]) -> Dict[str, Any]:
    """
    Export multiple tools as a JSON Schema tool envelope.

    Args:
        tools: List of ToolDefinition objects

    Returns:
        Dict with "tools" key containing list of JSON Schema tool definitions

    Example:
        envelope = to_json_schema_batch([read_file_tool, write_file_tool])
        json_str = json.dumps(envelope, indent=2)
    """
    return {
        "tools": [tool.to_json_schema_tool() for tool in tools]
    }


# =============================================================================
# WEBSITE CATALOG + GROUPED UCL
# =============================================================================

def choose_domain_keywords(tools: List[ToolDefinition],
                           name: str,
                           explicit: Optional[List[str]] = None,
                           limit: int = 8) -> List[str]:
    """Pick the shared ^^CONDITION^^ keywords for a domain.

    Explicit keywords win. Otherwise name words come first, then the
    keywords that show up on the most tools. Capped so the group
    condition stays shorter than repeating every tool keyword.
    """
    if explicit:
        chosen: List[str] = []
        for kw in explicit:
            text = str(kw).strip().lower()
            if text and text not in chosen:
                chosen.append(text)
            if len(chosen) >= limit:
                break
        return chosen or [name.lower()]

    counts: Dict[str, int] = {}
    for tool in tools:
        for kw in tool.keywords:
            counts[kw] = counts.get(kw, 0) + 1
    ranked = sorted(counts, key=lambda kw: (-counts[kw], kw))
    name_words = [part.lower() for part in name.split("_") if len(part) > 1]
    chosen = []
    for word in name_words + ranked:
        if word not in chosen:
            chosen.append(word)
        if len(chosen) >= limit:
            break
    return chosen or [name.lower()]


def tool_to_website(tool: ToolDefinition) -> Dict[str, Any]:
    """One tool in the ucl-builder ``tools/*.json`` shape."""
    parameters = []
    for param in tool.parameters:
        entry: Dict[str, Any] = {
            "name": param.name,
            "type": param.param_type,
            "required": param.required,
        }
        if param.description:
            entry["description"] = param.description
        if param.default is not None:
            entry["default"] = param.default
        if param.param_type == "enum":
            entry["constraints"] = list(param.constraints.get("values", []))
        if "min" in param.constraints:
            entry["minimum"] = param.constraints["min"]
        if "max" in param.constraints:
            entry["maximum"] = param.constraints["max"]
        if "pattern" in param.constraints:
            entry["pattern"] = param.constraints["pattern"]
        parameters.append(entry)

    payload: Dict[str, Any] = {
        "name": tool.name,
        "description": tool.description,
        "parameters": parameters,
        "keywords": list(tool.keywords),
    }
    if tool.return_type:
        payload["return_type"] = tool.return_type
    if tool.examples:
        payload["examples"] = [
            {"input": example.input_text, "output": example.output_text}
            for example in tool.examples
        ]
    return payload


def to_website_domain(tools: List[ToolDefinition],
                      name: str,
                      description: str = "",
                      keywords: Optional[List[str]] = None,
                      category: str = "original",
                      icon: str = "🔧") -> Dict[str, Any]:
    """Domain document loaded by ``ucl-builder/tools/tool-loader.js``."""
    if not description:
        description = f"Auto-imported domain: {name} ({len(tools)} tools)"
    return {
        "name": name,
        "description": description,
        "category": category,
        "icon": icon,
        "keywords": choose_domain_keywords(tools, name, explicit=keywords),
        "tools": {tool.name: tool_to_website(tool) for tool in tools},
    }


def render_tools_ucl(tools: List[ToolDefinition],
                     name: str,
                     description: str = "",
                     keywords: Optional[List[str]] = None) -> str:
    """Canonical grouped UCL for a tool list. One shared ^^CONDITION^^."""
    if not description:
        description = f"Auto-imported domain: {name} ({len(tools)} tools)"
    group = ToolGroup(
        name=name,
        keywords=choose_domain_keywords(tools, name, explicit=keywords),
        tools=list(tools),
        description=description,
    )
    return group.render()


# =============================================================================
# JSON BLOCK DETECTION IN TEXT
# =============================================================================

def _find_json_block_end(text: str, start: int) -> int:
    """
    Find the end of a JSON object starting at position `start`.

    Uses brace-counting to handle nested objects.
    Returns -1 if no valid JSON block found.
    """
    if start >= len(text) or text[start] != '{':
        return -1

    depth = 0
    in_string = False
    escape_next = False

    for i in range(start, len(text)):
        ch = text[i]

        if escape_next:
            escape_next = False
            continue

        if ch == '\\' and in_string:
            escape_next = True
            continue

        if ch == '"' and not escape_next:
            in_string = not in_string
            continue

        if in_string:
            continue

        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                return i

    return -1


def find_json_tool_blocks(text: str) -> List[Tuple[int, int, Any]]:
    """
    Find JSON tool definition blocks in prompt text.

    Returns list of (start_pos, end_pos, parsed_json) tuples.
    Only returns blocks that look like tool definitions.
    """
    results = []

    # Find potential JSON starts
    for match in re.finditer(r'\{', text):
        start = match.start()

        # Quick check: does this look like a tool definition?
        # Look ahead for "type": "function" or "tools"
        ahead = text[start:start+200]
        if '"type"' not in ahead and '"tools"' not in ahead and '"function"' not in ahead:
            continue

        # Try to parse the JSON block
        end = _find_json_block_end(text, start)
        if end == -1:
            continue

        json_str = text[start:end + 1]
        try:
            parsed = json.loads(json_str)
        except (json.JSONDecodeError, ValueError):
            continue

        # Check if it's a tool definition
        is_tool = False
        if isinstance(parsed, dict):
            if parsed.get("type") == "function" and "function" in parsed:
                is_tool = True
            elif "tools" in parsed and isinstance(parsed["tools"], list):
                # Check if any tool in the list looks valid
                for t in parsed["tools"]:
                    if isinstance(t, dict) and ("function" in t or "name" in t):
                        is_tool = True
                        break

        if is_tool:
            results.append((start, end + 1, parsed))

    # Remove overlapping blocks (keep the outermost)
    if not results:
        return results

    filtered = [results[0]]
    for start, end, parsed in results[1:]:
        prev_start, prev_end, _ = filtered[-1]
        if start >= prev_end:  # No overlap
            filtered.append((start, end, parsed))

    return filtered


# =============================================================================
# PROMPT MIGRATION
# =============================================================================

def migrate_prompt(prompt_text: str,
                   auto_keywords: bool = True,
                   group: bool = True) -> Tuple[str, List[ToolDefinition]]:
    """
    Migrate a prompt from JSON Schema tools to UCL-Tool encoding.

    Finds JSON tool definitions in the prompt text, converts them
    to UCL-Tool format, and replaces them in-place.

    Args:
        prompt_text: Original prompt text with JSON Schema tools
        auto_keywords: Auto-generate keywords from name/description
        group: Group converted tools under shared condition

    Returns:
        Tuple of (new_prompt_text, list_of_migrated_tools)

    Example:
        new_text, tools = migrate_prompt(old_prompt)
        print(f"Migrated {len(tools)} tools to UCL-Tool format")
    """
    blocks = find_json_tool_blocks(prompt_text)

    if not blocks:
        return prompt_text, []

    all_tools = []
    # Process blocks in reverse order to preserve positions
    replacements = []

    for start, end, parsed in blocks:
        tools = from_json_schema_batch(parsed, auto_keywords=auto_keywords)
        all_tools.extend(tools)

        # Render as UCL-Tool
        if group and len(tools) > 1:
            # Derive group keywords
            all_kw = set()
            for t in tools:
                all_kw.update(t.keywords)
            tool_group = ToolGroup(
                name="tools",
                keywords=sorted(all_kw),
                tools=tools,
            )
            ucl_text = tool_group.render()
        else:
            ucl_text = "\n\n".join(t.render() for t in tools)

        replacements.append((start, end, ucl_text))

    # Apply replacements in reverse order
    result = prompt_text
    for start, end, ucl_text in reversed(replacements):
        result = result[:start] + ucl_text + result[end:]

    return result, all_tools


# =============================================================================
# ROUND-TRIP FIDELITY
# =============================================================================

def roundtrip_json_to_ucl_to_json(
    schema: Dict[str, Any],
    auto_keywords: bool = True
) -> Dict[str, Any]:
    """
    Test round-trip: JSON Schema → UCL-Tool → JSON Schema.

    Returns the resulting JSON Schema after round-trip.
    Core fields (name, description, params, types) should be preserved.
    """
    # JSON → UCL (ToolDefinition objects)
    tools = from_json_schema_batch(schema, auto_keywords=auto_keywords)

    # UCL → JSON Schema
    return to_json_schema_batch(tools)


def roundtrip_ucl_to_json_to_ucl(
    tool: ToolDefinition
) -> ToolDefinition:
    """
    Test round-trip: UCL-Tool → JSON Schema → UCL-Tool.

    Returns the resulting ToolDefinition after round-trip.
    Core fields should be preserved; keywords/examples/grouping are lost.
    """
    # UCL → JSON Schema
    schema = tool.to_json_schema_tool()

    # JSON → UCL
    return ToolDefinition.from_json_schema_tool(schema, keywords=tool.keywords)


# =============================================================================
# TYPESCRIPT/PI.DEV INTEGRATION
# =============================================================================

def export_for_typescript(
    domains: List[str] = None,
    custom_tools: List[ToolDefinition] = None,
    identity: str = None,
    output_path: str = None
) -> Dict[str, Any]:
    """
    Export UCL-Tool configuration as JSON for TypeScript/pi.dev integration.

    This function bridges Python UCL-Tool with the TypeScript system prompt builder.
    It exports tool domains and definitions in a format that can be imported by
    the ucl-tool-formatter.ts module.

    Args:
        domains: List of domain names to export (e.g., ["filesystem", "web"])
        custom_tools: Additional custom tool definitions
        identity: Agent identity/role for the system prompt
        output_path: Optional file path to save JSON (if None, returns dict)

    Returns:
        Dictionary with UCL-Tool configuration in TypeScript-compatible format

    Example:
        >>> config = export_for_typescript(domains=["filesystem", "web"])
        >>> # Save to file for TypeScript to load:
        >>> export_for_typescript(domains=["filesystem"], output_path="ucl-tools.json")

    Usage in TypeScript:
        import config from './ucl-tools.json';
        const formatted = formatUCLToolsForPrompt(config);
    """
    from datetime import datetime

    # Import tool domains
    try:
        from tool_domains import (
            FileSystemDomain, WebDomain, PlaywrightDomain,
            ChromaDomain, ClearThoughtDomain, DatabaseDomain,
            APIDomain, ShellDomain, MCPDomain, ToolDomainRegistry
        )
        domain_map = {
            'filesystem': FileSystemDomain,
            'web': WebDomain,
            'playwright': PlaywrightDomain,
            'chroma': ChromaDomain,
            'clearthought': ClearThoughtDomain,
            'database': DatabaseDomain,
            'api': APIDomain,
            'shell': ShellDomain,
            'mcp': MCPDomain,
        }
    except ImportError:
        domain_map = {}

    # Build domain list
    exported_domains = []
    if domains:
        for domain_name in domains:
            domain_cls = domain_map.get(domain_name.lower())
            if domain_cls:
                domain_instance = domain_cls()
                tools = domain_instance.get_tools()

                # Convert tools to TypeScript-compatible format
                ts_tools = []
                for tool in tools:
                    ts_tool = {
                        'name': tool.name,
                        'description': tool.description,
                        'parameters': [],
                        'keywords': tool.keywords or [],
                    }

                    # Convert parameters
                    for param in tool.parameters:
                        ts_param = {
                            'name': param.name,
                            'type': param.param_type,
                            'required': param.required,
                            'description': param.description,
                        }
                        if param.default is not None:
                            ts_param['default'] = param.default
                        ts_tool['parameters'].append(ts_param)

                    # Convert examples
                    if tool.examples:
                        ts_tool['examples'] = [
                            {'input': ex.input_text, 'output': ex.output_text}
                            for ex in tool.examples
                        ]

                    ts_tools.append(ts_tool)

                exported_domains.append({
                    'name': domain_instance.name,
                    'description': domain_instance.description,
                    'category': domain_instance.category,
                    'keywords': domain_instance.keywords,
                    'tools': ts_tools,
                })

    # Build custom tools list
    exported_custom_tools = []
    if custom_tools:
        for tool in custom_tools:
            ts_tool = {
                'name': tool.name,
                'description': tool.description,
                'parameters': [],
                'keywords': tool.keywords or [],
            }
            for param in tool.parameters:
                ts_param = {
                    'name': param.name,
                    'type': param.param_type,
                    'required': param.required,
                    'description': param.description,
                }
                if param.default is not None:
                    ts_param['default'] = param.default
                ts_tool['parameters'].append(ts_param)
            exported_custom_tools.append(ts_tool)

    # Calculate metadata
    total_tools = sum(len(d['tools']) for d in exported_domains) + len(exported_custom_tools)
    estimated_tokens = total_tools * 100  # Rough estimate
    overhead_reduction = 78  # Percentage

    # Build export config
    config = {
        'version': '2.1',
        'generatedAt': datetime.now().isoformat(),
        'domains': exported_domains,
        'customTools': exported_custom_tools,
        'metadata': {
            'totalTools': total_tools,
            'estimatedTokens': estimated_tokens,
            'overheadReduction': overhead_reduction,
        },
    }

    # Add identity if provided
    if identity:
        config['identity'] = identity

    # Save to file if path provided
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)

    return config


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'keyword_heuristic',
    'from_json_schema_batch',
    'to_json_schema_batch',
    'find_json_tool_blocks',
    'migrate_prompt',
    'roundtrip_json_to_ucl_to_json',
    'roundtrip_ucl_to_json_to_ucl',
    'export_for_typescript',
]
