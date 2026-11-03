#!/usr/bin/env python3
"""
UCL-Tool Export & Migration Test Suite (Phase 6)
=================================================

Tests for tool_export.py bidirectional JSON Schema interop:
1. keyword_heuristic() — keyword inference from name/description
2. from_json_schema_batch() — parse JSON Schema envelopes
3. to_json_schema_batch() — generate JSON Schema envelopes
4. find_json_tool_blocks() — detect JSON blocks in text
5. migrate_prompt() — replace JSON Schema with UCL-Tool in prompts
6. roundtrip_json_to_ucl_to_json() — JSON -> UCL -> JSON fidelity
7. roundtrip_ucl_to_json_to_ucl() — UCL -> JSON -> UCL fidelity
8. Edge cases — empty inputs, no params, malformed data
9. Integration — full migration + analysis pipeline

Run: PYTHONIOENCODING=utf-8 python tests/test_tool_export.py
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import ToolParameter, ToolExample, ToolDefinition, ToolGroup
from tool_export import (
    keyword_heuristic,
    from_json_schema_batch,
    to_json_schema_batch,
    find_json_tool_blocks,
    migrate_prompt,
    roundtrip_json_to_ucl_to_json,
    roundtrip_ucl_to_json_to_ucl,
)


# =============================================================================
# TEST DATA
# =============================================================================

SAMPLE_JSON_SCHEMA_TOOL = {
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
}

SAMPLE_JSON_SCHEMA_ENVELOPE = {
    "tools": [
        SAMPLE_JSON_SCHEMA_TOOL,
        {
            "type": "function",
            "function": {
                "name": "write_file",
                "description": "Write content to a file on disk",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to the file"
                        },
                        "content": {
                            "type": "string",
                            "description": "Content to write"
                        }
                    },
                    "required": ["path", "content"]
                }
            }
        }
    ]
}

SAMPLE_PROMPT_WITH_JSON = """You are a helpful assistant.

Here are your available tools:

{"type": "function", "function": {"name": "search_web", "description": "Search the web for information", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "Search query"}}, "required": ["query"]}}}

Use these tools when needed to help the user.
"""


# =============================================================================
# TEST 1: keyword_heuristic()
# =============================================================================

def test_keyword_heuristic():
    print("\n" + "="*70)
    print("TEST 1: keyword_heuristic()")
    print("="*70)

    # Basic snake_case splitting
    kw = keyword_heuristic("read_file", "Read file content from disk")
    assert "read" in kw, f"Expected 'read' in keywords, got {kw}"
    assert "file" in kw, f"Expected 'file' in keywords, got {kw}"
    print(f"  read_file -> {kw}")

    # Stopwords filtered from description
    kw2 = keyword_heuristic("send_email", "Send an email to the user with a message")
    assert "send" in kw2
    assert "email" in kw2
    # Stopwords should be filtered
    for stopword in ["the", "an", "to", "with", "a"]:
        assert stopword not in kw2, f"Stopword '{stopword}' should be filtered"
    print(f"  send_email -> {kw2}")

    # Name words have priority over description words
    kw3 = keyword_heuristic("delete_record", "Remove an entry from the database permanently")
    assert kw3[0] == "delete", f"First keyword should be from name, got {kw3}"
    assert kw3[1] == "record", f"Second keyword should be from name, got {kw3}"
    print(f"  delete_record -> {kw3}")

    # Deduplication: word in both name and description
    kw4 = keyword_heuristic("fetch_data", "Fetch data from remote server")
    assert kw4.count("fetch") == 1, "No duplicate keywords"
    assert kw4.count("data") == 1, "No duplicate keywords"
    print(f"  fetch_data -> {kw4}")

    # max_keywords limit
    kw5 = keyword_heuristic("a_b_c_d", "extra words here now please thanks", max_keywords=3)
    assert len(kw5) <= 3, f"Should respect max_keywords=3, got {len(kw5)}"
    print(f"  max_keywords=3 -> {kw5} (limited)")

    # Empty description — only name words
    kw6 = keyword_heuristic("ping_service", "")
    assert "ping" in kw6
    assert "service" in kw6
    print(f"  ping_service (no desc) -> {kw6}")

    # Single-word name
    kw7 = keyword_heuristic("search", "Search for items in catalog")
    assert "search" in kw7
    print(f"  search -> {kw7}")

    # Short words filtered from name (single-char parts)
    kw8 = keyword_heuristic("a_b_read", "Test")
    # "a" and "b" should be filtered (len <= 1)
    for k in kw8:
        assert len(k) > 1, f"Single-char keyword should be filtered: {k}"
    print(f"  a_b_read -> {kw8}")

    print("  keyword_heuristic() passed")


# =============================================================================
# TEST 2: from_json_schema_batch()
# =============================================================================

def test_from_json_schema_batch():
    print("\n" + "="*70)
    print("TEST 2: from_json_schema_batch()")
    print("="*70)

    # Parse envelope with 2 tools
    tools = from_json_schema_batch(SAMPLE_JSON_SCHEMA_ENVELOPE)
    assert len(tools) == 2, f"Expected 2 tools, got {len(tools)}"
    assert tools[0].name == "read_file"
    assert tools[1].name == "write_file"
    print(f"  Envelope parsed: {len(tools)} tools")

    # Auto-generated keywords
    assert len(tools[0].keywords) > 0, "Auto keywords should be generated"
    assert "read" in tools[0].keywords or "file" in tools[0].keywords
    print(f"  Auto keywords for read_file: {tools[0].keywords}")

    # Parameters parsed correctly
    assert len(tools[0].parameters) == 2
    path_param = next(p for p in tools[0].parameters if p.name == "path")
    assert path_param.param_type == "string"
    assert path_param.required is True
    encoding_param = next(p for p in tools[0].parameters if p.name == "encoding")
    assert encoding_param.required is False
    print(f"  Parameters: path (required), encoding (optional)")

    # Single tool schema (no envelope)
    single = from_json_schema_batch(SAMPLE_JSON_SCHEMA_TOOL)
    assert len(single) == 1
    assert single[0].name == "read_file"
    print(f"  Single tool schema: {single[0].name}")

    # List of tools
    tool_list = [SAMPLE_JSON_SCHEMA_TOOL, SAMPLE_JSON_SCHEMA_ENVELOPE["tools"][1]]
    from_list = from_json_schema_batch(tool_list)
    assert len(from_list) == 2
    print(f"  List input: {len(from_list)} tools")

    # auto_keywords=False
    no_kw = from_json_schema_batch(SAMPLE_JSON_SCHEMA_TOOL, auto_keywords=False)
    assert no_kw[0].keywords == [], f"Expected empty keywords, got {no_kw[0].keywords}"
    print(f"  auto_keywords=False: keywords={no_kw[0].keywords}")

    # Invalid envelope raises ValueError
    try:
        from_json_schema_batch({"invalid_key": "value"})
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"  Invalid envelope raises ValueError: {e}")

    print("  from_json_schema_batch() passed")


# =============================================================================
# TEST 3: to_json_schema_batch()
# =============================================================================

def test_to_json_schema_batch():
    print("\n" + "="*70)
    print("TEST 3: to_json_schema_batch()")
    print("="*70)

    tools = [
        ToolDefinition(
            name="read_file",
            description="Read file content",
            parameters=[
                ToolParameter("path", "string", required=True),
            ],
            keywords=["read"],
        ),
        ToolDefinition(
            name="write_file",
            description="Write file content",
            parameters=[
                ToolParameter("path", "string", required=True),
                ToolParameter("content", "string", required=True),
            ],
            keywords=["write"],
        ),
    ]

    envelope = to_json_schema_batch(tools)

    assert "tools" in envelope
    assert len(envelope["tools"]) == 2
    print(f"  Envelope has {len(envelope['tools'])} tools")

    # Each tool has correct structure
    for t in envelope["tools"]:
        assert t["type"] == "function"
        assert "function" in t
        assert "name" in t["function"]
        assert "parameters" in t["function"]
    print(f"  All tools have type=function, function.name, function.parameters")

    # Verify specific tool
    read_tool = envelope["tools"][0]
    assert read_tool["function"]["name"] == "read_file"
    assert "path" in read_tool["function"]["parameters"]["properties"]
    assert "path" in read_tool["function"]["parameters"]["required"]
    print(f"  read_file: path param is required [OK]")

    # JSON serializable
    json_str = json.dumps(envelope, indent=2)
    reparsed = json.loads(json_str)
    assert reparsed == envelope
    print(f"  JSON serializable and round-trips correctly")

    # Empty tool list
    empty = to_json_schema_batch([])
    assert empty == {"tools": []}
    print(f"  Empty list -> {{'tools': []}}")

    print("  to_json_schema_batch() passed")


# =============================================================================
# TEST 4: find_json_tool_blocks()
# =============================================================================

def test_find_json_tool_blocks():
    print("\n" + "="*70)
    print("TEST 4: find_json_tool_blocks()")
    print("="*70)

    # Find single tool in text
    blocks = find_json_tool_blocks(SAMPLE_PROMPT_WITH_JSON)
    assert len(blocks) == 1, f"Expected 1 block, got {len(blocks)}"
    start, end, parsed = blocks[0]
    assert parsed["function"]["name"] == "search_web"
    print(f"  Found 1 tool block: {parsed['function']['name']}")

    # Position is correct
    json_fragment = SAMPLE_PROMPT_WITH_JSON[start:end]
    reparsed = json.loads(json_fragment)
    assert reparsed["function"]["name"] == "search_web"
    print(f"  Position [{start}:{end}] extracts valid JSON")

    # Find envelope in text
    text_with_envelope = f"Tools:\n{json.dumps(SAMPLE_JSON_SCHEMA_ENVELOPE)}\nEnd."
    blocks2 = find_json_tool_blocks(text_with_envelope)
    assert len(blocks2) == 1
    _, _, parsed2 = blocks2[0]
    assert "tools" in parsed2
    assert len(parsed2["tools"]) == 2
    print(f"  Envelope block: {len(parsed2['tools'])} tools inside")

    # No JSON in plain text
    plain_text = "This is just a regular prompt with no tools."
    blocks3 = find_json_tool_blocks(plain_text)
    assert len(blocks3) == 0
    print(f"  Plain text: 0 blocks found")

    # JSON without tool structure is ignored
    non_tool_json = "Here is some data: {\"key\": \"value\", \"nested\": {\"a\": 1}}"
    blocks4 = find_json_tool_blocks(non_tool_json)
    assert len(blocks4) == 0, f"Non-tool JSON should be ignored, got {len(blocks4)}"
    print(f"  Non-tool JSON: correctly ignored")

    # Nested JSON (tool with nested objects in parameters)
    nested_text = """Use this tool:
{"type": "function", "function": {"name": "query_db", "parameters": {"type": "object", "properties": {"filter": {"type": "object", "properties": {"age": {"type": "integer"}, "name": {"type": "string"}}}}}}}"""
    blocks5 = find_json_tool_blocks(nested_text)
    assert len(blocks5) == 1
    assert blocks5[0][2]["function"]["name"] == "query_db"
    print(f"  Deeply nested JSON: correctly parsed")

    print("  find_json_tool_blocks() passed")


# =============================================================================
# TEST 5: migrate_prompt()
# =============================================================================

def test_migrate_prompt():
    print("\n" + "="*70)
    print("TEST 5: migrate_prompt()")
    print("="*70)

    # Migrate prompt with single tool
    new_text, tools = migrate_prompt(SAMPLE_PROMPT_WITH_JSON)
    assert len(tools) == 1
    assert tools[0].name == "search_web"
    print(f"  Migrated {len(tools)} tool(s)")

    # JSON replaced with UCL-Tool syntax
    assert "[[TOOL: search_web]]" in new_text, \
        f"Expected UCL-Tool syntax in output"
    assert '"type": "function"' not in new_text, \
        "JSON Schema should be removed"
    print(f"  JSON Schema replaced with [[TOOL: ...]] syntax")

    # Surrounding text preserved
    assert "You are a helpful assistant" in new_text
    assert "Use these tools when needed" in new_text
    print(f"  Surrounding text preserved")

    # Migrate prompt with envelope (2 tools -> grouped)
    envelope_text = f"Here are tools:\n{json.dumps(SAMPLE_JSON_SCHEMA_ENVELOPE)}\nUse them."
    new_text2, tools2 = migrate_prompt(envelope_text, group=True)
    assert len(tools2) == 2
    assert "[[TOOL: read_file]]" in new_text2
    assert "[[TOOL: write_file]]" in new_text2
    print(f"  Envelope migrated: 2 tools grouped")

    # With grouping, should have single condition wrapper
    assert new_text2.count("^^CONDITION:") == 1, \
        f"Expected 1 condition for grouped tools, got {new_text2.count('^^CONDITION:')}"
    print(f"  Single ^^CONDITION wrapper for grouped tools")

    # Without grouping
    new_text3, tools3 = migrate_prompt(envelope_text, group=False)
    assert len(tools3) == 2
    print(f"  Ungrouped migration: {len(tools3)} tools")

    # No JSON in prompt -> no changes
    plain = "Just a regular prompt with no tools."
    new_text4, tools4 = migrate_prompt(plain)
    assert new_text4 == plain
    assert len(tools4) == 0
    print(f"  No JSON -> no changes, 0 tools returned")

    # auto_keywords=False
    new_text5, tools5 = migrate_prompt(SAMPLE_PROMPT_WITH_JSON, auto_keywords=False)
    assert len(tools5) == 1
    assert tools5[0].keywords == []
    print(f"  auto_keywords=False: tool has no keywords")

    print("  migrate_prompt() passed")


# =============================================================================
# TEST 6: roundtrip_json_to_ucl_to_json()
# =============================================================================

def test_roundtrip_json_to_ucl_to_json():
    print("\n" + "="*70)
    print("TEST 6: roundtrip_json_to_ucl_to_json()")
    print("="*70)

    # Round-trip: JSON -> UCL -> JSON
    result = roundtrip_json_to_ucl_to_json(SAMPLE_JSON_SCHEMA_ENVELOPE)

    assert "tools" in result
    assert len(result["tools"]) == 2
    print(f"  Round-trip produced {len(result['tools'])} tools")

    # Core fields preserved
    read_tool = result["tools"][0]
    assert read_tool["type"] == "function"
    assert read_tool["function"]["name"] == "read_file"
    assert read_tool["function"]["description"] == "Read file content from disk"
    print(f"  Name preserved: {read_tool['function']['name']}")
    print(f"  Description preserved: {read_tool['function']['description']}")

    # Parameters preserved
    props = read_tool["function"]["parameters"]["properties"]
    assert "path" in props
    assert "encoding" in props
    assert props["path"]["type"] == "string"
    print(f"  Parameters preserved: {list(props.keys())}")

    # Required preserved
    required = read_tool["function"]["parameters"]["required"]
    assert "path" in required
    assert "encoding" not in required
    print(f"  Required preserved: {required}")

    # Second tool
    write_tool = result["tools"][1]
    assert write_tool["function"]["name"] == "write_file"
    write_required = write_tool["function"]["parameters"]["required"]
    assert "path" in write_required
    assert "content" in write_required
    print(f"  write_file: both params required [OK]")

    # Single tool round-trip
    single_result = roundtrip_json_to_ucl_to_json(SAMPLE_JSON_SCHEMA_TOOL)
    assert len(single_result["tools"]) == 1
    assert single_result["tools"][0]["function"]["name"] == "read_file"
    print(f"  Single tool round-trip: name preserved")

    print("  roundtrip_json_to_ucl_to_json() passed")


# =============================================================================
# TEST 7: roundtrip_ucl_to_json_to_ucl()
# =============================================================================

def test_roundtrip_ucl_to_json_to_ucl():
    print("\n" + "="*70)
    print("TEST 7: roundtrip_ucl_to_json_to_ucl()")
    print("="*70)

    # Create a UCL tool with keywords
    original = ToolDefinition(
        name="search_web",
        description="Search the internet for information",
        parameters=[
            ToolParameter("query", "string", required=True,
                         description="Search query string"),
            ToolParameter("limit", "integer", required=False,
                         description="Max results to return"),
        ],
        keywords=["search", "web", "internet"],
    )

    # Round-trip: UCL -> JSON -> UCL
    result = roundtrip_ucl_to_json_to_ucl(original)

    # Core fields preserved
    assert result.name == "search_web"
    assert result.description == "Search the internet for information"
    print(f"  Name preserved: {result.name}")
    print(f"  Description preserved: {result.description}")

    # Keywords preserved (passed through)
    assert result.keywords == ["search", "web", "internet"]
    print(f"  Keywords preserved: {result.keywords}")

    # Parameters preserved
    assert len(result.parameters) == 2
    query_param = next(p for p in result.parameters if p.name == "query")
    assert query_param.param_type == "string"
    assert query_param.required is True
    limit_param = next(p for p in result.parameters if p.name == "limit")
    assert limit_param.param_type == "integer"
    assert limit_param.required is False
    print(f"  Parameters preserved: query (string, req), limit (integer, opt)")

    # Render and verify UCL syntax
    rendered = result.render()
    assert "[[TOOL: search_web]]" in rendered
    assert "[[PARAM: query, string, required]]" in rendered
    print(f"  Rendered UCL contains tool and param directives")

    # Tool with no parameters
    minimal = ToolDefinition(
        name="ping",
        description="Check if service is alive",
        parameters=[],
        keywords=["ping", "health"],
    )
    minimal_result = roundtrip_ucl_to_json_to_ucl(minimal)
    assert minimal_result.name == "ping"
    assert minimal_result.keywords == ["ping", "health"]
    assert len(minimal_result.parameters) == 0
    print(f"  Minimal tool (no params) round-trips correctly")

    print("  roundtrip_ucl_to_json_to_ucl() passed")


# =============================================================================
# TEST 8: Edge cases
# =============================================================================

def test_edge_cases():
    print("\n" + "="*70)
    print("TEST 8: Edge Cases")
    print("="*70)

    # Empty envelope
    tools = from_json_schema_batch({"tools": []})
    assert len(tools) == 0
    print(f"  Empty envelope -> 0 tools")

    # Tool with no parameters
    no_params = {
        "type": "function",
        "function": {
            "name": "list_items",
            "description": "List all items",
            "parameters": {"type": "object", "properties": {}}
        }
    }
    parsed = from_json_schema_batch(no_params)
    assert len(parsed) == 1
    assert len(parsed[0].parameters) == 0
    print(f"  Tool with no parameters: parsed correctly")

    # Tool with no description
    no_desc = {
        "type": "function",
        "function": {
            "name": "mystery_tool",
            "parameters": {"type": "object", "properties": {}}
        }
    }
    parsed2 = from_json_schema_batch(no_desc)
    assert parsed2[0].name == "mystery_tool"
    assert parsed2[0].description == ""
    print(f"  Tool with no description: handled")

    # keyword_heuristic with very long name
    long_name = "very_long_tool_name_with_many_parts"
    kw = keyword_heuristic(long_name, "description")
    assert len(kw) <= 6  # max_keywords default
    assert all(len(w) > 1 for w in kw)
    print(f"  Long name '{long_name}' -> {kw}")

    # to_json_schema_batch with single tool
    single = [ToolDefinition("ping", "Ping", [ToolParameter("host", "string", required=True)], ["ping"])]
    env = to_json_schema_batch(single)
    assert len(env["tools"]) == 1
    print(f"  Single tool envelope: correct")

    # migrate_prompt with empty string
    text, tools = migrate_prompt("")
    assert text == ""
    assert tools == []
    print(f"  Empty prompt: no changes")

    # migrate_prompt with only JSON (no surrounding text)
    json_only = json.dumps(SAMPLE_JSON_SCHEMA_TOOL)
    text2, tools2 = migrate_prompt(json_only)
    assert len(tools2) == 1
    assert "[[TOOL: read_file]]" in text2
    print(f"  JSON-only prompt: migrated correctly")

    # Round-trip with enum parameter
    enum_tool = {
        "type": "function",
        "function": {
            "name": "set_color",
            "description": "Set the color",
            "parameters": {
                "type": "object",
                "properties": {
                    "color": {
                        "type": "string",
                        "enum": ["red", "green", "blue"]
                    }
                },
                "required": ["color"]
            }
        }
    }
    result = roundtrip_json_to_ucl_to_json(enum_tool)
    color_prop = result["tools"][0]["function"]["parameters"]["properties"]["color"]
    assert "enum" in color_prop
    assert color_prop["enum"] == ["red", "green", "blue"]
    print(f"  Enum parameter preserved in round-trip")

    print("  Edge cases passed")


# =============================================================================
# TEST 9: Integration — migrate + analyze
# =============================================================================

def test_integration():
    print("\n" + "="*70)
    print("TEST 9: Integration — Full Migration + Analysis Pipeline")
    print("="*70)

    from prompt_analyzer import analyze_prompt

    # Start with a JSON Schema prompt
    original_prompt = """You are a helpful coding assistant.

Available tools:

{"tools": [{"type": "function", "function": {"name": "run_code", "description": "Execute Python code in a sandbox", "parameters": {"type": "object", "properties": {"code": {"type": "string", "description": "Python code to execute"}, "timeout": {"type": "integer", "description": "Timeout in seconds"}}, "required": ["code"]}}}, {"type": "function", "function": {"name": "read_file", "description": "Read a file from the workspace", "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "File path"}, "lines": {"type": "integer", "description": "Number of lines to read"}}, "required": ["path"]}}}]}

Help the user with their coding tasks.
"""

    # Migrate to UCL-Tool
    migrated_text, tools = migrate_prompt(original_prompt)

    assert len(tools) == 2
    tool_names = [t.name for t in tools]
    assert "run_code" in tool_names
    assert "read_file" in tool_names
    print(f"  Migrated {len(tools)} tools: {tool_names}")

    # UCL syntax present
    assert "[[TOOL: run_code]]" in migrated_text
    assert "[[TOOL: read_file]]" in migrated_text
    print(f"  UCL-Tool syntax present in migrated text")

    # JSON Schema removed from migrated text
    assert '"type": "function"' not in migrated_text, \
        "JSON Schema should be completely removed from migrated text"
    print(f"  JSON Schema completely removed")

    # Analyze the migrated prompt
    analysis = analyze_prompt(migrated_text)

    assert analysis.tool_count == 2, \
        f"Analyzer should detect 2 tools, got {analysis.tool_count}"
    print(f"  Analyzer detected {analysis.tool_count} tools")

    assert analysis.tools_with_keywords == 2, \
        f"All tools should have keywords, got {analysis.tools_with_keywords}"
    print(f"  All {analysis.tools_with_keywords} tools have keywords")

    # Architecture should be UCL-TOOL-CONDITIONAL (grouped tools)
    assert analysis.architecture in ("UCL-TOOL-CONDITIONAL", "UCL-TOOL"), \
        f"Expected UCL-TOOL architecture, got {analysis.architecture}"
    print(f"  Architecture: {analysis.architecture}")

    # Compare sizes
    original_size = len(original_prompt)
    migrated_size = len(migrated_text)
    reduction = ((original_size - migrated_size) / original_size) * 100
    print(f"  Size: {original_size} -> {migrated_size} chars ({reduction:.1f}% reduction)")
    assert migrated_size < original_size, "Migrated prompt should be smaller"

    # Surrounding text preserved
    assert "helpful coding assistant" in migrated_text
    assert "Help the user with their coding tasks" in migrated_text
    print(f"  Surrounding prompt text preserved")

    print("  Integration test passed")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "="*70)
    print("  UCL-TOOL EXPORT & MIGRATION TEST SUITE (Phase 6)")
    print("="*70)

    try:
        test_keyword_heuristic()
        test_from_json_schema_batch()
        test_to_json_schema_batch()
        test_find_json_tool_blocks()
        test_migrate_prompt()
        test_roundtrip_json_to_ucl_to_json()
        test_roundtrip_ucl_to_json_to_ucl()
        test_edge_cases()
        test_integration()

        print("\n" + "="*70)
        print("  ALL PHASE 6 TESTS PASSED!")
        print("="*70)
        print("\n  Phase 6 JSON Schema Interop Verified:")
        print("    [ok] keyword_heuristic() — infers keywords from name/desc")
        print("    [ok] from_json_schema_batch() — parses all envelope formats")
        print("    [ok] to_json_schema_batch() — generates valid envelopes")
        print("    [ok] find_json_tool_blocks() — detects JSON in text")
        print("    [ok] migrate_prompt() — replaces JSON with UCL-Tool")
        print("    [ok] roundtrip_json_to_ucl_to_json() — core fields preserved")
        print("    [ok] roundtrip_ucl_to_json_to_ucl() — keywords preserved")
        print("    [ok] Edge cases — empty, no params, enums, long names")
        print("    [ok] Integration — migrate + analyze pipeline works")
        print("\n  Phase 6 Capabilities:")
        print("    • Auto-generate keywords from tool name/description")
        print("    • Parse OpenAI/Anthropic JSON Schema tool definitions")
        print("    • Export UCL-Tool definitions as JSON Schema")
        print("    • Migrate existing prompts in-place")
        print("    • Bidirectional round-trip with fidelity")
        print("\n  Ready for Phase 7: Documentation")

    except AssertionError as e:
        print(f"\n  TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"\n  ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
