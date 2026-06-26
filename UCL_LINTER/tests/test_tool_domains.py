#!/usr/bin/env python3
"""
UCL-Tool Domains Test Suite (Phase 4)
======================================

Tests for pre-built tool domain templates:
1. ToolDomain base class
2. ToolDomainRegistry
3. FileSystemDomain — 7 tools
4. WebDomain — 5 tools
5. DatabaseDomain — 6 tools
6. APIDomain — 4 tools
7. MCPDomain — 5 tools
8. ShellDomain — 4 tools
9. Builder integration with add_tool_domain()
10. O_s savings validation
11. Analyzer integration with domain tools

Run: PYTHONIOENCODING=utf-8 python tests/test_tool_domains.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import ToolGroup, ToolDefinition, ToolParameter
from tool_domains import (
    ToolDomain, ToolDomainRegistry,
    FileSystemDomain, WebDomain, DatabaseDomain,
    APIDomain, MCPDomain, ShellDomain,
    PlaywrightDomain, ChromaDomain, ClearThoughtDomain,
    MCPServerTemplate,
)
from builder import UCLPromptBuilder
from prompt_analyzer import analyze_prompt


# =============================================================================
# TEST 1: ToolDomain base class
# =============================================================================

def test_base_class():
    print("\n" + "="*70)
    print("TEST 1: ToolDomain Base Class")
    print("="*70)

    domain = FileSystemDomain()

    assert domain.name == "filesystem"
    assert domain.description != ""
    assert len(domain.keywords) > 0
    print(f"  Name: {domain.name}")
    print(f"  Description: {domain.description}")
    print(f"  Keywords: {domain.keywords}")

    # get_tools returns list of ToolDefinition
    tools = domain.get_tools()
    assert len(tools) > 0
    for t in tools:
        assert t.name
        assert t.keywords
    print(f"  Tools: {len(tools)}")

    # create_tool_group returns ToolGroup
    group = domain.create_tool_group()
    assert isinstance(group, ToolGroup)
    assert group.name == "filesystem"
    assert len(group.tools) == len(tools)
    print(f"  create_tool_group() → ToolGroup with {len(group.tools)} tools")

    # O_s methods
    individual = domain.get_individual_Os()
    grouped = domain.get_grouped_Os()
    savings = domain.get_Os_savings()
    print(f"  Individual O_s: {individual:.3f}")
    print(f"  Grouped O_s:    {grouped:.3f}")
    print(f"  Savings:        {savings:.1f}%")
    assert grouped < individual
    assert savings > 0

    print("  Base class passed")


# =============================================================================
# TEST 2: ToolDomainRegistry
# =============================================================================

def test_registry():
    print("\n" + "="*70)
    print("TEST 2: ToolDomainRegistry")
    print("="*70)

    # List all registered domains
    names = ToolDomainRegistry.list_domains()
    print(f"  Registered domains: {names}")
    assert len(names) == 9
    assert "filesystem" in names
    assert "web" in names
    assert "database" in names
    assert "api" in names
    assert "mcp" in names
    assert "shell" in names
    assert "playwright" in names
    assert "chroma" in names
    assert "clear_thought" in names

    # Get by name
    fs = ToolDomainRegistry.get("filesystem")
    assert isinstance(fs, FileSystemDomain)
    print(f"  get('filesystem') → {fs.name} ({fs.get_tool_count()} tools)")

    # Get all
    all_domains = ToolDomainRegistry.get_all()
    assert len(all_domains) == 9
    print(f"  get_all() -> {len(all_domains)} domains")

    # Total tools
    total = ToolDomainRegistry.get_total_tools()
    print(f"  Total tools across all domains: {total}")
    assert total == 71, f"Expected 71, got {total}"

    # Unknown domain raises ValueError
    try:
        ToolDomainRegistry.get("nonexistent")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"  Unknown domain raises ValueError: {e}")

    print("  Registry passed")


# =============================================================================
# TEST 3-8: Each domain
# =============================================================================

def test_filesystem_domain():
    print("\n" + "="*70)
    print("TEST 3: FileSystemDomain (7 tools)")
    print("="*70)

    domain = FileSystemDomain()
    tools = domain.get_tools()
    assert len(tools) == 7

    tool_names = [t.name for t in tools]
    expected = ["read_file", "write_file", "list_dir", "delete_file",
                "move_file", "copy_file", "file_info"]
    for name in expected:
        assert name in tool_names, f"Missing tool: {name}"
    print(f"  All {len(expected)} expected tools present")

    # Each tool has keywords and description
    for t in tools:
        assert t.keywords, f"Tool {t.name} has no keywords"
        assert t.description, f"Tool {t.name} has no description"
        assert len(t.parameters) > 0, f"Tool {t.name} has no parameters"
    print(f"  All tools have keywords, descriptions, and parameters")

    # read_file should have an example
    read_file = next(t for t in tools if t.name == "read_file")
    assert len(read_file.examples) > 0
    print(f"  read_file has {len(read_file.examples)} example(s)")

    savings = domain.get_Os_savings()
    print(f"  Grouping savings: {savings:.1f}%")
    assert savings > 15, f"Expected >15% savings, got {savings:.1f}%"

    print("  FileSystemDomain passed")


def test_web_domain():
    print("\n" + "="*70)
    print("TEST 4: WebDomain (5 tools)")
    print("="*70)

    domain = WebDomain()
    tools = domain.get_tools()
    assert len(tools) == 5

    tool_names = [t.name for t in tools]
    for name in ["http_get", "http_post", "web_search", "fetch_url", "download_file"]:
        assert name in tool_names
    print(f"  All 5 expected tools present")

    # http_get should have an example
    http_get = next(t for t in tools if t.name == "http_get")
    assert len(http_get.examples) > 0
    print(f"  http_get has examples")

    print("  WebDomain passed")


def test_database_domain():
    print("\n" + "="*70)
    print("TEST 5: DatabaseDomain (6 tools)")
    print("="*70)

    domain = DatabaseDomain()
    tools = domain.get_tools()
    assert len(tools) == 6

    tool_names = [t.name for t in tools]
    for name in ["query_sql", "insert_row", "update_row",
                 "delete_row", "create_table", "list_tables"]:
        assert name in tool_names
    print(f"  All 6 expected tools present")

    # query_sql should have an example
    query = next(t for t in tools if t.name == "query_sql")
    assert len(query.examples) > 0
    print(f"  query_sql has examples")

    # list_tables has no parameters
    list_t = next(t for t in tools if t.name == "list_tables")
    assert len(list_t.parameters) == 0
    print(f"  list_tables has no parameters (correct)")

    print("  DatabaseDomain passed")


def test_api_domain():
    print("\n" + "="*70)
    print("TEST 6: APIDomain (4 tools)")
    print("="*70)

    domain = APIDomain()
    tools = domain.get_tools()
    assert len(tools) == 4

    tool_names = [t.name for t in tools]
    for name in ["rest_call", "graphql_query", "webhook_send", "auth_token"]:
        assert name in tool_names
    print(f"  All 4 expected tools present")

    # rest_call should have enum parameter for method
    rest = next(t for t in tools if t.name == "rest_call")
    method_param = next(p for p in rest.parameters if p.name == "method")
    assert method_param.param_type == "enum"
    print(f"  rest_call has enum parameter for method")

    print("  APIDomain passed")


def test_mcp_domain():
    print("\n" + "="*70)
    print("TEST 7: MCPDomain (5 tools)")
    print("="*70)

    domain = MCPDomain()
    tools = domain.get_tools()
    assert len(tools) == 5

    tool_names = [t.name for t in tools]
    for name in ["mcp_call", "mcp_list_tools", "mcp_list_resources",
                 "mcp_read_resource", "mcp_list_prompts"]:
        assert name in tool_names
    print(f"  All 5 MCP protocol tools present")

    # mcp_call should have an example
    call = next(t for t in tools if t.name == "mcp_call")
    assert len(call.examples) > 0
    assert "mcp_call" in call.examples[0].input_text
    print(f"  mcp_call has example showing MCP invocation")

    # All tools should have 'mcp' keyword
    for t in tools:
        assert "mcp" in t.keywords, f"Tool {t.name} missing 'mcp' keyword"
    print(f"  All tools have 'mcp' keyword for activation")

    print("  MCPDomain passed")


def test_shell_domain():
    print("\n" + "="*70)
    print("TEST 8: ShellDomain (4 tools)")
    print("="*70)

    domain = ShellDomain()
    tools = domain.get_tools()
    assert len(tools) == 4

    tool_names = [t.name for t in tools]
    for name in ["run_command", "run_script", "get_env", "set_env"]:
        assert name in tool_names
    print(f"  All 4 expected tools present")

    # run_command should have an example
    cmd = next(t for t in tools if t.name == "run_command")
    assert len(cmd.examples) > 0
    print(f"  run_command has examples")

    print("  ShellDomain passed")


# =============================================================================
# TEST 8b: MCP Server Domains
# =============================================================================

def test_playwright_domain():
    print("\n" + "="*70)
    print("TEST 8b: PlaywrightDomain (20 browser tools)")
    print("="*70)

    domain = PlaywrightDomain()
    tools = domain.get_tools()
    assert len(tools) == 20
    print(f"  {len(tools)} browser automation tools")

    # Check key tools exist
    tool_names = [t.name for t in tools]
    for name in ["browser_navigate", "browser_click", "browser_type",
                 "browser_snapshot", "browser_take_screenshot", "browser_close"]:
        assert name in tool_names, f"Missing tool: {name}"
    print(f"  Key browser tools present: navigate, click, type, snapshot, screenshot, close")

    # browser_navigate has an example
    nav = next(t for t in tools if t.name == "browser_navigate")
    assert len(nav.examples) > 0
    print(f"  browser_navigate has examples")

    # browser_click has enum parameter for button
    click = next(t for t in tools if t.name == "browser_click")
    button_param = next(p for p in click.parameters if p.name == "button")
    assert button_param.param_type == "enum"
    assert "left" in button_param.constraints.get("values", [])
    print(f"  browser_click has enum button param")

    # All tools have keywords
    for t in tools:
        assert len(t.keywords) > 0, f"Tool {t.name} has no keywords"
    print(f"  All tools have activation keywords")

    savings = domain.get_Os_savings()
    print(f"  Grouping savings: {savings:.1f}%")
    assert savings > 20

    print("  PlaywrightDomain passed")


def test_chroma_domain():
    print("\n" + "="*70)
    print("TEST 8c: ChromaDomain (10 vector DB tools)")
    print("="*70)

    domain = ChromaDomain()
    tools = domain.get_tools()
    assert len(tools) == 10
    print(f"  {len(tools)} vector database tools")

    tool_names = [t.name for t in tools]
    for name in ["chroma_create_collection", "chroma_add_documents",
                 "chroma_query_documents", "chroma_delete_documents"]:
        assert name in tool_names, f"Missing tool: {name}"
    print(f"  Key vector DB tools present")

    # chroma_query_documents has similarity search keywords
    query = next(t for t in tools if t.name == "chroma_query_documents")
    assert any("similarity" in kw or "search" in kw or "query" in kw
               for kw in query.keywords)
    print(f"  chroma_query_documents has search keywords")

    # chroma_list_tables has no parameters
    list_colls = next(t for t in tools if t.name == "chroma_list_collections")
    assert len(list_colls.parameters) == 0 or all(
        not p.required for p in list_colls.parameters
    )
    print(f"  chroma_list_collections has optional params only")

    # All tools have keywords
    for t in tools:
        assert len(t.keywords) > 0, f"Tool {t.name} has no keywords"
    print(f"  All tools have activation keywords")

    print("  ChromaDomain passed")


def test_clear_thought_domain():
    print("\n" + "="*70)
    print("TEST 8d: ClearThoughtDomain (10 reasoning tools)")
    print("="*70)

    domain = ClearThoughtDomain()
    tools = domain.get_tools()
    assert len(tools) == 10
    print(f"  {len(tools)} reasoning/analysis tools")

    tool_names = [t.name for t in tools]
    for name in ["sequential_thinking", "debugging_approach",
                 "decision_framework", "mental_model", "scientific_method"]:
        assert name in tool_names, f"Missing tool: {name}"
    print(f"  Key reasoning tools present")

    # sequential_thinking has an example
    thinking = next(t for t in tools if t.name == "sequential_thinking")
    assert len(thinking.examples) > 0
    print(f"  sequential_thinking has examples")

    # debugging_approach has enum parameter
    debug = next(t for t in tools if t.name == "debugging_approach")
    approach_param = next(p for p in debug.parameters if p.name == "approachName")
    assert approach_param.param_type == "enum"
    assert "binary_search" in approach_param.constraints.get("values", [])
    print(f"  debugging_approach has enum approach param")

    # decision_framework has multiple enum params
    decision = next(t for t in tools if t.name == "decision_framework")
    enum_params = [p for p in decision.parameters if p.param_type == "enum"]
    assert len(enum_params) >= 2
    print(f"  decision_framework has {len(enum_params)} enum params")

    print("  ClearThoughtDomain passed")


def test_mcp_server_template():
    print("\n" + "="*70)
    print("TEST 8e: MCPServerTemplate (user-extensible)")
    print("="*70)

    # Create a custom MCP server
    custom = MCPServerTemplate(
        server_name="weather_api",
        description="Weather data MCP server",
        tools=[
            ToolDefinition(
                name="get_weather",
                description="Get current weather for a location",
                parameters=[
                    ToolParameter("location", "string", required=True,
                                 description="City name or coordinates"),
                    ToolParameter("units", "string", required=False,
                                 description="celsius or fahrenheit"),
                ],
                keywords=["weather", "temperature", "forecast"],
            ),
            ToolDefinition(
                name="get_forecast",
                description="Get weather forecast",
                parameters=[
                    ToolParameter("location", "string", required=True),
                    ToolParameter("days", "integer", required=False),
                ],
                keywords=["forecast", "weather", "prediction"],
            ),
        ],
        keywords=["weather", "forecast", "temperature"],
    )

    assert custom.name == "weather_api"
    assert custom.get_tool_count() == 2
    print(f"  Custom server: {custom.name} with {custom.get_tool_count()} tools")

    tools = custom.get_tools()
    assert tools[0].name == "get_weather"
    assert tools[1].name == "get_forecast"
    print(f"  Tools: {tools[0].name}, {tools[1].name}")

    # Works with builder
    builder = (UCLPromptBuilder()
        .set_identity("Weather Agent")
        .add_critical()
        .add_tool_domain(custom)
    )
    built = builder.build()
    assert "weather_api" in built.tool_groups
    assert len(built.tools) == 2
    print(f"  Builder integration: weather_api domain added")

    # O_s calculation works
    grouped = custom.get_grouped_Os()
    individual = custom.get_individual_Os()
    assert grouped < individual
    print(f"  O_s: individual={individual:.3f}, grouped={grouped:.3f}")

    # Empty template
    empty = MCPServerTemplate(server_name="empty_server")
    assert empty.get_tool_count() == 0
    print(f"  Empty template: 0 tools (correct)")

    print("  MCPServerTemplate passed")


# =============================================================================
# TEST 9: Builder Integration
# =============================================================================

def test_builder_integration():
    print("\n" + "="*70)
    print("TEST 9: Builder Integration with add_tool_domain()")
    print("="*70)

    builder = (UCLPromptBuilder()
        .set_identity("Agent")
        .add_critical()
        .set_task("Help the user with various tasks")
        .add_tool_domain(FileSystemDomain())
        .add_tool_domain(WebDomain())
        .add_tool_domain(MCPDomain())
    )

    built = builder.build()

    # All three domain groups present
    assert "filesystem" in built.tool_groups
    assert "web" in built.tool_groups
    assert "mcp" in built.tool_groups
    print(f"  Tool groups: {built.tool_groups}")

    # Total tools
    assert len(built.tools) == 17  # 7 + 5 + 5
    print(f"  Total tools: {len(built.tools)}")

    # Each domain renders as a section tag inside a condition
    text = built.text
    assert "<filesystem>" in text
    assert "<web>" in text
    assert "<mcp>" in text
    print(f"  Domain section tags present in output")

    # Condition wrappers = 3 (one per domain)
    cond_count = text.count("^^CONDITION:")
    assert cond_count == 3, f"Expected 3 conditions, got {cond_count}"
    print(f"  3 condition wrappers (one per domain)")

    print(f"\n  Quality: {built.score.Q_final:.1%}")
    print(f"  O_s:     {built.score.Os:.3f}")
    print(f"  Bonuses: {built.score.bonuses}")

    print("  Builder integration passed")


# =============================================================================
# TEST 10: O_s savings validation
# =============================================================================

def test_os_savings():
    print("\n" + "="*70)
    print("TEST 10: O_s Savings Validation")
    print("="*70)

    # Compare FileSystemDomain: grouped vs individual
    domain = FileSystemDomain()
    individual = domain.get_individual_Os()
    grouped = domain.get_grouped_Os()
    savings = domain.get_Os_savings()

    print(f"  FileSystemDomain:")
    print(f"    Individual: {individual:.3f}")
    print(f"    Grouped:    {grouped:.3f}")
    print(f"    Savings:    {savings:.1f}%")
    assert savings > 15

    # All domains combined
    all_domains = ToolDomainRegistry.get_all()
    total_individual = sum(d.get_individual_Os() for d in all_domains)
    total_grouped = sum(d.get_grouped_Os() for d in all_domains)
    total_savings = ((total_individual - total_grouped) / total_individual) * 100

    print(f"\n  All 9 domains (71 tools):")
    print(f"    Individual: {total_individual:.3f}")
    print(f"    Grouped:    {total_grouped:.3f}")
    print(f"    Savings:    {total_savings:.1f}%")
    assert total_savings > 20, f"Expected >20% total savings, got {total_savings:.1f}%"

    # Compare with JSON Schema estimate (3.0 per tool)
    json_schema_os = 71 * 3.0
    ucl_os = total_grouped
    reduction = ((json_schema_os - ucl_os) / json_schema_os) * 100
    print(f"\n  vs JSON Schema (71 * 3.0 = {json_schema_os:.1f}):")
    print(f"    UCL-Tool O_s: {ucl_os:.3f}")
    print(f"    Reduction:    {reduction:.1f}%")

    print("  O_s savings passed")


# =============================================================================
# TEST 11: Analyzer integration
# =============================================================================

def test_analyzer_integration():
    print("\n" + "="*70)
    print("TEST 11: Analyzer Detects Domain Tools")
    print("="*70)

    # Build a prompt with FileSystemDomain and analyze it
    builder = (UCLPromptBuilder()
        .set_identity("Agent")
        .add_critical()
        .add_tool_domain(FileSystemDomain())
    )
    built = builder.build()

    analysis = analyze_prompt(built.text)

    assert analysis.tool_count == 7
    print(f"  Analyzer detected {analysis.tool_count} tools")

    assert analysis.tools_with_keywords == 7
    print(f"  All {analysis.tools_with_keywords} tools have keywords")

    assert len(analysis.tool_groups_found) >= 1
    print(f"  Tool groups found: {len(analysis.tool_groups_found)}")

    assert analysis.architecture == "UCL-TOOL-CONDITIONAL"
    print(f"  Architecture: {analysis.architecture}")

    # No tool warnings (all tools well-formed)
    tool_warnings = [w for w in analysis.warnings if 'tool' in w['message'].lower()
                     and 'no activation' in w['message'].lower()]
    assert len(tool_warnings) == 0
    print(f"  No tool anti-pattern warnings")

    print("  Analyzer integration passed")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "="*70)
    print("  UCL-TOOL DOMAINS TEST SUITE (Phase 4)")
    print("="*70)

    try:
        test_base_class()
        test_registry()
        test_filesystem_domain()
        test_web_domain()
        test_database_domain()
        test_api_domain()
        test_mcp_domain()
        test_shell_domain()
        test_playwright_domain()
        test_chroma_domain()
        test_clear_thought_domain()
        test_mcp_server_template()
        test_builder_integration()
        test_os_savings()
        test_analyzer_integration()

        print("\n" + "="*70)
        print("  ALL PHASE 4 TESTS PASSED!")
        print("="*70)
        print("\n  Phase 4 Tool Domains Verified:")
        print("    [ok] ToolDomain base class works")
        print("    [ok] ToolDomainRegistry with 9 domains registered")
        print("    [ok] FileSystemDomain: 7 tools")
        print("    [ok] WebDomain: 5 tools")
        print("    [ok] DatabaseDomain: 6 tools")
        print("    [ok] APIDomain: 4 tools")
        print("    [ok] MCPDomain: 5 tools (MCP protocol)")
        print("    [ok] ShellDomain: 4 tools")
        print("    [ok] PlaywrightDomain: 20 tools (browser automation)")
        print("    [ok] ChromaDomain: 10 tools (vector DB)")
        print("    [ok] ClearThoughtDomain: 10 tools (reasoning)")
        print("    [ok] MCPServerTemplate: user-extensible")
        print("    [ok] Builder integration: add_tool_domain()")
        print("    [ok] O_s savings: >20% from grouping")
        print("    [ok] Analyzer detects domain tools")
        print(f"\n  Total: {ToolDomainRegistry.get_total_tools()} pre-built tools")
        print("\n  Ready for Phase 5: O_s Validation")

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
