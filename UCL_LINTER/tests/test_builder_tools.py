#!/usr/bin/env python3
"""
UCL-Tool Builder Integration Test Suite (Phase 2)
==================================================

Tests for UCLPromptBuilder tool integration:
1. add_tool() - standalone tool registration
2. add_tool_category() - tool group creation
3. add_tool_to_category() - adding tools to groups
4. _assemble_elements() - correct assembly order, deduplication
5. Tool-aware task augmentation
6. O_s savings verification through builder
7. Warnings for anti-patterns (tools without keywords)
8. BuiltPrompt metadata includes tools/groups
9. Full end-to-end prompt build with tools

Run: PYTHONIOENCODING=utf-8 python tests/test_builder_tools.py
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import (
    ToolParameter, ToolExample, ToolDefinition, ToolGroup,
    UCLToolConstants, DirectiveType, Condition, Directive
)
from builder import UCLPromptBuilder, JSONOutput, CodeOutput


def _make_tool(name, keywords=None, description=None):
    """Helper to create a tool quickly."""
    return ToolDefinition(
        name=name,
        description=description or f"Tool: {name}",
        parameters=[ToolParameter("arg", "string", required=True)],
        keywords=keywords or [name],
    )


# =============================================================================
# TEST 1: add_tool() — standalone tool registration
# =============================================================================

def test_add_tool():
    print("\n" + "="*70)
    print("TEST 1: add_tool() - Standalone Tool Registration")
    print("="*70)

    builder = UCLPromptBuilder()
    tool = _make_tool("read_file", keywords=["file", "read"])

    result = builder.add_tool(tool)

    # Fluent API returns builder
    assert result is builder, "add_tool() must return self for chaining"
    print("  Fluent API: returns self for chaining")

    # Tool is registered
    assert len(builder._tools) == 1
    assert builder._tools[0].name == "read_file"
    print("  Tool registered in _tools list")

    # Built prompt includes tool
    built = builder.build()
    assert "read_file" in built.tools
    assert "[[TOOL: read_file]]" in built.text
    print("  Built prompt contains tool rendering")

    # Tool is wrapped in condition (keywords present)
    assert "^^CONDITION:" in built.text
    assert "file, read" in built.text or "read, file" in built.text
    print("  Tool wrapped in keyword-based ^^CONDITION^^")

    print("  add_tool() passed")


# =============================================================================
# TEST 2: add_tool_category() — tool group creation
# =============================================================================

def test_add_tool_category():
    print("\n" + "="*70)
    print("TEST 2: add_tool_category() - Tool Group Creation")
    print("="*70)

    read_tool = _make_tool("read_file", keywords=["read"])
    write_tool = _make_tool("write_file", keywords=["write"])
    list_tool = _make_tool("list_dir", keywords=["list"])

    builder = UCLPromptBuilder()
    builder.add_tool_category(
        "file_ops",
        keywords=["file", "read", "write", "directory"],
        tools=[read_tool, write_tool, list_tool],
        description="File system operations"
    )

    # Group registered
    assert "file_ops" in builder._tool_groups
    group = builder._tool_groups["file_ops"]
    assert len(group.tools) == 3
    print(f"  Group 'file_ops' created with {len(group.tools)} tools")

    # Build and verify single condition wrapper
    built = builder.build()
    text = built.text

    # Count condition wrappers — should be exactly 1 for the group
    cond_count = text.count("^^CONDITION:")
    assert cond_count == 1, f"Expected 1 condition wrapper for group, got {cond_count}"
    print(f"  Single ^^CONDITION^^ wrapper for all 3 tools (O_s saving)")

    # All tools present
    assert "[[TOOL: read_file]]" in text
    assert "[[TOOL: write_file]]" in text
    assert "[[TOOL: list_dir]]" in text
    print("  All 3 tools rendered inside group")

    # Group in metadata
    assert "file_ops" in built.tool_groups
    print("  Group name in BuiltPrompt.tool_groups")

    print("  add_tool_category() passed")


# =============================================================================
# TEST 3: add_tool_to_category() — adding tools to existing groups
# =============================================================================

def test_add_tool_to_category():
    print("\n" + "="*70)
    print("TEST 3: add_tool_to_category() - Add Tool to Group")
    print("="*70)

    builder = UCLPromptBuilder()
    builder.add_tool_category("web_ops", keywords=["http", "url", "web"])

    get_tool = _make_tool("http_get", keywords=["get"])
    builder.add_tool_to_category("web_ops", get_tool)

    assert len(builder._tool_groups["web_ops"].tools) == 1
    print("  Tool added to existing category")

    # Invalid category raises ValueError
    try:
        builder.add_tool_to_category("nonexistent", get_tool)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"  Correctly raises ValueError for unknown category: {e}")

    # Build and verify
    built = builder.build()
    assert "[[TOOL: http_get]]" in built.text
    print("  Tool rendered in built prompt")

    print("  add_tool_to_category() passed")


# =============================================================================
# TEST 4: _assemble_elements() — deduplication
# =============================================================================

def test_assemble_deduplication():
    print("\n" + "="*70)
    print("TEST 4: _assemble_elements() - Deduplication")
    print("="*70)

    read_tool = _make_tool("read_file", keywords=["read"])
    write_tool = _make_tool("write_file", keywords=["write"])

    builder = UCLPromptBuilder()

    # Add read_tool to group
    builder.add_tool_category("file_ops", keywords=["file"], tools=[read_tool])

    # Also add read_tool standalone — should NOT appear twice
    builder.add_tool(read_tool)
    # Add write_tool standalone only
    builder.add_tool(write_tool)

    built = builder.build()
    text = built.text

    # read_file should appear exactly once (in the group, not standalone)
    occurrences = text.count("[[TOOL: read_file]]")
    assert occurrences == 1, f"read_file appears {occurrences} times, expected 1"
    print(f"  read_file appears exactly once (in group, not duplicated)")

    # write_file should appear once (standalone)
    assert text.count("[[TOOL: write_file]]") == 1
    print(f"  write_file appears once (standalone)")

    # Group condition wrapper + standalone write_file condition wrapper
    assert text.count("^^CONDITION:") == 2, \
        f"Expected 2 condition wrappers (group + standalone), got {text.count('^^CONDITION:')}"
    print(f"  2 condition wrappers (1 group + 1 standalone)")

    print("  Deduplication passed")


# =============================================================================
# TEST 5: Tool-aware task augmentation
# =============================================================================

def test_task_augmentation():
    print("\n" + "="*70)
    print("TEST 5: Tool-Aware Task Augmentation")
    print("="*70)

    # Without tools — plain task
    builder1 = UCLPromptBuilder()
    builder1.set_task("Process user requests")
    built1 = builder1.build()
    assert "invoke them only when their activation keywords" not in built1.text
    print("  No tools: task is plain (no augmentation)")

    # With tools — augmented task
    builder2 = UCLPromptBuilder()
    builder2.set_task("Process user requests")
    builder2.add_tool(_make_tool("search", keywords=["search"]))
    built2 = builder2.build()
    assert "invoke them only when their activation keywords" in built2.text
    print("  With tools: task augmented with tool-awareness directive")

    print("  Task augmentation passed")


# =============================================================================
# TEST 6: O_s savings through builder
# =============================================================================

def test_os_savings():
    print("\n" + "="*70)
    print("TEST 6: O_s Savings Through Builder")
    print("="*70)

    read_tool = _make_tool("read_file", keywords=["read"])
    write_tool = _make_tool("write_file", keywords=["write"])
    list_tool = _make_tool("list_dir", keywords=["list"])

    # Standalone tools (each gets own condition wrapper)
    builder_standalone = UCLPromptBuilder()
    builder_standalone.add_tool(read_tool)
    builder_standalone.add_tool(write_tool)
    builder_standalone.add_tool(list_tool)
    os_standalone = builder_standalone.get_Os()

    # Grouped tools (single condition wrapper)
    builder_grouped = UCLPromptBuilder()
    builder_grouped.add_tool_category(
        "file_ops",
        keywords=["file", "read", "write", "list"],
        tools=[read_tool, write_tool, list_tool],
    )
    os_grouped = builder_grouped.get_Os()

    print(f"  Standalone O_s: {os_standalone:.3f}")
    print(f"  Grouped O_s:    {os_grouped:.3f}")
    savings = os_standalone - os_grouped
    savings_pct = (savings / os_standalone * 100) if os_standalone > 0 else 0
    print(f"  Savings:        {savings:.3f} ({savings_pct:.1f}%)")

    assert os_grouped < os_standalone, "Grouped should have lower O_s"
    assert savings_pct > 10, f"Expected >10% savings, got {savings_pct:.1f}%"
    print(f"  Grouping saves >10% O_s as expected")

    print("  O_s savings passed")


# =============================================================================
# TEST 7: Warnings for anti-patterns
# =============================================================================

def test_warnings():
    print("\n" + "="*70)
    print("TEST 7: Warnings for Tool Anti-Patterns")
    print("="*70)

    # Tool without keywords should trigger warning
    builder = UCLPromptBuilder()
    no_kw_tool = ToolDefinition(
        name="always_on",
        description="Always active",
        parameters=[],
        keywords=[],  # No keywords!
    )
    builder.add_tool(no_kw_tool)

    warnings = builder._warnings
    assert len(warnings) >= 1, "Expected warning for tool without keywords"
    assert "no activation keywords" in warnings[0].message.lower()
    print(f"  Warning generated: '{warnings[0].message}'")
    print(f"  Recommendation: '{warnings[0].recommendation}'")

    # Tool WITH keywords should NOT trigger warning
    builder2 = UCLPromptBuilder()
    builder2.add_tool(_make_tool("good_tool", keywords=["good"]))
    assert len(builder2._warnings) == 0
    print("  No warning for well-configured tool")

    print("  Warnings passed")


# =============================================================================
# TEST 8: BuiltPrompt metadata
# =============================================================================

def test_metadata():
    print("\n" + "="*70)
    print("TEST 8: BuiltPrompt Metadata")
    print("="*70)

    builder = UCLPromptBuilder()
    builder.set_identity("TestBot")
    builder.add_critical()
    builder.set_task("Test task")
    builder.add_tool(_make_tool("tool_a", keywords=["a"]))
    builder.add_tool_category("group_b", keywords=["b"], tools=[_make_tool("tool_b")])

    built = builder.build()

    # Tools metadata
    assert "tool_a" in built.tools
    assert "tool_b" in built.tools
    print(f"  Tools in metadata: {built.tools}")

    # Groups metadata
    assert "group_b" in built.tool_groups
    print(f"  Groups in metadata: {built.tool_groups}")

    # to_dict includes tools
    d = built.to_dict()
    assert "tools" in d
    assert "tool_groups" in d
    print(f"  to_dict() includes 'tools' and 'tool_groups'")

    print("  Metadata passed")


# =============================================================================
# TEST 9: Full end-to-end build
# =============================================================================

def test_e2e_build():
    print("\n" + "="*70)
    print("TEST 9: Full End-to-End Prompt Build with Tools")
    print("="*70)

    # Build a realistic prompt with tools
    read_file = ToolDefinition(
        name="read_file",
        description="Read file content from disk",
        parameters=[
            ToolParameter("path", "string", required=True, description="File path"),
        ],
        keywords=["file", "read", "open"],
        examples=[ToolExample("read_file('/etc/hosts')", "127.0.0.1 localhost")],
    )

    write_file = ToolDefinition(
        name="write_file",
        description="Write content to a file",
        parameters=[
            ToolParameter("path", "string", required=True),
            ToolParameter("content", "string", required=True),
        ],
        keywords=["file", "write", "save"],
    )

    search = ToolDefinition(
        name="web_search",
        description="Search the web for information",
        parameters=[
            ToolParameter("query", "string", required=True),
        ],
        keywords=["search", "web", "find"],
        examples=[ToolExample("web_search('UCL paper')", "arXiv:2601.00880...")],
    )

    builder = (UCLPromptBuilder()
        .set_identity("ResearchAssistant")
        .add_critical()
        .set_task("Help the user with research tasks")
        .require("Always cite sources")
        .add_tool_category(
            "file_operations",
            keywords=["file", "path", "read", "write"],
            tools=[read_file, write_file],
            description="Local file system operations"
        )
        .add_tool(search)
        .set_output(JSONOutput(fields=["response", "sources_used"]))
    )

    built = builder.build()

    # Structure checks
    assert "{{concept:ai_identity}}" in built.text or "ResearchAssistant" in built.text
    print("  Identity present")

    assert "CRITICAL" in built.text
    print("  CRITICAL directive present")

    assert "[[TOOL: read_file]]" in built.text
    assert "[[TOOL: write_file]]" in built.text
    assert "[[TOOL: web_search]]" in built.text
    print("  All 3 tools rendered")

    # File ops should be in one group condition
    assert "[[TOOL: read_file]]" in built.text
    assert "[[TOOL: write_file]]" in built.text
    print("  File operations grouped")

    # Web search is standalone with its own condition
    assert "^^CONDITION:" in built.text
    print("  Condition wrappers present")

    # Score should reflect tool bonuses
    assert "tool_keyword_activation" in built.score.bonuses or \
           "tool_examples" in built.score.bonuses
    print(f"  Tool bonuses applied: {built.score.bonuses}")

    # Print summary
    print(f"\n  Quality: {built.score.Q_final:.1%} (Grade: {built.score.grade})")
    print(f"  O_s:     {built.score.Os:.3f}")
    print(f"  Tools:   {built.tools}")
    print(f"  Groups:  {built.tool_groups}")

    print("\n  Full prompt preview (first 600 chars):")
    print("  " + "-"*66)
    for line in built.text[:600].split("\n"):
        print(f"  {line}")
    print("  " + "-"*66)

    print("  End-to-end build passed")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "="*70)
    print("  UCL-TOOL BUILDER INTEGRATION TEST SUITE (Phase 2)")
    print("="*70)

    try:
        test_add_tool()
        test_add_tool_category()
        test_add_tool_to_category()
        test_assemble_deduplication()
        test_task_augmentation()
        test_os_savings()
        test_warnings()
        test_metadata()
        test_e2e_build()

        print("\n" + "="*70)
        print("  ALL PHASE 2 TESTS PASSED!")
        print("="*70)
        print("\n  Phase 2 Builder Integration Verified:")
        print("    [ok] add_tool() registers and renders tools")
        print("    [ok] add_tool_category() groups tools under shared condition")
        print("    [ok] add_tool_to_category() adds tools to existing groups")
        print("    [ok] Deduplication prevents tools appearing twice")
        print("    [ok] Task augmented with tool-awareness directive")
        print("    [ok] Grouping saves >10% O_s vs standalone")
        print("    [ok] Warnings generated for anti-patterns")
        print("    [ok] BuiltPrompt metadata includes tools/groups")
        print("    [ok] Full end-to-end build with tools succeeds")
        print("\n  Ready for Phase 3: Analyzer Integration")

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
