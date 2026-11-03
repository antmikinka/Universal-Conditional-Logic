#!/usr/bin/env python3
"""
UCL-Tool Primitives Test Suite
===============================

Comprehensive tests for UCL-Tool extensions to core.py.

Tests cover:
1. ToolParameter - rendering and O_s calculation
2. ToolExample - rendering and O_s calculation
3. ToolDefinition - rendering, condition wrapping, O_s
4. ToolGroup - shared condition rendering, O_s savings
5. calculate_score() - tool bonuses/penalties
6. JSON Schema export/import
7. Backward compatibility

Run: python tests/test_tool_primitives.py
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import (
    ToolParameter, ToolExample, ToolDefinition, ToolGroup,
    UCLToolConstants, calculate_score, DirectiveType
)


def test_tool_parameter():
    """Test ToolParameter rendering and O_s calculation."""
    print("\n" + "="*70)
    print("TEST 1: ToolParameter")
    print("="*70)

    # Test 1.1: Basic required parameter
    param1 = ToolParameter("path", "string", required=True)
    rendered = param1.render()
    print(f"\n1.1 Basic required parameter:")
    print(f"    Rendered: {rendered}")
    assert "[[PARAM: path, string, required]]" in rendered
    print(f"    ✅ Renders correctly")

    os_value = param1.calculate_Os()
    print(f"    O_s: {os_value:.3f}")
    expected_os = UCLToolConstants.PARAM_OS + UCLToolConstants.REQUIRED_PARAM_DISCOUNT
    assert abs(os_value - expected_os) < 0.01
    print(f"    ✅ O_s calculation correct (expected {expected_os:.3f})")

    # Test 1.2: Optional parameter with description
    param2 = ToolParameter(
        "encoding", "string", required=False,
        description="File encoding",
        default="utf-8"
    )
    rendered = param2.render()
    print(f"\n1.2 Optional parameter with description:")
    print(f"    Rendered:\n{rendered}")
    assert "[[PARAM: encoding, string, optional]]" in rendered
    assert "[[DESC: File encoding]]" in rendered
    assert "[[DESC: Default: utf-8]]" in rendered
    print(f"    ✅ Renders with description and default")

    # Test 1.3: Enum parameter
    param3 = ToolParameter(
        "method", "enum", required=True,
        constraints={"values": ["GET", "POST", "PUT"]}
    )
    rendered = param3.render()
    print(f"\n1.3 Enum parameter:")
    print(f"    Rendered: {rendered}")
    os_value = param3.calculate_Os()
    print(f"    O_s: {os_value:.3f}")
    # Should have enum overhead: 0.05 * 3 values = 0.15
    assert os_value > UCLToolConstants.PARAM_OS
    print(f"    ✅ Enum overhead applied correctly")

    # Test 1.4: Invalid type validation
    print(f"\n1.4 Invalid type validation:")
    try:
        bad_param = ToolParameter("test", "invalid_type")
        print(f"    ❌ Should have raised ValueError")
        assert False
    except ValueError as e:
        print(f"    ✅ Correctly raises ValueError: {e}")

    print(f"\n✅ ToolParameter tests passed!")


def test_tool_example():
    """Test ToolExample rendering and O_s calculation."""
    print("\n" + "="*70)
    print("TEST 2: ToolExample")
    print("="*70)

    # Test 2.1: Basic example
    example1 = ToolExample(
        input_text="read_file('/etc/hosts')",
        output_text="127.0.0.1 localhost"
    )
    rendered = example1.render()
    print(f"\n2.1 Basic example:")
    print(f"    Rendered: {rendered}")
    assert "[[EXAMPLE: read_file('/etc/hosts') -> 127.0.0.1 localhost]]" in rendered
    print(f"    ✅ Renders correctly")

    os_value = example1.calculate_Os()
    print(f"    O_s: {os_value:.3f}")
    assert os_value >= UCLToolConstants.EXAMPLE_OS
    print(f"    ✅ O_s calculation correct")

    # Test 2.2: Example with description
    example2 = ToolExample(
        input_text="read_file path=/etc/hosts",
        output_text="127.0.0.1 localhost\n::1 localhost",
        description="Reading hosts file"
    )
    rendered = example2.render()
    print(f"\n2.2 Example with description:")
    print(f"    Rendered:\n{rendered}")
    assert "[[DESC: Example: Reading hosts file]]" in rendered
    assert "[[EXAMPLE:" in rendered
    print(f"    ✅ Renders with description")

    print(f"\n✅ ToolExample tests passed!")


def test_tool_definition():
    """Test ToolDefinition rendering and O_s calculation."""
    print("\n" + "="*70)
    print("TEST 3: ToolDefinition")
    print("="*70)

    # Test 3.1: Basic tool with keywords
    tool1 = ToolDefinition(
        name="read_file",
        description="Reads file content from disk",
        parameters=[
            ToolParameter("path", "string", required=True,
                         description="Path to the file"),
        ],
        keywords=["file", "read", "open"],
        return_type="string"
    )

    rendered = tool1.render()
    print(f"\n3.1 Basic tool with keywords:")
    print(f"    Rendered:\n{rendered}")

    # Verify structure
    assert "[[TOOL: read_file]]" in rendered
    assert "[[DESC: Reads file content from disk]]" in rendered
    assert "[[PARAM: path, string, required]]" in rendered
    assert "[[WHEN: file, read, open]]" in rendered
    assert "^^CONDITION:" in rendered
    assert "^^/CONDITION^^" in rendered
    print(f"    ✅ Renders with condition wrapper")

    os_value = tool1.calculate_Os()
    print(f"    O_s: {os_value:.3f}")
    # Should be: base + param + when
    expected_min = UCLToolConstants.TOOL_BASE_OS + UCLToolConstants.WHEN_OS
    assert os_value >= expected_min
    print(f"    ✅ O_s calculation correct (≥ {expected_min:.3f})")

    # Test 3.2: Tool without keywords (always-on)
    tool2 = ToolDefinition(
        name="always_on_tool",
        description="Always active tool",
        parameters=[],
        keywords=[]  # No keywords!
    )
    rendered = tool2.render()
    print(f"\n3.2 Tool without keywords:")
    print(f"    Rendered:\n{rendered}")
    assert "^^CONDITION:" not in rendered
    print(f"    ✅ No condition wrapper when keywords empty")

    # Test 3.3: Tool with examples
    tool3 = ToolDefinition(
        name="calculator",
        description="Performs calculations",
        parameters=[
            ToolParameter("expression", "string", required=True),
        ],
        keywords=["calculate", "math"],
        examples=[
            ToolExample("2 + 2", "4"),
            ToolExample("10 * 5", "50"),
        ]
    )
    os_value = tool3.calculate_Os()
    print(f"\n3.3 Tool with examples:")
    print(f"    O_s: {os_value:.3f}")
    # Should include example overhead
    assert os_value > UCLToolConstants.TOOL_BASE_OS
    print(f"    ✅ Example overhead included")

    # Test 3.4: Invalid tool name
    print(f"\n3.4 Invalid tool name validation:")
    try:
        bad_tool = ToolDefinition(name="123invalid")
        print(f"    ❌ Should have raised ValueError")
        assert False
    except ValueError as e:
        print(f"    ✅ Correctly raises ValueError: {e}")

    # Test 3.5: JSON Schema export
    print(f"\n3.5 JSON Schema export:")
    json_schema = tool1.to_json_schema_tool()
    print(f"    Exported: {json_schema}")
    assert json_schema["type"] == "function"
    assert json_schema["function"]["name"] == "read_file"
    assert "path" in json_schema["function"]["parameters"]["properties"]
    print(f"    ✅ Exports to JSON Schema correctly")

    print(f"\n✅ ToolDefinition tests passed!")


def test_tool_group():
    """Test ToolGroup rendering and O_s savings."""
    print("\n" + "="*70)
    print("TEST 4: ToolGroup")
    print("="*70)

    # Create 3 related tools
    read_tool = ToolDefinition(
        name="read_file",
        description="Read file",
        parameters=[ToolParameter("path", "string", required=True)],
        keywords=["read"],
        category="file_ops"
    )

    write_tool = ToolDefinition(
        name="write_file",
        description="Write file",
        parameters=[
            ToolParameter("path", "string", required=True),
            ToolParameter("content", "string", required=True),
        ],
        keywords=["write"],
        category="file_ops"
    )

    list_tool = ToolDefinition(
        name="list_dir",
        description="List directory",
        parameters=[ToolParameter("path", "string", required=True)],
        keywords=["list"],
        category="file_ops"
    )

    # Test 4.1: Individual tools (baseline O_s)
    individual_os = (read_tool.calculate_Os() +
                     write_tool.calculate_Os() +
                     list_tool.calculate_Os())
    print(f"\n4.1 Individual tools O_s:")
    print(f"    Total: {individual_os:.3f}")

    # Test 4.2: Grouped tools (should save O_s)
    group = ToolGroup(
        name="file_operations",
        keywords=["file", "read", "write", "directory"],
        tools=[read_tool, write_tool, list_tool],
        description="File system operations"
    )

    rendered = group.render()
    print(f"\n4.2 Grouped tools:")
    print(f"    Rendered:\n{rendered}")

    # Verify single condition wrapper
    assert rendered.count("^^CONDITION:") == 1
    assert rendered.count("^^/CONDITION^^") == 1
    print(f"    ✅ Single condition wrapper for all tools")

    group_os = group.calculate_Os()
    print(f"\n4.3 O_s comparison:")
    print(f"    Individual: {individual_os:.3f}")
    print(f"    Grouped:    {group_os:.3f}")
    print(f"    Savings:    {individual_os - group_os:.3f}")

    # Grouped should be lower (saved duplicate condition overhead)
    assert group_os < individual_os
    savings_pct = ((individual_os - group_os) / individual_os) * 100
    print(f"    ✅ Grouping saves {savings_pct:.1f}% O_s")

    print(f"\n✅ ToolGroup tests passed!")


def test_calculate_score_with_tools():
    """Test calculate_score() with tool definitions."""
    print("\n" + "="*70)
    print("TEST 5: calculate_score() with Tools")
    print("="*70)

    # Test 5.1: Prompt with well-formed tools
    tool1 = ToolDefinition(
        name="read_file",
        description="Read file",
        parameters=[ToolParameter("path", "string", required=True)],
        keywords=["file", "read"],
        examples=[ToolExample("read_file('test.txt')", "content...")]
    )

    tool2 = ToolDefinition(
        name="write_file",
        description="Write file",
        parameters=[
            ToolParameter("path", "string", required=True),
            ToolParameter("content", "string", required=True),
        ],
        keywords=["file", "write"],
    )

    elements = [tool1, tool2]
    score = calculate_score(elements, has_critical=True, critical_at_top=True)

    print(f"\n5.1 Score with well-formed tools:")
    print(f"    Quality: {score.Q_final:.1%}")
    print(f"    Grade: {score.grade}")
    print(f"    O_s: {score.Os:.3f}")
    print(f"    Bonuses: {score.bonuses}")

    # Should have tool bonuses
    assert "tool_keyword_activation" in score.bonuses
    assert "tool_examples" in score.bonuses
    print(f"    ✅ Tool bonuses applied")

    # Test 5.2: Prompt with anti-patterns (no keywords)
    bad_tool = ToolDefinition(
        name="bad_tool",
        description="",  # No description!
        parameters=[],
        keywords=[]  # No keywords!
    )

    elements = [bad_tool]
    score = calculate_score(elements)

    print(f"\n5.2 Score with anti-patterns:")
    print(f"    Penalties: {score.penalties}")
    assert "tools_without_activation" in score.penalties
    assert "tools_without_description" in score.penalties
    print(f"    ✅ Penalties applied for missing keywords/description")

    # Test 5.3: Duplicate tool names
    dup1 = ToolDefinition(name="duplicate", keywords=["test"])
    dup2 = ToolDefinition(name="duplicate", keywords=["test"])

    elements = [dup1, dup2]
    score = calculate_score(elements)

    print(f"\n5.3 Score with duplicate names:")
    print(f"    Penalties: {score.penalties}")
    assert "tool_name_conflicts" in score.penalties
    print(f"    ✅ Penalty applied for duplicate names")

    print(f"\n✅ calculate_score() tests passed!")


def test_backward_compatibility():
    """Test that existing UCL elements still work."""
    print("\n" + "="*70)
    print("TEST 6: Backward Compatibility")
    print("="*70)

    from core import Directive, DirectiveType, Condition, calculate_score

    # Test existing elements still work
    directive = Directive(DirectiveType.REQUIRE, "Test requirement")
    condition = Condition(keywords=["test"], body=[directive])

    elements = [condition]
    score = calculate_score(elements)

    print(f"\n6.1 Existing elements:")
    print(f"    Quality: {score.Q_final:.1%}")
    print(f"    O_s: {score.Os:.3f}")
    print(f"    ✅ Existing elements work unchanged")

    # Test new DirectiveType values don't break enum
    print(f"\n6.2 New DirectiveType values:")
    print(f"    TOOL: {DirectiveType.TOOL.value}")
    print(f"    PARAM: {DirectiveType.PARAM.value}")
    print(f"    WHEN: {DirectiveType.WHEN.value}")
    print(f"    DESC: {DirectiveType.DESC.value}")
    print(f"    EXAMPLE: {DirectiveType.EXAMPLE.value}")
    print(f"    ✅ New enum values accessible")

    print(f"\n✅ Backward compatibility verified!")


def main():
    """Run all tests."""
    print("\n" + "█"*70)
    print("  UCL-TOOL PRIMITIVES TEST SUITE")
    print("  Testing Phase 1 Implementation")
    print("█"*70)

    try:
        test_tool_parameter()
        test_tool_example()
        test_tool_definition()
        test_tool_group()
        test_calculate_score_with_tools()
        test_backward_compatibility()

        print("\n" + "="*70)
        print("✅ ALL TESTS PASSED!")
        print("="*70)
        print("\nPhase 1 Implementation Verified:")
        print("  ✅ ToolParameter renders and calculates O_s correctly")
        print("  ✅ ToolExample renders and calculates O_s correctly")
        print("  ✅ ToolDefinition wraps in conditions, exports to JSON Schema")
        print("  ✅ ToolGroup saves O_s through shared conditions")
        print("  ✅ calculate_score() applies tool bonuses/penalties")
        print("  ✅ Backward compatibility maintained")
        print("\n🎯 Ready for Phase 2: Builder Integration")

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
