#!/usr/bin/env python3
"""
UCL-Tool Analyzer Test Suite (Phase 3)
=======================================

Tests for prompt_analyzer.py tool detection and analysis:
1. detect_tools() — state-machine parser
2. detect_tool_groups() — section tag parsing
3. detect_json_schema_tools() — JSON Schema detection
4. estimate_tool_Os() — tool overhead calculation
5. analyze_prompt() with tools — full analysis
6. Tool-specific warnings and suggestions
7. Architecture classification with tools
8. Report generation with tool section
9. Backward compatibility — non-tool prompts unchanged

Run: PYTHONIOENCODING=utf-8 python tests/test_analyzer_tools.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prompt_analyzer import (
    PromptPatternDetector, PromptAnalysis,
    analyze_prompt, ucl_score
)


# =============================================================================
# TEST 1: detect_tools() — state-machine parser
# =============================================================================

def test_detect_tools():
    print("\n" + "="*70)
    print("TEST 1: detect_tools() — State-Machine Parser")
    print("="*70)

    prompt = """
^^CONDITION: {{concept:user_input}} CONTAINS "file" OR "read"^^
[[TOOL: read_file]]
[[DESC: Read file content from disk]]
[[PARAM: path, string, required]]
[[WHEN: file, read]]
[[EXAMPLE: read_file('/etc/hosts') -> content...]]
^^/CONDITION^^

[[TOOL: calculator]]
[[DESC: Performs math calculations]]
[[PARAM: expression, string, required]]
[[WHEN: calculate, math]]
[[EXAMPLE: 2+2 -> 4]]
[[EXAMPLE: 10*5 -> 50]]
"""

    detector = PromptPatternDetector(prompt)
    tools = detector.detect_tools()

    assert len(tools) == 2, f"Expected 2 tools, got {len(tools)}"
    print(f"  Detected {len(tools)} tools")

    # Tool 1: read_file
    t1 = tools[0]
    assert t1['name'] == 'read_file'
    assert len(t1['params']) == 1
    assert t1['params'][0]['name'] == 'path'
    assert t1['params'][0]['type'] == 'string'
    assert t1['params'][0]['required'] == True
    assert 'file' in t1['keywords']
    assert 'read' in t1['keywords']
    assert len(t1['descriptions']) == 1
    assert len(t1['examples']) == 1
    print(f"  read_file: 1 param, {len(t1['keywords'])} keywords, 1 example")

    # Tool 2: calculator
    t2 = tools[1]
    assert t2['name'] == 'calculator'
    assert len(t2['params']) == 1
    assert len(t2['examples']) == 2  # Two examples!
    print(f"  calculator: 1 param, {len(t2['keywords'])} keywords, 2 examples")

    # count_tools quick check
    assert detector.count_tools() == 2
    print(f"  count_tools() = 2")

    print("  detect_tools() passed")


# =============================================================================
# TEST 2: detect_tool_groups()
# =============================================================================

def test_detect_tool_groups():
    print("\n" + "="*70)
    print("TEST 2: detect_tool_groups() — Section Tag Parsing")
    print("="*70)

    prompt = """
^^CONDITION: {{concept:user_input}} CONTAINS "file"^^
<file_operations>
[[DESC: File system operations]]
    [[TOOL: read_file]]
    [[PARAM: path, string, required]]
    [[WHEN: read]]
    [[TOOL: write_file]]
    [[PARAM: path, string, required]]
    [[PARAM: content, string, required]]
    [[WHEN: write]]
    [[TOOL: list_dir]]
    [[PARAM: path, string, required]]
    [[WHEN: list]]
</file_operations>
^^/CONDITION^^
"""

    detector = PromptPatternDetector(prompt)
    groups = detector.detect_tool_groups()

    assert len(groups) == 1, f"Expected 1 group, got {len(groups)}"
    print(f"  Detected {len(groups)} group(s)")

    g = groups[0]
    assert g['name'] == 'file_operations'
    assert len(g['tools']) == 3
    assert 'read_file' in g['tools']
    assert 'write_file' in g['tools']
    assert 'list_dir' in g['tools']
    print(f"  <file_operations>: {len(g['tools'])} tools")

    # Tools should be detected with group association
    tools = detector.detect_tools()
    for t in tools:
        assert t['in_group'] == 'file_operations', \
            f"Tool {t['name']} not associated with group"
    print(f"  All tools correctly associated with group")

    print("  detect_tool_groups() passed")


# =============================================================================
# TEST 3: detect_json_schema_tools()
# =============================================================================

def test_detect_json_schema_tools():
    print("\n" + "="*70)
    print("TEST 3: detect_json_schema_tools() — JSON Schema Detection")
    print("="*70)

    prompt = """
You have access to the following tools:

{
  "type": "function",
  "function": {
    "name": "read_file",
    "description": "Read a file",
    "parameters": {
      "type": "object",
      "properties": {
        "path": {"type": "string"}
      },
      "required": ["path"]
    }
  }
}

{
  "type": "function",
  "function": {
    "name": "write_file",
    "description": "Write a file",
    "parameters": {}
  }
}
"""

    detector = PromptPatternDetector(prompt)
    json_tools = detector.detect_json_schema_tools()

    assert len(json_tools) == 2
    assert 'read_file' in json_tools
    assert 'write_file' in json_tools
    print(f"  Detected {len(json_tools)} JSON Schema tools: {json_tools}")

    assert detector.count_json_schema_tools() == 2
    print(f"  count_json_schema_tools() = 2")

    # Architecture should be JSON-SCHEMA
    arch = detector.detect_architecture()
    assert arch == "JSON-SCHEMA", f"Expected JSON-SCHEMA, got {arch}"
    print(f"  Architecture: {arch}")

    print("  detect_json_schema_tools() passed")


# =============================================================================
# TEST 4: estimate_tool_Os()
# =============================================================================

def test_estimate_tool_Os():
    print("\n" + "="*70)
    print("TEST 4: estimate_tool_Os() — Tool Overhead Calculation")
    print("="*70)

    # Minimal tool (no keywords, no params)
    prompt_minimal = """
[[TOOL: simple_tool]]
"""
    d1 = PromptPatternDetector(prompt_minimal)
    os_minimal = d1.estimate_tool_Os()
    print(f"  Minimal tool O_s: {os_minimal:.3f}")
    assert abs(os_minimal - 0.40) < 0.01, f"Expected 0.40, got {os_minimal}"

    # Full tool (params, keywords, description, example)
    prompt_full = """
[[TOOL: read_file]]
[[DESC: Read file content]]
[[PARAM: path, string, required]]
[[WHEN: file, read]]
[[EXAMPLE: read_file('x') -> content]]
"""
    d2 = PromptPatternDetector(prompt_full)
    os_full = d2.estimate_tool_Os()
    # Expected: 0.40 (base) + 0.13 (required param) + 0.50 (WHEN) + 0.08 (DESC) + 0.30 (EXAMPLE)
    expected = 0.40 + 0.13 + 0.50 + 0.08 + 0.30
    print(f"  Full tool O_s: {os_full:.3f} (expected ~{expected:.3f})")
    assert abs(os_full - expected) < 0.05

    # Grouped tools should have lower O_s than individual
    prompt_group = """
^^CONDITION: {{concept:user_input}} CONTAINS "file"^^
<file_ops>
    [[TOOL: read_file]]
    [[PARAM: path, string, required]]
    [[WHEN: read]]
    [[TOOL: write_file]]
    [[PARAM: path, string, required]]
    [[WHEN: write]]
</file_ops>
^^/CONDITION^^
"""
    d3 = PromptPatternDetector(prompt_group)
    os_group = d3.estimate_tool_Os()
    print(f"  Grouped (2 tools) O_s: {os_group:.3f}")

    # Compare with individual (no group)
    prompt_individual = """
[[TOOL: read_file]]
[[PARAM: path, string, required]]
[[WHEN: read]]

[[TOOL: write_file]]
[[PARAM: path, string, required]]
[[WHEN: write]]
"""
    d4 = PromptPatternDetector(prompt_individual)
    os_individual = d4.estimate_tool_Os()
    print(f"  Individual (2 tools) O_s: {os_individual:.3f}")

    assert os_group < os_individual, "Grouped should have lower O_s"
    savings = (os_individual - os_group) / os_individual * 100
    print(f"  Grouping saves: {savings:.1f}%")

    print("  estimate_tool_Os() passed")


# =============================================================================
# TEST 5: analyze_prompt() with tools
# =============================================================================

def test_analyze_with_tools():
    print("\n" + "="*70)
    print("TEST 5: analyze_prompt() — Full Analysis with Tools")
    print("="*70)

    prompt = """
[[CRITICAL: Return ONLY JSON output]]

[[LLM: Help the user with file operations]]

^^CONDITION: {{concept:user_input}} CONTAINS "file" OR "read" OR "write"^^
<file_operations>
[[DESC: File system tools]]
    [[TOOL: read_file]]
    [[DESC: Read file content from disk]]
    [[PARAM: path, string, required]]
    [[WHEN: file, read]]
    [[EXAMPLE: read_file('/etc/hosts') -> content...]]

    [[TOOL: write_file]]
    [[DESC: Write content to a file]]
    [[PARAM: path, string, required]]
    [[PARAM: content, string, required]]
    [[WHEN: file, write]]
</file_operations>
^^/CONDITION^^
"""

    analysis = analyze_prompt(prompt, "test_tool_prompt")

    print(f"  Quality: {analysis.quality_score:.1%} (Grade: {analysis.grade})")
    print(f"  Architecture: {analysis.architecture}")
    print(f"  O_s: {analysis.Os:.3f}")
    print(f"  Tool count: {analysis.tool_count}")
    print(f"  Tools with keywords: {analysis.tools_with_keywords}")
    print(f"  Tools with examples: {analysis.tools_with_examples}")
    print(f"  Tool groups: {len(analysis.tool_groups_found)}")

    # Assertions
    assert analysis.tool_count == 2
    assert analysis.tools_with_keywords == 2
    assert analysis.tools_with_examples == 1
    assert len(analysis.tool_groups_found) == 1
    assert analysis.architecture == "UCL-TOOL-CONDITIONAL"
    assert analysis.has_critical == True
    assert analysis.critical_at_top == True

    print(f"  Architecture: {analysis.architecture}")

    # to_dict includes tool data
    d = analysis.to_dict()
    assert 'tool_count' in d
    assert 'tools_found' in d
    assert 'tool_groups_found' in d
    print(f"  to_dict() includes tool data")

    print("  analyze_prompt() with tools passed")


# =============================================================================
# TEST 6: Tool-specific warnings
# =============================================================================

def test_tool_warnings():
    print("\n" + "="*70)
    print("TEST 6: Tool-Specific Warnings")
    print("="*70)

    # Tool without keywords
    prompt1 = """
[[TOOL: always_on]]
[[DESC: Always active tool]]
[[PARAM: x, string, required]]
"""
    analysis1 = analyze_prompt(prompt1)
    high_warnings = [w for w in analysis1.warnings if w['severity'] == 'HIGH']
    kw_warnings = [w for w in high_warnings if 'no activation keywords' in w['message']]
    assert len(kw_warnings) >= 1, "Expected warning for tool without keywords"
    print(f"  Warning for no keywords: '{kw_warnings[0]['message']}'")

    # Tool without description
    prompt2 = """
[[TOOL: no_desc]]
[[PARAM: x, string, required]]
[[WHEN: test]]
"""
    analysis2 = analyze_prompt(prompt2)
    med_warnings = [w for w in analysis2.warnings if w['severity'] == 'MEDIUM']
    desc_warnings = [w for w in med_warnings if 'no description' in w['message']]
    assert len(desc_warnings) >= 1, "Expected warning for tool without description"
    print(f"  Warning for no description: '{desc_warnings[0]['message']}'")

    # JSON Schema tools
    prompt3 = """
{"type": "function", "function": {"name": "json_tool", "parameters": {}}}
"""
    analysis3 = analyze_prompt(prompt3)
    json_warnings = [w for w in analysis3.warnings if 'JSON Schema' in w['message']]
    assert len(json_warnings) >= 1
    print(f"  Warning for JSON Schema: '{json_warnings[0]['message']}'")

    # Duplicate tool names
    prompt4 = """
[[TOOL: duplicate]]
[[WHEN: test]]
[[TOOL: duplicate]]
[[WHEN: test]]
"""
    analysis4 = analyze_prompt(prompt4)
    dup_warnings = [w for w in analysis4.warnings if 'duplicate' in w['message'].lower()]
    assert len(dup_warnings) >= 1
    print(f"  Warning for duplicates: '{dup_warnings[0]['message']}'")

    print("  Tool warnings passed")


# =============================================================================
# TEST 7: Architecture classification
# =============================================================================

def test_architecture_classification():
    print("\n" + "="*70)
    print("TEST 7: Architecture Classification with Tools")
    print("="*70)

    # UCL-TOOL-CONDITIONAL (tools + conditions)
    p1 = """
^^CONDITION: {{concept:user_input}} CONTAINS "test"^^
[[TOOL: my_tool]]
[[WHEN: test]]
^^/CONDITION^^
"""
    a1 = analyze_prompt(p1)
    assert a1.architecture == "UCL-TOOL-CONDITIONAL"
    print(f"  Tools + conditions → {a1.architecture}")

    # UCL-TOOL (tools without conditions)
    p2 = """
[[TOOL: always_on]]
[[WHEN: test]]
"""
    a2 = analyze_prompt(p2)
    assert a2.architecture == "UCL-TOOL"
    print(f"  Tools without conditions → {a2.architecture}")

    # JSON-SCHEMA
    p3 = """
{"type": "function", "function": {"name": "tool1"}}
"""
    a3 = analyze_prompt(p3)
    assert a3.architecture == "JSON-SCHEMA"
    print(f"  JSON Schema tools → {a3.architecture}")

    # Existing architectures still work
    p4 = """
^^CONDITION: {{concept:user_input}} CONTAINS "math"^^
[[LLM: Solve math problems]]
^^/CONDITION^^
"""
    a4 = analyze_prompt(p4)
    assert a4.architecture == "KEYWORD"
    print(f"  Conditions without tools → {a4.architecture}")

    p5 = """
^^SWITCH: mode^^
^^CASE: A^^
^^CASE: B^^
^^/SWITCH^^
"""
    a5 = analyze_prompt(p5)
    assert a5.architecture == "SWITCH"
    print(f"  Switch prompt → {a5.architecture}")

    print("  Architecture classification passed")


# =============================================================================
# TEST 8: Report generation
# =============================================================================

def test_report_generation():
    print("\n" + "="*70)
    print("TEST 8: Report Generation with Tool Section")
    print("="*70)

    prompt = """
[[CRITICAL: Return JSON]]
^^CONDITION: {{concept:user_input}} CONTAINS "file"^^
[[TOOL: read_file]]
[[DESC: Read file]]
[[PARAM: path, string, required]]
[[WHEN: file, read]]
[[EXAMPLE: read_file('x') -> content]]
^^/CONDITION^^
"""

    analysis = analyze_prompt(prompt)
    report = analysis.to_report()

    # Report should include tool section
    assert "TOOL ANALYSIS" in report
    print("  Report contains TOOL ANALYSIS section")

    # Report should show tool details
    assert "read_file" in report
    assert "UCL-Tool definitions: 1" in report
    print("  Report shows tool details")

    # Print first part of report
    print("\n  Report preview:")
    for line in report.split('\n')[:25]:
        print(f"    {line}")

    print("  Report generation passed")


# =============================================================================
# TEST 9: Backward compatibility
# =============================================================================

def test_backward_compatibility():
    print("\n" + "="*70)
    print("TEST 9: Backward Compatibility — Non-Tool Prompts")
    print("="*70)

    # Simple prompt without tools should work exactly as before
    prompt = """
[[CRITICAL: Return only JSON]]
[[LLM: Solve math problems]]
^^CONDITION: {{concept:user_input}} CONTAINS "algebra"^^
[[APPLY: Use algebraic methods]]
^^/CONDITION^^
"""

    analysis = analyze_prompt(prompt)

    assert analysis.tool_count == 0
    assert analysis.tools_found == []
    assert analysis.tool_groups_found == []
    assert analysis.json_schema_tools == []
    assert analysis.tool_os_contribution == 0.0
    print("  Non-tool prompt: all tool fields are empty/zero")

    # Score should be reasonable
    assert 0.0 <= analysis.quality_score <= 1.0
    print(f"  Quality score: {analysis.quality_score:.1%}")

    # Architecture should be KEYWORD (not tool-related)
    assert analysis.architecture == "KEYWORD"
    print(f"  Architecture: {analysis.architecture}")

    # ucl_score() simple function still works
    score = ucl_score(prompt)
    assert 0.0 <= score <= 1.0
    print(f"  ucl_score() = {score:.1%}")

    print("  Backward compatibility passed")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "="*70)
    print("  UCL-TOOL ANALYZER TEST SUITE (Phase 3)")
    print("="*70)

    try:
        test_detect_tools()
        test_detect_tool_groups()
        test_detect_json_schema_tools()
        test_estimate_tool_Os()
        test_analyze_with_tools()
        test_tool_warnings()
        test_architecture_classification()
        test_report_generation()
        test_backward_compatibility()

        print("\n" + "="*70)
        print("  ALL PHASE 3 TESTS PASSED!")
        print("="*70)
        print("\n  Phase 3 Analyzer Integration Verified:")
        print("    [ok] detect_tools() state-machine parser works")
        print("    [ok] detect_tool_groups() finds section tags")
        print("    [ok] detect_json_schema_tools() detects verbose format")
        print("    [ok] estimate_tool_Os() calculates overhead correctly")
        print("    [ok] analyze_prompt() includes full tool analysis")
        print("    [ok] Tool-specific warnings generated")
        print("    [ok] Architecture classification extended")
        print("    [ok] Report includes TOOL ANALYSIS section")
        print("    [ok] Backward compatibility maintained")
        print("\n  Ready for Phase 4: Tool Domains")

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
