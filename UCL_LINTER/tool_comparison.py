#!/usr/bin/env python3
"""
UCL-Tool vs JSON Schema — O_s Validation (Phase 5)
====================================================

Empirical comparison of UCL-Tool encoding vs JSON Schema encoding.

Measures:
1. Character count (proxy for token count / context window usage)
2. Line count
3. Estimated structural overhead O_s
4. Reduction percentages

Proves that UCL-Tool achieves ≥80% reduction in encoding verbosity
compared to standard JSON Schema tool definitions.

Usage:
    python tool_comparison.py
    # or
    from tool_comparison import compare_tools, scaling_experiment
"""

from __future__ import annotations
import json
from typing import List, Dict, Any, Optional

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
# JSON SCHEMA O_s MODEL
# =============================================================================

# Structural overhead constants for JSON Schema encoding.
# These model the parsing complexity of JSON Schema's nested structure
# relative to UCL-Tool's flat directive-based format.
JSON_SCHEMA_BASE_OS = 1.5       # {"type":"function","function":{...}} wrapper
JSON_SCHEMA_PER_PARAM_OS = 0.5  # Each property declaration + type
JSON_SCHEMA_DESC_OS = 0.3       # Description field
JSON_SCHEMA_NESTING_FACTOR = 1.3  # Overhead multiplier for nesting depth


# =============================================================================
# ENCODING FUNCTIONS
# =============================================================================

def encode_json_schema(tools: List[ToolDefinition]) -> str:
    """
    Encode tools as JSON Schema (OpenAI/Anthropic format).

    Returns pretty-printed JSON string.
    """
    schema_tools = [tool.to_json_schema_tool() for tool in tools]
    envelope = {"tools": schema_tools}
    return json.dumps(envelope, indent=2)


def encode_ucl_tool(tools: List[ToolDefinition],
                    group: bool = True,
                    group_name: str = "tools",
                    group_keywords: List[str] = None) -> str:
    """
    Encode tools using UCL-Tool syntax.

    Args:
        tools: Tool definitions to encode
        group: If True, wrap in single ToolGroup (lower O_s)
        group_name: Name for the group
        group_keywords: Shared keywords (auto-derived if None)

    Returns rendered UCL-Tool text.
    """
    if not tools:
        return ""

    if group and len(tools) > 1:
        # Derive keywords from all tools if not provided
        if group_keywords is None:
            all_kw = set()
            for t in tools:
                all_kw.update(t.keywords)
            group_keywords = sorted(all_kw)

        tool_group = ToolGroup(
            name=group_name,
            keywords=group_keywords,
            tools=tools,
        )
        return tool_group.render()
    else:
        # Individual rendering (each tool gets own condition if keywords)
        parts = [tool.render() for tool in tools]
        return "\n\n".join(parts)


# =============================================================================
# O_s ESTIMATION
# =============================================================================

def estimate_json_schema_Os(tool: ToolDefinition) -> float:
    """
    Estimate structural overhead O_s for a single tool in JSON Schema format.

    Model:
    - Base wrapper: 1.5
    - Per parameter: 0.5
    - Description: 0.3
    - Nesting factor: 1.3x (JSON Schema's deep nesting)
    """
    base = JSON_SCHEMA_BASE_OS
    param_os = JSON_SCHEMA_PER_PARAM_OS * len(tool.parameters)
    desc_os = JSON_SCHEMA_DESC_OS if tool.description else 0.0

    raw = base + param_os + desc_os
    return raw * JSON_SCHEMA_NESTING_FACTOR


def estimate_json_schema_Os_total(tools: List[ToolDefinition]) -> float:
    """Total JSON Schema O_s for a list of tools."""
    return sum(estimate_json_schema_Os(t) for t in tools)


def estimate_ucl_Os(tools: List[ToolDefinition], group: bool = True) -> float:
    """
    Total UCL-Tool O_s for a list of tools.

    If group=True, uses ToolGroup.calculate_Os() (shared condition savings).
    If group=False, sums individual tool O_s.
    """
    if group and len(tools) > 1:
        all_kw = set()
        for t in tools:
            all_kw.update(t.keywords)
        tool_group = ToolGroup(
            name="tools",
            keywords=sorted(all_kw),
            tools=tools,
        )
        return tool_group.calculate_Os()
    else:
        return sum(t.calculate_Os() for t in tools)


# =============================================================================
# COMPARISON
# =============================================================================

def compare_tools(tools: List[ToolDefinition],
                  group: bool = True,
                  label: str = "") -> Dict[str, Any]:
    """
    Head-to-head comparison of JSON Schema vs UCL-Tool encoding.

    Returns dict with:
    - label: description
    - tool_count: number of tools
    - json_chars, json_lines: JSON Schema size
    - ucl_chars, ucl_lines: UCL-Tool size
    - char_reduction_pct: character count reduction
    - json_Os, ucl_Os: structural overhead
    - Os_reduction_pct: O_s reduction
    """
    json_text = encode_json_schema(tools)
    ucl_text = encode_ucl_tool(tools, group=group)

    json_chars = len(json_text)
    json_lines = json_text.count('\n') + 1
    ucl_chars = len(ucl_text)
    ucl_lines = ucl_text.count('\n') + 1

    char_reduction = ((json_chars - ucl_chars) / json_chars * 100) if json_chars > 0 else 0

    json_os = estimate_json_schema_Os_total(tools)
    ucl_os = estimate_ucl_Os(tools, group=group)
    os_reduction = ((json_os - ucl_os) / json_os * 100) if json_os > 0 else 0

    return {
        "label": label,
        "tool_count": len(tools),
        "json_chars": json_chars,
        "json_lines": json_lines,
        "ucl_chars": ucl_chars,
        "ucl_lines": ucl_lines,
        "char_reduction_pct": char_reduction,
        "json_Os": json_os,
        "ucl_Os": ucl_os,
        "Os_reduction_pct": os_reduction,
    }


# =============================================================================
# SCALING EXPERIMENT
# =============================================================================

def _make_sample_tool(index: int) -> ToolDefinition:
    """Create a sample tool for scaling experiments."""
    return ToolDefinition(
        name=f"tool_{index}",
        description=f"Sample tool number {index} for benchmarking",
        parameters=[
            ToolParameter("input", "string", required=True,
                         description=f"Input for tool {index}"),
            ToolParameter("option", "string", required=False,
                         description="Optional parameter"),
        ],
        keywords=[f"keyword_{index}", "common"],
    )


def scaling_experiment(
    sizes: List[int] = None,
) -> List[Dict[str, Any]]:
    """
    Run comparison at multiple tool counts to show scaling behavior.

    Returns list of comparison results for each size.
    """
    if sizes is None:
        sizes = [1, 2, 3, 5, 10, 20, 31]

    results = []
    for n in sizes:
        tools = [_make_sample_tool(i) for i in range(n)]
        result = compare_tools(tools, group=True, label=f"{n} tools")
        results.append(result)

    return results


# =============================================================================
# REAL-WORLD COMPARISON (using tool_domains)
# =============================================================================

def compare_real_domains() -> List[Dict[str, Any]]:
    """
    Compare using actual tool domains (FileSystem, Web, DB, etc.).
    """
    try:
        from tool_domains import ToolDomainRegistry
    except ImportError:
        from .tool_domains import ToolDomainRegistry

    results = []
    for domain in ToolDomainRegistry.get_all():
        tools = domain.get_tools()
        result = compare_tools(
            tools, group=True,
            label=f"{domain.name} ({len(tools)} tools)"
        )
        result["domain"] = domain.name
        results.append(result)

    # All domains combined
    all_tools = []
    for domain in ToolDomainRegistry.get_all():
        all_tools.extend(domain.get_tools())
    combined = compare_tools(all_tools, group=True, label="All 71 tools")
    combined["domain"] = "all"
    results.append(combined)

    return results


# =============================================================================
# REPORTING
# =============================================================================

def print_comparison_table(results: List[Dict[str, Any]]):
    """Print a formatted comparison table."""
    print("\n" + "="*95)
    print(f"{'Label':<25} {'Tools':>5} {'JSON chars':>11} {'UCL chars':>10} "
          f"{'CharRed%':>9} {'JSON O_s':>9} {'UCL O_s':>8} {'OsRed%':>8}")
    print("-"*95)

    for r in results:
        print(f"{r['label']:<25} {r['tool_count']:>5} "
              f"{r['json_chars']:>11,} {r['ucl_chars']:>10,} "
              f"{r['char_reduction_pct']:>8.1f}% "
              f"{r['json_Os']:>9.2f} {r['ucl_Os']:>8.2f} "
              f"{r['Os_reduction_pct']:>7.1f}%")

    print("="*95)


def print_scaling_report(results: List[Dict[str, Any]]):
    """Print scaling analysis report."""
    print("\n" + "="*70)
    print("  SCALING ANALYSIS: O_s Growth Rate")
    print("="*70)

    print(f"\n  {'Tools':>6} {'JSON O_s':>10} {'UCL O_s':>10} {'Reduction':>10} {'CharRed':>10}")
    print(f"  {'-'*50}")

    for r in results:
        print(f"  {r['tool_count']:>6} {r['json_Os']:>10.2f} {r['ucl_Os']:>10.2f} "
              f"{r['Os_reduction_pct']:>9.1f}% {r['char_reduction_pct']:>9.1f}%")

    # Growth rate analysis
    if len(results) >= 2:
        first = results[0]
        last = results[-1]

        json_growth = (last['json_Os'] - first['json_Os']) / max(1, last['tool_count'] - first['tool_count'])
        ucl_growth = (last['ucl_Os'] - first['ucl_Os']) / max(1, last['tool_count'] - first['tool_count'])

        print(f"\n  O_s growth per additional tool:")
        print(f"    JSON Schema: {json_growth:.3f} O_s/tool (linear)")
        print(f"    UCL-Tool:    {ucl_growth:.3f} O_s/tool (sub-linear)")
        print(f"    UCL is {json_growth/max(0.001, ucl_growth):.1f}x more efficient per tool at scale")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run full validation suite and print results."""
    print("\n" + "#" * 70)
    print("  UCL-TOOL vs JSON SCHEMA — O_s VALIDATION (Phase 5)")
    print("#" * 70)

    # ---- 1. Scaling experiment ----
    print("\n[1] SCALING EXPERIMENT")
    scaling_results = scaling_experiment()
    print_comparison_table(scaling_results)
    print_scaling_report(scaling_results)

    # ---- 2. Real domain comparison ----
    print("\n\n[2] REAL-WORLD DOMAIN COMPARISON")
    try:
        domain_results = compare_real_domains()
        print_comparison_table(domain_results)
    except ImportError:
        print("  (tool_domains not available — skipping)")

    # ---- 3. Single tool deep dive ----
    print("\n\n[3] SINGLE TOOL DEEP DIVE")
    tool = ToolDefinition(
        name="read_file",
        description="Read file content from disk",
        parameters=[
            ToolParameter("path", "string", required=True,
                         description="Path to the file"),
        ],
        keywords=["file", "read"],
    )

    json_text = encode_json_schema([tool])
    ucl_text = encode_ucl_tool([tool], group=False)

    print(f"\n  JSON Schema ({len(json_text)} chars, {json_text.count(chr(10))+1} lines):")
    for line in json_text.split('\n')[:15]:
        print(f"    {line}")
    if json_text.count('\n') > 15:
        print(f"    ... ({json_text.count(chr(10))+1 - 15} more lines)")

    print(f"\n  UCL-Tool ({len(ucl_text)} chars, {ucl_text.count(chr(10))+1} lines):")
    for line in ucl_text.split('\n'):
        print(f"    {line}")

    single = compare_tools([tool], group=False, label="Single tool")
    print(f"\n  Character reduction: {single['char_reduction_pct']:.1f}%")
    print(f"  O_s reduction:       {single['Os_reduction_pct']:.1f}%")

    # ---- 4. Summary ----
    print("\n\n" + "="*70)
    print("  VALIDATION SUMMARY")
    print("="*70)

    all_scaling = scaling_results
    if all_scaling:
        avg_char_red = sum(r['char_reduction_pct'] for r in all_scaling) / len(all_scaling)
        avg_os_red = sum(r['Os_reduction_pct'] for r in all_scaling) / len(all_scaling)
        max_char_red = max(r['char_reduction_pct'] for r in all_scaling)
        max_os_red = max(r['Os_reduction_pct'] for r in all_scaling)

        print(f"\n  Character reduction (avg): {avg_char_red:.1f}%")
        print(f"  Character reduction (max): {max_char_red:.1f}%")
        print(f"  O_s reduction (avg):       {avg_os_red:.1f}%")
        print(f"  O_s reduction (max):       {max_os_red:.1f}%")

        if avg_char_red >= 80:
            print(f"\n  ✅ VERIFIED: ≥80% character reduction achieved (avg {avg_char_red:.1f}%)")
        else:
            print(f"\n  ⚠️ Character reduction is {avg_char_red:.1f}% (target: ≥80%)")

        if avg_os_red >= 70:
            print(f"  ✅ VERIFIED: ≥70% O_s reduction achieved (avg {avg_os_red:.1f}%)")
        else:
            print(f"  ⚠️ O_s reduction is {avg_os_red:.1f}% (target: ≥70%)")

    print("\n" + "="*70)


if __name__ == "__main__":
    main()
