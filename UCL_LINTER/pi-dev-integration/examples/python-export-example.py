#!/usr/bin/env python3
"""
UCL-Tool Export Example for pi.dev Integration

Demonstrates the Python side of the UCL-Tool x pi.dev bridge:
1. Export tool domains as JSON for TypeScript consumption
2. Build UCL-optimized prompts directly in Python
3. Generate sample exported configuration files

Usage:
    python pi-dev-integration/examples/python-export-example.py

    # Or from the UCL_LINTER root:
    cd UCL_LINTER
    python -m pi-dev-integration.examples.python-export-example
"""

import sys
import os
import json

# Add parent directories to path for imports
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(_script_dir))  # pi-dev-integration/ -> UCL_LINTER/
sys.path.insert(0, _project_root)

from tool_export import export_for_typescript, keyword_heuristic
from tool_domains import FileSystemDomain, WebDomain
from builder import UCLPromptBuilder


def example_1_basic_export():
    """Export tool domains as JSON for TypeScript."""
    print("=" * 70)
    print("EXAMPLE 1: Basic Export (Python -> JSON)")
    print("=" * 70)
    print()

    # Export filesystem and web domains
    config = export_for_typescript(
        domains=["filesystem", "web"],
        identity="CodeAssistant",
    )

    print(f"Exported {config['metadata']['totalTools']} tools")
    print(f"Domains: {[d['name'] for d in config['domains']]}")
    print(f"Estimated tokens: {config['metadata']['estimatedTokens']}")
    print(f"Overhead reduction: {config['metadata']['overheadReduction']}%")
    print()

    # Show structure of first domain
    domain = config["domains"][0]
    print(f"Domain: {domain['name']}")
    print(f"  Description: {domain['description']}")
    print(f"  Keywords: {domain['keywords']}")
    print(f"  Tools ({len(domain['tools'])}):")
    for tool in domain["tools"]:
        params = [p["name"] for p in tool["parameters"]]
        kw = tool.get("keywords", [])
        print(f"    - {tool['name']}({', '.join(params)})  keywords={kw}")
    print()


def example_2_save_to_file():
    """Export and save to JSON file for TypeScript import."""
    print("=" * 70)
    print("EXAMPLE 2: Save to File")
    print("=" * 70)
    print()

    output_path = os.path.join(
        os.path.dirname(_script_dir),  # pi-dev-integration/
        "ucl-tools-export.json",
    )

    config = export_for_typescript(
        domains=["filesystem", "web"],
        identity="CodeAssistant",
        output_path=output_path,
    )

    file_size = os.path.getsize(output_path)
    print(f"Saved to: {output_path}")
    print(f"File size: {file_size} bytes")
    print(f"Tools: {config['metadata']['totalTools']}")
    print()
    print("In TypeScript, import with:")
    print(f'  import uclConfig from "./ucl-tools-export.json";')
    print()


def example_3_python_prompt_builder():
    """Build a UCL-optimized prompt directly in Python."""
    print("=" * 70)
    print("EXAMPLE 3: Python Prompt Builder")
    print("=" * 70)
    print()

    builder = UCLPromptBuilder()
    builder.set_identity("CodeAssistant")
    builder.add_tool_domain(FileSystemDomain())
    builder.add_tool_domain(WebDomain())
    prompt = builder.build()

    prompt_text = str(prompt)
    print(f"Generated prompt: {len(prompt_text)} characters")
    print(f"Estimated tokens: {len(prompt_text) // 4}")
    print()
    print("--- Prompt preview (first 600 chars) ---")
    print(prompt_text[:600])
    print("...")
    print()


def example_4_keyword_heuristic():
    """Auto-generate keywords from tool names and descriptions."""
    print("=" * 70)
    print("EXAMPLE 4: Keyword Heuristic")
    print("=" * 70)
    print()

    tools = [
        ("read_file", "Read file content from disk"),
        ("http_get", "Send an HTTP GET request to a URL"),
        ("web_search", "Search the web for information"),
        ("execute_code", "Execute Python code in a sandbox"),
        ("send_email", "Send an email to a recipient"),
    ]

    for name, desc in tools:
        keywords = keyword_heuristic(name, desc)
        print(f"  {name:20s} -> {keywords}")
    print()


def example_5_custom_tools():
    """Export with custom tool definitions."""
    print("=" * 70)
    print("EXAMPLE 5: Custom Tools")
    print("=" * 70)
    print()

    from core import ToolDefinition, ToolParameter

    # Define a custom tool
    bmi_tool = ToolDefinition(
        name="calculate_bmi",
        description="Calculate Body Mass Index from weight and height",
        parameters=[
            ToolParameter(name="weight_kg", type="number", required=True,
                          description="Weight in kilograms"),
            ToolParameter(name="height_m", type="number", required=True,
                          description="Height in meters"),
        ],
        keywords=["bmi", "body", "mass", "weight", "height", "calculate"],
    )

    config = export_for_typescript(
        domains=["filesystem"],
        custom_tools=[bmi_tool],
        identity="HealthAssistant",
    )

    print(f"Total tools: {config['metadata']['totalTools']}")
    print(f"Domains: {[d['name'] for d in config['domains']]}")
    print(f"Custom tools: {len(config['customTools'])}")
    if config["customTools"]:
        ct = config["customTools"][0]
        print(f"  Custom tool: {ct['name']}")
        print(f"  Keywords: {ct['keywords']}")
    print()


def main():
    print()
    print("#" * 70)
    print("# UCL-Tool x pi.dev Integration: Python Examples")
    print("#" * 70)
    print()

    example_1_basic_export()
    example_2_save_to_file()
    example_3_python_prompt_builder()
    example_4_keyword_heuristic()
    example_5_custom_tools()

    print("#" * 70)
    print("# All examples complete")
    print("#" * 70)
    print()
    print("Next steps:")
    print("  1. Import ucl-tools-export.json in your TypeScript project")
    print("  2. Use formatUCLToolsForPrompt() to format for prompt injection")
    print("  3. Use buildSystemPrompt() with uclTools for full integration")
    print()
    print("See: pi-dev-integration/examples/typescript-usage-example.ts")
    print()


if __name__ == "__main__":
    main()
