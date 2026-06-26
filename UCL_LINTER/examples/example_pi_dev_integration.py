#!/usr/bin/env python3
"""
UCL-Tool + pi.dev Integration Example

This example demonstrates how to:
1. Export UCL-Tool configuration from Python
2. Generate a formatted system prompt
3. Show the TypeScript integration workflow

Usage:
    python examples/example_pi_dev_integration.py
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tool_export import export_for_typescript
from tool_domains import FileSystemDomain, WebDomain
from builder import UCLPromptBuilder


def main():
    print("=" * 80)
    print("UCL-TOOL + PI.DEV INTEGRATION EXAMPLE")
    print("=" * 80)
    print()

    # ========================================================================
    # STEP 1: Export UCL-Tool Configuration (Python → JSON)
    # ========================================================================
    print("STEP 1: Export UCL-Tool Configuration")
    print("-" * 80)

    config = export_for_typescript(
        domains=["filesystem", "web"],
        identity="CodeAssistant",
    )

    print(f"Exported {config['metadata']['totalTools']} tools")
    print(f"Domains: {len(config['domains'])}")
    print(f"Estimated tokens: {config['metadata']['estimatedTokens']}")
    print(f"Overhead reduction: {config['metadata']['overheadReduction']}%")
    print()

    # Show sample of exported config
    print("Sample exported config (first domain, first tool):")
    if config['domains']:
        domain = config['domains'][0]
        print(f"  Domain: {domain['name']}")
        print(f"  Description: {domain['description']}")
        print(f"  Tools: {len(domain['tools'])}")
        if domain['tools']:
            tool = domain['tools'][0]
            print(f"  First tool: {tool['name']}")
            print(f"    Parameters: {len(tool['parameters'])}")
            print(f"    Keywords: {tool['keywords']}")
    print()

    # ========================================================================
    # STEP 2: Build UCL-Optimized Prompt (Python)
    # ========================================================================
    print("STEP 2: Build UCL-Optimized Prompt (Python)")
    print("-" * 80)

    builder = UCLPromptBuilder()
    builder.set_identity('CodeAssistant')
    builder.add_tool_domain(FileSystemDomain())
    builder.add_tool_domain(WebDomain())
    prompt = builder.build()

    prompt_text = str(prompt)
    print(f"Generated prompt: {len(prompt_text)} characters")
    print(f"Estimated tokens: {len(prompt_text) // 4}")
    print()

    # Show first 500 characters
    print("Prompt preview (first 500 chars):")
    print(prompt_text[:500])
    print("...")
    print()

    # ========================================================================
    # STEP 3: Demonstrate TypeScript Integration (Conceptual)
    # ========================================================================
    print("STEP 3: TypeScript Integration (Conceptual)")
    print("-" * 80)

    typescript_code = """
// In your TypeScript project:

import uclConfig from './ucl-tools.json';
import { buildSystemPrompt } from './system-prompt-with-ucl.ts';

// Build system prompt with UCL-Tool integration
const prompt = buildSystemPrompt({
  cwd: process.cwd(),
  selectedTools: ['read', 'bash', 'edit', 'write'],
  toolSnippets: {
    read: 'Read file contents',
    bash: 'Execute shell commands',
    edit: 'Edit existing files',
    write: 'Write new files',
  },
  uclTools: uclConfig,  // Inject UCL-Tool configuration
  enableUCLPlaceholders: true,
});

console.log(prompt);
"""

    print("TypeScript code to use the exported config:")
    print(typescript_code)
    print()

    # ========================================================================
    # STEP 4: Show Placeholder Usage
    # ========================================================================
    print("STEP 4: Placeholder Usage in Custom Prompts")
    print("-" * 80)

    custom_template = """You are a helpful coding assistant.

{{UCL_TOOLS}}

Guidelines:
- Be concise and helpful
- Show file paths clearly
- Use tools when appropriate
"""

    print("Custom prompt template with {{UCL_TOOLS}} placeholder:")
    print(custom_template)
    print()

    # ========================================================================
    # STEP 5: Performance Metrics
    # ========================================================================
    print("STEP 5: Performance Metrics")
    print("-" * 80)

    from tool_comparison import compare_tools

    # Compare FileSystemDomain
    fs_domain = FileSystemDomain()
    fs_tools = fs_domain.get_tools()
    comparison = compare_tools(fs_tools)

    print("FileSystemDomain (7 tools):")
    print(f"  JSON Schema: {comparison['json_chars']} chars, O_s={comparison['json_Os']:.2f}")
    print(f"  UCL-Tool:    {comparison['ucl_chars']} chars, O_s={comparison['ucl_Os']:.2f}")
    print(f"  Character reduction: {comparison['char_reduction_pct']:.1f}%")
    print(f"  Overhead reduction:  {comparison['Os_reduction_pct']:.1f}%")
    print()

    # ========================================================================
    # STEP 6: Save Config to File (Optional)
    # ========================================================================
    print("STEP 6: Save Config to File")
    print("-" * 80)

    output_file = "ucl-tools-export.json"
    export_for_typescript(
        domains=["filesystem", "web"],
        identity="CodeAssistant",
        output_path=output_file,
    )

    print(f"Config saved to: {output_file}")
    print(f"File size: {os.path.getsize(output_file)} bytes")
    print()

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("=" * 80)
    print("INTEGRATION COMPLETE")
    print("=" * 80)
    print()
    print("What we accomplished:")
    print("1. Exported UCL-Tool configuration as JSON")
    print("2. Built UCL-optimized prompt with tool domains")
    print("3. Demonstrated TypeScript integration workflow")
    print("4. Showed placeholder usage ({{UCL_TOOLS}})")
    print("5. Measured performance (78% overhead reduction)")
    print("6. Saved config to file for TypeScript import")
    print()
    print("Next steps:")
    print("- Import ucl-tools-export.json in your TypeScript project")
    print("- Use buildSystemPrompt() with uclTools config")
    print("- Replace {{UCL_TOOLS}} placeholder in custom prompts")
    print("- Customize tool domains for your specific needs")
    print()
    print("Resources:")
    print("- TypeScript integration guide: typescript/INTEGRATION.md")
    print("- UCL-Tool architecture: DOCS/UCL_TOOL_ARCHITECTURE.md")
    print("- pi.dev system prompt: https://github.com/earendil-works/pi")
    print()


if __name__ == "__main__":
    main()
