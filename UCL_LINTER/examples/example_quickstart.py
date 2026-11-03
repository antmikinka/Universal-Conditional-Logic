#!/usr/bin/env python3
"""
UCL Framework Quick Start Example
===================================

This example demonstrates the easiest way to get started with UCL.
It creates a simple prompt and shows the quality scoring.
"""

import sys
import os

# Add parent directory to path to import UCL modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from builder import UCLPromptBuilder, JSONOutput
from domains import LinearAlgebraDomain


def main():
    print("=" * 70)
    print("  UCL FRAMEWORK - QUICK START EXAMPLE")
    print("=" * 70)
    
    # Build a simple optimized prompt
    print("\n📝 Building an optimized prompt...")
    
    prompt = (UCLPromptBuilder()
        .set_identity("MathAssistant")
        .add_critical()  # +9.3% quality bonus!
        .set_task("Solve linear algebra problems")
        .add_domain(LinearAlgebraDomain())
        .set_output(JSONOutput(fields=["solution", "explanation"]))
        .build()
    )
    
    # Show quality metrics
    print("\n✅ PROMPT BUILT SUCCESSFULLY!")
    print(f"\n📊 Quality Metrics:")
    print(f"   Quality Score: {prompt.score.Q_final:.1%}")
    print(f"   Letter Grade:  {prompt.score.grade}")
    print(f"   O_s Overhead:  {prompt.score.Os:.2f}")
    print(f"   Architecture:  {prompt.score.architecture.name}")
    
    # Show warnings (if any)
    if prompt.score.warnings:
        print(f"\n⚠️  Warnings: {len(prompt.score.warnings)}")
        for w in prompt.score.warnings:
            print(f"   - {w.message}")
    else:
        print(f"\n✨ No warnings - optimal prompt!")
    
    # Show the prompt text
    print(f"\n📄 Generated Prompt Preview:")
    print("-" * 70)
    lines = prompt.text.split('\n')
    preview_lines = lines[:15] if len(lines) > 15 else lines
    print('\n'.join(preview_lines))
    if len(lines) > 15:
        print(f"... ({len(lines) - 15} more lines)")
    
    print("\n" + "=" * 70)
    print("🎉 Quick start complete! See ucl_demo.py for more examples.")
    print("=" * 70)


if __name__ == "__main__":
    main()
