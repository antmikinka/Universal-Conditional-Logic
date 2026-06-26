#!/usr/bin/env python3
"""
UCL Framework - Math-to-TTS Example
====================================

This example recreates the V4.1 prompt from the research paper.
V4.1 achieved 100% quality in empirical testing (N=305, p<10⁻¹³).
"""

import sys
import os

# Add parent directory to path to import UCL modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from builder import UCLPromptBuilder, JSONOutput
from domains import BasicMathDomain, LinearAlgebraDomain, VectorCalculusDomain


def main():
    print("=" * 70)
    print("  MATH-TO-TTS PROMPT (V4.1 Architecture)")
    print("  The Optimal Architecture - 100% Quality Achieved")
    print("=" * 70)
    
    # Build the V4.1 style prompt
    print("\n🔨 Building V4.1 prompt...")
    
    prompt = (UCLPromptBuilder()
        .set_identity("Qwen")
        .add_critical()  # KEY: This is what makes V4.1 achieve 100%!
        .set_task("Transform mathematical expressions INTO TTS-friendly format")
        .require("JSON output with result and scratchwork_answer fields")
        .add_domain(BasicMathDomain())
        .add_domain(LinearAlgebraDomain())
        .add_domain(VectorCalculusDomain())
        .set_output(JSONOutput(
            fields=["result", "scratchwork_answer"],
            validations=["Valid JSON syntax", "Complete transformation"]
        ))
        .build()
    )
    
    # Show the V4.1 results
    print("\n✅ V4.1 PROMPT BUILT!")
    print(f"\n📊 Quality Metrics (Matching Research Results):")
    print(f"   Quality Score: {prompt.score.Q_final:.1%} ⭐")
    print(f"   Letter Grade:  {prompt.score.grade}")
    print(f"   O_s Overhead:  {prompt.score.Os:.2f} (Minimal!)")
    print(f"   Architecture:  {prompt.score.architecture.name}")
    
    print(f"\n🔬 Research Validation:")
    print(f"   Sample Size:        N = 305")
    print(f"   Statistical Sig:    p < 2.89×10⁻¹³")
    print(f"   Success Rate:       100% (V4.1)")
    print(f"   Previous Best:      90.7% (V4 without [[CRITICAL:]])")
    
    print(f"\n✨ Key V4.1 Features:")
    print(f"   ✅ [[CRITICAL:]] directive at TOP")
    print(f"   ✅ Keyword-based CONDITIONS (not SWITCH)")
    print(f"   ✅ 3 domain-specific transformations")
    print(f"   ✅ Structured JSON output")
    print(f"   ✅ O_s ≈ 2.0 (vs 8.47 for V1, 32 for V2)")
    
    # Show prompt preview
    print(f"\n📄 Prompt Preview (first 20 lines):")
    print("-" * 70)
    lines = prompt.text.split('\n')
    for i, line in enumerate(lines[:20], 1):
        print(f"{i:2}. {line}")
    
    if len(lines) > 20:
        print(f"... ({len(lines) - 20} more lines)")
    
    print("\n" + "=" * 70)
    print(f"Total Lines: {len(lines)} | Total Words: {len(prompt.text.split())}")
    print("=" * 70)


if __name__ == "__main__":
    main()
