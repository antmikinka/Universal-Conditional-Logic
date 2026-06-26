#!/usr/bin/env python3
"""
Example: Using the UCL Prompt Analyzer
=======================================

Demonstrates how to analyze any prompt text using UCL metrics.
"""

import sys
import os

# Add parent directory to path to import UCL modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prompt_analyzer import ucl_score, analyze_prompt, PromptAnalysis

# =============================================================================
# EXAMPLE 1: Simple Scoring
# =============================================================================

print("=" * 70)
print("EXAMPLE 1: Simple Quick Scoring")
print("=" * 70)

# Any prompt text
my_prompt = """
You are a helpful assistant.
Process the user's query and provide a clear response.
"""

score = ucl_score(my_prompt)
print(f"\nPrompt Quality: {score:.1%}")
print("(Simple one-line scoring!)")

# =============================================================================
# EXAMPLE 2: Detailed Analysis
# =============================================================================

print("\n" + "=" * 70)
print("EXAMPLE 2: Detailed Analysis")
print("=" * 70)

better_prompt = """
<system_role>
[[CRITICAL: Your ONLY output is JSON. Begin immediately with {]]
{{concept:ai_identity}} = "Assistant"
[[LLM: Process user queries efficiently]]
[[REQUIRE: Accuracy >= 95%]]
</system_role>

^^CONDITION: {{concept:problem_content}} CONTAINS "math" OR "equation"^^
    [[APPLY: Mathematical reasoning]]
    [[TRANSFORM: {{concept:fraction}} TO "numerator over denominator"]]
^^/CONDITION^^

<output_specification>
[[REQUIRE: JSON validity]]
[[REQUIRE: JSON fields: result, confidence]]
</output_specification>
"""

analysis = analyze_prompt(better_prompt, prompt_name="Example V4-style")

print(f"\nQuality Score: {analysis.quality_score:.1%}")
print(f"Grade: {analysis.grade}")
print(f"O_s: {analysis.Os:.2f}")
print(f"Architecture: {analysis.architecture}")
print(f"Has [[CRITICAL:]]: {analysis.has_critical}")
print(f"Warnings: {len(analysis.warnings)}")

# =============================================================================
# EXAMPLE 3: Print Full Report
# =============================================================================

print("\n" + "=" * 70)
print("EXAMPLE 3: Full Text Report")
print("=" * 70)

analysis.print_report()

# =============================================================================
# EXAMPLE 4: Export Results
# =============================================================================

print("\n" + "=" * 70)
print("EXAMPLE 4: Exporting Results")
print("=" * 70)

# Export as JSON
analysis.export("analysis_report.json")
print("✅ Exported to: analysis_report.json")

# Export as text
analysis.export("analysis_report.txt")
print("✅ Exported to: analysis_report.txt")

# =============================================================================
# EXAMPLE 5: Comparing Prompts
# =============================================================================

print("\n" + "=" * 70)
print("EXAMPLE 5: Comparing Multiple Prompts")
print("=" * 70)

# Bad prompt (SWITCH architecture)
bad_prompt = """
^^SWITCH: question_type^^
    ^^CASE: math^^
        Solve the math problem.
    ^^/CASE^^
    ^^CASE: physics^^
        Solve the physics problem.
    ^^/CASE^^
    ^^CASE: chemistry^^
        Solve the chemistry problem.
    ^^/CASE^^
^^/SWITCH^^
"""

# Good prompt (keyword CONDITIONS)
good_prompt = """
[[CRITICAL: Your ONLY output is the solution]]
^^CONDITION: {{concept:problem_content}} CONTAINS "math"^^
    [[APPLY: Mathematical reasoning]]
^^/CONDITION^^
^^CONDITION: {{concept:problem_content}} CONTAINS "physics"^^
    [[APPLY: Physics principles]]
^^/CONDITION^^
"""

bad_analysis = analyze_prompt(bad_prompt, "Bad (SWITCH)")
good_analysis = analyze_prompt(good_prompt, "Good (Conditions)")

print(f"\n{bad_analysis.prompt_name}:")
print(f"  Quality: {bad_analysis.quality_score:.1%}")
print(f"  O_s: {bad_analysis.Os:.2f}")
print(f"  Warnings: {len(bad_analysis.warnings)}")

print(f"\n{good_analysis.prompt_name}:")
print(f"  Quality: {good_analysis.quality_score:.1%}")
print(f"  O_s: {good_analysis.Os:.2f}")
print(f"  Warnings: {len(good_analysis.warnings)}")

improvement = good_analysis.quality_score - bad_analysis.quality_score
print(f"\n✨ Improvement: +{improvement:.1%}")

# =============================================================================
# EXAMPLE 6: Analyzing from File
# =============================================================================

print("\n" + "=" * 70)
print("EXAMPLE 6: Analyzing from File")
print("=" * 70)

# Create a sample file
sample_prompt = """[[CRITICAL: Return ONLY JSON]]
{{concept:ai_identity}} = "Qwen"
[[LLM: Transform math to TTS format]]
"""

with open("sample_prompt.txt", "w") as f:
    f.write(sample_prompt)

from prompt_analyzer import analyze_prompt_file
file_analysis = analyze_prompt_file("sample_prompt.txt")

print(f"\nAnalyzed: sample_prompt.txt")
print(f"Quality: {file_analysis.quality_score:.1%}")

print("\n" + "=" * 70)
print("🎉 All Examples Complete!")
print("=" * 70)
print("\nKey Functions:")
print("  ucl_score(text) - Quick scoring")
print(" analyze_prompt(text) - Detailed analysis")
print("  analyze_prompt_file(path) - Analyze from file")
print("  analysis.export(path) - Save results")
print("\nCLI Usage:")
print("  python prompt_analyzer.py my_prompt.txt")
print("=" * 70)
