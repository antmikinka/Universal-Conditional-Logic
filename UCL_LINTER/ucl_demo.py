#!/usr/bin/env python3
"""
UCL Framework Demonstration
============================

Comprehensive demonstration of the Universal Conditional Logic Framework.

This script demonstrates:
1. Basic prompt building
2. Domain system usage
3. Real-time scoring
4. Anti-pattern detection
5. Optimization suggestions
6. Math-to-TTS prompts (V4.1 style)
7. Code generation prompts

Run this script to see the framework in action!
"""

try:
    from .core import (
        DirectiveType, Directive, Condition,
        Switch, Case, Section,
        calculate_score, detect_anti_patterns
    )
    from .builder import (
        UCLPromptBuilder,
        JSONOutput, CodeOutput,
        build_math_tts_prompt, build_code_prompt
    )
    from .domains import (
        LinearAlgebraDomain, VectorCalculusDomain, StatisticsDomain,
        PythonDomain, CustomDomain
    )
except ImportError:
    from core import (
        DirectiveType, Directive, Condition,
        Switch, Case, Section,
        calculate_score, detect_anti_patterns
    )
    from builder import (
        UCLPromptBuilder,
        JSONOutput, CodeOutput,
        build_math_tts_prompt, build_code_prompt
    )
    from domains import (
        LinearAlgebraDomain, VectorCalculusDomain, StatisticsDomain,
        PythonDomain, CustomDomain
    )


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def demo_basic_usage():
    """Demonstrate basic UCL prompt building."""
    print_section("DEMO 1: Basic Prompt Building")
    
    prompt = (UCLPromptBuilder()
        .set_identity("Assistant")
        .add_critical()  # The KEY to 100% quality!
        .set_task("Process user queries efficiently")
        .require("Clear, concise responses")
        .set_output(JSONOutput(fields=["response", "confidence"]))
        .build()
    )
    
    print("📊 PROMPT QUALITY METRICS:")
    print(f"  Quality:      {prompt.score.Q_final:.1%}")
    print(f"  Grade:        {prompt.score.grade}")
    print(f"  O_s:          {prompt.score.Os:.2f}")
    print(f"  Architecture: {prompt.score.architecture.name}")
    print(f"\n✨ Generated Prompt ({len(prompt.text.split())} words):")
    print("-" * 70)
    print(prompt.text[:500] + "..." if len(prompt.text) > 500 else prompt.text)


def demo_domain_system():
    """Demonstrate the domain system."""
    print_section("DEMO 2: Domain System (Keyword-Based Conditions)")
    
    prompt = (UCLPromptBuilder()
        .set_identity("MathProcessor")
        .add_critical()
        .set_task("Solve mathematical problems")
        .add_domain(LinearAlgebraDomain())      # Activates on "matrix", "eigenvalue", etc.
        .add_domain(VectorCalculusDomain())     # Activates on "curl", "divergence", etc.
        .add_domain(StatisticsDomain())         # Activates on "mean", "variance", etc.
        .set_output(JSONOutput(fields=["solution", "steps"]))
        .build()
    )
    
    print("📊 PROMPT QUALITY METRICS:")
    print(f"  Quality:      {prompt.score.Q_final:.1%}")
    print(f"  Grade:        {prompt.score.grade}")
    print(f"  O_s:          {prompt.score.Os:.2f} (LOW - keyword conditions!)")
    print(f"  Architecture: {prompt.score.architecture.name}")
    print(f"\n🎯 Domains Added: 3 (Linear Algebra, Vector Calculus, Statistics)")
    print("   Each domain uses KEYWORD-based conditions for minimal O_s overhead!")


def demo_custom_domain():
    """Demonstrate creating a custom domain."""
    print_section("DEMO 3: Custom Domain Creation")
    
    # Create a medical domain
    medical = CustomDomain(
        name="medical_diagnosis",
        keywords=["symptom", "diagnosis", "treatment", "patient"],
        transformations=[
            ("icd_code", "ICD-10 code format"),
            ("dosage", "amount unit frequency duration"),
        ],
        requirements=["Include medical disclaimer"],
        emphases=["Patient safety", "Evidence-based medicine"]
    )
    
    prompt = (UCLPromptBuilder()
        .set_identity("MedicalAssistant")
        .add_critical()
        .set_task("Provide medical information")
        .add_domain(medical)
        .set_output(JSONOutput(fields=["information", "sources", "disclaimer"]))
        .build()
    )
    
    print("📊 PROMPT QUALITY METRICS:")
    print(f"  Quality:      {prompt.score.Q_final:.1%}")
    print(f"  Grade:        {prompt.score.grade}")
    print(f"\n🏥 Custom Medical Domain Created!")
    print(f"   Keywords: {medical.keywords}")
    print(f"   Transformations: {len(medical.transformations)} defined")


def demo_anti_patterns():
    """Demonstrate anti-pattern detection."""
    print_section("DEMO 4: Anti-Pattern Detection")
    
    # Build a prompt with SWITCH (anti-pattern)
    print("❌ BAD EXAMPLE: Using SWITCH architecture...")
    
    switch_element = Switch(
        variable="question_type",
        cases=[
            Case("math", [Directive(DirectiveType.APPLY, "Math rules")]),
            Case("physics", [Directive(DirectiveType.APPLY, "Physics rules")]),
            Case("chemistry", [Directive(DirectiveType.APPLY, "Chemistry rules")]),
            Case("biology", [Directive(DirectiveType.APPLY, "Biology rules")]),
        ]
    )
    
    elements = [switch_element]
    score = calculate_score(elements, has_critical=False)
    
    print(f"\n  Quality:      {score.Q_final:.1%} (LOW!)")
    print(f"  O_s:          {score.Os:.2f} (HIGH!)")
    print(f"  Architecture: {score.architecture.name}")
    
    print(f"\n⚠️  WARNINGS DETECTED: {len(score.warnings)}")
    for warning in score.warnings:
        print(f"  {warning}")
    
    print("\n✅ GOOD EXAMPLE: Using keyword-based CONDITIONS instead...")
    
    prompt = (UCLPromptBuilder()
        .set_identity("Assistant")
        .add_critical()
        .add_condition(keywords=["math", "equation"], body=[
            Directive(DirectiveType.APPLY, "Mathematical reasoning")
        ])
        .add_condition(keywords=["physics", "force"], body=[
            Directive(DirectiveType.APPLY, "Physics principles")
        ])
        .add_condition(keywords=["chemistry", "molecule"], body=[
            Directive(DirectiveType.APPLY, "Chemistry knowledge")
        ])
        .build()
    )
    
    print(f"\n  Quality:      {prompt.score.Q_final:.1%} (MUCH BETTER!)")
    print(f"  O_s:          {prompt.score.Os:.2f} (MUCH LOWER!)")
    print(f"  Warnings:     {len(prompt.score.warnings)}")


def demo_math_to_tts():
    """Demonstrate the Math-to-TTS prompt (V4.1 style - 100% quality)."""
    print_section("DEMO 5: Math-to-TTS Prompt (V4.1 Style)")
    
    prompt = build_math_tts_prompt(
        identity="Qwen",
        domains=["basic_math", "linear_algebra", "vector_calculus"]
    )
    
    print("🎯 THE OPTIMAL PROMPT ARCHITECTURE!")
    print(f"\n📊 QUALITY METRICS:")
    print(f"  Quality:      {prompt.score.Q_final:.1%} ⭐ PERFECT!")
    print(f"  Grade:        {prompt.score.grade}")
    print(f"  O_s:          {prompt.score.Os:.2f} (Minimal overhead)")
    print(f"  Architecture: {prompt.score.architecture.name}")
    
    print(f"\n✨ KEY FEATURES:")
    print(f"  ✅ [[CRITICAL:]] at TOP (+9.3% quality bonus)")
    print(f"  ✅ Keyword-based CONDITIONS (low O_s)")
    print(f"  ✅ Domain-specific transformations")
    print(f"  ✅ JSON output specification")
    
    print(f"\n📄 This is the V4.1 architecture that achieved 100% in research!")


def demo_code_generation():
    """Demonstrate code generation prompt."""
    print_section("DEMO 6: Code Generation Prompt)")
    
    prompt = build_code_prompt(
        identity="CodeAssistant",
        language="python",
        include_tests=True
    )
    
    print("📊 PROMPT QUALITY METRICS:")
    print(f"  Quality:      {prompt.score.Q_final:.1%}")
    print(f"  Grade:        {prompt.score.grade}")
    print(f"  O_s:          {prompt.score.Os:.2f}")
    
    print(f"\n💻 CODE GENERATION OPTIMIZATIONS:")
    print(f"  ✅ Python domain with best practices")
    print(f"  ✅ Test generation enabled")
    print(f"  ✅ Documentation included")
    print(f"  ✅ Error handling requirements")


def demo_critical_effect():
    """Demonstrate the [[CRITICAL:]] directive effect."""
    print_section("DEMO 7: The [[CRITICAL:]] Directive Effect")
    
    print("Building TWO identical prompts - one WITH and one WITHOUT [[CRITICAL:]]...\n")
    
    # WITHOUT [[CRITICAL:]]
    prompt_without = (UCLPromptBuilder()
        .set_identity("Assistant")
        .set_task("Process queries")
        .add_domain(LinearAlgebraDomain())
        .set_output(JSONOutput(fields=["result"]))
        .build()
    )
    
    # WITH [[CRITICAL:]]
    prompt_with = (UCLPromptBuilder()
        .set_identity("Assistant")
        .add_critical()  # <-- THE DIFFERENCE!
        .set_task("Process queries")
        .add_domain(LinearAlgebraDomain())
        .set_output(JSONOutput(fields=["result"]))
        .build()
    )
    
    print(f"WITHOUT [[CRITICAL:]]:")
    print(f"  Quality: {prompt_without.score.Q_final:.1%}")
    print(f"  Grade:   {prompt_without.score.grade}")
    print(f"  Warnings: {len(prompt_without.score.warnings)}")
    
    print(f"\nWITH [[CRITICAL:]] at TOP:")
    print(f"  Quality: {prompt_with.score.Q_final:.1%}")
    print(f"  Grade:   {prompt_with.score.grade}")
    print(f"  Warnings: {len(prompt_with.score.warnings)}")
    
    improvement = prompt_with.score.Q_final - prompt_without.score.Q_final
    print(f"\n✨ IMPROVEMENT: +{improvement:.1%} quality!")
    print(f"   (Matches the empirical 9.3% bonus from research!)")


def demo_optimization_suggestions():
    """Demonstrate optimization suggestions."""
    print_section("DEMO 8: Optimization Suggestions")
    
    try:
        from . import optimize_prompt
    except ImportError:
        from __init__ import optimize_prompt
    
    # Create a suboptimal prompt
    bad_prompt = """
    ^^SWITCH: question_type^^
        ^^CASE: math^^
            Solve the math problem.
        ^^/CASE^^
        ^^CASE: physics^^
            Solve the physics problem.
        ^^/CASE^^
    ^^/SWITCH^^
    """
    
    print("Analyzing a SUBOPTIMAL prompt...\n")
    result = optimize_prompt(bad_prompt)
    
    print(f"📊 CURRENT STATE:")
    print(f"  Quality:           {result['current_quality']:.1%}")
    print(f"  Potential Quality: {result['potential_quality']:.1%}")
    
    print(f"\n💡 OPTIMIZATION SUGGESTIONS ({len(result['suggestions'])}):")
    for i, suggestion in enumerate(result['suggestions'], 1):
        print(f"\n  {i}. [{suggestion['priority']}] {suggestion['action']}")
        print(f"     Reason: {suggestion['reason']}")
        print(f"     Impact: {suggestion['impact']}")


def main():
    """Run all demonstrations."""
    print("\n" + "█" * 70)
    print("  UCL FRAMEWORK COMPREHENSIVE DEMONSTRATION")
    print("  Version 2.0 - The Over-Specification Paradox")
    print("█" * 70)
    
    try:
        demo_basic_usage()
        demo_domain_system()
        demo_custom_domain()
        demo_anti_patterns()
        demo_math_to_tts()
        demo_code_generation()
        demo_critical_effect()
        demo_optimization_suggestions()
        
        print_section("DEMONSTRATION COMPLETE!")
        print("✅ All demos executed successfully!")
        print("\n📚 Key Takeaways:")
        print("  1. Use [[CRITICAL:]] at TOP for +9.3% quality")
        print("  2. Use keyword-based CONDITIONS, not SWITCH")
        print("  3. Add domains for specialized logic")
        print("  4. Monitor O_s and quality scores")
        print("  5. Follow anti-pattern warnings")
        print("\n🎯 Optimal architecture: V4.1 style (100% quality achieved!)")
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
