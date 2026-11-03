#!/usr/bin/env python3
"""
UNIVERSAL CONDITIONAL LOGIC (UCL) FRAMEWORK
============================================
A Mathematical Framework for AI Prompt Optimization

Based on research: "The Over-Specification Paradox" (N=305, p<10⁻¹³)

Quick Start:
    from ucl_linter import UCLPromptBuilder, LinearAlgebraDomain, JSONOutput

    prompt = (UCLPromptBuilder()
        .set_identity("Assistant")
        .add_critical()  # +9.3% quality!
        .add_domain(LinearAlgebraDomain())
        .set_output(JSONOutput(fields=["result"]))
        .build()
    )

    print(f"Quality: {prompt.score.Q_final:.1%}")

Author: Anthony Mikinka
Version: 2.0
License: MIT
"""

__version__ = "2.0.0"
__author__ = "Anthony Mikinka"
__license__ = "MIT"

# =============================================================================
# DUAL-MODE IMPORTS
# Works as both a package (pip install) and direct script execution
# =============================================================================

try:
    # Package mode: from .core import ...
    from .core import (
        UCLConstants,
        DirectiveType, ArchitectureType, Severity,
        UCLElement,
        Concept, Directive, Transform, Condition, Case, Switch, Section,
        Assignment, RawText,
        create_critical_directive,
        UCLScore, compute_Q, compute_Os, calculate_score,
        Warning, detect_anti_patterns,
    )
    from .domains import (
        Domain, DomainRegistry, CustomDomain,
        BasicMathDomain, LinearAlgebraDomain, VectorCalculusDomain,
        IntegralsDomain, DifferentialEquationsDomain, StatisticsDomain,
        PhysicsDomain,
        CodeGenerationDomain, PythonDomain, JavaScriptDomain,
        DataAnalysisDomain, TechnicalWritingDomain,
    )
    from .builder import (
        UCLPromptBuilder, OutputFormat, OutputSpec, JSONOutput, CodeOutput,
    )
    from .prompt_analyzer import (
        analyze_prompt, ucl_score as analyze_and_score,
        analyze_prompt_file, PromptAnalysis, PromptPatternDetector,
    )
except ImportError:
    # Script mode: from core import ...
    from core import (
        UCLConstants,
        DirectiveType, ArchitectureType, Severity,
        UCLElement,
        Concept, Directive, Transform, Condition, Case, Switch, Section,
        Assignment, RawText,
        create_critical_directive,
        UCLScore, compute_Q, compute_Os, calculate_score,
        Warning, detect_anti_patterns,
    )
    from domains import (
        Domain, DomainRegistry, CustomDomain,
        BasicMathDomain, LinearAlgebraDomain, VectorCalculusDomain,
        IntegralsDomain, DifferentialEquationsDomain, StatisticsDomain,
        PhysicsDomain,
        CodeGenerationDomain, PythonDomain, JavaScriptDomain,
        DataAnalysisDomain, TechnicalWritingDomain,
    )
    from builder import (
        UCLPromptBuilder, OutputFormat, OutputSpec, JSONOutput, CodeOutput,
    )
    from prompt_analyzer import (
        analyze_prompt, ucl_score as analyze_and_score,
        analyze_prompt_file, PromptAnalysis, PromptPatternDetector,
    )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def score_prompt(prompt_text: str) -> UCLScore:
    """
    Score an existing prompt text.

    Args:
        prompt_text: The prompt text to score

    Returns:
        UCLScore with quality metrics

    Example:
        score = score_prompt(my_prompt)
        print(f"Quality: {score.Q_final:.1%}")
        print(f"Architecture: {score.architecture.name}")
    """
    import re

    lines = prompt_text.split('\n')
    total_lines = len(lines)

    has_critical = '[[CRITICAL:' in prompt_text
    critical_at_top = has_critical and any('[[CRITICAL:' in line for line in lines[:10])

    elements = []

    directives = re.findall(r'\[\[(\w+):', prompt_text)
    for d in directives:
        try:
            dtype = DirectiveType[d]
            elements.append(Directive(dtype, "content"))
        except KeyError:
            pass

    switch_count = prompt_text.count('^^SWITCH:')
    for _ in range(switch_count):
        elements.append(Switch("var", [Case("case1", [])]))

    condition_count = prompt_text.count('^^CONDITION:')
    for _ in range(condition_count):
        elements.append(Condition(keywords=["keyword"], body=[]))

    return calculate_score(
        elements=elements,
        has_critical=has_critical,
        critical_at_top=critical_at_top,
        total_lines=total_lines
    )


def optimize_prompt(prompt_text: str) -> dict:
    """
    Analyze a prompt and provide optimization suggestions.

    Args:
        prompt_text: The prompt text to optimize

    Returns:
        Dictionary with score and suggestions

    Example:
        result = optimize_prompt(my_prompt)
        for suggestion in result["suggestions"]:
            print(f"[{suggestion['priority']}] {suggestion['action']}")
    """
    score = score_prompt(prompt_text)

    suggestions = []

    for warning in score.warnings:
        priority = "HIGH" if warning.severity in [Severity.CRITICAL, Severity.HIGH] else "MEDIUM"
        suggestions.append({
            "priority": priority,
            "action": warning.recommendation,
            "reason": warning.message,
            "impact": f"+{warning.estimated_impact:.1%} quality"
        })

    if score.Q_final < 0.95:
        if not any('CRITICAL' in s['action'] for s in suggestions):
            suggestions.insert(0, {
                "priority": "HIGH",
                "action": "Add [[CRITICAL:]] directive at TOP of prompt",
                "reason": "Can boost quality by 9.3%",
                "impact": "+9.3% quality"
            })

    return {
        "score": score.to_dict(),
        "suggestions": suggestions,
        "current_quality": score.Q_final,
        "potential_quality": min(1.0, score.Q_final + sum(s.estimated_impact for s in score.warnings))
    }


def build_math_tts_prompt(identity: str = "Qwen",
                          domains: list = None):
    """
    Build a Math-to-TTS prompt (V4.1 style).

    Args:
        identity: AI identity name
        domains: List of domain names or Domain instances

    Returns:
        Built prompt with 100% quality score

    Example:
        prompt = build_math_tts_prompt(
            identity="Qwen",
            domains=["basic_math", "linear_algebra", "vector_calculus"]
        )
    """
    if domains is None:
        domains = ["basic_math", "linear_algebra", "vector_calculus"]

    builder = (UCLPromptBuilder()
        .set_identity(identity)
        .add_critical()
        .set_task("Transform mathematical expressions INTO TTS-friendly format")
        .require("JSON output with result and scratchwork_answer fields")
    )

    for domain in domains:
        builder.add_domain(domain)

    builder.set_output(JSONOutput(
        fields=["result", "scratchwork_answer"],
        validations=["Valid JSON syntax", "Complete mathematical transformation"]
    ))

    return builder.build()


def build_code_prompt(identity: str = "CodeAssistant",
                      language: str = "python",
                      include_tests: bool = True):
    """
    Build a code generation prompt.

    Args:
        identity: AI identity name
        language: Programming language
        include_tests: Include test generation

    Returns:
        Built prompt optimized for code generation

    Example:
        prompt = build_code_prompt(identity="CodeAssistant", language="python")
    """
    builder = (UCLPromptBuilder()
        .set_identity(identity)
        .add_critical()
        .set_task(f"Generate high-quality {language} code")
        .require("Clean, readable code")
        .require("Proper error handling")
    )

    if language.lower() == "python":
        builder.add_domain(PythonDomain())
    elif language.lower() in ["javascript", "js"]:
        builder.add_domain(JavaScriptDomain())
    else:
        builder.add_domain(CodeGenerationDomain())

    builder.set_output(CodeOutput(
        language=language,
        include_tests=include_tests,
        include_docs=True
    ))

    return builder.build()


# =============================================================================
# PACKAGE METADATA
# =============================================================================

__all__ = [
    '__version__', '__author__', '__license__',
    'UCLConstants',
    'DirectiveType', 'ArchitectureType', 'Severity', 'OutputFormat',
    'UCLElement', 'Domain',
    'Concept', 'Directive', 'Transform', 'Condition', 'Case',
    'Switch', 'Section', 'Assignment', 'RawText',
    'UCLPromptBuilder', 'OutputSpec', 'JSONOutput', 'CodeOutput',
    'DomainRegistry', 'CustomDomain',
    'BasicMathDomain', 'LinearAlgebraDomain', 'VectorCalculusDomain',
    'IntegralsDomain', 'DifferentialEquationsDomain', 'StatisticsDomain',
    'PhysicsDomain',
    'CodeGenerationDomain', 'PythonDomain', 'JavaScriptDomain',
    'DataAnalysisDomain', 'TechnicalWritingDomain',
    'UCLScore', 'compute_Q', 'compute_Os', 'calculate_score',
    'Warning', 'detect_anti_patterns',
    'create_critical_directive',
    'score_prompt', 'optimize_prompt',
    'build_math_tts_prompt', 'build_code_prompt',
    'analyze_prompt', 'analyze_and_score', 'analyze_prompt_file',
    'PromptAnalysis', 'PromptPatternDetector',
]
