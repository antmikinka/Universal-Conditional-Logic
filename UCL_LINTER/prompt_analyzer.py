#!/usr/bin/env python3
"""
UCL Prompt Analyzer
===================

Analyze ANY prompt text and score it using UCL metrics.

This tool can:
- Score existing prompts (not built with UCL)
- Detect UCL patterns and anti-patterns
- Calculate quality metrics
- Export analysis results
- Provide optimization suggestions

Usage:
    # Quick scoring
    score = ucl_score("Your prompt text here")
    
    # Detailed analysis
    analysis = analyze_prompt("Your prompt text here")
    analysis.export("report.json")
    
    # CLI
    python prompt_analyzer.py my_prompt.txt
"""

import re
import json
import math
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path

try:
    from .core import (
        UCLScore, calculate_score, detect_anti_patterns,
        Directive, DirectiveType, Condition, Switch, Case,
        Severity, Warning, UCLToolConstants
    )
except ImportError:
    from core import (
        UCLScore, calculate_score, detect_anti_patterns,
        Directive, DirectiveType, Condition, Switch, Case,
        Severity, Warning, UCLToolConstants
    )


# =============================================================================
# PATTERN DETECTION
# =============================================================================

class PromptPatternDetector:
    """Detect UCL patterns in raw prompt text"""

    # Regex patterns for UCL syntax
    DIRECTIVE_PATTERN = r'\[\[(\w+):\s*([^\]]+)\]\]'
    CONDITION_PATTERN = r'\^\^CONDITION:\s*([^\^]+)\^\^'
    SWITCH_PATTERN = r'\^\^SWITCH:\s*([^\^]+)\^\^'
    CASE_PATTERN = r'\^\^CASE:\s*([^\^]+)\^\^'
    CONCEPT_PATTERN = r'\{\{concept:(\w+)(?::(\w+))?\}\}'
    TRANSFORM_PATTERN = r'\[\[TRANSFORM:\s*([^\]]+)\]\]'

    # UCL-Tool patterns (Phase 3)
    TOOL_PATTERN = r'\[\[TOOL:\s*(\w+)\]\]'
    PARAM_PATTERN = r'\[\[PARAM:\s*(\w+),\s*(\w+),\s*(required|optional)\]\]'
    WHEN_PATTERN = r'\[\[WHEN:\s*([^\]]+)\]\]'
    EXAMPLE_PATTERN = r'\[\[EXAMPLE:\s*([^\]]+)\]\]'
    DESC_PATTERN = r'\[\[DESC:\s*([^\]]+)\]\]'

    # Tool group section tags: <group_name>...</group_name>
    GROUP_SECTION_PATTERN = r'<(\w+)>'

    # JSON Schema tool detection (verbose format)
    JSON_SCHEMA_TOOL_PATTERN = r'"type"\s*:\s*"function"'
    JSON_SCHEMA_FUNC_NAME = r'"function"\s*:\s*\{\s*"name"\s*:\s*"(\w+)"'

    def __init__(self, text: str):
        self.text = text
        self.lines = text.split('\n')
        self.total_lines = len(self.lines)

    def detect_directives(self) -> Dict[str, int]:
        """Count each type of directive"""
        directives = re.findall(self.DIRECTIVE_PATTERN, self.text)
        counts = {}
        for dtype, _ in directives:
            counts[dtype] = counts.get(dtype, 0) + 1
        return counts

    def has_critical(self) -> bool:
        """Check if [[CRITICAL:]] exists"""
        return 'CRITICAL' in self.detect_directives()

    def critical_at_top(self) -> bool:
        """Check if [[CRITICAL:]] is in first 10 lines"""
        first_10_lines = '\n'.join(self.lines[:10])
        return '[[CRITICAL:' in first_10_lines

    def count_conditions(self) -> int:
        """Count ^^CONDITION:^^ blocks"""
        return len(re.findall(self.CONDITION_PATTERN, self.text))

    def count_switches(self) -> int:
        """Count ^^SWITCH:^^ blocks"""
        return len(re.findall(self.SWITCH_PATTERN, self.text))

    def count_cases(self) -> int:
        """Count ^^CASE:^^ blocks"""
        return len(re.findall(self.CASE_PATTERN, self.text))

    def estimate_Os(self) -> float:
        """Estimate structural overhead O_s"""
        Os = 0.0

        # Directive overhead
        directives = self.detect_directives()
        directive_weights = {
            'CRITICAL': -0.05,  # Reduces overhead!
            'LLM': 0.3,
            'REQUIRE': 0.5,
            'TRANSFORM': 1.0,
            'VALIDATE': 0.5,
            'ENFORCE': 0.5,
            'APPLY': 0.3,
            'EMPHASIZE': 0.2,
        }
        for dtype, count in directives.items():
            weight = directive_weights.get(dtype, 0.3)
            Os += weight * count

        # Switch overhead: gamma * ln(cases)
        num_switches = self.count_switches()
        num_cases = self.count_cases()
        if num_switches > 0:
            avg_cases_per_switch = max(1, num_cases / num_switches)
            Os += 1.0 * math.log(avg_cases_per_switch) * num_switches

        # Condition overhead (lower than switch)
        num_conditions = self.count_conditions()
        Os += 0.5 * num_conditions

        # Tool overhead (UCL-Tool encoding)
        Os += self.estimate_tool_Os()

        return Os

    def estimate_specification_level(self) -> float:
        """Estimate S (specification level) from text"""
        # Multiple factors
        word_count = len(self.text.split())
        line_count = self.total_lines

        # Normalize to 0-1 scale
        word_factor = min(1.0, word_count / 500)  # 500 words = high spec
        line_factor = min(1.0, line_count / 150)   # 150 lines = high spec

        # Average
        S = (word_factor + line_factor) / 2
        return S

    # =========================================================================
    # UCL-TOOL DETECTION (Phase 3)
    # =========================================================================

    def detect_tools(self) -> List[Dict[str, Any]]:
        """
        Detect UCL-Tool definitions using state-machine parser.

        Scans line-by-line, associating [[PARAM]], [[WHEN]], [[DESC]],
        [[EXAMPLE]] with the most recent [[TOOL:]].

        Returns:
            List of dicts, each with:
            - name: str
            - params: List[Dict] with keys (name, type, required)
            - keywords: List[str]
            - descriptions: List[str]
            - examples: List[str]
            - in_group: str or None (group name if inside section tags)
        """
        tools = []
        current_tool = None
        group_stack = []  # Stack of group names from <section> tags

        for line in self.lines:
            stripped = line.strip()

            # Track group sections: <group_name>
            group_open = re.match(r'^<(\w+)>$', stripped)
            if group_open:
                group_stack.append(group_open.group(1))
                continue
            group_close = re.match(r'^</(\w+)>$', stripped)
            if group_close and group_stack and group_stack[-1] == group_close.group(1):
                group_stack.pop()
                continue

            # Check for [[TOOL: name]]
            tool_match = re.match(r'.*\[\[TOOL:\s*(\w+)\]\]', stripped)
            if tool_match:
                # Finalize previous tool
                if current_tool is not None:
                    tools.append(current_tool)
                current_tool = {
                    'name': tool_match.group(1),
                    'params': [],
                    'keywords': [],
                    'descriptions': [],
                    'examples': [],
                    'in_group': group_stack[-1] if group_stack else None,
                }
                continue

            if current_tool is None:
                continue

            # Check for [[PARAM: name, type, required|optional]]
            param_match = re.match(r'.*\[\[PARAM:\s*(\w+),\s*(\w+),\s*(required|optional)\]\]', stripped)
            if param_match:
                current_tool['params'].append({
                    'name': param_match.group(1),
                    'type': param_match.group(2),
                    'required': param_match.group(3) == 'required',
                })
                continue

            # Check for [[WHEN: kw1, kw2, ...]]
            when_match = re.match(r'.*\[\[WHEN:\s*([^\]]+)\]\]', stripped)
            if when_match:
                keywords = [k.strip() for k in when_match.group(1).split(',') if k.strip()]
                current_tool['keywords'].extend(keywords)
                continue

            # Check for [[DESC: text]]
            desc_match = re.match(r'.*\[\[DESC:\s*([^\]]+)\]\]', stripped)
            if desc_match:
                current_tool['descriptions'].append(desc_match.group(1).strip())
                continue

            # Check for [[EXAMPLE: input -> output]]
            example_match = re.match(r'.*\[\[EXAMPLE:\s*([^\]]+)\]\]', stripped)
            if example_match:
                current_tool['examples'].append(example_match.group(1).strip())
                continue

        # Finalize last tool
        if current_tool is not None:
            tools.append(current_tool)

        return tools

    def count_tools(self) -> int:
        """Count UCL-Tool definitions (quick regex, no association)"""
        return len(re.findall(self.TOOL_PATTERN, self.text))

    def detect_tool_groups(self) -> List[Dict[str, Any]]:
        """
        Detect tool groups (section tags inside ^^CONDITION^^ blocks).

        Returns:
            List of dicts with:
            - name: str (group/section name)
            - tools: List[str] (tool names found inside)
        """
        groups = []
        # Find all section tags
        section_pattern = r'<(\w+)>(.*?)</\1>'
        for match in re.finditer(section_pattern, self.text, re.DOTALL):
            group_name = match.group(1)
            group_body = match.group(2)
            # Count tools inside this section
            tool_names = re.findall(self.TOOL_PATTERN, group_body)
            if tool_names:  # Only report if it contains tools
                groups.append({
                    'name': group_name,
                    'tools': tool_names,
                })
        return groups

    def detect_json_schema_tools(self) -> List[str]:
        """
        Detect JSON Schema tool definitions (verbose format).

        Returns list of tool names found in JSON Schema format.
        """
        names = re.findall(self.JSON_SCHEMA_FUNC_NAME, self.text)
        return names

    def count_json_schema_tools(self) -> int:
        """Count JSON Schema tool definitions (quick check)"""
        return len(re.findall(self.JSON_SCHEMA_TOOL_PATTERN, self.text))

    def estimate_tool_Os(self) -> float:
        """
        Estimate O_s contribution from UCL-Tool definitions.

        Uses UCLToolConstants for per-element overhead.
        """
        tools = self.detect_tools()
        total_os = 0.0

        for tool in tools:
            # Base overhead
            total_os += UCLToolConstants.TOOL_BASE_OS

            # Parameter overhead
            for param in tool['params']:
                p_os = UCLToolConstants.PARAM_OS
                if param['required']:
                    p_os += UCLToolConstants.REQUIRED_PARAM_DISCOUNT
                total_os += p_os

            # Keyword activation overhead
            if tool['keywords']:
                total_os += UCLToolConstants.WHEN_OS

            # Description overhead
            total_os += UCLToolConstants.DESC_OS * len(tool['descriptions'])

            # Example overhead
            total_os += UCLToolConstants.EXAMPLE_OS * len(tool['examples'])

        # Group discount: subtract shared condition savings
        groups = self.detect_tool_groups()
        for group in groups:
            n_tools = len(group['tools'])
            if n_tools > 1:
                # Save WHEN_OS for each duplicate condition eliminated
                total_os -= UCLToolConstants.SHARED_CONDITION_SAVINGS * (n_tools - 1)
                total_os += UCLToolConstants.TOOL_GROUP_DISCOUNT * n_tools

        return max(0.0, total_os)

    def estimate_json_schema_Os(self) -> float:
        """
        Estimate O_s if tools were encoded as JSON Schema.

        JSON Schema is significantly more verbose.
        Estimated ~3.0 O_s per tool (based on nesting + token count).
        """
        n_json = self.count_json_schema_tools()
        n_ucl = self.count_tools()
        total_tools = n_json + n_ucl
        return total_tools * 3.0

    def detect_architecture(self) -> str:
        """Detect prompt architecture type"""
        has_switch = self.count_switches() > 0
        has_condition = self.count_conditions() > 0
        has_tools = self.count_tools() > 0
        has_json_tools = self.count_json_schema_tools() > 0

        # Tool-specific architectures (highest priority)
        if has_json_tools and not has_tools:
            return "JSON-SCHEMA"
        if has_tools and has_condition:
            return "UCL-TOOL-CONDITIONAL"
        if has_tools:
            return "UCL-TOOL"

        # Standard architectures
        if has_switch and has_condition:
            return "MIXED"
        elif has_switch:
            return "SWITCH"
        elif has_condition:
            return "KEYWORD"
        else:
            return "MINIMAL"


# =============================================================================
# ANALYSIS RESULT
# =============================================================================

@dataclass
class PromptAnalysis:
    """Complete analysis of a prompt"""

    # Input
    prompt_text: str
    prompt_name: Optional[str] = None

    # Metrics
    quality_score: float = 0.0
    grade: str = "F"
    Os: float = 0.0
    specification_level: float = 0.0
    architecture: str = "UNKNOWN"

    # Breakdown
    total_lines: int = 0
    total_words: int = 0
    directives_found: Dict[str, int] = None
    has_critical: bool = False
    critical_at_top: bool = False
    num_conditions: int = 0
    num_switches: int = 0
    num_cases: int = 0

    # UCL-Tool analysis (Phase 3)
    tools_found: List[Dict[str, Any]] = None
    tool_count: int = 0
    tools_with_keywords: int = 0
    tools_without_keywords: int = 0
    tools_with_examples: int = 0
    tool_groups_found: List[Dict[str, Any]] = None
    json_schema_tools: List[str] = None
    tool_os_contribution: float = 0.0

    # Warnings
    warnings: List[Dict[str, Any]] = None

    # Suggestions
    suggestions: List[Dict[str, Any]] = None

    def __post_init__(self):
        if self.directives_found is None:
            self.directives_found = {}
        if self.warnings is None:
            self.warnings = []
        if self.suggestions is None:
            self.suggestions = []
        if self.tools_found is None:
            self.tools_found = []
        if self.tool_groups_found is None:
            self.tool_groups_found = []
        if self.json_schema_tools is None:
            self.json_schema_tools = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Export as dictionary"""
        return asdict(self)
    
    def to_json(self, indent: int = 2) -> str:
        """Export as JSON string"""
        return json.dumps(self.to_dict(), indent=indent)
    
    def export(self, filepath: str):
        """Export to file (JSON or TXT based on extension)"""
        path = Path(filepath)
        
        if path.suffix.lower() == '.json':
            with open(path, 'w', encoding='utf-8') as f:
                f.write(self.to_json())
        else:
            # Text report
            with open(path, 'w', encoding='utf-8') as f:
                f.write(self.to_report())
    
    def to_report(self) -> str:
        """Generate human-readable text report"""
        report = []
        report.append("=" * 70)
        report.append("UCL PROMPT ANALYSIS REPORT")
        if self.prompt_name:
            report.append(f"Prompt: {self.prompt_name}")
        report.append("=" * 70)
        report.append("")
        
        # Overall Score
        report.append("📊 OVERALL QUALITY")
        report.append(f"   Score: {self.quality_score:.1%}")
        report.append(f"   Grade: {self.grade}")
        report.append(f"   Architecture: {self.architecture}")
        report.append("")
        
        # Metrics
        report.append("📐 METRICS")
        report.append(f"   Structural Overhead (O_s): {self.Os:.2f}")
        report.append(f"   Specification Level (S): {self.specification_level:.3f}")
        report.append(f"   Total Lines: {self.total_lines}")
        report.append(f"   Total Words: {self.total_words}")
        report.append("")
        
        # Patterns Found
        report.append("🔍 PATTERNS DETECTED")
        report.append(f"   [[CRITICAL:]]: {'YES ✅' if self.has_critical else 'NO ❌'}")
        if self.has_critical:
            report.append(f"   - Location: {'TOP ✅' if self.critical_at_top else 'Not at top ⚠️'}")
        report.append(f"   Conditions: {self.num_conditions}")
        report.append(f"   Switches: {self.num_switches}")
        if self.num_switches > 0:
            report.append(f"   - Cases: {self.num_cases}")
        report.append("")
        
        if self.directives_found:
            report.append("   Directives Found:")
            for dtype, count in sorted(self.directives_found.items()):
                report.append(f"      [[{dtype}:]]: {count}")
            report.append("")

        # Tool Analysis (Phase 3)
        if self.tool_count > 0 or self.json_schema_tools:
            report.append("🔧 TOOL ANALYSIS")
            if self.tool_count > 0:
                report.append(f"   UCL-Tool definitions: {self.tool_count}")
                report.append(f"   - With keywords: {self.tools_with_keywords}")
                report.append(f"   - Without keywords: {self.tools_without_keywords}")
                report.append(f"   - With examples: {self.tools_with_examples}")
                report.append(f"   Tool O_s contribution: {self.tool_os_contribution:.3f}")
                if self.tool_groups_found:
                    report.append(f"   Tool groups: {len(self.tool_groups_found)}")
                    for g in self.tool_groups_found:
                        report.append(f"      <{g['name']}>: {len(g['tools'])} tools")
                # List tools
                report.append("   Tools detected:")
                for t in self.tools_found:
                    status = "✅" if t['keywords'] else "⚠️ no keywords"
                    ex = f" ({len(t['examples'])} examples)" if t['examples'] else ""
                    grp = f" [group: {t['in_group']}]" if t['in_group'] else ""
                    report.append(f"      - {t['name']}: {len(t['params'])} params{ex}{grp} {status}")
            if self.json_schema_tools:
                report.append(f"   JSON Schema tools: {len(self.json_schema_tools)}")
                for name in self.json_schema_tools:
                    report.append(f"      - {name} (high O_s format)")
            report.append("")
        
        # Warnings
        if self.warnings:
            report.append(f"⚠️  WARNINGS ({len(self.warnings)})")
            for i, w in enumerate(self.warnings, 1):
                report.append(f"   {i}. [{w['severity']}] {w['message']}")
                report.append(f"      → {w['recommendation']}")
                report.append(f"      Impact: {w['impact']}")
            report.append("")
        else:
            report.append("✅ NO WARNINGS - Well optimized!")
            report.append("")
        
        # Suggestions
        if self.suggestions:
            report.append(f"💡 OPTIMIZATION SUGGESTIONS ({len(self.suggestions)})")
            for i, s in enumerate(self.suggestions, 1):
                report.append(f"   {i}. [{s['priority']}] {s['action']}")
                if 'impact' in s:
                    report.append(f"      Impact: {s['impact']}")
            report.append("")
        
        report.append("=" * 70)
        report.append(f"Potential Quality: {self.quality_score + sum(float(w.get('estimated_impact', 0)) for w in self.warnings):.1%}")
        report.append("(if all suggestions applied)")
        report.append("=" * 70)
        
        return '\n'.join(report)
    
    def print_report(self):
        """Print the report to console"""
        print(self.to_report())


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def analyze_prompt(prompt_text: str, prompt_name: Optional[str] = None) -> PromptAnalysis:
    """
    Analyze any prompt text and return detailed analysis.
    
    Args:
        prompt_text: The prompt text to analyze
        prompt_name: Optional name for the prompt
        
    Returns:
        PromptAnalysis object with complete breakdown
        
    Example:
        analysis = analyze_prompt(my_prompt, "V4.1")
        print(f"Quality: {analysis.quality_score:.1%}")
        analysis.export("analysis.json")
    """
    # Detect patterns
    detector = PromptPatternDetector(prompt_text)

    # Basic stats
    total_lines = detector.total_lines
    total_words = len(prompt_text.split())

    # Pattern detection
    directives = detector.detect_directives()
    has_critical = detector.has_critical()
    critical_at_top = detector.critical_at_top()
    num_conditions = detector.count_conditions()
    num_switches = detector.count_switches()
    num_cases = detector.count_cases()

    # UCL-Tool detection (Phase 3)
    tools_found = detector.detect_tools()
    tool_count = len(tools_found)
    tools_with_keywords = sum(1 for t in tools_found if t['keywords'])
    tools_without_keywords = sum(1 for t in tools_found if not t['keywords'])
    tools_with_examples = sum(1 for t in tools_found if t['examples'])
    tool_groups_found = detector.detect_tool_groups()
    json_schema_tools = detector.detect_json_schema_tools()
    tool_os_contribution = detector.estimate_tool_Os()

    # Calculate metrics
    Os = detector.estimate_Os()
    S = detector.estimate_specification_level()
    architecture = detector.detect_architecture()

    # Calculate quality using UCL formula
    try:
        from .core import compute_Q, UCLConstants
    except ImportError:
        from core import compute_Q, UCLConstants
    Q_base = compute_Q(S)

    # Apply bonuses/penalties
    bonuses = 0.0
    penalties = 0.0

    if has_critical:
        if critical_at_top:
            bonuses += UCLConstants.CRITICAL_BONUS
        else:
            bonuses += UCLConstants.CRITICAL_BONUS * 0.7

    if architecture == "KEYWORD":
        bonuses += UCLConstants.KEYWORD_BONUS

    if architecture == "UCL-TOOL-CONDITIONAL":
        bonuses += UCLConstants.KEYWORD_BONUS  # Same bonus as keyword arch

    if num_switches > 0:
        penalties += UCLConstants.SWITCH_PENALTY_PER_CASE * num_cases

    # Tool bonuses
    tool_bonuses = tools_with_keywords * UCLToolConstants.TOOL_KEYWORD_BONUS
    tool_bonuses += tools_with_examples * UCLToolConstants.EXAMPLE_BONUS
    bonuses += tool_bonuses

    # Tool penalties
    if tools_without_keywords > 0:
        penalties += UCLToolConstants.TOOL_NO_KEYWORDS_PENALTY * tools_without_keywords
    # Tools without descriptions
    tools_without_desc = sum(1 for t in tools_found if not t['descriptions'])
    if tools_without_desc > 0:
        penalties += UCLToolConstants.TOOL_NO_DESC_PENALTY * tools_without_desc
    # Duplicate tool names
    tool_names = [t['name'] for t in tools_found]
    duplicate_names = len(tool_names) - len(set(tool_names))
    if duplicate_names > 0:
        penalties += UCLToolConstants.TOOL_NAME_CONFLICT_PENALTY * duplicate_names
    # JSON Schema tools penalty (high O_s)
    if json_schema_tools:
        penalties += 0.05 * len(json_schema_tools)

    quality = max(0.0, min(1.0, Q_base + bonuses - penalties))

    # Grade
    if quality >= 0.95:
        grade = "A"
    elif quality >= 0.85:
        grade = "B"
    elif quality >= 0.70:
        grade = "C"
    elif quality >= 0.50:
        grade = "D"
    else:
        grade = "F"

    # Generate warnings
    warnings = []

    if num_switches > 0:
        warnings.append({
            "severity": "HIGH",
            "message": f"SWITCH architecture detected ({num_switches} switches, {num_cases} cases)",
            "recommendation": "Convert to keyword-based CONDITIONS for lower O_s",
            "impact": f"-{penalties:.1%}",
            "estimated_impact": penalties
        })

    if not has_critical and (total_lines > 20 or num_conditions > 0 or num_switches > 0 or tool_count > 0):
        warnings.append({
            "severity": "HIGH",
            "message": "Missing [[CRITICAL:]] directive",
            "recommendation": "Add [[CRITICAL:]] at TOP of prompt for +9.3% quality",
            "impact": "+9.3%",
            "estimated_impact": UCLConstants.CRITICAL_BONUS
        })

    if has_critical and not critical_at_top:
        warnings.append({
            "severity": "MEDIUM",
            "message": "[[CRITICAL:]] not at top of prompt",
            "recommendation": "Move [[CRITICAL:]] to first 10 lines",
            "impact": "+3%",
            "estimated_impact": 0.03
        })

    # Tool-specific warnings
    for tool in tools_found:
        if not tool['keywords']:
            warnings.append({
                "severity": "HIGH",
                "message": f"Tool '{tool['name']}' has no activation keywords ([[WHEN:]])",
                "recommendation": f"Add keywords: [[WHEN: keyword1, keyword2]] to enable conditional activation",
                "impact": f"+{UCLToolConstants.WHEN_OS:.2f} O_s",
                "estimated_impact": UCLToolConstants.TOOL_NO_KEYWORDS_PENALTY
            })
        if not tool['descriptions']:
            warnings.append({
                "severity": "MEDIUM",
                "message": f"Tool '{tool['name']}' has no description ([[DESC:]])",
                "recommendation": f"Add description: [[DESC: What this tool does]]",
                "impact": "LLM may misuse tool",
                "estimated_impact": UCLToolConstants.TOOL_NO_DESC_PENALTY
            })

    if json_schema_tools:
        json_os = detector.estimate_json_schema_Os()
        warnings.append({
            "severity": "HIGH",
            "message": f"{len(json_schema_tools)} tools encoded as JSON Schema (high O_s)",
            "recommendation": "Convert to UCL-Tool format for 80%+ O_s reduction. "
                              f"Current JSON Schema O_s estimate: {json_os:.1f}",
            "impact": f"~{json_os - tool_os_contribution:.1f} O_s savings potential",
            "estimated_impact": 0.05 * len(json_schema_tools)
        })

    if duplicate_names > 0:
        warnings.append({
            "severity": "HIGH",
            "message": f"{duplicate_names} duplicate tool name(s) detected",
            "recommendation": "Ensure all tools have unique names to avoid activation conflicts",
            "impact": "Ambiguous tool routing",
            "estimated_impact": UCLToolConstants.TOOL_NAME_CONFLICT_PENALTY * duplicate_names
        })

    # Generate suggestions
    suggestions = []
    for w in warnings:
        suggestions.append({
            "priority": w["severity"],
            "action": w["recommendation"],
            "impact": w["impact"]
        })

    # Group suggestion
    standalone_with_similar_keywords = []
    keyword_sets = {t['name']: set(t['keywords']) for t in tools_found if t['keywords'] and not t['in_group']}
    names_list = list(keyword_sets.keys())
    for i in range(len(names_list)):
        for j in range(i + 1, len(names_list)):
            overlap = keyword_sets[names_list[i]] & keyword_sets[names_list[j]]
            if len(overlap) >= 2:
                standalone_with_similar_keywords.append((names_list[i], names_list[j]))
    if standalone_with_similar_keywords:
        suggestions.append({
            "priority": "LOW",
            "action": "Tools with overlapping keywords could be grouped for O_s savings",
            "impact": "Reduced O_s through shared condition"
        })

    return PromptAnalysis(
        prompt_text=prompt_text,
        prompt_name=prompt_name,
        quality_score=quality,
        grade=grade,
        Os=Os,
        specification_level=S,
        architecture=architecture,
        total_lines=total_lines,
        total_words=total_words,
        directives_found=directives,
        has_critical=has_critical,
        critical_at_top=critical_at_top,
        num_conditions=num_conditions,
        num_switches=num_switches,
        num_cases=num_cases,
        tools_found=tools_found,
        tool_count=tool_count,
        tools_with_keywords=tools_with_keywords,
        tools_without_keywords=tools_without_keywords,
        tools_with_examples=tools_with_examples,
        tool_groups_found=tool_groups_found,
        json_schema_tools=json_schema_tools,
        tool_os_contribution=tool_os_contribution,
        warnings=warnings,
        suggestions=suggestions
    )


def ucl_score(prompt_text: str) -> float:
    """
    Simple function to get just the quality score.
    
    Args:
        prompt_text: The prompt text to score
        
    Returns:
        Quality score as float (0.0 to 1.0)
        
    Example:
        score = ucl_score("Your prompt here")
        print(f"Quality: {score:.1%}")
    """
    analysis = analyze_prompt(prompt_text)
    return analysis.quality_score


def analyze_prompt_file(filepath: str) -> PromptAnalysis:
    """
    Analyze a prompt from a file.
    
    Args:
        filepath: Path to prompt text file
        
    Returns:
        PromptAnalysis object
    """
    path = Path(filepath)
    with open(path, 'r', encoding='utf-8') as f:
        prompt_text = f.read()
    
    return analyze_prompt(prompt_text, prompt_name=path.name)


# =============================================================================
# CLI
# =============================================================================

def main():
    """CLI interface for prompt analysis"""
    import sys
    
    if len(sys.argv) < 2:
        print("UCL Prompt Analyzer")
        print("=" * 50)
        print("\nUsage:")
        print("  python prompt_analyzer.py <prompt_file.txt>")
        print("  python prompt_analyzer.py <prompt_file.txt> -o report.json")
        print("\nExample:")
        print("  python prompt_analyzer.py my_prompt.txt")
        print("  python prompt_analyzer.py my_prompt.txt -o analysis.json")
        sys.exit(1)
    
    filepath = sys.argv[1]
    output_file = None
    
    if len(sys.argv) > 3 and sys.argv[2] == '-o':
        output_file = sys.argv[3]
    
    # Analyze
    print(f"\nAnalyzing: {filepath}...")
    analysis = analyze_prompt_file(filepath)
    
    # Print report
    analysis.print_report()
    
    # Export if requested
    if output_file:
        analysis.export(output_file)
        print(f"\n✅ Analysis exported to: {output_file}")


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'analyze_prompt',
    'ucl_score',
    'analyze_prompt_file',
    'PromptAnalysis',
    'PromptPatternDetector',
]


if __name__ == "__main__":
    main()
