#!/usr/bin/env python3
"""
UCL Prompt Builder
==================

Fluent API for building optimized UCL prompts.

The builder provides:
- Intuitive chainable methods
- Real-time quality scoring
- Anti-pattern warnings
- Optimization suggestions
- Multiple output formats

Author: Anthony Mikinka
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union
from enum import Enum, auto
import json

try:
    from .core import (
        UCLElement, UCLConstants, UCLScore,
        Concept, Directive, DirectiveType, Transform, Condition,
        Switch, Case, Section, Assignment, RawText,
        create_critical_directive, calculate_score, detect_anti_patterns,
        ArchitectureType, Warning,
        ToolDefinition, ToolGroup, ToolParameter, ToolExample
    )
    from .domains import Domain, DomainRegistry
    from .tool_domains import ToolDomain, ToolDomainRegistry
except ImportError:
    from core import (
        UCLElement, UCLConstants, UCLScore,
        Concept, Directive, DirectiveType, Transform, Condition,
        Switch, Case, Section, Assignment, RawText,
        create_critical_directive, calculate_score, detect_anti_patterns,
        ArchitectureType, Warning,
        ToolDefinition, ToolGroup, ToolParameter, ToolExample
    )
    from domains import Domain, DomainRegistry
    try:
        from tool_domains import ToolDomain, ToolDomainRegistry
    except ImportError:
        ToolDomain = None
        ToolDomainRegistry = None


# =============================================================================
# OUTPUT FORMATS
# =============================================================================

class OutputFormat(Enum):
    """Supported output format types"""
    JSON = auto()
    TEXT = auto()
    CODE = auto()
    MARKDOWN = auto()
    CUSTOM = auto()


@dataclass
class OutputSpec:
    """Output specification for the prompt"""
    format: OutputFormat
    fields: List[str] = field(default_factory=list)
    validations: List[str] = field(default_factory=list)
    custom_template: Optional[str] = None
    
    def to_elements(self) -> List[UCLElement]:
        """Convert to UCL elements"""
        elements = []
        
        if self.format == OutputFormat.JSON:
            elements.append(Directive(DirectiveType.REQUIRE, "JSON validity"))
            if self.fields:
                fields_str = ", ".join(self.fields)
                elements.append(Directive(DirectiveType.REQUIRE, f"JSON fields: {fields_str}"))
        
        for validation in self.validations:
            elements.append(Directive(DirectiveType.VALIDATE, validation))
        
        if self.custom_template:
            elements.append(RawText(f"<json_template>\n{self.custom_template}\n</json_template>"))
        
        return elements


def JSONOutput(fields: List[str] = None, 
               validations: List[str] = None,
               template: str = None) -> OutputSpec:
    """Create JSON output specification"""
    return OutputSpec(
        format=OutputFormat.JSON,
        fields=fields or [],
        validations=validations or [],
        custom_template=template
    )


def CodeOutput(language: str = "python",
               include_tests: bool = False,
               include_docs: bool = True) -> OutputSpec:
    """Create code output specification"""
    validations = []
    if include_tests:
        validations.append("Include unit tests")
    if include_docs:
        validations.append("Include docstrings/documentation")
    validations.append(f"Valid {language} syntax")
    
    return OutputSpec(
        format=OutputFormat.CODE,
        validations=validations
    )


# =============================================================================
# PROMPT BUILDER
# =============================================================================

class UCLPromptBuilder:
    """
    Fluent API for building UCL prompts.
    
    Usage:
        prompt = (UCLPromptBuilder()
            .set_identity("Assistant")
            .add_critical()
            .set_task("Process mathematical expressions")
            .add_domain(LinearAlgebraDomain())
            .set_output(JSONOutput(fields=["result"]))
            .build()
        )
    
    The builder:
    - Enforces best practices automatically
    - Warns about anti-patterns
    - Tracks O_s in real-time
    - Suggests optimizations
    """
    
    def __init__(self):
        self._identity: Optional[str] = None
        self._critical: Optional[Directive] = None
        self._critical_position: int = -1
        self._task: Optional[str] = None
        self._requirements: List[Directive] = []
        self._domains: List[Domain] = []
        self._conditions: List[Condition] = []
        self._switches: List[Switch] = []  # Discouraged but supported
        self._output_spec: Optional[OutputSpec] = None
        self._custom_sections: List[Section] = []
        self._raw_elements: List[UCLElement] = []

        # UCL-Tool: Tool definitions
        self._tools: List[ToolDefinition] = []
        self._tool_groups: Dict[str, ToolGroup] = {}

        # Tracking
        self._warnings: List[Warning] = []
        self._os_running: float = 0.0
    
    # =========================================================================
    # FLUENT API METHODS
    # =========================================================================
    
    def set_identity(self, name: str) -> UCLPromptBuilder:
        """
        Set the AI identity.
        
        Example:
            .set_identity("Qwen")
            -> {{concept:ai_identity}} = "Qwen"
        """
        self._identity = name
        return self
    
    def add_critical(self, content: Optional[str] = None) -> UCLPromptBuilder:
        """
        Add [[CRITICAL:]] directive (HIGHLY RECOMMENDED).
        
        This is THE KEY to achieving 100% quality.
        Place at TOP of prompt for +9.3% quality bonus.
        
        Args:
            content: Custom content or None for optimal template
        
        Example:
            .add_critical()  # Uses optimal template
            .add_critical("Return ONLY JSON")  # Custom
        """
        self._critical = create_critical_directive(content)
        self._critical_position = 0  # Mark as early
        return self
    
    def set_task(self, description: str) -> UCLPromptBuilder:
        """
        Set the main task directive.
        
        Example:
            .set_task("Transform mathematical expressions INTO TTS format")
        """
        self._task = description
        return self
    
    def require(self, requirement: str) -> UCLPromptBuilder:
        """
        Add a requirement directive.
        
        Example:
            .require("JSON output")
            .require("Mathematical accuracy >= 95%")
        """
        self._requirements.append(Directive(DirectiveType.REQUIRE, requirement))
        return self
    
    def add_domain(self, domain: Union[Domain, str]) -> UCLPromptBuilder:
        """
        Add a domain (creates keyword-based condition automatically).

        This is the RECOMMENDED way to add domain-specific logic.
        Uses keyword CONDITIONS for optimal O_s.

        Args:
            domain: Domain instance or name from registry

        Example:
            .add_domain(LinearAlgebraDomain())
            .add_domain("vector_calculus")  # From registry
        """
        if isinstance(domain, str):
            domain = DomainRegistry.get(domain)
        self._domains.append(domain)
        return self

    # =========================================================================
    # UCL-TOOL METHODS (v2.1)
    # =========================================================================

    def add_tool(self, tool: ToolDefinition) -> UCLPromptBuilder:
        """
        Add a tool definition to the prompt.

        The tool will be wrapped in a keyword-based ^^CONDITION^^ block
        using its activation keywords. Tools without keywords are always-on
        (higher O_s).

        Args:
            tool: ToolDefinition instance

        Example:
            read_file = ToolDefinition(
                name="read_file",
                description="Read file content",
                parameters=[ToolParameter("path", "string", required=True)],
                keywords=["file", "read", "open"],
            )
            builder.add_tool(read_file)
        """
        self._tools.append(tool)

        # Warn if tool has no keywords (always-on = higher O_s)
        if not tool.keywords:
            self._warnings.append(Warning(
                severity="MEDIUM",
                message=f"Tool '{tool.name}' has no activation keywords",
                recommendation="Add keywords= parameter for keyword-based activation. "
                              "Tools without keywords are always-on (higher O_s).",
                estimated_impact=0.05
            ))

        return self

    def add_tool_category(self,
                         category_name: str,
                         keywords: List[str],
                         tools: List[ToolDefinition] = None,
                         description: str = "") -> UCLPromptBuilder:
        """
        Add a tool group (category) for efficient multi-tool organization.

        Groups tools under a single ^^CONDITION^^ block, saving O_s
        by eliminating redundant keyword conditions.

        Args:
            category_name: Group identifier
            keywords: Shared activation keywords
            tools: Initial list of tools (can add more later)
            description: Optional group description

        Example:
            builder.add_tool_category(
                "file_operations",
                keywords=["file", "path", "directory", "read", "write"],
                tools=[read_file_tool, write_file_tool, list_dir_tool],
                description="File system operations"
            )
        """
        group = ToolGroup(
            name=category_name,
            keywords=keywords,
            tools=tools or [],
            description=description
        )
        self._tool_groups[category_name] = group
        return self

    def add_tool_to_category(self,
                             category_name: str,
                             tool: ToolDefinition) -> UCLPromptBuilder:
        """
        Add a tool to an existing category.

        Args:
            category_name: Name of existing category
            tool: ToolDefinition to add

        Raises:
            ValueError: If category doesn't exist
        """
        if category_name not in self._tool_groups:
            raise ValueError(
                f"Category '{category_name}' not found. "
                f"Call add_tool_category() first."
            )
        self._tool_groups[category_name].add_tool(tool)
        return self

    def add_tool_domain(self, domain) -> UCLPromptBuilder:
        """
        Add a pre-built tool domain (e.g., FileSystemDomain, WebDomain).

        One line adds a complete tool ecosystem with optimal O_s grouping.

        Args:
            domain: ToolDomain instance

        Example:
            from tool_domains import FileSystemDomain, WebDomain, MCPDomain

            builder = (UCLPromptBuilder()
                .add_tool_domain(FileSystemDomain())   # 7 file tools
                .add_tool_domain(WebDomain())          # 5 web tools
                .add_tool_domain(MCPDomain())          # 5 MCP tools
            )
        """
        group = domain.create_tool_group()
        self._tool_groups[group.name] = group
        return self

    def add_condition(self,
                      keywords: List[str] = None,
                      expression: str = None,
                      body: List[UCLElement] = None) -> UCLPromptBuilder:
        """
        Add a custom condition.
        
        Prefer keyword-based conditions for lower O_s.
        
        Example:
            .add_condition(
                keywords=["special", "custom"],
                body=[Directive(DirectiveType.APPLY, "custom logic")]
            )
        """
        condition = Condition(
            keywords=keywords,
            expression=expression,
            body=body or []
        )
        self._conditions.append(condition)
        
        # Warn if not keyword-based
        if not condition.is_keyword_based:
            self._warnings.append(Warning(
                severity=Warning.Severity.MEDIUM if hasattr(Warning, 'Severity') else "MEDIUM",
                message="Non-keyword condition added",
                recommendation="Use keywords when possible for lower O_s",
                estimated_impact=0.02
            ))
        
        return self
    
    def add_switch(self, 
                   variable: str,
                   cases: Dict[str, List[UCLElement]],
                   default: List[UCLElement] = None) -> UCLPromptBuilder:
        """
        Add a SWITCH statement (DISCOURAGED).
        
        ⚠️ WARNING: SWITCH creates high O_s. Use keyword CONDITIONS instead.
        
        This method is provided for compatibility but will generate warnings.
        """
        case_objects = [
            Case(value=name, body=body) 
            for name, body in cases.items()
        ]
        switch = Switch(variable=variable, cases=case_objects, default=default)
        self._switches.append(switch)
        
        # Add warning
        case_count = len(cases) + (1 if default else 0)
        try:
            from .core import Warning as CoreWarning, Severity
        except ImportError:
            from core import Warning as CoreWarning, Severity
        self._warnings.append(CoreWarning(
            severity=Severity.HIGH,
            message=f"SWITCH with {case_count} cases added (O_s += {1.0 * __import__('math').log(case_count):.2f})",
            recommendation="Convert to keyword-based CONDITIONS",
            estimated_impact=UCLConstants.SWITCH_PENALTY_PER_CASE * case_count
        ))
        
        return self
    
    def set_output(self, spec: OutputSpec) -> UCLPromptBuilder:
        """
        Set output specification.
        
        Example:
            .set_output(JSONOutput(fields=["result", "scratchwork"]))
            .set_output(CodeOutput(language="python"))
        """
        self._output_spec = spec
        return self
    
    def add_section(self, name: str, elements: List[UCLElement]) -> UCLPromptBuilder:
        """
        Add a custom section.
        
        Example:
            .add_section("custom_rules", [
                Directive(DirectiveType.ENFORCE, "Custom rule 1"),
            ])
        """
        self._custom_sections.append(Section(name, elements))
        return self
    
    def add_element(self, element: UCLElement) -> UCLPromptBuilder:
        """
        Add a raw UCL element.
        
        For advanced usage when other methods don't suffice.
        """
        self._raw_elements.append(element)
        return self
    
    def add_transform(self, source: str, target: str) -> UCLPromptBuilder:
        """
        Add a standalone transformation.
        
        Example:
            .add_transform("fraction", "numerator over denominator")
        """
        self._raw_elements.append(Transform(source, target))
        return self
    
    # =========================================================================
    # BUILD AND SCORING
    # =========================================================================
    
    def build(self) -> BuiltPrompt:
        """
        Build the final prompt.
        
        Returns:
            BuiltPrompt object with text, score, and metadata
        """
        elements = self._assemble_elements()
        text = self._render_elements(elements)
        
        # Calculate score
        score = calculate_score(
            elements,
            has_critical=self._critical is not None,
            critical_at_top=self._critical_position == 0,
            total_lines=text.count('\n') + 1
        )
        
        # Add builder warnings
        all_warnings = score.warnings + self._warnings
        
        # Collect tool names for metadata (from both standalone and groups)
        tool_names = [t.name for t in self._tools]
        for group in self._tool_groups.values():
            for tool in group.tools:
                if tool.name not in tool_names:
                    tool_names.append(tool.name)
        group_names = list(self._tool_groups.keys())

        return BuiltPrompt(
            text=text,
            score=score,
            elements=elements,
            warnings=all_warnings,
            domains=[d.name for d in self._domains],
            tools=tool_names,
            tool_groups=group_names,
        )
    
    def preview(self) -> str:
        """Preview the prompt without full scoring"""
        elements = self._assemble_elements()
        return self._render_elements(elements)
    
    def get_score(self) -> UCLScore:
        """Get current score without building"""
        elements = self._assemble_elements()
        return calculate_score(
            elements,
            has_critical=self._critical is not None,
            critical_at_top=self._critical_position == 0
        )
    
    def get_Os(self) -> float:
        """Get current O_s estimate"""
        elements = self._assemble_elements()
        return sum(e.calculate_Os() for e in elements)
    
    def get_warnings(self) -> List[Warning]:
        """Get current warnings"""
        elements = self._assemble_elements()
        return detect_anti_patterns(elements, self._critical is not None) + self._warnings
    
    # =========================================================================
    # INTERNAL METHODS
    # =========================================================================
    
    def _assemble_elements(self) -> List[UCLElement]:
        """
        Assemble all elements in optimal order.

        Assembly order (optimized for O_s and LLM comprehension):
        1. System role (identity, CRITICAL, task, requirements)
        2. Domain conditions (keyword-based)
        3. Tool groups (shared ^^CONDITION^^ for related tools)
        4. Standalone tools (not in any group)
        5. Custom conditions
        6. Switches (discouraged)
        7. Custom sections
        8. Raw elements
        9. Output specification
        """
        elements = []

        # --- 1. System role section ---
        system_children = []

        # Identity first
        if self._identity:
            system_children.append(Assignment("ai_identity", self._identity))

        # CRITICAL directive (MUST be early!)
        if self._critical:
            system_children.append(self._critical)

        # Task directive (augment with tool awareness)
        if self._task:
            task_text = self._task
            # If tools are present, augment task with tool-awareness directive
            if self._tools or self._tool_groups:
                task_text += (
                    ". You have access to tools activated by keyword conditions; "
                    "invoke them only when their activation keywords are relevant"
                )
            system_children.append(
                Directive(DirectiveType.LLM, task_text)
            )

        # Requirements
        system_children.extend(self._requirements)

        if system_children:
            elements.append(Section("system_role", system_children))

        # --- 2. Domain conditions (keyword-based) ---
        for domain in self._domains:
            elements.append(domain.create_condition())

        # --- 3. Tool groups (shared condition wrapper saves O_s) ---
        grouped_tool_names = set()  # Track to avoid duplicates
        for group in self._tool_groups.values():
            elements.append(group)
            for tool in group.tools:
                grouped_tool_names.add(tool.name)

        # --- 4. Standalone tools (not in any group) ---
        for tool in self._tools:
            if tool.name not in grouped_tool_names:
                elements.append(tool)

        # --- 5. Custom conditions ---
        elements.extend(self._conditions)

        # --- 6. Switches (discouraged) ---
        elements.extend(self._switches)

        # --- 7. Custom sections ---
        elements.extend(self._custom_sections)

        # --- 8. Raw elements ---
        elements.extend(self._raw_elements)

        # --- 9. Output specification ---
        if self._output_spec:
            output_elements = self._output_spec.to_elements()
            elements.append(Section("output_specification", output_elements))

        return elements
    
    def _render_elements(self, elements: List[UCLElement]) -> str:
        """Render elements to text"""
        rendered = [elem.render() for elem in elements]
        return "\n\n".join(rendered)


# =============================================================================
# BUILT PROMPT
# =============================================================================

@dataclass
class BuiltPrompt:
    """
    A built UCL prompt with metadata.

    Contains:
    - text: The final prompt text
    - score: Quality score breakdown
    - elements: UCL elements used
    - warnings: Anti-pattern warnings
    - domains: Domains used
    - tools: Tool definition names
    - tool_groups: Tool group names
    """
    text: str
    score: UCLScore
    elements: List[UCLElement]
    warnings: List[Warning]
    domains: List[str]
    tools: List[str] = field(default_factory=list)
    tool_groups: List[str] = field(default_factory=list)
    
    def __str__(self) -> str:
        return self.text
    
    def to_dict(self) -> Dict[str, Any]:
        """Export to dictionary"""
        return {
            "prompt_text": self.text,
            "score": self.score.to_dict(),
            "domains": self.domains,
            "tools": self.tools,
            "tool_groups": self.tool_groups,
            "warnings_count": len(self.warnings),
            "element_count": len(self.elements),
            "lines": self.text.count('\n') + 1,
        }
    
    def save(self, path: str):
        """Save prompt to file"""
        with open(path, 'w', encoding='utf-8') as f:
            f.write(self.text)
    
    def print_report(self):
        """Print a detailed report"""
        print("=" * 70)
        print("UCL PROMPT REPORT")
        print("=" * 70)
        print()
        
        print("📊 SCORE:")
        print(f"   Quality: {self.score.Q_final:.1%} (Grade: {self.score.grade})")
        print(f"   Structural Overhead (O_s): {self.score.Os:.2f}")
        print(f"   Architecture: {self.score.architecture.name}")
        print()
        
        print("📐 BREAKDOWN:")
        print(f"   Base Quality Q(S): {self.score.Q_base:.3f}")
        if self.score.penalties:
            print("   Penalties:")
            for name, value in self.score.penalties.items():
                print(f"      - {name}: -{value:.3f}")
        if self.score.bonuses:
            print("   Bonuses:")
            for name, value in self.score.bonuses.items():
                print(f"      + {name}: +{value:.3f}")
        print()
        
        if self.warnings:
            print("⚠️ WARNINGS:")
            for w in self.warnings:
                print(f"   {w}")
            print()
        else:
            print("✅ No warnings - prompt is well optimized!")
            print()
        
        print("📝 DOMAINS USED:")
        for d in self.domains:
            print(f"   - {d}")
        print()

        if self.tools or self.tool_groups:
            print("🔧 TOOLS:")
            if self.tool_groups:
                for g in self.tool_groups:
                    print(f"   [group] {g}")
            for t in self.tools:
                print(f"   - {t}")
            print()

        print("📄 PROMPT PREVIEW (first 500 chars):")
        print("-" * 70)
        print(self.text[:500])
        if len(self.text) > 500:
            print(f"... ({len(self.text) - 500} more characters)")
        print("-" * 70)


# =============================================================================
# QUICK BUILDERS
# =============================================================================

def build_math_tts_prompt(identity: str = "Qwen",
                          domains: List[str] = None) -> BuiltPrompt:
    """
    Quick builder for math-to-TTS prompts (V4.1 style).
    
    Args:
        identity: AI identity name
        domains: List of domain names (defaults to all math domains)
    
    Returns:
        Optimized math-to-TTS prompt
    """
    # Lazy import to avoid circular dependency
    try:
        from .domains import DomainRegistry
    except ImportError:
        from domains import DomainRegistry
    
    builder = (UCLPromptBuilder()
        .set_identity(identity)
        .add_critical()
        .set_task("Transform mathematical expressions INTO TTS-compatible natural spoken language while preserving mathematical accuracy")
        .require("JSON output with validation_status")
    )
    
    # Add domains
    if domains:
        for d in domains:
            builder.add_domain(d)
    else:
        # Default: all math domains
        for domain in DomainRegistry.get_math_domains():
            builder.add_domain(domain)
    
    # Output spec
    builder.set_output(JSONOutput(
        fields=["problem_number", "problem_statement", "question_type", 
                "math_content_check", "final_answers"],
        validations=["TTS compatibility = 100%", "Zero mathematical symbols remaining"]
    ))
    
    return builder.build()


def build_code_prompt(identity: str = "CodeAssistant",
                      language: str = "python",
                      include_tests: bool = True) -> BuiltPrompt:
    """
    Quick builder for code generation prompts.
    
    Args:
        identity: AI identity name
        language: Target programming language
    
    Returns:
        Optimized code generation prompt
    """
    # Lazy imports to avoid circular dependency
    try:
        from .domains import CodeGenerationDomain, PythonDomain, JavaScriptDomain
    except ImportError:
        from domains import CodeGenerationDomain, PythonDomain, JavaScriptDomain
    
    builder = (UCLPromptBuilder()
        .set_identity(identity)
        .add_critical("Return ONLY code. No explanations unless in comments.")
        .set_task(f"Generate production-quality {language} code")
        .require("Syntactically valid code")
        .require("Error handling included")
    )
    
    builder.add_domain(CodeGenerationDomain())
    
    if language.lower() == "python":
        builder.add_domain(PythonDomain())
    elif language.lower() in ["javascript", "js", "typescript", "ts"]:
        builder.add_domain(JavaScriptDomain())
    
    builder.set_output(CodeOutput(language=language, include_tests=True))
    
    return builder.build()




# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Builder
    'UCLPromptBuilder', 'BuiltPrompt',
    
    # Output
    'OutputFormat', 'OutputSpec', 'JSONOutput', 'CodeOutput',
    
    # Quick builders
    'build_math_tts_prompt', 'build_code_prompt',
]
