#!/usr/bin/env python3
"""
UNIVERSAL CONDITIONAL LOGIC (UCL) FRAMEWORK
============================================
A Modular, Scalable System for AI Prompt Engineering

This framework implements the UCL mathematical model for creating
optimized prompts for any domain. It provides:

1. UCL Syntax Primitives - Building blocks for prompts
2. Domain System - Pre-built and custom domain support
3. Mathematical Scoring - Real-time quality prediction
4. Anti-Pattern Detection - Prevents common mistakes
5. Optimization Engine - Suggests improvements

Based on research: "The Over-Specification Paradox" (N=305, p<10⁻¹³)

Author: Anthony Mikinka
Version: 2.0
License: MIT
"""

from __future__ import annotations
import math
import re
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any, Union, Callable
from enum import Enum, auto


# =============================================================================
# MATHEMATICAL CONSTANTS (Empirically Derived)
# =============================================================================

class UCLConstants:
    """Mathematical constants from empirical research (N=305)"""

    # Quality Function Q(S)
    S_STAR = 0.509          # Optimal specification threshold
    Q_MAX = 1.0             # Maximum quality
    A_COEF = 1.96           # Linear coefficient (Q_MAX / S_STAR)
    B_COEF = 4.0            # Quadratic degradation rate

    # Structural Overhead O_s(A)
    GAMMA = 1.0             # Branching complexity weight
    DELTA = 0.1             # Procedural token weight

    # Bonuses and Penalties
    CRITICAL_BONUS = 0.093  # [[CRITICAL:]] effect (V4.1 - V4)
    KEYWORD_BONUS = 0.05    # Keyword architecture bonus
    ROLE_PENALTY = 0.15     # Role-questioning penalty
    UNCONDITIONAL_PENALTY = 0.06  # Unconditional procedures
    SWITCH_PENALTY_PER_CASE = 0.02  # Per-case SWITCH overhead


class UCLToolConstants:
    """Mathematical constants for UCL-Tool extensions (v2.1).

    These extend UCLConstants with tool-specific coefficients.
    Derived from comparative analysis of JSON Schema vs. keyword-conditional
    tool definitions (projected from Anthropic Tool Search Tool results).
    """
    # Tool overhead parameters
    TOOL_BASE_OS = 0.40          # Base overhead per tool declaration
    PARAM_OS = 0.15              # Per-parameter overhead
    REQUIRED_PARAM_DISCOUNT = -0.02  # Required params parse faster
    ENUM_PARAM_OVERHEAD = 0.05   # Per enum-value overhead
    DESC_OS = 0.08               # Description annotation overhead
    EXAMPLE_OS = 0.30            # Few-shot example overhead
    WHEN_OS = 0.50               # WHEN condition (keyword-based)

    # Grouping efficiency
    TOOL_GROUP_DISCOUNT = 0.10   # Per-tool discount when grouped
    SHARED_CONDITION_SAVINGS = 0.50  # Saved per eliminated duplicate condition

    # Quality bonuses
    TOOL_KEYWORD_BONUS = 0.03    # Bonus per tool with keyword activation
    EXAMPLE_BONUS = 0.02         # Bonus per few-shot example (improves accuracy)
    MCP_NATIVE_BONUS = 0.04      # Bonus for native MCP protocol encoding

    # Penalties
    TOOL_NO_KEYWORDS_PENALTY = 0.05   # Penalty for tools without WHEN (always-on)
    TOOL_NO_DESC_PENALTY = 0.03       # Penalty for tools without description
    TOOL_NAME_CONFLICT_PENALTY = 0.10 # Penalty for duplicate tool names
    DEPRECATED_TOOL_PENALTY = 0.08    # Penalty per deprecated tool retained

    # Comparative efficiency
    JSON_SCHEMA_OS_MULTIPLIER = 7.5   # JSON Schema is ~7.5x more overhead
    TOOL_COUNT_S_STAR_SHIFT = 0.002   # S* shifts up per tool (tools add useful spec)


# =============================================================================
# ENUMERATIONS
# =============================================================================

class DirectiveType(Enum):
    """Types of UCL directives [[TYPE: content]]"""
    # Existing directives (v2.0)
    LLM = "LLM"
    REQUIRE = "REQUIRE"
    TRANSFORM = "TRANSFORM"
    CRITICAL = "CRITICAL"
    VALIDATE = "VALIDATE"
    ENFORCE = "ENFORCE"
    APPLY = "APPLY"
    EMPHASIZE = "EMPHASIZE"
    SET = "SET"
    LEARN = "LEARN"
    ADAPT = "ADAPT"
    IMPROVE = "IMPROVE"
    REDUCE = "REDUCE"
    REMOVE = "REMOVE"
    ANALYZE = "ANALYZE"
    REVALIDATE = "REVALIDATE"

    # UCL-Tool directives (v2.1)
    TOOL = "TOOL"          # Tool declaration header
    PARAM = "PARAM"        # Parameter schema definition
    WHEN = "WHEN"          # Activation condition (keyword list)
    DESC = "DESC"          # Description/annotation
    EXAMPLE = "EXAMPLE"    # Few-shot input->output example


class ArchitectureType(Enum):
    """Prompt architecture classification"""
    KEYWORD = auto()    # V4/V4.1 style - optimal
    SWITCH = auto()     # V1/V2/V3 style - avoid
    MIXED = auto()      # Both present
    MINIMAL = auto()    # No branching


class Severity(Enum):
    """Warning/anti-pattern severity"""
    CRITICAL = "🔴 CRITICAL"
    HIGH = "🟠 HIGH"
    MEDIUM = "🟡 MEDIUM"
    LOW = "🟢 LOW"
    INFO = "ℹ️ INFO"


# =============================================================================
# BASE CLASSES
# =============================================================================

class UCLElement(ABC):
    """Abstract base class for all UCL elements"""
    
    @abstractmethod
    def render(self, indent: int = 0) -> str:
        """Render element to UCL syntax string"""
        pass
    
    @abstractmethod
    def calculate_Os(self) -> float:
        """Calculate structural overhead contribution"""
        pass
    
    def _indent(self, text: str, level: int) -> str:
        """Apply indentation to text"""
        if level == 0:
            return text
        prefix = "    " * level
        return "\n".join(prefix + line if line.strip() else line 
                        for line in text.split("\n"))


# =============================================================================
# UCL SYNTAX ELEMENTS
# =============================================================================

@dataclass
class Concept(UCLElement):
    """
    UCL Concept: {{concept:name:attribute}}
    
    Represents a variable or concept reference in UCL prompts.
    
    Example:
        Concept("ai_identity", "mathematical_processor")
        -> {{concept:ai_identity:mathematical_processor}}
    """
    name: str
    attribute: str = ""
    
    def render(self, indent: int = 0) -> str:
        if self.attribute:
            result = f"{{{{concept:{self.name}:{self.attribute}}}}}"
        else:
            result = f"{{{{concept:{self.name}}}}}"
        return self._indent(result, indent)
    
    def calculate_Os(self) -> float:
        return 0.1  # Minimal overhead


@dataclass
class Directive(UCLElement):
    """
    UCL Directive: [[TYPE: content]]
    
    Core instruction element in UCL. Different types have different effects.
    
    Types and their O_s contributions:
        CRITICAL: -0.05 (reduces P_perceived)
        LLM: 0.3
        REQUIRE: 0.5
        TRANSFORM: 1.0 (procedural)
        VALIDATE: 0.5
        ENFORCE: 0.5
        APPLY: 0.3
        EMPHASIZE: 0.2
    
    Example:
        Directive(DirectiveType.CRITICAL, "Your ONLY output is JSON")
        -> [[CRITICAL: Your ONLY output is JSON]]
    """
    dtype: DirectiveType
    content: str
    
    # O_s contributions by type
    OS_MAP = {
        DirectiveType.CRITICAL: -0.05,  # Reduces P_perceived!
        DirectiveType.LLM: 0.3,
        DirectiveType.REQUIRE: 0.5,
        DirectiveType.TRANSFORM: 1.0,
        DirectiveType.VALIDATE: 0.5,
        DirectiveType.ENFORCE: 0.5,
        DirectiveType.APPLY: 0.3,
        DirectiveType.EMPHASIZE: 0.2,
        DirectiveType.SET: 0.2,
        DirectiveType.LEARN: 0.3,
        DirectiveType.ADAPT: 0.3,
        DirectiveType.IMPROVE: 0.3,
        DirectiveType.REDUCE: 0.2,
        DirectiveType.REMOVE: 0.2,
        DirectiveType.ANALYZE: 0.4,
        DirectiveType.REVALIDATE: 0.4,
    }
    
    def render(self, indent: int = 0) -> str:
        result = f"[[{self.dtype.value}: {self.content}]]"
        return self._indent(result, indent)
    
    def calculate_Os(self) -> float:
        return self.OS_MAP.get(self.dtype, 0.3)


@dataclass  
class Transform(UCLElement):
    """
    UCL Transformation: [[TRANSFORM: source TO target USING pattern]]
    
    Defines a transformation rule for converting one representation to another.
    
    Example:
        Transform("fraction", "numerator over denominator")
        -> [[TRANSFORM: {{concept:fraction}} TO "numerator over denominator"]]
    """
    source: str
    target: str
    using: Optional[str] = None
    
    def render(self, indent: int = 0) -> str:
        source_ref = f"{{{{concept:{self.source}}}}}"
        if self.using:
            result = f'[[TRANSFORM: {source_ref} TO "{self.target}" USING "{self.using}"]]'
        else:
            result = f'[[TRANSFORM: {source_ref} TO "{self.target}"]]'
        return self._indent(result, indent)
    
    def calculate_Os(self) -> float:
        return 1.0  # Procedural content


@dataclass
class Condition(UCLElement):
    """
    UCL Condition: ^^CONDITION: expr^^ ... ^^/CONDITION^^
    
    Conditional block that activates based on expression evaluation.
    
    Two types:
    1. Keyword-based (RECOMMENDED): Activates when keywords present
       O_s ≈ 0.5 (efficient - only parses when needed)
    
    2. Forced: Always evaluated
       O_s ≈ 2.0 (inefficient - always parsed)
    
    Example (keyword-based):
        Condition(
            keywords=["integral", "derivative"],
            body=[Directive(DirectiveType.APPLY, "calculus rules")]
        )
        -> ^^CONDITION: {{concept:problem_content}} CONTAINS "integral" OR "derivative"^^
               [[APPLY: calculus rules]]
           ^^/CONDITION^^
    """
    expression: Optional[str] = None
    keywords: Optional[List[str]] = None
    body: List[UCLElement] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.expression and not self.keywords:
            raise ValueError("Condition requires either expression or keywords")
    
    @property
    def is_keyword_based(self) -> bool:
        return self.keywords is not None and len(self.keywords) > 0
    
    def render(self, indent: int = 0) -> str:
        # Build expression
        if self.keywords:
            kw_expr = ' OR '.join(f'"{kw}"' for kw in self.keywords)
            expr = f"{{{{concept:problem_content}}}} CONTAINS {kw_expr}"
        else:
            expr = self.expression
        
        # Build body
        body_lines = [elem.render(indent=1) for elem in self.body]
        body_str = "\n".join(body_lines)
        
        result = f"^^CONDITION: {expr}^^\n{body_str}\n^^/CONDITION^^"
        return self._indent(result, indent)
    
    def calculate_Os(self) -> float:
        # Keyword-based is much more efficient
        base = 0.5 if self.is_keyword_based else 2.0
        # Add body overhead
        body_os = sum(elem.calculate_Os() for elem in self.body)
        # Keyword conditions only add body when activated (lower effective Os)
        if self.is_keyword_based:
            body_os *= 0.3  # Only 30% of body parses on average
        return base + body_os


@dataclass
class Case(UCLElement):
    """
    UCL Case: ^^CASE: value^^ ... ^^/CASE^^
    
    A case within a SWITCH statement.
    
    WARNING: SWITCH/CASE architecture is DEPRECATED.
    Use keyword-based CONDITIONS instead for lower O_s.
    """
    value: str
    body: List[UCLElement] = field(default_factory=list)
    
    def render(self, indent: int = 0) -> str:
        body_lines = [elem.render(indent=1) for elem in self.body]
        body_str = "\n".join(body_lines)
        result = f"^^CASE: {{{{concept:{self.value}}}}}^^\n{body_str}\n^^/CASE:{self.value}^^"
        return self._indent(result, indent)
    
    def calculate_Os(self) -> float:
        return sum(elem.calculate_Os() for elem in self.body)


@dataclass
class Switch(UCLElement):
    """
    UCL Switch: ^^SWITCH: var^^ ... ^^/SWITCH^^
    
    Multi-branch selection structure.
    
    ⚠️ WARNING: SWITCH creates HIGH structural overhead!
    O_s contribution: γ × ln(num_cases)
    
    For 8 cases: O_s += 1.0 × ln(8) = 2.08
    
    RECOMMENDATION: Use keyword-based CONDITIONS instead.
    
    Example of BAD pattern (avoid):
        ^^SWITCH: question_type^^
            ^^CASE: math^^ ... ^^/CASE^^
            ^^CASE: physics^^ ... ^^/CASE^^
        ^^/SWITCH^^
    
    GOOD pattern (use instead):
        ^^CONDITION: content CONTAINS "math"^^ ... ^^/CONDITION^^
        ^^CONDITION: content CONTAINS "physics"^^ ... ^^/CONDITION^^
    """
    variable: Union[str, Concept]
    cases: List[Case] = field(default_factory=list)
    default: Optional[List[UCLElement]] = None
    _warned: bool = field(default=False, repr=False)
    
    def render(self, indent: int = 0) -> str:
        # Variable reference
        if isinstance(self.variable, Concept):
            var_ref = self.variable.render()
        else:
            var_ref = f"{{{{concept:{self.variable}}}}}"
        
        # Cases
        case_lines = [case.render(indent=1) for case in self.cases]
        
        # Default
        if self.default:
            default_body = "\n".join(elem.render(indent=2) for elem in self.default)
            case_lines.append(f"    ^^DEFAULT^^\n{default_body}\n    ^^/DEFAULT^^")
        
        cases_str = "\n\n".join(case_lines)
        result = f"^^SWITCH: {var_ref}^^\n{cases_str}\n^^/SWITCH:{self.variable if isinstance(self.variable, str) else self.variable.name}^^"
        return self._indent(result, indent)
    
    def calculate_Os(self) -> float:
        num_cases = len(self.cases) + (1 if self.default else 0)
        # Mathematical formula: γ × ln(C)
        branching_os = UCLConstants.GAMMA * math.log(max(1, num_cases))
        # Body overhead
        body_os = sum(case.calculate_Os() for case in self.cases)
        if self.default:
            body_os += sum(elem.calculate_Os() for elem in self.default)
        return branching_os + body_os


@dataclass
class Section(UCLElement):
    """
    UCL Section: <name> ... </name>
    
    Structural container for grouping related elements.
    Minimal O_s overhead.
    
    Example:
        Section("system_role", [
            Concept("ai_identity", "processor"),
            Directive(DirectiveType.LLM, "Process input")
        ])
        -> <system_role>
               {{concept:ai_identity:processor}}
               [[LLM: Process input]]
           </system_role>
    """
    name: str
    children: List[UCLElement] = field(default_factory=list)
    
    def render(self, indent: int = 0) -> str:
        children_lines = [child.render(indent=1) for child in self.children]
        children_str = "\n".join(children_lines)
        result = f"<{self.name}>\n{children_str}\n</{self.name}>"
        return self._indent(result, indent)
    
    def calculate_Os(self) -> float:
        # Section overhead plus children
        return 0.1 + sum(child.calculate_Os() for child in self.children)
    
    def add(self, element: UCLElement) -> Section:
        """Fluent API for adding elements"""
        self.children.append(element)
        return self


@dataclass
class Assignment(UCLElement):
    """
    UCL Assignment: {{concept:name}} = "value"
    
    Assigns a value to a concept variable.
    
    Example:
        Assignment("ai_identity", "Qwen")
        -> {{concept:ai_identity}} = "Qwen"
    """
    concept_name: str
    value: str
    
    def render(self, indent: int = 0) -> str:
        result = f'{{{{concept:{self.concept_name}}}}} = "{self.value}"'
        return self._indent(result, indent)
    
    def calculate_Os(self) -> float:
        return 0.05


@dataclass
class RawText(UCLElement):
    """
    Raw text element for custom content.

    Use sparingly - prefer structured UCL elements.
    """
    text: str

    def render(self, indent: int = 0) -> str:
        return self._indent(self.text, indent)

    def calculate_Os(self) -> float:
        # Estimate based on word count
        words = len(self.text.split())
        return words * 0.01


# =============================================================================
# UCL-TOOL ELEMENTS (v2.1)
# =============================================================================

@dataclass
class ToolParameter(UCLElement):
    """
    UCL Tool Parameter: [[PARAM: name, type, required|optional]]

    Defines a single parameter within a tool declaration.
    Maps to JSON Schema properties but at ~60% lower token cost.

    Fields:
        name:        Parameter identifier (snake_case convention)
        param_type:  Type string: "string" | "integer" | "float" | "boolean" |
                     "array" | "object" | "enum" | "file"
        required:    Whether parameter is mandatory
        description: Human-readable description (rendered via [[DESC:]])
        default:     Default value (None = no default)
        constraints: Validation constraints (min, max, pattern, enum values)

    O_s Model:
        Base overhead:  0.15 per parameter (vs 0.5 for JSON Schema property)
        Required bonus: -0.02 (required params are more efficiently parsed)
        Enum overhead:  +0.05 * len(enum_values) if type == "enum"

    Example:
        ToolParameter("path", "string", required=True,
                      description="File path to read")
        -> [[PARAM: path, string, required]]
           [[DESC: File path to read]]
    """
    name: str
    param_type: str = "string"
    required: bool = True
    description: str = ""
    default: Optional[Any] = None
    constraints: Dict[str, Any] = field(default_factory=dict)

    # Valid type strings
    VALID_TYPES = frozenset({
        "string", "integer", "float", "boolean",
        "array", "object", "enum", "file"
    })

    def __post_init__(self):
        if self.param_type not in self.VALID_TYPES:
            raise ValueError(
                f"Invalid param_type '{self.param_type}'. "
                f"Valid types: {self.VALID_TYPES}"
            )
        if self.param_type == "enum" and "values" not in self.constraints:
            raise ValueError("Enum type requires 'values' in constraints")

    def render(self, indent: int = 0) -> str:
        req_str = "required" if self.required else "optional"
        lines = [f"[[PARAM: {self.name}, {self.param_type}, {req_str}]]"]
        if self.description:
            lines.append(f"[[DESC: {self.description}]]")
        if self.default is not None:
            lines.append(f"[[DESC: Default: {self.default}]]")
        if self.constraints:
            constraint_str = ", ".join(
                f"{k}={v}" for k, v in self.constraints.items()
            )
            lines.append(f"[[DESC: Constraints: {constraint_str}]]")
        result = "\n".join(lines)
        return self._indent(result, indent)

    def calculate_Os(self) -> float:
        os_value = UCLToolConstants.PARAM_OS
        if self.required:
            os_value += UCLToolConstants.REQUIRED_PARAM_DISCOUNT
        if self.param_type == "enum":
            n_values = len(self.constraints.get("values", []))
            os_value += UCLToolConstants.ENUM_PARAM_OVERHEAD * n_values
        if self.param_type in ("array", "object"):
            os_value += 0.10  # Nested structure overhead
        return max(0.05, os_value)

    def to_json_schema_property(self) -> Dict[str, Any]:
        """Convert to JSON Schema property definition."""
        prop: Dict[str, Any] = {}
        type_map = {
            "string": "string", "integer": "integer",
            "float": "number", "boolean": "boolean",
            "array": "array", "object": "object",
            "file": "string",  # file paths as strings
        }
        prop["type"] = type_map.get(self.param_type, "string")
        if self.description:
            prop["description"] = self.description
        if self.default is not None:
            prop["default"] = self.default
        if self.param_type == "enum":
            prop["enum"] = self.constraints.get("values", [])
        if "min" in self.constraints:
            prop["minimum"] = self.constraints["min"]
        if "max" in self.constraints:
            prop["maximum"] = self.constraints["max"]
        if "pattern" in self.constraints:
            prop["pattern"] = self.constraints["pattern"]
        return prop


@dataclass
class ToolExample(UCLElement):
    """
    UCL Tool Example: [[EXAMPLE: input -> output]]

    Provides few-shot demonstration of tool usage.
    Critical for LLM disambiguation of tool intent.

    Fields:
        input_text:   Example input/invocation
        output_text:  Example output/result
        description:  Optional annotation of what this demonstrates

    O_s Model:
        Base overhead: 0.3 per example (procedural content)
        Word scaling:  +0.005 * total_words (beyond 20 words)

    Example:
        ToolExample(
            input_text="read_file path=/etc/hosts",
            output_text="127.0.0.1 localhost\\n..."
        )
        -> [[EXAMPLE: read_file path=/etc/hosts -> 127.0.0.1 localhost...]]
    """
    input_text: str
    output_text: str
    description: str = ""

    def render(self, indent: int = 0) -> str:
        if self.description:
            line1 = f"[[DESC: Example: {self.description}]]"
            line2 = f"[[EXAMPLE: {self.input_text} -> {self.output_text}]]"
            result = f"{line1}\n{line2}"
        else:
            result = f"[[EXAMPLE: {self.input_text} -> {self.output_text}]]"
        return self._indent(result, indent)

    def calculate_Os(self) -> float:
        total_words = len(self.input_text.split()) + len(self.output_text.split())
        base = UCLToolConstants.EXAMPLE_OS
        if total_words > 20:
            base += 0.005 * (total_words - 20)
        return base


@dataclass
class ToolDefinition(UCLElement):
    """
    UCL Tool Definition: [[TOOL: name]] ... [[/TOOL]]

    Complete tool declaration with parameters, description,
    activation keywords, and usage examples.

    This is the CORE primitive of UCL-Tool. It replaces the verbose
    JSON Schema tool definition pattern with keyword-conditional activation.

    Fields:
        name:         Tool identifier (snake_case, unique within prompt)
        description:  Human-readable tool description
        parameters:   List of ToolParameter objects
        keywords:     Activation keywords for WHEN condition
        examples:     List of ToolExample few-shot demos
        return_type:  Description of return value
        category:     Logical grouping category (e.g., "file_ops", "web")
        deprecated:   Whether tool is deprecated (generates warning)

    O_s Model:
        Base tool overhead:       0.40  (vs 2.0+ for JSON Schema equivalent)
        Per-parameter overhead:   sum(param.calculate_Os())
        WHEN condition overhead:  0.5   (keyword-based, same as Condition)
        Per-example overhead:     sum(example.calculate_Os())
        Category discount:        -0.10 if category is set (grouping efficiency)

    Activation Pattern:
        The tool is wrapped in a keyword-based ^^CONDITION^^ block:
        ^^CONDITION: {{concept:user_input}} CONTAINS "keyword1" OR "keyword2"^^
            [[TOOL: tool_name]]
            [[PARAM: ...]]
            ...
        ^^/CONDITION^^

        This means the tool definition is only "active" when the user's
        input contains the activation keywords, matching LLM tool-routing
        behavior while minimizing O_s.

    Example:
        ToolDefinition(
            name="read_file",
            description="Reads file content from disk",
            parameters=[
                ToolParameter("path", "string", required=True,
                              description="Path to the file"),
                ToolParameter("encoding", "string", required=False,
                              default="utf-8"),
            ],
            keywords=["file", "read", "open", "load"],
            examples=[
                ToolExample("read_file('/tmp/data.csv')", "col1,col2\\n1,2\\n...")
            ],
            category="file_ops"
        )
    """
    name: str
    description: str = ""
    parameters: List[ToolParameter] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    examples: List[ToolExample] = field(default_factory=list)
    return_type: str = ""
    category: str = ""
    deprecated: bool = False

    def __post_init__(self):
        if not self.name or not self.name.strip():
            raise ValueError("ToolDefinition requires a non-empty name")
        # Validate name is snake_case compatible
        if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', self.name):
            raise ValueError(
                f"Tool name '{self.name}' must be snake_case "
                f"(letters, digits, underscores, starting with letter/underscore)"
            )

    @property
    def required_parameters(self) -> List[ToolParameter]:
        return [p for p in self.parameters if p.required]

    @property
    def optional_parameters(self) -> List[ToolParameter]:
        return [p for p in self.parameters if not p.required]

    @property
    def has_condition(self) -> bool:
        return len(self.keywords) > 0

    def render(self, indent: int = 0) -> str:
        lines = []

        # Tool header
        lines.append(f"[[TOOL: {self.name}]]")

        # Description
        if self.description:
            lines.append(f"[[DESC: {self.description}]]")

        # Return type
        if self.return_type:
            lines.append(f"[[DESC: Returns: {self.return_type}]]")

        # Deprecated marker
        if self.deprecated:
            lines.append(f"[[DESC: DEPRECATED - use alternative tool]]")

        # Parameters
        for param in self.parameters:
            lines.append(param.render(indent=0))

        # Activation keywords (rendered as WHEN directive)
        if self.keywords:
            kw_str = ", ".join(self.keywords)
            lines.append(f"[[WHEN: {kw_str}]]")

        # Examples
        for example in self.examples:
            lines.append(example.render(indent=0))

        body = "\n".join(lines)

        # Wrap in keyword condition if keywords present
        if self.keywords:
            kw_expr = ' OR '.join(f'"{kw}"' for kw in self.keywords)
            condition_header = (
                f"^^CONDITION: {{{{concept:user_input}}}} CONTAINS {kw_expr}^^"
            )
            result = f"{condition_header}\n{body}\n^^/CONDITION^^"
        else:
            result = body

        return self._indent(result, indent)

    def render_flat(self, indent: int = 0) -> str:
        """Render without condition wrapper (for grouped/section use)."""
        lines = [f"[[TOOL: {self.name}]]"]
        if self.description:
            lines.append(f"[[DESC: {self.description}]]")
        if self.return_type:
            lines.append(f"[[DESC: Returns: {self.return_type}]]")
        for param in self.parameters:
            lines.append(param.render(indent=0))
        if self.keywords:
            kw_str = ", ".join(self.keywords)
            lines.append(f"[[WHEN: {kw_str}]]")
        for example in self.examples:
            lines.append(example.render(indent=0))
        result = "\n".join(lines)
        return self._indent(result, indent)

    def calculate_Os(self) -> float:
        # Base tool overhead (much lower than JSON Schema equivalent)
        os_value = UCLToolConstants.TOOL_BASE_OS

        # Parameter overhead
        for param in self.parameters:
            os_value += param.calculate_Os()

        # WHEN/condition overhead (keyword-based is efficient)
        if self.keywords:
            os_value += UCLToolConstants.WHEN_OS

        # Example overhead
        for example in self.examples:
            os_value += example.calculate_Os()

        # Category discount (grouped tools parse more efficiently)
        if self.category:
            os_value -= UCLToolConstants.TOOL_GROUP_DISCOUNT

        return max(0.20, os_value)

    def to_json_schema_tool(self) -> Dict[str, Any]:
        """Export as OpenAI/Anthropic-compatible tool definition."""
        properties = {}
        required_list = []

        for param in self.parameters:
            properties[param.name] = param.to_json_schema_property()
            if param.required:
                required_list.append(param.name)

        tool = {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required_list,
                }
            }
        }
        return tool

    @classmethod
    def from_json_schema_tool(cls, schema: Dict[str, Any],
                              keywords: List[str] = None) -> ToolDefinition:
        """Import from OpenAI/Anthropic tool definition format."""
        func = schema.get("function", schema)
        name = func["name"]
        description = func.get("description", "")

        params_schema = func.get("parameters", {})
        properties = params_schema.get("properties", {})
        required_names = set(params_schema.get("required", []))

        parameters = []
        for pname, pdef in properties.items():
            type_reverse_map = {
                "string": "string", "integer": "integer",
                "number": "float", "boolean": "boolean",
                "array": "array", "object": "object",
            }
            ptype = type_reverse_map.get(pdef.get("type", "string"), "string")

            constraints = {}
            if "enum" in pdef:
                ptype = "enum"
                constraints["values"] = pdef["enum"]
            if "minimum" in pdef:
                constraints["min"] = pdef["minimum"]
            if "maximum" in pdef:
                constraints["max"] = pdef["maximum"]
            if "pattern" in pdef:
                constraints["pattern"] = pdef["pattern"]

            parameters.append(ToolParameter(
                name=pname,
                param_type=ptype,
                required=(pname in required_names),
                description=pdef.get("description", ""),
                default=pdef.get("default"),
                constraints=constraints,
            ))

        return cls(
            name=name,
            description=description,
            parameters=parameters,
            keywords=keywords or [],
        )


@dataclass
class ToolGroup(UCLElement):
    """
    UCL Tool Group: Groups related tools under a shared condition.

    When multiple tools share activation keywords, grouping them
    under a single ^^CONDITION^^ block avoids redundant keyword
    evaluation and reduces O_s.

    O_s Savings vs. individual tools:
        N tools, each with K shared keywords:
        Individual: N * (tool_Os + 0.5 condition) = N * tool_Os + 0.5N
        Grouped:    group_condition(0.5) + N * tool_Os_flat
        Savings:    0.5 * (N - 1)

    Example:
        ToolGroup(
            name="file_operations",
            keywords=["file", "read", "write", "directory"],
            tools=[read_file_tool, write_file_tool, list_dir_tool]
        )
    """
    name: str
    keywords: List[str] = field(default_factory=list)
    tools: List[ToolDefinition] = field(default_factory=list)
    description: str = ""

    def render(self, indent: int = 0) -> str:
        lines = []

        # Group header
        if self.description:
            lines.append(f"<{self.name}>")
            lines.append(f"[[DESC: {self.description}]]")

        # Each tool rendered flat (no individual condition wrappers)
        for tool in self.tools:
            lines.append(tool.render_flat(indent=1 if self.description else 0))

        if self.description:
            lines.append(f"</{self.name}>")

        body = "\n".join(lines)

        # Wrap entire group in condition
        if self.keywords:
            kw_expr = ' OR '.join(f'"{kw}"' for kw in self.keywords)
            condition_header = (
                f"^^CONDITION: {{{{concept:user_input}}}} CONTAINS {kw_expr}^^"
            )
            result = f"{condition_header}\n{body}\n^^/CONDITION^^"
        else:
            result = body

        return self._indent(result, indent)

    def calculate_Os(self) -> float:
        # Single condition for the group
        os_value = UCLToolConstants.WHEN_OS if self.keywords else 0.0
        # Group header overhead
        if self.description:
            os_value += 0.1  # Section overhead
        # Tools without individual condition wrappers
        for tool in self.tools:
            tool_os = UCLToolConstants.TOOL_BASE_OS  # Base
            tool_os += sum(p.calculate_Os() for p in tool.parameters)
            tool_os += sum(e.calculate_Os() for e in tool.examples)
            if tool.category:
                tool_os -= UCLToolConstants.TOOL_GROUP_DISCOUNT
            os_value += max(0.20, tool_os)
        return os_value

    def add_tool(self, tool: ToolDefinition) -> ToolGroup:
        """Fluent API for adding tools."""
        self.tools.append(tool)
        return self


# =============================================================================
# CRITICAL DIRECTIVE TEMPLATE
# =============================================================================

def create_critical_directive(custom_content: Optional[str] = None) -> Directive:
    """
    Create a [[CRITICAL:]] directive using the optimal template.
    
    The [[CRITICAL:]] directive is THE KEY to achieving 100% quality.
    It MUST be placed at the TOP of system_role to block P_perceived.
    
    Args:
        custom_content: Custom message (uses template if None)
    
    Returns:
        Directive configured for CRITICAL type
    """
    if custom_content:
        content = custom_content
    else:
        content = """Your ONLY output is JSON. Begin your response IMMEDIATELY with the opening brace { character.
DO NOT output:
- Greeting or casual language ("Okay, let's tackle...")
- Reasoning or explanation of your process
- Meta-commentary about what you're doing
- Validation or checking steps
Internal calculations and scratchwork belong in the scratchwork_answer field INSIDE the JSON structure, not as exposition before the JSON."""
    
    return Directive(DirectiveType.CRITICAL, content)


# =============================================================================
# WARNINGS AND ANTI-PATTERNS
# =============================================================================

@dataclass
class Warning:
    """Anti-pattern warning"""
    severity: Severity
    message: str
    recommendation: str
    estimated_impact: float  # Quality loss estimate
    
    def __str__(self) -> str:
        return f"{self.severity.value}: {self.message}\n   → {self.recommendation}"


def detect_anti_patterns(elements: List[UCLElement], 
                         has_critical: bool = False) -> List[Warning]:
    """
    Detect anti-patterns in UCL element list.
    
    Returns list of warnings sorted by severity.
    """
    warnings = []
    
    # Check for SWITCH usage
    switches = [e for e in elements if isinstance(e, Switch)]
    for switch in switches:
        case_count = len(switch.cases) + (1 if switch.default else 0)
        warnings.append(Warning(
            severity=Severity.HIGH,
            message=f"SWITCH with {case_count} cases detected (O_s += {math.log(case_count):.2f})",
            recommendation="Convert to keyword-based CONDITIONS for lower O_s",
            estimated_impact=UCLConstants.SWITCH_PENALTY_PER_CASE * case_count
        ))
    
    # Check for forced conditions
    forced = [e for e in elements if isinstance(e, Condition) and not e.is_keyword_based]
    if forced:
        warnings.append(Warning(
            severity=Severity.MEDIUM,
            message=f"{len(forced)} forced (non-keyword) CONDITIONS detected",
            recommendation="Use keyword-based CONDITIONS when possible",
            estimated_impact=0.03 * len(forced)
        ))
    
    # Check for missing CRITICAL
    total_elements = len(elements)
    is_complex = total_elements > 5 or any(isinstance(e, (Switch, Condition)) for e in elements)
    if is_complex and not has_critical:
        warnings.append(Warning(
            severity=Severity.HIGH,
            message="Complex prompt without [[CRITICAL:]] directive",
            recommendation="Add [[CRITICAL:]] at TOP of system_role for +9.3% quality",
            estimated_impact=UCLConstants.CRITICAL_BONUS
        ))
    
    # Sort by severity
    severity_order = [Severity.CRITICAL, Severity.HIGH, Severity.MEDIUM, Severity.LOW, Severity.INFO]
    warnings.sort(key=lambda w: severity_order.index(w.severity))
    
    return warnings


# =============================================================================
# MATHEMATICAL SCORING
# =============================================================================

@dataclass
class UCLScore:
    """Complete UCL quality score"""
    Os: float                       # Structural overhead
    specification_level: float      # S value [0,1]
    Q_base: float                   # Base quality Q(S)
    penalties: Dict[str, float]     # Breakdown of penalties
    bonuses: Dict[str, float]       # Breakdown of bonuses
    Q_final: float                  # Final quality
    grade: str                      # Letter grade
    warnings: List[Warning]         # Anti-pattern warnings
    architecture: ArchitectureType  # Detected architecture
    
    def to_dict(self) -> Dict:
        return {
            "structural_overhead": round(self.Os, 3),
            "specification_level": round(self.specification_level, 3),
            "quality_base": round(self.Q_base, 3),
            "penalties": {k: round(v, 3) for k, v in self.penalties.items()},
            "bonuses": {k: round(v, 3) for k, v in self.bonuses.items()},
            "quality_final": round(self.Q_final, 3),
            "predicted_success_rate": f"{self.Q_final:.1%}",
            "grade": self.grade,
            "architecture": self.architecture.name,
            "warnings_count": len(self.warnings)
        }
    
    def __str__(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


def compute_Q(S: float) -> float:
    """
    Compute base quality Q(S) - The Over-Specification Paradox
    
    Q(S) = aS           for S ≤ S*
    Q(S) = Q_max - b(S-S*)²   for S > S*
    """
    if S <= UCLConstants.S_STAR:
        return UCLConstants.A_COEF * S
    else:
        return max(0, UCLConstants.Q_MAX - UCLConstants.B_COEF * (S - UCLConstants.S_STAR) ** 2)


def compute_Os(elements: List[UCLElement]) -> float:
    """Compute total structural overhead"""
    return sum(elem.calculate_Os() for elem in elements)


def calculate_score(elements: List[UCLElement],
                    has_critical: bool = False,
                    critical_at_top: bool = False,
                    total_lines: int = 0) -> UCLScore:
    """
    Calculate complete UCL score for a prompt.

    Args:
        elements: List of UCL elements
        has_critical: Whether [[CRITICAL:]] is present
        critical_at_top: Whether [[CRITICAL:]] is in first 10 lines
        total_lines: Total lines in prompt (for S estimation)

    Returns:
        UCLScore with complete breakdown
    """
    # Calculate O_s
    Os = compute_Os(elements)

    # Detect architecture
    has_switch = any(isinstance(e, Switch) for e in elements)
    has_keyword_cond = any(isinstance(e, Condition) and e.is_keyword_based for e in elements)

    if has_switch and has_keyword_cond:
        architecture = ArchitectureType.MIXED
    elif has_switch:
        architecture = ArchitectureType.SWITCH
    elif has_keyword_cond:
        architecture = ArchitectureType.KEYWORD
    else:
        architecture = ArchitectureType.MINIMAL

    # === UCL-TOOL: Collect tool definitions ===
    tool_definitions = [e for e in elements if isinstance(e, ToolDefinition)]
    tool_groups = [e for e in elements if isinstance(e, ToolGroup)]

    # Collect all tools (from direct definitions and groups)
    all_tools = list(tool_definitions)
    for group in tool_groups:
        all_tools.extend(group.tools)

    tool_count = len(all_tools)

    # Estimate specification level S
    # Based on element count and complexity
    element_factor = min(1.0, len(elements) / 20)
    os_factor = min(1.0, Os / 30)
    line_factor = min(1.0, total_lines / 200) if total_lines > 0 else element_factor

    # UCL-TOOL: Tool richness factor (tools add useful specification)
    tool_factor = min(1.0, tool_count / 20)  # 20+ tools = max richness

    S = (element_factor * 0.25 + os_factor * 0.35
         + line_factor * 0.25 + tool_factor * 0.15)

    # UCL-TOOL: Adjust S* upward for tool-heavy prompts
    # More tools = more useful specification capacity
    adjusted_s_star = UCLConstants.S_STAR + (
        UCLToolConstants.TOOL_COUNT_S_STAR_SHIFT * tool_count
    )

    # Base quality (using adjusted S*)
    if S <= adjusted_s_star:
        Q_base = UCLConstants.A_COEF * S
    else:
        Q_base = max(0, UCLConstants.Q_MAX - UCLConstants.B_COEF * (S - adjusted_s_star) ** 2)

    # Calculate penalties
    penalties = {}

    # SWITCH penalty
    switches = [e for e in elements if isinstance(e, Switch)]
    if switches:
        total_cases = sum(len(s.cases) + (1 if s.default else 0) for s in switches)
        penalties["switch_overhead"] = UCLConstants.SWITCH_PENALTY_PER_CASE * total_cases

    # === UCL-TOOL: Tool-specific penalties ===
    if tool_count > 0:
        # No-keywords penalty: tools without WHEN are always-on (higher Os)
        no_kw = sum(1 for t in all_tools if not t.keywords)
        if no_kw > 0:
            penalties["tools_without_activation"] = (
                UCLToolConstants.TOOL_NO_KEYWORDS_PENALTY * no_kw
            )

        # No-description penalty
        no_desc = sum(1 for t in all_tools if not t.description)
        if no_desc > 0:
            penalties["tools_without_description"] = (
                UCLToolConstants.TOOL_NO_DESC_PENALTY * no_desc
            )

        # Name conflict detection
        names = [t.name for t in all_tools]
        duplicates = len(names) - len(set(names))
        if duplicates > 0:
            penalties["tool_name_conflicts"] = (
                UCLToolConstants.TOOL_NAME_CONFLICT_PENALTY * duplicates
            )

        # Deprecated tools retained
        deprecated = sum(1 for t in all_tools if t.deprecated)
        if deprecated > 0:
            penalties["deprecated_tools_retained"] = (
                UCLToolConstants.DEPRECATED_TOOL_PENALTY * deprecated
            )

    # Calculate bonuses
    bonuses = {}

    if has_critical:
        if critical_at_top:
            bonuses["critical_directive"] = UCLConstants.CRITICAL_BONUS
        else:
            bonuses["critical_directive"] = UCLConstants.CRITICAL_BONUS * 0.7

    if architecture == ArchitectureType.KEYWORD:
        bonuses["keyword_architecture"] = UCLConstants.KEYWORD_BONUS

    # === UCL-TOOL: Tool-specific bonuses ===
    if tool_count > 0:
        # Keyword activation bonus: +0.03 per tool with WHEN keywords
        keyword_tools = sum(1 for t in all_tools if t.keywords)
        if keyword_tools > 0:
            bonuses["tool_keyword_activation"] = (
                UCLToolConstants.TOOL_KEYWORD_BONUS * keyword_tools
            )

        # Example bonus: +0.02 per tool with examples
        example_count = sum(len(t.examples) for t in all_tools)
        if example_count > 0:
            bonuses["tool_examples"] = (
                UCLToolConstants.EXAMPLE_BONUS * example_count
            )

    # Final quality
    total_penalty = sum(penalties.values())
    total_bonus = sum(bonuses.values())
    Q_final = max(0.0, min(1.0, Q_base - total_penalty + total_bonus))

    # Grade
    if Q_final >= 0.95:
        grade = "A"
    elif Q_final >= 0.85:
        grade = "B"
    elif Q_final >= 0.70:
        grade = "C"
    elif Q_final >= 0.50:
        grade = "D"
    else:
        grade = "F"

    # Warnings
    warnings = detect_anti_patterns(elements, has_critical)

    return UCLScore(
        Os=Os,
        specification_level=S,
        Q_base=Q_base,
        penalties=penalties,
        bonuses=bonuses,
        Q_final=Q_final,
        grade=grade,
        warnings=warnings,
        architecture=architecture
    )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Constants
    'UCLConstants', 'UCLToolConstants',

    # Enums
    'DirectiveType', 'ArchitectureType', 'Severity',

    # Base
    'UCLElement',

    # Elements (v2.0)
    'Concept', 'Directive', 'Transform', 'Condition', 'Case',
    'Switch', 'Section', 'Assignment', 'RawText',

    # UCL-Tool Elements (v2.1)
    'ToolParameter', 'ToolExample', 'ToolDefinition', 'ToolGroup',

    # Helpers
    'create_critical_directive',

    # Scoring
    'UCLScore', 'compute_Q', 'compute_Os', 'calculate_score',

    # Warnings
    'Warning', 'detect_anti_patterns',
]
