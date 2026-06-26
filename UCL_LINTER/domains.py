#!/usr/bin/env python3
"""
UCL Domain System
=================

Pre-built domains and custom domain builder for UCL prompts.

Domains encapsulate:
- Keywords for activation (used in keyword CONDITIONS)
- Transformations specific to the domain
- Requirements and validations
- Best practices for the domain

Author: Anthony Mikinka
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

try:
    from .core import (
        UCLElement, Condition, Section, Transform, Directive,
        DirectiveType, UCLConstants
    )
except ImportError:
    from core import (
        UCLElement, Condition, Section, Transform, Directive,
        DirectiveType, UCLConstants
    )


# =============================================================================
# DOMAIN BASE CLASS
# =============================================================================

@dataclass
class Domain(ABC):
    """
    Abstract base class for UCL domains.
    
    A domain represents a specific field or topic area with its own
    keywords, transformations, and requirements.
    
    Subclass this to create custom domains.
    """
    name: str
    description: str = ""
    
    @property
    @abstractmethod
    def keywords(self) -> List[str]:
        """Keywords that trigger this domain's condition"""
        pass
    
    @property
    @abstractmethod
    def transformations(self) -> List[Transform]:
        """Domain-specific transformations"""
        pass
    
    @property
    def requirements(self) -> List[Directive]:
        """Domain-specific requirements (optional)"""
        return []
    
    @property
    def emphases(self) -> List[Directive]:
        """Domain-specific emphases (optional)"""
        return []
    
    def create_condition(self) -> Condition:
        """
        Create a keyword-based condition for this domain.
        
        This is the RECOMMENDED way to add domain logic.
        O_s is minimized because the condition only activates
        when relevant keywords are present.
        """
        body = []
        
        # Add section with transformations
        section = Section(f"{self.name}_procedures", [])
        for transform in self.transformations:
            section.children.append(transform)
        for req in self.requirements:
            section.children.append(req)
        for emp in self.emphases:
            section.children.append(emp)
        
        body.append(section)
        
        return Condition(
            keywords=self.keywords,
            body=body
        )
    
    def get_section(self) -> Section:
        """
        Get domain content as a standalone section.
        
        WARNING: This creates UNCONDITIONAL content.
        Prefer create_condition() for lower O_s.
        """
        section = Section(f"{self.name}_procedures", [])
        for transform in self.transformations:
            section.children.append(transform)
        for req in self.requirements:
            section.children.append(req)
        return section


# =============================================================================
# MATH-TO-TTS DOMAINS (From original research)
# =============================================================================

@dataclass
class BasicMathDomain(Domain):
    """
    Basic mathematics domain for TTS conversion.
    
    Covers: fractions, exponents, roots, basic operations
    """
    name: str = "basic_math"
    description: str = "Basic mathematical notation to speech conversion"
    
    @property
    def keywords(self) -> List[str]:
        return ["math", "equation", "formula", "calculate", "solve"]
    
    @property
    def transformations(self) -> List[Transform]:
        return [
            Transform("negative_number", "negative absolute_value"),
            Transform("decimal", "integer_part point fractional_digits"),
            Transform("fraction", "numerator over denominator"),
            Transform("exponent", "base to the power of exponent"),
            Transform("square_root", "square root of radicand"),
            Transform("function_notation", "function_name of parameters"),
        ]


@dataclass
class LinearAlgebraDomain(Domain):
    """
    Linear algebra domain.
    
    Covers: matrices, eigenvalues, determinants, Gram-Schmidt, QR
    """
    name: str = "linear_algebra"
    description: str = "Linear algebra notation and procedures"
    
    @property
    def keywords(self) -> List[str]:
        return ["matrix", "eigenvalue", "eigenvector", "determinant", 
                "gram", "schmidt", "qr", "orthogonal", "orthonormal"]
    
    @property
    def transformations(self) -> List[Transform]:
        return [
            Transform("matrix_element", "matrix_name sub row sub column"),
            Transform("eigenvalue", "lambda sub index"),
            Transform("eigenvector", "eigenvector corresponding to lambda sub index"),
            Transform("determinant", "determinant of matrix matrix_name"),
            Transform("trace", "trace of matrix matrix_name"),
            Transform("norm", "norm of vector_expression"),
            Transform("inner_product", "inner product of vector_a and vector_b"),
        ]
    
    @property
    def emphases(self) -> List[Directive]:
        return [
            Directive(DirectiveType.EMPHASIZE, 
                     'For Gram-Schmidt: "r sub i sub j" for R-matrix, "q sub i" for Q-vectors')
        ]


@dataclass
class VectorCalculusDomain(Domain):
    """
    Vector calculus domain.
    
    Covers: curl, divergence, gradient, partial derivatives
    """
    name: str = "vector_calculus"
    description: str = "Vector calculus operations and notation"
    
    @property
    def keywords(self) -> List[str]:
        return ["curl", "divergence", "gradient", "del", "nabla", 
                "partial derivative", "vector field"]
    
    @property
    def transformations(self) -> List[Transform]:
        return [
            Transform("curl", "curl of vector_field"),
            Transform("divergence", "divergence of vector_field"),
            Transform("gradient", "gradient of scalar_field"),
            Transform("partial_derivative", "partial derivative of function with respect to variable"),
            Transform("del_operator", "del operator"),
        ]


@dataclass
class IntegralsDomain(Domain):
    """
    Integrals domain (single, double, triple, line, surface).
    """
    name: str = "integrals"
    description: str = "Integration notation and bounds"
    
    @property
    def keywords(self) -> List[str]:
        return ["integral", "integrate", "double integral", "triple integral",
                "line integral", "surface integral", "bounds", "limits"]
    
    @property
    def transformations(self) -> List[Transform]:
        return [
            Transform("integral", "integral from lower to upper of integrand d_variable"),
            Transform("double_integral", "double integral over region of integrand"),
            Transform("triple_integral", "triple integral over volume of integrand"),
            Transform("line_integral", "line integral over curve of integrand"),
            Transform("surface_integral", "surface integral over surface of integrand"),
        ]
    
    @property
    def requirements(self) -> List[Directive]:
        return [
            Directive(DirectiveType.REQUIRE, 
                     "Explicit vocalization of bounds: from lower_bound to upper_bound")
        ]


@dataclass
class DifferentialEquationsDomain(Domain):
    """
    Differential equations domain.
    """
    name: str = "differential_equations"
    description: str = "ODE and PDE notation"
    
    @property
    def keywords(self) -> List[str]:
        return ["differential equation", "ode", "pde", "derivative", 
                "laplace", "laplacian", "boundary"]
    
    @property
    def transformations(self) -> List[Transform]:
        return [
            Transform("first_derivative", "d function d variable"),
            Transform("second_derivative", "d squared function d variable squared"),
            Transform("nth_derivative", "d to the n function d variable to the n"),
            Transform("laplacian", "laplacian of function"),
            Transform("partial", "partial function partial variable"),
        ]


@dataclass
class StatisticsDomain(Domain):
    """
    Statistics and probability domain.
    """
    name: str = "statistics"
    description: str = "Statistical notation and probability"
    
    @property
    def keywords(self) -> List[str]:
        return ["probability", "statistics", "mean", "variance", 
                "standard deviation", "expected", "distribution"]
    
    @property
    def transformations(self) -> List[Transform]:
        return [
            Transform("expected_value", "expected value of random_variable"),
            Transform("variance", "variance of random_variable"),
            Transform("std_dev", "standard deviation of random_variable"),
            Transform("probability", "probability of event"),
            Transform("conditional", "probability of A given B"),
        ]


@dataclass
class PhysicsDomain(Domain):
    """
    Physics domain with units and dimensional analysis.
    """
    name: str = "physics"
    description: str = "Physical quantities and units"
    
    @property
    def keywords(self) -> List[str]:
        return ["unit", "dimension", "kg", "meter", "second", "newton",
                "joule", "watt", "ampere", "volt", "physics"]
    
    @property
    def transformations(self) -> List[Transform]:
        return [
            Transform("quantity_with_unit", "value unit_name"),
            Transform("vector_quantity", "magnitude direction"),
        ]
    
    @property
    def requirements(self) -> List[Directive]:
        return [
            Directive(DirectiveType.REQUIRE, "Dimensional consistency in calculations"),
            Directive(DirectiveType.APPLY, "Standard SI unit vocalization"),
        ]


# =============================================================================
# CODE GENERATION DOMAINS
# =============================================================================

@dataclass
class CodeGenerationDomain(Domain):
    """
    General code generation domain.
    """
    name: str = "code_generation"
    description: str = "Code generation and programming"
    
    @property
    def keywords(self) -> List[str]:
        return ["code", "function", "class", "implement", "write",
                "program", "script", "module", "method"]
    
    @property
    def transformations(self) -> List[Transform]:
        return [
            Transform("function_def", "function name takes parameters returns type"),
            Transform("class_def", "class name inherits from parent"),
            Transform("variable", "variable name of type"),
        ]
    
    @property
    def requirements(self) -> List[Directive]:
        return [
            Directive(DirectiveType.REQUIRE, "Production-quality code with error handling"),
            Directive(DirectiveType.REQUIRE, "Clear documentation and comments"),
        ]


@dataclass
class PythonDomain(Domain):
    """
    Python-specific code domain.
    """
    name: str = "python"
    description: str = "Python code generation"
    
    @property
    def keywords(self) -> List[str]:
        return ["python", "def", "class", "import", "pip", "venv",
                "django", "flask", "pandas", "numpy"]
    
    @property
    def transformations(self) -> List[Transform]:
        return [
            Transform("decorator", "at decorator_name above function"),
            Transform("list_comp", "expression for item in iterable if condition"),
            Transform("context_manager", "with resource as alias"),
            Transform("type_hint", "parameter colon type"),
        ]
    
    @property
    def requirements(self) -> List[Directive]:
        return [
            Directive(DirectiveType.ENFORCE, "PEP 8 style compliance"),
            Directive(DirectiveType.ENFORCE, "Type hints for function signatures"),
        ]


@dataclass
class JavaScriptDomain(Domain):
    """
    JavaScript/TypeScript domain.
    """
    name: str = "javascript"
    description: str = "JavaScript and TypeScript code"
    
    @property
    def keywords(self) -> List[str]:
        return ["javascript", "typescript", "js", "ts", "react", 
                "node", "npm", "async", "await", "promise"]
    
    @property
    def transformations(self) -> List[Transform]:
        return [
            Transform("arrow_function", "parameters arrow expression"),
            Transform("destructure", "curly properties close curly equals object"),
            Transform("template_literal", "backtick text dollar curly expression close curly text backtick"),
        ]


# =============================================================================
# ANALYSIS AND WRITING DOMAINS
# =============================================================================

@dataclass
class DataAnalysisDomain(Domain):
    """
    Data analysis and visualization domain.
    """
    name: str = "data_analysis"
    description: str = "Data analysis, visualization, and insights"
    
    @property
    def keywords(self) -> List[str]:
        return ["analyze", "data", "chart", "graph", "visualization",
                "trend", "correlation", "regression", "insight"]
    
    @property
    def transformations(self) -> List[Transform]:
        return [
            Transform("metric", "metric_name colon value"),
            Transform("trend", "trend_direction from start to end"),
            Transform("correlation", "correlation between var_a and var_b"),
        ]


@dataclass
class TechnicalWritingDomain(Domain):
    """
    Technical documentation domain.
    """
    name: str = "technical_writing"
    description: str = "Technical documentation and specifications"
    
    @property
    def keywords(self) -> List[str]:
        return ["document", "specification", "requirement", "api",
                "readme", "guide", "tutorial", "reference"]
    
    @property
    def transformations(self) -> List[Transform]:
        return [
            Transform("section_ref", "see section section_name"),
            Transform("code_ref", "see code_element in module"),
        ]
    
    @property
    def requirements(self) -> List[Directive]:
        return [
            Directive(DirectiveType.ENFORCE, "Clear structure with headings"),
            Directive(DirectiveType.ENFORCE, "Examples for complex concepts"),
        ]


# =============================================================================
# CUSTOM DOMAIN BUILDER
# =============================================================================

class CustomDomain(Domain):
    """
    User-defined custom domain.
    
    Create domains for any field by specifying:
    - Name and description
    - Keywords for activation
    - Custom transformations
    - Requirements and emphases
    
    Example:
        medical = CustomDomain(
            name="medical",
            description="Medical terminology processing",
            keywords=["diagnosis", "symptom", "treatment"],
            transformations=[
                ("dosage", "amount unit frequency"),
                ("icd_code", "ICD code code_number"),
            ],
            requirements=["HIPAA compliance"],
        )
    """
    
    def __init__(self,
                 name: str,
                 keywords: List[str],
                 transformations: List[tuple] = None,
                 requirements: List[str] = None,
                 emphases: List[str] = None,
                 description: str = ""):
        """
        Create a custom domain.
        
        Args:
            name: Domain identifier
            keywords: List of trigger keywords
            transformations: List of (source, target) tuples
            requirements: List of requirement strings
            emphases: List of emphasis strings
            description: Optional description
        """
        self._name = name
        self._description = description
        self._keywords = keywords
        self._transformations = transformations or []
        self._requirements = requirements or []
        self._emphases = emphases or []
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def description(self) -> str:
        return self._description
    
    @property
    def keywords(self) -> List[str]:
        return self._keywords
    
    @property
    def transformations(self) -> List[Transform]:
        return [Transform(src, tgt) for src, tgt in self._transformations]
    
    @property
    def requirements(self) -> List[Directive]:
        return [Directive(DirectiveType.REQUIRE, req) for req in self._requirements]
    
    @property
    def emphases(self) -> List[Directive]:
        return [Directive(DirectiveType.EMPHASIZE, emp) for emp in self._emphases]


# =============================================================================
# DOMAIN REGISTRY
# =============================================================================

class DomainRegistry:
    """
    Registry of available domains.
    
    Use to discover and instantiate domains by name.
    """
    
    _builtin_domains = {
        # Math/Science
        "basic_math": BasicMathDomain,
        "linear_algebra": LinearAlgebraDomain,
        "vector_calculus": VectorCalculusDomain,
        "integrals": IntegralsDomain,
        "differential_equations": DifferentialEquationsDomain,
        "statistics": StatisticsDomain,
        "physics": PhysicsDomain,
        
        # Code
        "code_generation": CodeGenerationDomain,
        "python": PythonDomain,
        "javascript": JavaScriptDomain,
        
        # Analysis/Writing
        "data_analysis": DataAnalysisDomain,
        "technical_writing": TechnicalWritingDomain,
    }
    
    _custom_domains: Dict[str, Domain] = {}
    
    @classmethod
    def list_domains(cls) -> List[str]:
        """List all available domain names"""
        return list(cls._builtin_domains.keys()) + list(cls._custom_domains.keys())
    
    @classmethod
    def get(cls, name: str) -> Domain:
        """Get a domain by name"""
        if name in cls._builtin_domains:
            return cls._builtin_domains[name]()
        elif name in cls._custom_domains:
            return cls._custom_domains[name]
        else:
            raise ValueError(f"Unknown domain: {name}. Available: {cls.list_domains()}")
    
    @classmethod
    def register(cls, domain: Domain):
        """Register a custom domain"""
        cls._custom_domains[domain.name] = domain
    
    @classmethod
    def get_math_domains(cls) -> List[Domain]:
        """Get all math-related domains"""
        return [
            BasicMathDomain(),
            LinearAlgebraDomain(),
            VectorCalculusDomain(),
            IntegralsDomain(),
            DifferentialEquationsDomain(),
            StatisticsDomain(),
            PhysicsDomain(),
        ]
    
    @classmethod
    def get_code_domains(cls) -> List[Domain]:
        """Get all code-related domains"""
        return [
            CodeGenerationDomain(),
            PythonDomain(),
            JavaScriptDomain(),
        ]


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Base
    'Domain', 'CustomDomain', 'DomainRegistry',
    
    # Math domains
    'BasicMathDomain', 'LinearAlgebraDomain', 'VectorCalculusDomain',
    'IntegralsDomain', 'DifferentialEquationsDomain', 'StatisticsDomain',
    'PhysicsDomain',
    
    # Code domains
    'CodeGenerationDomain', 'PythonDomain', 'JavaScriptDomain',
    
    # Other domains
    'DataAnalysisDomain', 'TechnicalWritingDomain',
]
