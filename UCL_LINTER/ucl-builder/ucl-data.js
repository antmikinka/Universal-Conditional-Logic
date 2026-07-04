// ============================================================================
// UCL SYNTAX DATA
// Complete reference of Universal Conditional Logic syntax elements
// ============================================================================

const UCL_SYNTAX = {
    'Getting Started': {
        description: 'Introduction to UCL principles',
        items: {
            'What is UCL?': {
                syntax: '',
                description: 'Universal Conditional Logic - A mathematical framework for prompt optimization based on empirical research (N=305, p < 2.89×10⁻¹³)',
                example: '',
                insertable: false
            },
            'Key Principle': {
                syntax: '',
                description: 'The Over-Specification Paradox: More detail can degrade performance. Optimal threshold S* ≈ 0.509',
                example: '',
                insertable: false
            },
            'Always Use CRITICAL': {
                syntax: '[[CRITICAL: content]]',
                description: '⭐ +9.3% quality bonus - Place at the top of every prompt!',
                example: '[[CRITICAL: Return ONLY JSON format]]',
                insertable: true,
                priority: 'HIGH'
            }
        }
    },

    'Core Syntax': {
        description: 'Fundamental UCL elements',
        items: {
            'Concept': {
                syntax: '{{concept:name:attribute}}',
                description: 'Variable or concept reference',
                example: '{{concept:ai_identity:processor}}',
                insertable: true
            },
            'Section': {
                syntax: '<section_name>\n  content\n</section_name>',
                description: 'Named content sections for organization',
                example: '<output_spec>\n  [[REQUIRE: JSON format]]\n</output_spec>',
                insertable: true
            }
        }
    },

    'Directives': {
        description: 'Core instruction elements [[TYPE: content]]',
        items: {
            'CRITICAL': {
                syntax: '[[CRITICAL: content]]',
                description: '⭐ HIGHEST PRIORITY - Blocks P_perceived penalty (+9.3% quality). Use at top of prompt!',
                example: '[[CRITICAL: Your ONLY output is JSON]]',
                insertable: true,
                priority: 'HIGH',
                os_impact: '-0.05'
            },
            'REQUIRE': {
                syntax: '[[REQUIRE: requirement]]',
                description: 'Specify required conditions or constraints',
                example: '[[REQUIRE: Accuracy >= 95%]]',
                insertable: true,
                os_impact: '0.5'
            },
            'TRANSFORM': {
                syntax: '[[TRANSFORM: source TO target]]',
                description: 'Define transformation rules',
                example: '[[TRANSFORM: {{concept:fraction}} TO "numerator over denominator"]]',
                insertable: true,
                os_impact: '1.0'
            },
            'VALIDATE': {
                syntax: '[[VALIDATE: validation_rule]]',
                description: 'Specify validation requirements',
                example: '[[VALIDATE: JSON syntax correctness]]',
                insertable: true,
                os_impact: '0.5'
            },
            'ENFORCE': {
                syntax: '[[ENFORCE: enforcement_rule]]',
                description: 'Strictly enforce a rule or constraint',
                example: '[[ENFORCE: PEP 8 style compliance]]',
                insertable: true,
                os_impact: '0.5'
            },
            'APPLY': {
                syntax: '[[APPLY: action_or_rule]]',
                description: 'Apply specific logic or rules',
                example: '[[APPLY: linear algebra rules]]',
                insertable: true,
                os_impact: '0.3'
            },
            'EMPHASIZE': {
                syntax: '[[EMPHASIZE: emphasis]]',
                description: 'Highlight important aspects',
                example: '[[EMPHASIZE: Clarity over brevity]]',
                insertable: true,
                os_impact: '0.2'
            },
            'SET': {
                syntax: '[[SET: variable = value]]',
                description: 'Set a variable or parameter',
                example: '[[SET: precision = high]]',
                insertable: true,
                os_impact: '0.2'
            },
            'LLM': {
                syntax: '[[LLM: instruction]]',
                description: 'Direct LLM-specific instruction',
                example: '[[LLM: Process mathematical expressions]]',
                insertable: true,
                os_impact: '0.3'
            },
            'LEARN': {
                syntax: '[[LEARN: learning_directive]]',
                description: 'Adaptive learning instruction',
                example: '[[LEARN: User preferences from context]]',
                insertable: true,
                os_impact: '0.3'
            },
            'ADAPT': {
                syntax: '[[ADAPT: adaptation_rule]]',
                description: 'Adaptive behavior specification',
                example: '[[ADAPT: Tone based on input formality]]',
                insertable: true,
                os_impact: '0.3'
            },
            'IMPROVE': {
                syntax: '[[IMPROVE: improvement_target]]',
                description: 'Continuous improvement directive',
                example: '[[IMPROVE: Response clarity]]',
                insertable: true,
                os_impact: '0.3'
            },
            'REDUCE': {
                syntax: '[[REDUCE: reduction_target]]',
                description: 'Minimize or reduce something',
                example: '[[REDUCE: Verbosity]]',
                insertable: true,
                os_impact: '0.2'
            },
            'REMOVE': {
                syntax: '[[REMOVE: removal_target]]',
                description: 'Eliminate specific elements',
                example: '[[REMOVE: Redundant explanations]]',
                insertable: true,
                os_impact: '0.2'
            },
            'ANALYZE': {
                syntax: '[[ANALYZE: analysis_subject]]',
                description: 'Perform analysis on specified subject',
                example: '[[ANALYZE: Input complexity]]',
                insertable: true,
                os_impact: '0.4'
            },
            'REVALIDATE': {
                syntax: '[[REVALIDATE: revalidation_rule]]',
                description: 'Re-check validation conditions',
                example: '[[REVALIDATE: Output format compliance]]',
                insertable: true,
                os_impact: '0.4'
            }
        }
    },

    'Conditional Logic': {
        description: 'Conditional execution patterns',
        items: {
            'Keyword Condition (Recommended)': {
                syntax: '^^CONDITION: keyword1, keyword2^^\n  [[ACTION: response]]\n^^/CONDITION^^',
                description: '✅ RECOMMENDED - Low O_s overhead. Activates when keywords present in input',
                example: '^^CONDITION: matrix, eigenvalue^^\n  [[APPLY: Linear algebra methods]]\n^^/CONDITION^^',
                insertable: true,
                priority: 'HIGH'
            },
            'Expression Condition': {
                syntax: '^^CONDITION: expression^^\n  body\n^^/CONDITION^^',
                description: 'Conditional based on expression evaluation',
                example: '^^CONDITION: input_length > 100^^\n  [[APPLY: Summarization]]\n^^/CONDITION^^',
                insertable: true
            },
            'SWITCH Statement': {
                syntax: '^^SWITCH: variable^^\n  ^^CASE: value1^^ action1 ^^/CASE^^\n  ^^CASE: value2^^ action2 ^^/CASE^^\n^^/SWITCH^^',
                description: '⚠️ NOT RECOMMENDED - High O_s penalty (avoid >4 cases)',
                example: '^^SWITCH: domain^^\n  ^^CASE: math^^ [[APPLY: Math rules]] ^^/CASE^^\n  ^^CASE: physics^^ [[APPLY: Physics]] ^^/CASE^^\n^^/SWITCH^^',
                insertable: true,
                warning: 'High structural overhead - prefer keyword conditions'
            }
        }
    },

    'Domains': {
        description: 'Pre-built domain knowledge systems',
        items: {
            'Mathematics': {
                syntax: '',
                description: 'Basic Math, Linear Algebra, Vector Calculus, Integrals, Statistics',
                example: 'Keywords: matrix, eigenvalue, derivative, integral, mean, variance',
                insertable: false
            },
            'Physics': {
                syntax: '',
                description: 'Classical mechanics, thermodynamics, electromagnetism',
                example: 'Keywords: force, energy, momentum, field, wave',
                insertable: false
            },
            'Programming': {
                syntax: '',
                description: 'Python, JavaScript, Code Generation',
                example: 'Keywords: function, class, async, promise, decorator',
                insertable: false
            },
            'Data Analysis': {
                syntax: '',
                description: 'Statistical analysis, data visualization',
                example: 'Keywords: correlation, regression, visualization, pandas',
                insertable: false
            },
            'Custom Domain Template': {
                syntax: '// Domain: domain_name\n// Keywords: keyword1, keyword2\n^^CONDITION: keyword1, keyword2^^\n  [[APPLY: domain-specific rules]]\n^^/CONDITION^^',
                description: 'Template for creating custom domains',
                example: '// Domain: medical\n// Keywords: diagnosis, treatment\n^^CONDITION: diagnosis, treatment^^\n  [[REQUIRE: Medical accuracy disclaimer]]\n^^/CONDITION^^',
                insertable: true
            }
        }
    },

    'Output Formats': {
        description: 'Structured output specifications',
        items: {
            'JSON Output': {
                syntax: '<output_format>\n  [[CRITICAL: Return ONLY valid JSON]]\n  [[REQUIRE: Fields: field1, field2]]\n  [[VALIDATE: JSON syntax]]\n</output_format>',
                description: 'Strict JSON output format',
                example: '<output_format>\n  [[CRITICAL: Return ONLY valid JSON]]\n  [[REQUIRE: Fields: result, confidence]]\n  [[VALIDATE: JSON syntax]]\n</output_format>',
                insertable: true
            },
            'Code Output': {
                syntax: '<output_format>\n  [[CRITICAL: Return ONLY code]]\n  [[REQUIRE: Language: language_name]]\n  [[ENFORCE: Style guide compliance]]\n</output_format>',
                description: 'Code-only output format',
                example: '<output_format>\n  [[CRITICAL: Return ONLY Python code]]\n  [[REQUIRE: Language: Python]]\n  [[ENFORCE: PEP 8 compliance]]\n</output_format>',
                insertable: true
            },
            'Structured Data': {
                syntax: '<output_format>\n  [[REQUIRE: Structure: description]]\n  [[VALIDATE: Format compliance]]\n</output_format>',
                description: 'Custom structured output',
                example: '<output_format>\n  [[REQUIRE: Structure: Markdown with headings]]\n  [[VALIDATE: Heading hierarchy]]\n</output_format>',
                insertable: true
            }
        }
    },

    'Best Practices': {
        description: 'UCL optimization guidelines',
        items: {
            '✅ DO: Start with CRITICAL': {
                syntax: '[[CRITICAL: content]]',
                description: 'ALWAYS place [[CRITICAL:]] at the very top of your prompt',
                example: '[[CRITICAL: Return ONLY the requested format]]',
                insertable: true
            },
            '✅ DO: Use Keyword CONDITIONS': {
                syntax: '^^CONDITION: keyword1, keyword2^^',
                description: 'Keyword-based conditions have lower O_s than SWITCH',
                example: '^^CONDITION: matrix, linear^^\n  [[APPLY: Linear algebra]]\n^^/CONDITION^^',
                insertable: true
            },
            '✅ DO: Keep O_s Low': {
                syntax: '',
                description: 'Minimize structural overhead - fewer branches, simpler logic',
                example: 'Use keyword CONDITIONS instead of complex SWITCH statements',
                insertable: false
            },
            '❌ AVOID: Role-Questioning': {
                syntax: '',
                description: 'Don\'t use conditions like "if answer provided" - causes -15% penalty',
                example: 'BAD: ^^CONDITION: answer_provided^^ (role-questioning)',
                insertable: false
            },
            '❌ AVOID: Large SWITCH': {
                syntax: '',
                description: 'SWITCH with >4 cases has high O_s penalty (+0.02 per case)',
                example: 'Use multiple keyword CONDITIONS instead',
                insertable: false
            },
            '❌ AVOID: Unconditional Procedures': {
                syntax: '',
                description: 'Always wrap domain logic in CONDITIONS (-6% penalty if unconditional)',
                example: 'Put [[APPLY: rules]] inside ^^CONDITION^^',
                insertable: false
            }
        }
    },

    'Anti-Patterns': {
        description: 'Common mistakes to avoid',
        items: {
            'Missing CRITICAL': {
                syntax: '',
                description: '🔴 CRITICAL SEVERITY: -9.3% quality penalty',
                example: 'Solution: Add [[CRITICAL:]] at the top of every prompt',
                insertable: false,
                severity: 'CRITICAL'
            },
            'SWITCH Architecture': {
                syntax: '',
                description: '🟠 HIGH SEVERITY: Increases O_s significantly',
                example: 'Solution: Replace with keyword-based CONDITIONS',
                insertable: false,
                severity: 'HIGH'
            },
            'Role-Questioning Conditionals': {
                syntax: '',
                description: '🔴 CRITICAL SEVERITY: -15% quality penalty',
                example: 'Avoid: ^^CONDITION: answer_provided^^',
                insertable: false,
                severity: 'CRITICAL'
            },
            'Unconditional Procedures': {
                syntax: '',
                description: '🟠 HIGH SEVERITY: -6% quality penalty',
                example: 'Solution: Wrap procedures in keyword CONDITIONS',
                insertable: false,
                severity: 'HIGH'
            }
        }
    },

    'Tool Syntax': {
        description: 'UCL tool integration syntax for 71 pre-built tools across 9 domains',
        items: {
            'Tool Directive': {
                syntax: '[[TOOL: tool_name]]',
                description: 'Reference a specific tool by name. Activates within a ^^CONDITION^^ block',
                example: '[[TOOL: read_file]]',
                insertable: true
            },
            'Parameter Directive': {
                syntax: '[[PARAM: name:type]]',
                description: 'Define a tool parameter with name and type (string, integer, boolean, object, array)',
                example: '[[PARAM: path:string]] [[PARAM: encoding:string]]',
                insertable: true
            },
            'Tool Condition Block': {
                syntax: '^^CONDITION: keyword1, keyword2^^\n  [[TOOL: tool_name]]\n  [[PARAM: name:type]]\n^^/CONDITION^^',
                description: 'Wrap tool references in keyword conditions for activation',
                example: '^^CONDITION: file, read, open^^\n  [[TOOL: read_file]]\n  [[PARAM: path:string]]\n^^/CONDITION^^',
                insertable: true,
                priority: 'HIGH'
            },
            'Tool Group': {
                syntax: '^^CONDITION: shared_keywords^^\n  [[TOOL: tool1]] ...\n  [[TOOL: tool2]] ...\n^^/CONDITION^^',
                description: 'Group multiple tools under shared condition for lower O_s overhead',
                example: '^^CONDITION: file, path, directory^^\n  [[TOOL: read_file]]\n  [[TOOL: write_file]]\n  [[TOOL: list_dir]]\n^^/CONDITION^^',
                insertable: true
            },
            '{{UCL_TOOLS}} Placeholder': {
                syntax: '{{UCL_TOOLS}}',
                description: 'Placeholder that expands to full tool definitions for selected domains (78% overhead reduction vs JSON Schema)',
                example: '{{UCL_TOOLS}}',
                insertable: true,
                priority: 'HIGH'
            },
            'Required Parameter': {
                syntax: '[[PARAM: name:type:required]]',
                description: 'Mark a parameter as required',
                example: '[[PARAM: path:string:required]]',
                insertable: true
            },
            'Tool with Return Type': {
                syntax: '[[TOOL: name]] → return_type',
                description: 'Specify expected return type (string, boolean, object, array, integer)',
                example: '[[TOOL: read_file]] → string',
                insertable: true
            }
        }
    }
};

// Export for use in main app
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { UCL_SYNTAX };
}
