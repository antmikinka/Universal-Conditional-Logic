# Universal Conditional Logic (UCL) - Research Code

**Experimental Data and Code for the UCL arXiv Paper**

[![arXiv](https://img.shields.io/badge/arXiv-2601.00880-b31b1b.svg)](https://arxiv.org/abs/2601.00880)
[![DOI](https://img.shields.io/badge/DOI-10.48550%2FarXiv.2601.00880-blue)](https://doi.org/10.48550/arXiv.2601.00880)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Abstract

We present **Universal Conditional Logic (UCL)**, a mathematical framework for prompt optimization that transforms prompt engineering from heuristic practice into systematic optimization. Through systematic evaluation (N=305, 11 models, 4 iterations), we demonstrate:

- **29.8% token reduction** (t(10)=6.36, p < 0.001, Cohen's d = 2.01)
- Significant cost savings with maintained or improved output quality
- The **Over-Specification Paradox**: Beyond threshold S* = 0.509, additional specification degrades performance quadratically

This repository contains all experimental code, prompts, and raw model responses supporting these findings.

## Key Findings

### The Over-Specification Paradox

UCL reveals a counterintuitive phenomenon: **more detailed prompts can degrade AI performance**. Quality follows an inverted-U relationship:

```
Q(S) = a×S              for S ≤ S* (linear growth)
Q(S) = Q_max - b(S-S*)²  for S > S* (quadratic degradation)
```

Where S* ≈ 0.509 represents the optimal specification threshold.

### Validated Mechanisms

- **Indicator functions** (I_i ∈ {0,1}) for conditional logic
- **Structural overhead** (O_s = γ × Σ ln C_k) quantifying complexity
- **Early binding** for efficient LLM processing
- **Model-specific optimization**: Different architectures require version-specific adaptations (e.g., Llama 4 Scout with V4.1)

## Repository Structure

```
00_ARXiv_Code/
└── 0_MATH_TUTOR_UCL_OG_NO_PROMPT_13_models/
    ├── run_models.py                    # Test harness script
    ├── math_problem.jpeg                # Test input image
    │
    ├── prompts/                         # System prompts
    │   ├── baseline prompt.txt          # Traditional detailed prompt (17.6KB)
    │   ├── claude_hybrid_ucl_math_prompt_v1.txt    # Initial UCL (11.7KB)
    │   ├── claude_hybrid_ucl_math_prompt_v2.txt    # Expanded (18.3KB) - *Fails*
    │   ├── claude_hybrid_ucl_math_prompt_v3.txt    # Refined (15.1KB)
    │   ├── claude_hybrid_ucl_math_prompt_v4.txt    # Optimized (9.6KB)
    │   └── claude_hybrid_ucl_math_prompt_v4.1.txt  # Model-adapted (9.6KB)
    │
    ├── responses/                       # Model outputs organized by condition
    │   ├── baseline/        (86 files)  # Baseline prompt responses
    │   ├── no_prompt/       (86 files)  # Control (no system prompt)
    │   ├── ucl_v1/          (88 files)  # UCL V1 responses
    │   ├── ucl_v2/          (88 files)  # UCL V2 responses (over-specified)
    │   ├── ucl_v3/          (88 files)  # UCL V3 responses
    │   ├── ucl_v4/          (86 files)  # UCL V4 responses
    │   └── ucl_v4.1/        (88 files)  # UCL V4.1 (model-specific)
    │
    └── data/                            # Analysis data
        ├── TEST_SUITE_DATA.csv          # Complete experimental results (N=305)
        └── TEST_SUITE_REPORT.md         # Detailed analysis and findings
```

## Experimental Design

### Models Tested (N=13)

| Provider | Model | Type |
|----------|-------|------|
| Google | Gemini 3 Pro Preview | Standard |
| Google | Gemma 3 27B | Standard |
| OpenAI | GPT-5 Mini | Standard |
| Meta | Llama 4 Scout | Standard |
| Mistral | Medium 3 | Standard |
| Mistral | Small 3.2 24B | Standard |
| X.AI | Grok 4 | Standard |
| Baidu | ERNIE 4.5 21B | Reasoning |
| Baidu | ERNIE 4.5 VL 424B | Vision+Reasoning |
| Alibaba | Qwen 3 VL 235B | Vision+Reasoning |
| Zhipu | GLM 4.6V | Vision |

### Experimental Conditions

Each model was tested under 7 conditions × 4 iterations = 28 runs per model:

1. **No Prompt (Control)**: Zero system prompt - measures baseline model behavior
2. **Baseline**: Traditional detailed prompt - 17,637 characters of explicit instructions
3. **UCL V1**: Initial structured UCL - keyword-based conditions, moderate O_s
4. **UCL V2**: Expanded specification - **Over-specified (S > S*), demonstrates paradox**
5. **UCL V3**: Refined structure - balanced specification, improved clarity
6. **UCL V4**: Optimized conciseness - minimal O_s, maximum efficiency
7. **UCL V4.1**: Model-adapted - fine-tuning for specific architectures (Llama 4 Scout)

**Total Runs**: 13 models × 7 conditions × 4 iterations = **364 model invocations**

### Data Format

Each response file follows the naming convention:
```
{condition}_{provider}_{model}_ITER_{n}_{type}.{ext}
```

- **Condition**: `baseline`, `no_prompt`, `ucl_v1`, `ucl_v2`, `ucl_v3`, `ucl_v4`, `ucl_v4.1`
- **Provider**: `google`, `openai`, `meta-llama`, `mistralai`, `x-ai`, `baidu`, `qwen`, `z-ai`
- **Model**: Specific model identifier
- **Iteration**: 1-4
- **Type**: `OUTPUT` (response text) or `META` (metadata JSON)

## Reproducing the Experiments

### Prerequisites

```bash
# Python 3.8+
pip install openai requests python-dotenv
```

### Setup

1. **Configure API Key**:
   ```bash
   cd 00_ARXiv_Code/0_MATH_TUTOR_UCL_OG_NO_PROMPT_13_models/
   cp .env.example .env  # Create from template
   # Edit .env and add your OpenRouter API key:
   # OPENROUTER_API_KEY=your_key_here
   ```

2. **Run Experiments**:
   ```bash
   python run_models.py
   ```

   The script will:
   - Load all prompt variants from `prompts/`
   - Test each against all 13 models
   - Run 4 iterations per model/prompt combination
   - Save outputs to `responses/{condition}/`
   - Generate analysis in `data/`

### Analyzing Results

The `data/TEST_SUITE_DATA.csv` contains:
- Token counts (input/output)
- Response quality metrics
- Execution times
- Model metadata

Use `data/TEST_SUITE_REPORT.md` for comprehensive statistical analysis.

## Key Results Summary

| Metric | Baseline | UCL V4 | Improvement |
|--------|----------|--------|-------------|
| Avg Output Tokens | 4,521 | 3,173 | **-29.8%** |
| Prompt Size | 17.6 KB | 9.6 KB | **-45.5%** |
| API Cost (est.) | $0.045 | $0.032 | **-28.9%** |
| Quality Score | 0.82 | 0.89 | **+8.5%** |

*Statistical significance: t(10)=6.36, p < 0.001, Cohen's d = 2.01*

## Notable Findings

### V2 Catastrophic Failure
UCL V2 demonstrates the Over-Specification Paradox in action - excessive structural overhead (S > S*) led to:
- Verbose, unfocused responses
- Degraded task adherence
- Validation of quadratic degradation hypothesis

### Model-Specific Adaptation
Llama 4 Scout required V4.1 adaptation, revealing:
- Architecture-specific optimization needs
- Variable S* thresholds across model families
- Future research direction: per-model calibration

## Citation

If you use this code or data in your research, please cite:

### APA Format

Mikinka, A. (2025). Universal Conditional Logic: A formal language for prompt engineering. *arXiv preprint arXiv:2601.00880*. https://doi.org/10.48550/arXiv.2601.00880

### BibTeX

```bibtex
@article{mikinka2025ucl,
  title={Universal Conditional Logic: A Formal Language for Prompt Engineering},
  author={Mikinka, Anthony},
  journal={arXiv preprint arXiv:2601.00880},
  year={2025},
  url={https://arxiv.org/abs/2601.00880},
  doi={10.48550/arXiv.2601.00880},
  note={Supporting code and data: \url{https://github.com/antmikinka/Universal-Conditional-Logic}}
}
```

### MLA Format

Mikinka, Anthony. "Universal Conditional Logic: A Formal Language for Prompt Engineering." *arXiv preprint arXiv:2601.00880*, 2025, doi:10.48550/arXiv.2601.00880.

## Paper Information

- **arXiv ID**: [2601.00880](https://arxiv.org/abs/2601.00880)
- **DOI**: [10.48550/arXiv.2601.00880](https://doi.org/10.48550/arXiv.2601.00880)
- **Categories**: cs.AI, cs.CL, cs.LG, cs.PL, cs.SE
- **Pages**: 25 pages, 15 figures, 5 tables
- **Supplementary**: Prompt source code, 305 model responses (this repository)

## Author

**Anthony Mikinka**

## License

MIT License - See [LICENSE](LICENSE) file for details.

## Acknowledgments

This research validates UCL through systematic empirical evaluation across major LLM providers. All experiments were conducted via OpenRouter API for reproducibility.
