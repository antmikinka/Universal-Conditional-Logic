# Universal Conditional Logic (UCL) - Research Code

**Experimental Data and Code for the UCL arxiv Paper**

[![arXiv](https://img.shields.io/badge/arXiv-2025-b31b1b.svg)]()
[![License](https://img.shields.io/badge/license-MIT-blue.svg)]()

## Overview

This repository contains the experimental code and raw output data supporting the UCL (Universal Conditional Logic) research paper. The study investigates how structured prompt engineering affects LLM output quality across multiple models.

## Repository Structure

```
00_ARXiv_Code/
└── 0_MATH_TUTOR_UCL_OG_NO_PROMPT_13_models/
    ├── run_models.py              # Test harness script
    ├── TEST_SUITE_DATA.csv        # Complete experimental results
    ├── TEST_SUITE_REPORT.md       # Analysis report
    │
    ├── claude_hybrid_ucl_math_prompt_v*.txt   # UCL prompt versions (V1-V4.1)
    ├── baseline prompt.txt        # Baseline (non-UCL) prompt
    ├── math_problem.jpeg          # Test input image
    │
    ├── baseline_*_OUTPUT.txt      # Baseline prompt outputs
    ├── no_prompt_*_OUTPUT.txt     # Control (no system prompt) outputs  
    ├── ucl_v*_*_OUTPUT.txt        # UCL version outputs
    └── *_META.json                # Metadata for each run
```

## Models Tested

| Provider | Model |
|----------|-------|
| Google | Gemini 3 Pro Preview, Gemma 3 27B |
| OpenAI | GPT-5 Mini |
| Anthropic | Claude (hybrid prompts) |
| Meta | Llama 4 Scout |
| Mistral | Medium 3, Small 3.2 24B |
| X.AI | Grok 4 |
| Baidu | ERNIE 4.5 (21B, VL 424B) |
| Alibaba | Qwen 3 VL 235B |
| Zhipu | GLM 4.6V |

## Experimental Conditions

1. **No Prompt (Control)**: No system prompt, user message only
2. **Baseline**: Traditional detailed system prompt
3. **UCL V1**: Initial UCL structured prompt
4. **UCL V2**: Expanded specification
5. **UCL V3**: Refined structure
6. **UCL V4**: Optimized for conciseness
7. **UCL V4.1**: Fine-tuned final version

Each condition was run **4 iterations** per model to assess consistency.

## Key Files

- **`run_models.py`**: Python script to run tests via OpenRouter API
- **`TEST_SUITE_DATA.csv`**: Raw numerical results
- **`TEST_SUITE_REPORT.md`**: Detailed analysis and findings

## Citation

```bibtex
@article{mikinka2025ucl,
  title={The Over-Specification Paradox: A Mathematical Framework for 
         Prompt Optimization in Large Language Models},
  author={Mikinka, Anthony},
  journal={arXiv preprint},
  year={2025}
}
```

## Author

Anthony Mikinka

## License

MIT License
