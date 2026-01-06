# UCL Architecture Validation - Comprehensive Analytics Report
Generated at: 2025-12-15 05:10:35

## 1. Experimental Design

| Category | Description | Purpose |
|----------|-------------|---------|
| **Baseline** | `math_gpt_system_prompt.txt` | Target prompt that UCL attempts to replicate |
| **Control** | No prompt (raw) | Measures raw model behavior |
| **UCL v1-v4.1** | Progressive UCL versions | Tests UCL architecture evolution |

> **Reference Model**: `qwen/qwen3-vl-235b-a22b-thinking` — The model UCL architecture was evolved with. Expected to have best performance.

## 2. Reference Model Performance (Benchmark)

The UCL architecture was developed and evolved using **`qwen/qwen3-vl-235b-a22b-thinking`**.
This model's performance represents the expected/optimal results.

### Reference Model Summary Stats

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Total Runs | 28 | Across all prompt conditions |
| JSON Validity | 85.7% | ⚠️ Needs improvement |
| Schema Compliance | 85.7% | ⚠️ Needs improvement |
| TTS Ready | 85.7% | ✅ Excellent |
| Has Main Answer | 85.7% | ⚠️ Needs improvement |
| Avg Duration | 56.76s | Response time |
| Avg Tokens | 5042 | Total token usage |
| TTS Keyword Density | 12.72% | Higher = better TTS formatting |

### Reference Model Per-Prompt Breakdown

| Prompt | JSON Valid | Schema % | TTS Ready | Duration | Tokens |
|--------|------------|----------|-----------|----------|--------|
| ucl_v1 | 100% | 100% | ✅ | 69.63s | 4660 |
| ucl_v2 | 100% | 100% | ✅ | 47.69s | 6202 |
| ucl_v3 | 100% | 100% | ✅ | 48.67s | 5461 |
| ucl_v4 | 100% | 100% | ✅ | 68.83s | 4985 |
| ucl_v4.1 | 100% | 100% | ✅ | 49.55s | 4408 |
| baseline | 100% | 100% | ✅ | 55.36s | 7000 |
| no_prompt | 0% | 0% | ❌ | 57.56s | 2575 |

## 3. Overall Summary

- **Total API Calls**: 305
- **Models Tested**: 13 (including reference model)
- **Prompt Conditions**: 7
- **JSON Validity Rate**: 233/305 (76.4%)
- **Full Schema Compliance**: 233/305 (76.4%)
- **TTS Ready Rate**: 233/305 (76.4%)

## 4. Category Comparison

| Category | Runs | JSON Valid % | Schema Compliance % | TTS Ready % | Avg Tokens | Avg Duration |
|----------|------|--------------|---------------------|-------------|------------|--------------|
| Baseline | 43 | 100.0% | 100.0% | 100.0% | 7228 | 36.13s |
| Control | 43 | 0.0% | 0.0% | 0.0% | 3597 | 54.52s |
| UCL | 219 | 86.8% | 86.8% | 86.8% | 6214 | 46.53s |

## 5. UCL Version Evolution Analysis

This section tracks the progression of UCL architecture across versions.

| Version | Runs | JSON Valid % | Schema % | TTS Ready % | Avg Completion Tokens | TTS Keyword Density |
|---------|------|--------------|----------|-------------|----------------------|---------------------|
| ucl_v1 | 44 | 72.7% | 72.7% | 72.7% | 2022 | 7.14% |
| ucl_v2 | 44 | 84.1% | 84.1% | 84.1% | 2539 | 11.03% |
| ucl_v3 | 44 | 86.4% | 86.4% | 86.4% | 2306 | 12.94% |
| ucl_v4 | 43 | 90.7% | 90.7% | 90.7% | 1886 | 12.81% |
| ucl_v4.1 | 44 | 100.0% | 100.0% | 100.0% | 2774 | 14.06% |

## 6. Baseline Gap Analysis

How close is each UCL version to matching baseline performance?

**Baseline Reference Values:**
- JSON Valid: 100.0%
- Schema Compliance: 100.0%
- TTS Ready: 100.0%

| Version | JSON Gap | Schema Gap | TTS Gap | Overall Gap |
|---------|----------|------------|---------|-------------|
| ucl_v1 | 27.3% | 27.3% | 27.3% | 27.3% |
| ucl_v2 | 15.9% | 15.9% | 15.9% | 15.9% |
| ucl_v3 | 13.6% | 13.6% | 13.6% | 13.6% |
| ucl_v4 | 9.3% | 9.3% | 9.3% | 9.3% |
| ucl_v4.1 | 0.0% | 0.0% | 0.0% | 0.0% |

## 7. Per-Model Performance

| Model | Total Runs | JSON Valid % | Avg Duration | Avg Tokens | Success Rate |
|-------|------------|--------------|--------------|------------|--------------|
| baidu/ernie-4.5-21b-a3b-thinking | 27 | 70.4% | 59.30s | 8618 | 70.4% |
| baidu/ernie-4.5-vl-424b-a47b | 27 | 74.1% | 182.69s | 11674 | 74.1% |
| google/gemini-3-pro-preview | 27 | 85.2% | 39.19s | 8166 | 85.2% |
| google/gemma-3-27b-it:free | 28 | 78.6% | 32.94s | 3322 | 78.6% |
| meta-llama/llama-4-scout | 28 | 28.6% | 10.91s | 4424 | 28.6% |
| mistralai/mistral-medium-3 | 28 | 85.7% | 8.70s | 4639 | 85.7% |
| mistralai/mistral-small-3.2-24b-instruct | 28 | 85.7% | 11.90s | 4447 | 85.7% |
| openai/gpt-5-mini | 28 | 85.7% | 43.90s | 5564 | 85.7% |
| qwen/qwen3-vl-235b-a22b-thinking | 28 | 85.7% | 56.76s | 5042 | 85.7% |
| x-ai/grok-4 | 28 | 85.7% | 36.23s | 5043 | 85.7% |
| z-ai/glm-4.6v | 28 | 75.0% | 30.68s | 5299 | 75.0% |

## 8. Efficiency Analysis

| Category | Avg Prompt Tokens | Avg Completion Tokens | Token Ratio | Cost Proxy |
|----------|-------------------|----------------------|-------------|------------|
| Baseline | 5318 | 1910 | 0.36 | 11048 |
| Control | 723 | 2874 | 3.97 | 9344 |
| UCL | 3906 | 2307 | 0.59 | 10828 |

## 9. Statistical Summary

| Metric | Mean | Std Dev | CV (%) | Min | Max |
|--------|------|---------|--------|-----|-----|
| Duration (s) | 46.19 | 58.98 | 127.69 | 1.67 | 352.79 |
| Total Tokens | 5987.67 | 3016.59 | 50.38 | 263 | 18312 |

---
*Lower CV (Coefficient of Variation) indicates more consistent performance.*

## 10. Model Input Capability Analysis

Shows which input mode was used for each model and fallback behavior.

| Model | Configured | Vision Used | Text Used | Vision Fallbacks |
|-------|------------|-------------|-----------|-----------------|
| qwen/qwen3-vl-235b-a22b-thinking | 🔄 auto | 28 | 0 | 0 |
| baidu/ernie-4.5-21b-a3b-thinking | 🔄 auto | 0 | 27 | 27 |
| baidu/ernie-4.5-vl-424b-a47b | 🔄 auto | 27 | 0 | 0 |
| google/gemini-3-pro-preview | 🔄 auto | 27 | 0 | 0 |
| google/gemma-3-27b-it:free | 🔄 auto | 27 | 1 | 1 |
| meta-llama/llama-4-scout | 🔄 auto | 28 | 0 | 0 |
| mistralai/mistral-medium-3 | 🔄 auto | 28 | 0 | 0 |
| mistralai/mistral-small-3.2-24b-instruct | 🔄 auto | 28 | 0 | 0 |
| openai/gpt-5-mini | 🔄 auto | 28 | 0 | 0 |
| x-ai/grok-4 | 🔄 auto | 28 | 0 | 0 |
| z-ai/glm-4.6v | 🔄 auto | 28 | 0 | 0 |

*Legend: 🔍=vision, 📝=text_only, 🔄=auto*

## 11. Cross-Iteration Consistency Analysis

Measures reproducibility across 4 iterations.
**100% consistency** = same JSON validity result every iteration.

- **Perfect Consistency (100%)**: 54 combinations
- **Partial Consistency**: 8 combinations
- **Zero Consistency (0%)**: 15 combinations

### Combinations with Variance (Needs Investigation)

| Model | Prompt | Iter 1 | Iter 2 | Iter 3 | Iter 4 | Consistency |
|-------|--------|-----|-----|-----|-----|-------------|
| ernie-4.5-21b-a3b-th | ucl_v1 | ✅ | ✅ | ❌ | ❌ | 50% |
| ernie-4.5-21b-a3b-th | ucl_v2 | ❌ | ✅ | ✅ | ❌ | 50% |
| ernie-4.5-vl-424b-a4 | ucl_v1 | ✅ | ❌ | ✅ | ❌ | 50% |
| gemma-3-27b-it:free | ucl_v1 | ❌ | ✅ | ✅ | ❌ | 50% |
| glm-4.6v | ucl_v1 | ✅ | ❌ | ❌ | ✅ | 50% |
| ernie-4.5-21b-a3b-th | ucl_v3 | ✅ | ✅ | ❌ | ✅ | 75% |
| ernie-4.5-vl-424b-a4 | ucl_v3 | ✅ | ❌ | ✅ | ✅ | 75% |
| glm-4.6v | ucl_v2 | ✅ | ❌ | ✅ | ✅ | 75% |

## 12. Iteration-by-Iteration Performance

Side-by-side comparison of each iteration's aggregate performance.

| Iteration | Runs | JSON Valid % | Avg Duration | Avg Tokens | TTS Ready % |
|-----------|------|--------------|--------------|------------|-------------|
| 1 | 77 | 77.9% | 45.94s | 5864 | 77.9% |
| 2 | 77 | 75.3% | 45.75s | 6288 | 75.3% |
| 3 | 75 | 76.0% | 47.45s | 5807 | 76.0% |
| 4 | 76 | 76.3% | 45.64s | 5987 | 76.3% |

**Trend Analysis (Iter 1 → Iter 4):**
- JSON Validity: ➡️ Stable (-1.6%)
- Duration: ➡️ Stable (-0.30s)

## 13. Response Variance Analysis

Coefficient of Variation (CV) measures consistency. Lower = more reliable.

| Reliability | CV Range | Description |
|-------------|----------|-------------|
| ✅ High | < 10% | Very consistent responses |
| ⚠️ Medium | 10-25% | Moderate variance |
| ❌ Low | > 25% | High variance, investigate |

### High Variance Combinations (CV > 25%)

| Model | Prompt | Duration CV | Token CV | Reliability |
|-------|--------|-------------|----------|-------------|
| glm-4.6v | ucl_v4.1 | 122.2% | 81.7% | ❌ Low |
| gemma-3-27b-it:free | ucl_v1 | 99.8% | 0.0% | ❌ Low |
| llama-4-scout | ucl_v4 | 85.2% | 1.6% | ❌ Low |
| llama-4-scout | ucl_v2 | 79.5% | 1.9% | ❌ Low |
| gemma-3-27b-it:free | ucl_v4.1 | 68.0% | 5.8% | ❌ Low |
| llama-4-scout | ucl_v1 | 65.1% | 2.5% | ❌ Low |
| ernie-4.5-vl-424b-a47b | ucl_v2 | 61.6% | 33.1% | ❌ Low |
| mistral-small-3.2-24b-ins | ucl_v2 | 59.8% | 0.7% | ❌ Low |
| ernie-4.5-21b-a3b-thinkin | ucl_v3 | 58.6% | 33.0% | ❌ Low |
| ernie-4.5-vl-424b-a47b | ucl_v3 | 58.0% | 20.5% | ❌ Low |
| mistral-small-3.2-24b-ins | baseline | 54.5% | 2.7% | ❌ Low |
| qwen3-vl-235b-a22b-thinki | ucl_v1 | 53.7% | 6.0% | ❌ Low |
| mistral-small-3.2-24b-ins | ucl_v1 | 52.3% | 1.5% | ❌ Low |
| ernie-4.5-21b-a3b-thinkin | ucl_v1 | 51.7% | 31.9% | ❌ Low |
| llama-4-scout | ucl_v4.1 | 51.0% | 1.0% | ❌ Low |

## 14. Study Reproducibility Summary

Overall quality score for academic publication readiness.

| Metric | Value | Weight |
|--------|-------|--------|
| JSON Consistency | 76.3% | 50% |
| Duration CV (inverted) | 32.8% | 25% |
| Token CV (inverted) | 8.6% | 25% |
|--------|-------|--------|
| **Reproducibility Score** | **67.5/100** | ⚠️ Moderate - Some variance |

### Score Interpretation

| Score Range | Quality | Recommendation |
|-------------|---------|----------------|
| 90-100 | ✅ Excellent | Ready for publication |
| 75-89 | ✅ Good | Publishable with notes |
| 60-74 | ⚠️ Moderate | Address variance issues |
| < 60 | ❌ Low | Investigate root causes |

## 15. Model Architecture Classification (4-Quadrant Analysis)

Empirical classification based on actual test run data.
Models are classified into 4 quadrants based on two dimensions:
- **Reasoning**: Model produced reasoning tokens (chain-of-thought)
- **Input Type**: Vision (image+text) or Text-Only

### 15.1 Quadrant Distribution

| Quadrant | Description | Models | Count |
|----------|-------------|--------|-------|
| Q1 | 🧠🔍 Reasoning + Vision | 6 | Best UCL support expected |
| Q2 | 🧠📝 Reasoning + Text | 1 | Strong reasoning, no image |
| Q3 | 📊🔍 Standard + Vision | 4 | Direct image, simpler logic |
| Q4 | 📊📝 Standard + Text | 0 | May need simpler UCL |

### 15.2 Per-Model Classification

| Model | Quadrant | Reasoning Tokens | Vision Rate | JSON % | Schema % |
|-------|----------|------------------|-------------|--------|----------|
| qwen3-vl-235b-a22b-thinking | 🧠🔍 Reasoning + Vision | 1363 | 100% | 85.7% | 85.7% |
| ernie-4.5-21b-a3b-thinking | 🧠📝 Reasoning + Text | 4602 | 0% | 70.4% | 70.4% |
| ernie-4.5-vl-424b-a47b | 🧠🔍 Reasoning + Vision | 6559 | 100% | 74.1% | 74.1% |
| gemini-3-pro-preview | 🧠🔍 Reasoning + Vision | 3480 | 100% | 85.2% | 85.2% |
| gemma-3-27b-it:free | 📊🔍 Standard + Vision | 0 | 96% | 78.6% | 78.6% |
| llama-4-scout | 📊🔍 Standard + Vision | 0 | 100% | 28.6% | 28.6% |
| mistral-medium-3 | 📊🔍 Standard + Vision | 0 | 100% | 85.7% | 85.7% |
| mistral-small-3.2-24b-instru | 📊🔍 Standard + Vision | 0 | 100% | 85.7% | 85.7% |
| gpt-5-mini | 🧠🔍 Reasoning + Vision | 1735 | 100% | 85.7% | 85.7% |
| grok-4 | 🧠🔍 Reasoning + Vision | 1060 | 100% | 85.7% | 85.7% |
| glm-4.6v | 🧠🔍 Reasoning + Vision | 856 | 100% | 75.0% | 75.0% |

## 16. UCL Performance by Model Quadrant

Compares UCL effectiveness across all 4 model architecture combinations.
*Critical for understanding how to optimize UCL per model type.*

### 16.1 Overall Performance by Quadrant

| Quadrant | Description | Models | Runs | JSON % | Schema % | TTS % |
|----------|-------------|--------|------|--------|----------|-------|
| Q1 | 🧠🔍 Reasoning+Vision | 6 | 166 | 81.9% | 81.9% | 81.9% |
| Q2 | 🧠📝 Reasoning+Text | 1 | 27 | 70.4% | 70.4% | 70.4% |
| Q3 | 📊🔍 Standard+Vision | 4 | 112 | 69.6% | 69.6% | 69.6% |

### 16.2 Best UCL Version per Quadrant

Identifies which UCL version works best for each model architecture.

| Quadrant | Best UCL | JSON % | Schema % | TTS % | Score |
|----------|----------|--------|----------|-------|-------|
| 🧠🔍 Reasoning+Vision | **ucl_v4** | 100.0% | 100.0% | 100.0% | 100.0 |
| 🧠📝 Reasoning+Text | **ucl_v4** | 100.0% | 100.0% | 100.0% | 100.0 |
| 📊🔍 Standard+Vision | **ucl_v4.1** | 100.0% | 100.0% | 100.0% | 100.0 |

## 17. UCL Architecture Optimization Guide

**UCL (Universal Conditional Logic)** is a structured prompt architecture using:
- `{{concept:variable:domain}}` - Universal expressions
- `^^CONDITION^^...^^/CONDITION^^` - Conditional logic
- `<<REPEAT>>...</REPEAT>>` - Loop structures
- `[[LLM:...]]` - Meta-instructions

**Key Insight**: UCL may need to be tweaked for different model architectures.

### 17.1 Per-Quadrant Observations

#### Q1: 🧠🔍 Reasoning + Vision Models
- **Models**: qwen3-vl-235b-a22b-thinking, ernie-4.5-vl-424b-a47b, gemini-3-pro-preview, gpt-5-mini, grok-4, glm-4.6v
- **Best UCL Version (observed)**: `ucl_v4`

#### Q2: 🧠📝 Reasoning + Text-Only Models
- **Models**: ernie-4.5-21b-a3b-thinking
- **Best UCL Version (observed)**: `ucl_v4`

#### Q3: 📊🔍 Standard + Vision Models
- **Models**: gemma-3-27b-it:free, llama-4-scout, mistral-medium-3, mistral-small-3.2-24b-instruct
- **Best UCL Version (observed)**: `ucl_v4.1`

#### Q4: 📊📝 Standard + Text-Only Models
- *No models in this quadrant*

### 17.2 Key Findings

- **Vision Impact on Reasoning**: Vision adds 11.6% JSON validity for reasoning models
- **Best Quadrant for UCL**: 🧠🔍 Reasoning + Vision (81.9% JSON validity)

### 17.3 Limitations & Future Work

> **⚠️ IMPORTANT LIMITATIONS**

The following factors may influence UCL performance but require additional research:

1. **Insufficient Data for Per-Quadrant UCL Modifications**
   - Current testing does not provide sufficient evidence to recommend specific UCL prompt modifications per quadrant
   - Observed performance differences may be due to factors other than quadrant classification
   - Further controlled experiments needed before making optimization recommendations

2. **Underlying Model Architecture (PyTorch/Implementation)**
   - The literal PyTorch architecture and training data of each model could play a significant role
   - Models within the same quadrant may have vastly different internal implementations
   - Transformer variants (encoder-decoder, decoder-only, MoE, etc.) may process UCL differently
   - Tokenization strategies and attention mechanisms vary between models

3. **Model Family Differences**
   - Even with same quadrant classification, model families (OpenAI, Anthropic, Google, Meta, etc.) may have unique response patterns
   - Training methodologies (RLHF, DPO, SFT, etc.) could affect UCL interpretation

4. **Sample Size Considerations**
   - Results are based on limited test iterations
   - Statistical significance of quadrant differences should be validated with larger samples

**Recommended Future Work:**
- Conduct targeted experiments with UCL prompt variations per quadrant
- Control for model family and architecture when comparing quadrants
- Increase sample size for statistical significance
- Test on diverse problem domains beyond math/TTS

---
*This analysis provides observational data. UCL optimization recommendations require additional controlled experimentation.*
