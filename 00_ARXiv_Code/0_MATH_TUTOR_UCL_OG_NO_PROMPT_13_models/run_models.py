import os
import base64
import json
import time
import statistics
import csv
from openai import OpenAI
from dotenv import load_dotenv

# ================= CONFIGURATION =================

# 1. SETUP API KEY
load_dotenv()
API_KEY = os.environ.get("OPENROUTER_API_KEY")
if not API_KEY:
    raise ValueError("OPENROUTER_API_KEY environment variable is required. Set it in .env file or environment.")

# 2. SPECIFY YOUR IMAGE FILE AND TEXT PROBLEM FILE
IMAGE_FILENAME = "math_problem.jpeg"
TEXT_PROBLEM_FILE = "math_problem.txt"

# 3. PROMPT CONFIGURATIONS
# This test suite compares:
# - UCL Experimental Prompts (v1-v4.1): Testing the UCL architecture evolution
# - Baseline Prompt: The original prompt that UCL is trying to replicate
# - No Prompt (Control): Raw model behavior with just the problem
PROMPT_CONFIGS = [
    # UCL Experimental Prompts (Attempting to replicate baseline)
    {"file": "claude_hybrid_ucl_math_prompt_v1.txt", "label": "ucl_v1", "category": "ucl", "version": 1.0},
    {"file": "claude_hybrid_ucl_math_prompt_v2.txt", "label": "ucl_v2", "category": "ucl", "version": 2.0},
    {"file": "claude_hybrid_ucl_math_prompt_v3.txt", "label": "ucl_v3", "category": "ucl", "version": 3.0},
    {"file": "claude_hybrid_ucl_math_prompt_v4.txt", "label": "ucl_v4", "category": "ucl", "version": 4.0},
    {"file": "claude_hybrid_ucl_math_prompt_v4.1.txt", "label": "ucl_v4.1", "category": "ucl", "version": 4.1},
    
    # Baseline Prompt (Target that UCL is trying to replicate)
    {"file": "math_gpt_system_prompt.txt", "label": "baseline", "category": "baseline", "version": 0},
    
    # No Prompt - Raw Model Behavior (Control Group)
    {"file": None, "label": "no_prompt", "category": "control", "version": -1},
]

# 4. REFERENCE MODEL (The model UCL architecture was evolved with)
# This model should have the best/expected performance - it's the benchmark
REFERENCE_MODEL = "qwen/qwen3-vl-235b-a22b-thinking"

# 5. TARGET MODELS LIST (Including reference model + comparison models)
TARGET_MODELS = [
    # === REFERENCE MODEL (UCL was developed with this model) ===
    "qwen/qwen3-vl-235b-a22b-thinking",  # PRIMARY - UCL architecture evolved with this model
    
    # === COMPARISON MODELS ===
    "baidu/ernie-4.5-21b-a3b-thinking",
    "baidu/ernie-4.5-vl-424b-a47b",
    "google/gemini-3-pro-preview",
    "google/gemma-3-27b-it:free",
    "meta-llama/llama-4-scout",
    "mistralai/mistral-medium-3",
    "mistralai/mistral-small-3.2-24b-instruct",
    "nvidia/nemotron-nano-12b-v2-v1:free",
    "openai/gpt-5-mini",
    "qwen/qwen3-v1-30b-a3b-thinking",
    "x-ai/grok-4",
    "z-ai/glm-4.6v",
]

# 6. MODEL INPUT CAPABILITIES
# All models set to "auto" - we don't definitively know capabilities
# "auto" = Try vision first, fallback to text if it fails
# The results will empirically show which models support vision
MODEL_CAPABILITIES = {
    # All models set to AUTO for empirical testing
    "qwen/qwen3-vl-235b-a22b-thinking": "auto",
    "baidu/ernie-4.5-21b-a3b-thinking": "auto",  # Reasoning model (based on name)
    "baidu/ernie-4.5-vl-424b-a47b": "auto",      # Vision model (based on "vl" in name)
    "google/gemini-3-pro-preview": "auto",
    "google/gemma-3-27b-it:free": "auto",
    "meta-llama/llama-4-scout": "auto",
    "mistralai/mistral-medium-3": "auto",
    "mistralai/mistral-small-3.2-24b-instruct": "auto",
    "nvidia/nemotron-nano-12b-v2-v1:free": "auto",
    "openai/gpt-5-mini": "auto",
    "qwen/qwen3-v1-30b-a3b-thinking": "auto",
    "x-ai/grok-4": "auto",
    "z-ai/glm-4.6v": "auto",
}

# 7. ANALYTICS CONFIGURATION
# Required JSON fields for schema compliance checking
REQUIRED_JSON_FIELDS = ["problem_number", "problem_statement", "question_type", "math_content_check", "final_answers"]
REQUIRED_ANSWER_FIELDS = ["type", "values", "answer_format"]
TTS_KEYWORDS = ["quantity", "comma", "equals", "plus", "minus", "divided by", "times", "squared", "cubed", "root"]

# =================================================

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=API_KEY,
    timeout=120.0,
)

def encode_image(image_path):
    """Encodes the image to base64 for the API."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def read_text_file(file_path):
    """Reads the content of the prompt text file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: Could not find file {file_path}")
        return None

def get_model_capability(model_name):
    """
    Returns the input capability of a model.
    - 'vision': Supports image+text input
    - 'text_only': Only supports text input
    - 'auto': Unknown, try vision first then fallback to text
    """
    return MODEL_CAPABILITIES.get(model_name, "auto")

def get_output_filename(model_name, prompt_label, iteration):
    """Generate the output filename for resume capability checking."""
    safe_model_name = model_name.replace("/", "_").replace(":", "")
    return f"{prompt_label}_{safe_model_name}_ITER_{iteration}_OUTPUT.txt"

# ================= ANALYTICS FUNCTIONS =================

def calculate_json_schema_compliance(content):
    """
    Calculate how well the JSON response matches the expected schema.
    Returns a dict with compliance metrics.
    """
    result = {
        "is_valid_json": False,
        "json_parsed": None,
        "schema_compliance_pct": 0.0,
        "fields_present": [],
        "fields_missing": [],
        "has_main_answer": False,
        "has_scratchwork": False,
        "answer_format_correct": False,
        "tts_ready": False
    }
    
    try:
        # Extract JSON from markdown code blocks if present
        json_str = content
        if "```json" in content:
            parts = content.split("```json")
            if len(parts) > 1:
                json_str = parts[1].split("```")[0].strip()
        elif "```" in content:
            parts = content.split("```")
            if len(parts) > 1:
                json_str = parts[1].split("```")[0].strip()
        
        parsed = json.loads(json_str)
        result["is_valid_json"] = True
        result["json_parsed"] = parsed
        
        # Check required fields
        for field in REQUIRED_JSON_FIELDS:
            if field in parsed:
                result["fields_present"].append(field)
            else:
                result["fields_missing"].append(field)
        
        result["schema_compliance_pct"] = (len(result["fields_present"]) / len(REQUIRED_JSON_FIELDS)) * 100
        
        # Check final_answers structure
        if "final_answers" in parsed and isinstance(parsed["final_answers"], dict):
            fa = parsed["final_answers"]
            
            # Check main_answer
            if "main_answer" in fa and isinstance(fa["main_answer"], dict):
                result["has_main_answer"] = True
                ma = fa["main_answer"]
                
                # Check answer format fields
                answer_fields_present = sum(1 for f in REQUIRED_ANSWER_FIELDS if f in ma)
                result["answer_format_correct"] = answer_fields_present == len(REQUIRED_ANSWER_FIELDS)
                
                # Check TTS ready
                if ma.get("answer_format") == "tts_ready":
                    result["tts_ready"] = True
            
            # Check scratchwork
            if "scratchwork_answer" in fa:
                result["has_scratchwork"] = True
                
    except (json.JSONDecodeError, TypeError, AttributeError):
        pass
    
    return result

def calculate_content_metrics(content):
    """
    Calculate content-level metrics from the response text.
    """
    if not content:
        return {
            "char_count": 0,
            "word_count": 0,
            "line_count": 0,
            "tts_keyword_count": 0,
            "tts_keyword_density": 0.0
        }
    
    char_count = len(content)
    words = content.lower().split()
    word_count = len(words)
    line_count = content.count('\n') + 1
    
    # Count TTS keywords
    content_lower = content.lower()
    tts_keyword_count = sum(content_lower.count(kw) for kw in TTS_KEYWORDS)
    tts_keyword_density = (tts_keyword_count / word_count * 100) if word_count > 0 else 0
    
    return {
        "char_count": char_count,
        "word_count": word_count,
        "line_count": line_count,
        "tts_keyword_count": tts_keyword_count,
        "tts_keyword_density": round(tts_keyword_density, 2)
    }

def calculate_efficiency_metrics(usage_stats):
    """
    Calculate efficiency metrics from token usage.
    """
    prompt_tokens = usage_stats.get("prompt_tokens", 0)
    completion_tokens = usage_stats.get("completion_tokens", 0)
    total_tokens = usage_stats.get("total_tokens", 0)
    reasoning_tokens = usage_stats.get("reasoning_tokens", 0)
    duration = usage_stats.get("duration_seconds", 0)
    
    return {
        "token_efficiency_ratio": round(completion_tokens / prompt_tokens, 3) if prompt_tokens > 0 else 0,
        "reasoning_overhead_pct": round(reasoning_tokens / completion_tokens * 100, 2) if completion_tokens > 0 else 0,
        "ms_per_token": round(duration / total_tokens * 1000, 2) if total_tokens > 0 else 0,
        "tokens_per_second": round(total_tokens / duration, 1) if duration > 0 else 0,
        # Cost proxy: typical pricing ratio is ~3:1 output:input
        "cost_proxy": round(prompt_tokens * 1.0 + completion_tokens * 3.0, 0)
    }

def calculate_aggregate_stats(values):
    """
    Calculate aggregate statistics for a list of numeric values.
    """
    if not values:
        return {"mean": 0, "std_dev": 0, "cv": 0, "min": 0, "max": 0, "count": 0}
    
    n = len(values)
    mean = sum(values) / n
    
    if n > 1:
        std_dev = statistics.stdev(values)
        cv = (std_dev / mean * 100) if mean > 0 else 0  # Coefficient of Variation
    else:
        std_dev = 0
        cv = 0
    
    return {
        "mean": round(mean, 2),
        "std_dev": round(std_dev, 2),
        "cv": round(cv, 2),  # Lower = more consistent
        "min": round(min(values), 2),
        "max": round(max(values), 2),
        "count": n
    }

def calculate_consistency_by_combination(all_stats):
    """
    Groups results by (model, prompt) across iterations to measure reproducibility.
    Returns dict with consistency metrics for each combination.
    """
    combinations = {}
    
    for stat in all_stats:
        key = (stat['model'], stat['prompt_label'])
        if key not in combinations:
            combinations[key] = {
                'iterations': [],
                'json_valid': [],
                'durations': [],
                'tokens': [],
                'tts_ready': [],
                'schema_pct': []
            }
        
        combinations[key]['iterations'].append(stat['iteration'])
        combinations[key]['json_valid'].append(stat['schema_compliance']['is_valid_json'])
        combinations[key]['durations'].append(stat['usage']['duration_seconds'])
        combinations[key]['tokens'].append(stat['usage']['total_tokens'])
        combinations[key]['tts_ready'].append(stat['schema_compliance']['tts_ready'])
        combinations[key]['schema_pct'].append(stat['schema_compliance']['schema_compliance_pct'])
    
    # Calculate consistency metrics for each combination
    results = {}
    for key, data in combinations.items():
        n = len(data['iterations'])
        
        # JSON consistency = % of iterations with valid JSON
        json_valid_count = sum(1 for v in data['json_valid'] if v)
        json_consistency = (json_valid_count / n * 100) if n > 0 else 0
        
        # Duration variance
        duration_stats = calculate_aggregate_stats(data['durations'])
        
        # Token variance
        token_stats = calculate_aggregate_stats(data['tokens'])
        
        # TTS consistency
        tts_valid_count = sum(1 for v in data['tts_ready'] if v)
        tts_consistency = (tts_valid_count / n * 100) if n > 0 else 0
        
        results[key] = {
            'n_iterations': n,
            'json_valid_count': json_valid_count,
            'json_consistency_pct': round(json_consistency, 1),
            'json_valid_per_iter': data['json_valid'],
            'duration_mean': duration_stats['mean'],
            'duration_cv': duration_stats['cv'],
            'token_mean': token_stats['mean'],
            'token_cv': token_stats['cv'],
            'tts_consistency_pct': round(tts_consistency, 1),
            'schema_pct_mean': round(sum(data['schema_pct']) / n, 1) if n > 0 else 0
        }
    
    return results

def calculate_reproducibility_score(all_stats, consistency_data):
    """
    Calculates an overall reproducibility score (0-100) for the study.
    Weighted by: JSON consistency (50%), Duration CV (25%), Token CV (25%)
    """
    if not consistency_data:
        return {"score": 0, "interpretation": "No data"}
    
    # Calculate averages across all combinations
    n = len(consistency_data)
    
    avg_json_consistency = sum(c['json_consistency_pct'] for c in consistency_data.values()) / n
    
    # Lower CV is better, so we invert: high CV = low score
    avg_duration_cv = sum(c['duration_cv'] for c in consistency_data.values()) / n
    avg_token_cv = sum(c['token_cv'] for c in consistency_data.values()) / n
    
    # Convert CV to a 0-100 score (CV of 0 = 100, CV of 50+ = 0)
    duration_score = max(0, 100 - avg_duration_cv * 2)
    token_score = max(0, 100 - avg_token_cv * 2)
    
    # Weighted average: JSON consistency is most important
    overall_score = (avg_json_consistency * 0.50 + 
                     duration_score * 0.25 + 
                     token_score * 0.25)
    
    # Interpretation
    if overall_score >= 90:
        interpretation = "✅ Excellent - Highly reproducible"
    elif overall_score >= 75:
        interpretation = "✅ Good - Publishable quality"
    elif overall_score >= 60:
        interpretation = "⚠️ Moderate - Some variance"
    else:
        interpretation = "❌ Low - High variance, needs investigation"
    
    return {
        "score": round(overall_score, 1),
        "interpretation": interpretation,
        "avg_json_consistency": round(avg_json_consistency, 1),
        "avg_duration_cv": round(avg_duration_cv, 1),
        "avg_token_cv": round(avg_token_cv, 1),
        "n_combinations": n
    }

def classify_models_by_architecture(all_stats):
    """
    Empirically classifies models based on actual usage data:
    - reasoning_model: Has reasoning_tokens > 0 in any run
    - vision_capable: Successfully used vision mode in at least one run
    - text_only: Never successfully used vision (all text or fallbacks)
    
    This is critical for UCL optimization per model architecture.
    """
    model_classification = {}
    
    for stat in all_stats:
        model = stat['model']
        if model not in model_classification:
            model_classification[model] = {
                'has_reasoning': False,
                'vision_success_count': 0,
                'text_mode_count': 0,
                'total_runs': 0,
                'reasoning_tokens_list': [],
                'json_valid_list': [],
                'schema_pct_list': [],
                'tts_ready_list': []
            }
        
        mc = model_classification[model]
        mc['total_runs'] += 1
        
        # Check reasoning
        reasoning_tokens = stat['usage'].get('reasoning_tokens', 0) or 0
        if reasoning_tokens > 0:
            mc['has_reasoning'] = True
        mc['reasoning_tokens_list'].append(reasoning_tokens)
        
        # Check input mode
        if stat.get('input_mode') == 'vision':
            mc['vision_success_count'] += 1
        else:
            mc['text_mode_count'] += 1
        
        # Track performance metrics
        mc['json_valid_list'].append(stat['schema_compliance']['is_valid_json'])
        mc['schema_pct_list'].append(stat['schema_compliance']['schema_compliance_pct'])
        mc['tts_ready_list'].append(stat['schema_compliance']['tts_ready'])
    
    # Calculate derived metrics
    for model, mc in model_classification.items():
        n = mc['total_runs']
        mc['avg_reasoning_tokens'] = round(sum(mc['reasoning_tokens_list']) / n, 0) if n > 0 else 0
        mc['vision_capable'] = mc['vision_success_count'] > 0
        mc['text_only'] = mc['vision_success_count'] == 0
        mc['vision_rate'] = round(mc['vision_success_count'] / n * 100, 1) if n > 0 else 0
        mc['json_valid_pct'] = round(sum(1 for v in mc['json_valid_list'] if v) / n * 100, 1) if n > 0 else 0
        mc['schema_pct_avg'] = round(sum(mc['schema_pct_list']) / n, 1) if n > 0 else 0
        mc['tts_ready_pct'] = round(sum(1 for v in mc['tts_ready_list'] if v) / n * 100, 1) if n > 0 else 0
        
        # 4-Quadrant Classification
        # Q1: Reasoning + Vision | Q2: Reasoning + Text-Only
        # Q3: Standard + Vision  | Q4: Standard + Text-Only
        if mc['has_reasoning'] and mc['vision_capable']:
            mc['quadrant'] = 'reasoning_vision'
            mc['quadrant_label'] = '🧠🔍 Reasoning + Vision'
            mc['quadrant_short'] = 'Q1'
        elif mc['has_reasoning'] and mc['text_only']:
            mc['quadrant'] = 'reasoning_text'
            mc['quadrant_label'] = '🧠📝 Reasoning + Text'
            mc['quadrant_short'] = 'Q2'
        elif not mc['has_reasoning'] and mc['vision_capable']:
            mc['quadrant'] = 'standard_vision'
            mc['quadrant_label'] = '📊🔍 Standard + Vision'
            mc['quadrant_short'] = 'Q3'
        else:
            mc['quadrant'] = 'standard_text'
            mc['quadrant_label'] = '📊📝 Standard + Text'
            mc['quadrant_short'] = 'Q4'
    
    return model_classification

# ================= OUTPUT FUNCTIONS =================

def save_output(model_name, prompt_label, prompt_category, content, reasoning, usage_stats, 
                schema_metrics, content_metrics, efficiency_metrics, iteration):
    """Saves the response, reasoning, and comprehensive metadata to files."""
    safe_model_name = model_name.replace("/", "_").replace(":", "")
    
    filename_txt = f"{prompt_label}_{safe_model_name}_ITER_{iteration}_OUTPUT.txt"
    filename_meta = f"{prompt_label}_{safe_model_name}_ITER_{iteration}_META.json"
    
    # 1. Save Text Output
    with open(filename_txt, "w", encoding="utf-8") as f:
        f.write(f"{'='*60}\n")
        f.write(f" MODEL: {model_name}\n")
        f.write(f" PROMPT: {prompt_label} ({prompt_category})\n")
        f.write(f" ITERATION: {iteration}\n")
        f.write(f" INPUT MODE: {usage_stats.get('input_mode', 'unknown')}\n")
        f.write(f"{'='*60}\n\n")
        
        f.write(f"=== TOKEN USAGE ===\n")
        f.write(f"Prompt Tokens: {usage_stats['prompt_tokens']}\n")
        f.write(f"Completion Tokens: {usage_stats['completion_tokens']}\n")
        f.write(f"Total Tokens: {usage_stats['total_tokens']}\n")
        if usage_stats['reasoning_tokens'] > 0:
            f.write(f"Reasoning Tokens: {usage_stats['reasoning_tokens']}\n")
        f.write(f"Duration: {usage_stats['duration_seconds']}s\n\n")
        
        f.write(f"=== SCHEMA COMPLIANCE ===\n")
        f.write(f"Valid JSON: {schema_metrics['is_valid_json']}\n")
        f.write(f"Schema Compliance: {schema_metrics['schema_compliance_pct']:.1f}%\n")
        f.write(f"Fields Present: {schema_metrics['fields_present']}\n")
        f.write(f"Fields Missing: {schema_metrics['fields_missing']}\n")
        f.write(f"Has Main Answer: {schema_metrics['has_main_answer']}\n")
        f.write(f"Has Scratchwork: {schema_metrics['has_scratchwork']}\n")
        f.write(f"TTS Ready: {schema_metrics['tts_ready']}\n\n")
        
        f.write(f"=== CONTENT METRICS ===\n")
        f.write(f"Characters: {content_metrics['char_count']}\n")
        f.write(f"Words: {content_metrics['word_count']}\n")
        f.write(f"Lines: {content_metrics['line_count']}\n")
        f.write(f"TTS Keywords: {content_metrics['tts_keyword_count']}\n")
        f.write(f"TTS Keyword Density: {content_metrics['tts_keyword_density']}%\n\n")
        
        f.write(f"=== EFFICIENCY METRICS ===\n")
        f.write(f"Token Efficiency Ratio: {efficiency_metrics['token_efficiency_ratio']}\n")
        f.write(f"Reasoning Overhead: {efficiency_metrics['reasoning_overhead_pct']}%\n")
        f.write(f"Speed: {efficiency_metrics['tokens_per_second']} tok/s\n")
        f.write(f"Cost Proxy: {efficiency_metrics['cost_proxy']}\n\n")

        if reasoning:
            f.write("=== REASONING / THINKING ===\n")
            f.write(str(reasoning))
            f.write("\n\n" + "="*30 + "\n\n")
        
        f.write("=== FINAL RESPONSE ===\n")
        f.write(str(content))
    
    # 2. Save Comprehensive Metadata JSON
    metadata = {
        "model": model_name,
        "prompt_label": prompt_label,
        "prompt_category": prompt_category,
        "iteration": iteration,
        "input_mode": usage_stats.get('input_mode', 'unknown'),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "usage": usage_stats,
        "schema_compliance": {
            "is_valid_json": schema_metrics['is_valid_json'],
            "schema_compliance_pct": schema_metrics['schema_compliance_pct'],
            "fields_present": schema_metrics['fields_present'],
            "fields_missing": schema_metrics['fields_missing'],
            "has_main_answer": schema_metrics['has_main_answer'],
            "has_scratchwork": schema_metrics['has_scratchwork'],
            "tts_ready": schema_metrics['tts_ready']
        },
        "content_metrics": content_metrics,
        "efficiency_metrics": efficiency_metrics,
        "raw_content_length": len(content) if content else 0
    }
    
    with open(filename_meta, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print(f" > Saved: {filename_txt}")

def generate_comprehensive_report(all_stats):
    """Generates an enhanced markdown report with full analytics."""
    
    report_lines = []
    report_lines.append("# UCL Architecture Validation - Comprehensive Analytics Report")
    report_lines.append(f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    # ==================== EXPERIMENTAL DESIGN ====================
    report_lines.append("## 1. Experimental Design")
    report_lines.append("")
    report_lines.append("| Category | Description | Purpose |")
    report_lines.append("|----------|-------------|---------|")
    report_lines.append("| **Baseline** | `math_gpt_system_prompt.txt` | Target prompt that UCL attempts to replicate |")
    report_lines.append("| **Control** | No prompt (raw) | Measures raw model behavior |")
    report_lines.append("| **UCL v1-v4.1** | Progressive UCL versions | Tests UCL architecture evolution |")
    report_lines.append("")
    report_lines.append(f"> **Reference Model**: `{REFERENCE_MODEL}` — The model UCL architecture was evolved with. Expected to have best performance.")
    report_lines.append("")
    
    # ==================== REFERENCE MODEL PERFORMANCE (NEW SECTION) ====================
    report_lines.append("## 2. Reference Model Performance (Benchmark)")
    report_lines.append("")
    report_lines.append(f"The UCL architecture was developed and evolved using **`{REFERENCE_MODEL}`**.")
    report_lines.append("This model's performance represents the expected/optimal results.")
    report_lines.append("")
    
    # Extract reference model stats
    ref_stats = [s for s in all_stats if s['model'] == REFERENCE_MODEL]
    if ref_stats:
        n = len(ref_stats)
        json_valid = sum(1 for s in ref_stats if s['schema_compliance']['is_valid_json']) / n * 100
        schema_comp = sum(s['schema_compliance']['schema_compliance_pct'] for s in ref_stats) / n
        tts_ready = sum(1 for s in ref_stats if s['schema_compliance']['tts_ready']) / n * 100
        has_main = sum(1 for s in ref_stats if s['schema_compliance']['has_main_answer']) / n * 100
        avg_tokens = sum(s['usage']['total_tokens'] for s in ref_stats) / n
        avg_duration = sum(s['usage']['duration_seconds'] for s in ref_stats) / n
        avg_tts_density = sum(s['content_metrics']['tts_keyword_density'] for s in ref_stats) / n
        
        report_lines.append("### Reference Model Summary Stats")
        report_lines.append("")
        report_lines.append("| Metric | Value | Interpretation |")
        report_lines.append("|--------|-------|----------------|")
        report_lines.append(f"| Total Runs | {n} | Across all prompt conditions |")
        report_lines.append(f"| JSON Validity | {json_valid:.1f}% | {'✅ Excellent' if json_valid >= 90 else '⚠️ Needs improvement'} |")
        report_lines.append(f"| Schema Compliance | {schema_comp:.1f}% | {'✅ Excellent' if schema_comp >= 90 else '⚠️ Needs improvement'} |")
        report_lines.append(f"| TTS Ready | {tts_ready:.1f}% | {'✅ Excellent' if tts_ready >= 80 else '⚠️ Needs improvement'} |")
        report_lines.append(f"| Has Main Answer | {has_main:.1f}% | {'✅ Excellent' if has_main >= 90 else '⚠️ Needs improvement'} |")
        report_lines.append(f"| Avg Duration | {avg_duration:.2f}s | Response time |")
        report_lines.append(f"| Avg Tokens | {avg_tokens:.0f} | Total token usage |")
        report_lines.append(f"| TTS Keyword Density | {avg_tts_density:.2f}% | Higher = better TTS formatting |")
        report_lines.append("")
        
        # Reference model per-prompt breakdown
        report_lines.append("### Reference Model Per-Prompt Breakdown")
        report_lines.append("")
        report_lines.append("| Prompt | JSON Valid | Schema % | TTS Ready | Duration | Tokens |")
        report_lines.append("|--------|------------|----------|-----------|----------|--------|")
        
        # Group by prompt
        ref_by_prompt = {}
        for s in ref_stats:
            label = s['prompt_label']
            if label not in ref_by_prompt:
                ref_by_prompt[label] = []
            ref_by_prompt[label].append(s)
        
        prompt_order = ['ucl_v1', 'ucl_v2', 'ucl_v3', 'ucl_v4', 'ucl_v4.1', 'baseline', 'no_prompt']
        for prompt in prompt_order:
            if prompt not in ref_by_prompt:
                continue
            p_stats = ref_by_prompt[prompt]
            p_n = len(p_stats)
            p_json = sum(1 for s in p_stats if s['schema_compliance']['is_valid_json']) / p_n * 100
            p_schema = sum(s['schema_compliance']['schema_compliance_pct'] for s in p_stats) / p_n
            p_tts = "✅" if any(s['schema_compliance']['tts_ready'] for s in p_stats) else "❌"
            p_dur = sum(s['usage']['duration_seconds'] for s in p_stats) / p_n
            p_tok = sum(s['usage']['total_tokens'] for s in p_stats) / p_n
            
            report_lines.append(f"| {prompt} | {p_json:.0f}% | {p_schema:.0f}% | {p_tts} | {p_dur:.2f}s | {p_tok:.0f} |")
        
        report_lines.append("")
    else:
        report_lines.append("*No data available for reference model.*")
        report_lines.append("")
    
    # ==================== OVERALL SUMMARY ====================
    report_lines.append("## 3. Overall Summary")
    report_lines.append("")
    
    total_runs = len(all_stats)
    valid_json_count = sum(1 for s in all_stats if s['schema_compliance']['is_valid_json'])
    tts_ready_count = sum(1 for s in all_stats if s['schema_compliance']['tts_ready'])
    full_schema_count = sum(1 for s in all_stats if s['schema_compliance']['schema_compliance_pct'] == 100)
    
    report_lines.append(f"- **Total API Calls**: {total_runs}")
    report_lines.append(f"- **Models Tested**: {len(TARGET_MODELS)} (including reference model)")
    report_lines.append(f"- **Prompt Conditions**: {len(PROMPT_CONFIGS)}")
    report_lines.append(f"- **JSON Validity Rate**: {valid_json_count}/{total_runs} ({valid_json_count/total_runs*100:.1f}%)" if total_runs > 0 else "")
    report_lines.append(f"- **Full Schema Compliance**: {full_schema_count}/{total_runs} ({full_schema_count/total_runs*100:.1f}%)" if total_runs > 0 else "")
    report_lines.append(f"- **TTS Ready Rate**: {tts_ready_count}/{total_runs} ({tts_ready_count/total_runs*100:.1f}%)" if total_runs > 0 else "")
    report_lines.append("")
    
    # ==================== CATEGORY COMPARISON ====================
    report_lines.append("## 4. Category Comparison")
    report_lines.append("")
    report_lines.append("| Category | Runs | JSON Valid % | Schema Compliance % | TTS Ready % | Avg Tokens | Avg Duration |")
    report_lines.append("|----------|------|--------------|---------------------|-------------|------------|--------------|")
    
    categories = {"baseline": [], "control": [], "ucl": []}
    for stat in all_stats:
        cat = stat.get('prompt_category', 'unknown')
        if cat in categories:
            categories[cat].append(stat)
    
    for cat_name, cat_stats in [("Baseline", categories["baseline"]), ("Control", categories["control"]), ("UCL", categories["ucl"])]:
        if not cat_stats:
            continue
        n = len(cat_stats)
        json_valid = sum(1 for s in cat_stats if s['schema_compliance']['is_valid_json']) / n * 100
        schema_comp = sum(s['schema_compliance']['schema_compliance_pct'] for s in cat_stats) / n
        tts_ready = sum(1 for s in cat_stats if s['schema_compliance']['tts_ready']) / n * 100
        avg_tokens = sum(s['usage']['total_tokens'] for s in cat_stats) / n
        avg_duration = sum(s['usage']['duration_seconds'] for s in cat_stats) / n
        
        report_lines.append(f"| {cat_name} | {n} | {json_valid:.1f}% | {schema_comp:.1f}% | {tts_ready:.1f}% | {avg_tokens:.0f} | {avg_duration:.2f}s |")
    
    report_lines.append("")
    
    # ==================== UCL VERSION EVOLUTION ====================
    report_lines.append("## 5. UCL Version Evolution Analysis")
    report_lines.append("")
    report_lines.append("This section tracks the progression of UCL architecture across versions.")
    report_lines.append("")
    report_lines.append("| Version | Runs | JSON Valid % | Schema % | TTS Ready % | Avg Completion Tokens | TTS Keyword Density |")
    report_lines.append("|---------|------|--------------|----------|-------------|----------------------|---------------------|")
    
    # Group UCL stats by version
    ucl_by_version = {}
    for stat in all_stats:
        if stat['prompt_category'] == 'ucl':
            version = stat.get('prompt_label', 'unknown')
            if version not in ucl_by_version:
                ucl_by_version[version] = []
            ucl_by_version[version].append(stat)
    
    version_order = ['ucl_v1', 'ucl_v2', 'ucl_v3', 'ucl_v4', 'ucl_v4.1']
    for version in version_order:
        if version not in ucl_by_version:
            continue
        v_stats = ucl_by_version[version]
        n = len(v_stats)
        json_valid = sum(1 for s in v_stats if s['schema_compliance']['is_valid_json']) / n * 100
        schema_comp = sum(s['schema_compliance']['schema_compliance_pct'] for s in v_stats) / n
        tts_ready = sum(1 for s in v_stats if s['schema_compliance']['tts_ready']) / n * 100
        avg_completion = sum(s['usage']['completion_tokens'] for s in v_stats) / n
        avg_tts_density = sum(s['content_metrics']['tts_keyword_density'] for s in v_stats) / n
        
        report_lines.append(f"| {version} | {n} | {json_valid:.1f}% | {schema_comp:.1f}% | {tts_ready:.1f}% | {avg_completion:.0f} | {avg_tts_density:.2f}% |")
    
    report_lines.append("")
    
    # ==================== BASELINE GAP ANALYSIS ====================
    report_lines.append("## 6. Baseline Gap Analysis")
    report_lines.append("")
    report_lines.append("How close is each UCL version to matching baseline performance?")
    report_lines.append("")
    
    # Calculate baseline metrics
    baseline_stats = categories["baseline"]
    if baseline_stats:
        baseline_json_rate = sum(1 for s in baseline_stats if s['schema_compliance']['is_valid_json']) / len(baseline_stats) * 100
        baseline_schema = sum(s['schema_compliance']['schema_compliance_pct'] for s in baseline_stats) / len(baseline_stats)
        baseline_tts = sum(1 for s in baseline_stats if s['schema_compliance']['tts_ready']) / len(baseline_stats) * 100
        
        report_lines.append(f"**Baseline Reference Values:**")
        report_lines.append(f"- JSON Valid: {baseline_json_rate:.1f}%")
        report_lines.append(f"- Schema Compliance: {baseline_schema:.1f}%")
        report_lines.append(f"- TTS Ready: {baseline_tts:.1f}%")
        report_lines.append("")
        
        report_lines.append("| Version | JSON Gap | Schema Gap | TTS Gap | Overall Gap |")
        report_lines.append("|---------|----------|------------|---------|-------------|")
        
        for version in version_order:
            if version not in ucl_by_version:
                continue
            v_stats = ucl_by_version[version]
            n = len(v_stats)
            
            json_rate = sum(1 for s in v_stats if s['schema_compliance']['is_valid_json']) / n * 100
            schema = sum(s['schema_compliance']['schema_compliance_pct'] for s in v_stats) / n
            tts = sum(1 for s in v_stats if s['schema_compliance']['tts_ready']) / n * 100
            
            json_gap = abs(baseline_json_rate - json_rate)
            schema_gap = abs(baseline_schema - schema)
            tts_gap = abs(baseline_tts - tts)
            overall_gap = (json_gap + schema_gap + tts_gap) / 3
            
            report_lines.append(f"| {version} | {json_gap:.1f}% | {schema_gap:.1f}% | {tts_gap:.1f}% | {overall_gap:.1f}% |")
    
    report_lines.append("")
    
    # ==================== MODEL PERFORMANCE ====================
    report_lines.append("## 7. Per-Model Performance")
    report_lines.append("")
    report_lines.append("| Model | Total Runs | JSON Valid % | Avg Duration | Avg Tokens | Success Rate |")
    report_lines.append("|-------|------------|--------------|--------------|------------|--------------|")
    
    model_stats = {}
    for stat in all_stats:
        model = stat['model']
        if model not in model_stats:
            model_stats[model] = []
        model_stats[model].append(stat)
    
    for model, m_stats in sorted(model_stats.items()):
        n = len(m_stats)
        json_valid = sum(1 for s in m_stats if s['schema_compliance']['is_valid_json']) / n * 100
        avg_duration = sum(s['usage']['duration_seconds'] for s in m_stats) / n
        avg_tokens = sum(s['usage']['total_tokens'] for s in m_stats) / n
        # Success = valid JSON AND has main answer
        success = sum(1 for s in m_stats if s['schema_compliance']['is_valid_json'] and s['schema_compliance']['has_main_answer']) / n * 100
        
        report_lines.append(f"| {model} | {n} | {json_valid:.1f}% | {avg_duration:.2f}s | {avg_tokens:.0f} | {success:.1f}% |")
    
    report_lines.append("")
    
    # ==================== EFFICIENCY ANALYSIS ====================
    report_lines.append("## 8. Efficiency Analysis")
    report_lines.append("")
    report_lines.append("| Category | Avg Prompt Tokens | Avg Completion Tokens | Token Ratio | Cost Proxy |")
    report_lines.append("|----------|-------------------|----------------------|-------------|------------|")
    
    for cat_name, cat_stats in [("Baseline", categories["baseline"]), ("Control", categories["control"]), ("UCL", categories["ucl"])]:
        if not cat_stats:
            continue
        n = len(cat_stats)
        avg_prompt = sum(s['usage']['prompt_tokens'] for s in cat_stats) / n
        avg_completion = sum(s['usage']['completion_tokens'] for s in cat_stats) / n
        ratio = avg_completion / avg_prompt if avg_prompt > 0 else 0
        cost = avg_prompt * 1.0 + avg_completion * 3.0
        
        report_lines.append(f"| {cat_name} | {avg_prompt:.0f} | {avg_completion:.0f} | {ratio:.2f} | {cost:.0f} |")
    
    report_lines.append("")
    
    # ==================== STATISTICAL SUMMARY ====================
    report_lines.append("## 9. Statistical Summary")
    report_lines.append("")
    
    if all_stats:
        # Duration stats
        durations = [s['usage']['duration_seconds'] for s in all_stats]
        duration_stats = calculate_aggregate_stats(durations)
        
        # Token stats
        tokens = [s['usage']['total_tokens'] for s in all_stats]
        token_stats = calculate_aggregate_stats(tokens)
        
        report_lines.append("| Metric | Mean | Std Dev | CV (%) | Min | Max |")
        report_lines.append("|--------|------|---------|--------|-----|-----|")
        report_lines.append(f"| Duration (s) | {duration_stats['mean']} | {duration_stats['std_dev']} | {duration_stats['cv']} | {duration_stats['min']} | {duration_stats['max']} |")
        report_lines.append(f"| Total Tokens | {token_stats['mean']} | {token_stats['std_dev']} | {token_stats['cv']} | {token_stats['min']} | {token_stats['max']} |")
    
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("*Lower CV (Coefficient of Variation) indicates more consistent performance.*")
    report_lines.append("")
    
    # ==================== MODEL INPUT CAPABILITY ANALYSIS ====================
    report_lines.append("## 10. Model Input Capability Analysis")
    report_lines.append("")
    report_lines.append("Shows which input mode was used for each model and fallback behavior.")
    report_lines.append("")
    report_lines.append("| Model | Configured | Vision Used | Text Used | Vision Fallbacks |")
    report_lines.append("|-------|------------|-------------|-----------|-----------------|")
    
    for model in TARGET_MODELS:
        model_runs = [s for s in all_stats if s['model'] == model]
        if not model_runs:
            continue
        
        n = len(model_runs)
        configured = model_runs[0].get('model_capability', 'auto') if model_runs else 'unknown'
        vision_count = sum(1 for s in model_runs if s['input_mode'] == 'vision')
        text_count = sum(1 for s in model_runs if s['input_mode'] == 'text')
        fallback_count = sum(1 for s in model_runs if s.get('vision_fallback', False))
        
        # Status indicator
        cap_icon = "🔍" if configured == "vision" else ("📝" if configured == "text_only" else "🔄")
        
        report_lines.append(f"| {model} | {cap_icon} {configured} | {vision_count} | {text_count} | {fallback_count} |")
    
    report_lines.append("")
    report_lines.append("*Legend: 🔍=vision, 📝=text_only, 🔄=auto*")
    report_lines.append("")
    
    # ==================== CROSS-ITERATION ANALYTICS ====================
    # Calculate consistency data for iteration analysis
    consistency_data = calculate_consistency_by_combination(all_stats)
    
    # Get unique iterations
    iterations = sorted(set(s['iteration'] for s in all_stats))
    num_iterations = len(iterations)
    
    if num_iterations > 1:
        # ==================== SECTION 11: CROSS-ITERATION CONSISTENCY ====================
        report_lines.append("## 11. Cross-Iteration Consistency Analysis")
        report_lines.append("")
        report_lines.append(f"Measures reproducibility across {num_iterations} iterations.")
        report_lines.append("**100% consistency** = same JSON validity result every iteration.")
        report_lines.append("")
        
        # Count consistency levels
        perfect_consistency = sum(1 for c in consistency_data.values() if c['json_consistency_pct'] == 100)
        partial_consistency = sum(1 for c in consistency_data.values() if 0 < c['json_consistency_pct'] < 100)
        zero_consistency = sum(1 for c in consistency_data.values() if c['json_consistency_pct'] == 0)
        
        report_lines.append(f"- **Perfect Consistency (100%)**: {perfect_consistency} combinations")
        report_lines.append(f"- **Partial Consistency**: {partial_consistency} combinations")
        report_lines.append(f"- **Zero Consistency (0%)**: {zero_consistency} combinations")
        report_lines.append("")
        
        # Show table for combinations with variance (not 100% or 0%)
        report_lines.append("### Combinations with Variance (Needs Investigation)")
        report_lines.append("")
        report_lines.append("| Model | Prompt | " + " | ".join([f"Iter {i}" for i in iterations]) + " | Consistency |")
        report_lines.append("|-------|--------|" + "|".join(["-----" for _ in iterations]) + "|-------------|")
        
        # Sort by consistency (lowest first - most problematic)
        sorted_combinations = sorted(consistency_data.items(), key=lambda x: x[1]['json_consistency_pct'])
        
        shown_count = 0
        for (model, prompt), data in sorted_combinations:
            if 0 < data['json_consistency_pct'] < 100:  # Only show partial consistency
                iter_results = " | ".join(["✅" if v else "❌" for v in data['json_valid_per_iter']])
                report_lines.append(f"| {model.split('/')[-1][:20]} | {prompt} | {iter_results} | {data['json_consistency_pct']:.0f}% |")
                shown_count += 1
                if shown_count >= 20:  # Limit rows
                    report_lines.append(f"| ... | ({len(sorted_combinations) - 20} more) | ... | ... |")
                    break
        
        if shown_count == 0:
            report_lines.append("| *All combinations have either 100% or 0% consistency* | | | |")
        
        report_lines.append("")
        
        # ==================== SECTION 12: ITERATION-BY-ITERATION PERFORMANCE ====================
        report_lines.append("## 12. Iteration-by-Iteration Performance")
        report_lines.append("")
        report_lines.append("Side-by-side comparison of each iteration's aggregate performance.")
        report_lines.append("")
        report_lines.append("| Iteration | Runs | JSON Valid % | Avg Duration | Avg Tokens | TTS Ready % |")
        report_lines.append("|-----------|------|--------------|--------------|------------|-------------|")
        
        iteration_metrics = []
        for iteration in iterations:
            iter_stats = [s for s in all_stats if s['iteration'] == iteration]
            n = len(iter_stats)
            if n == 0:
                continue
            
            json_valid_pct = sum(1 for s in iter_stats if s['schema_compliance']['is_valid_json']) / n * 100
            avg_duration = sum(s['usage']['duration_seconds'] for s in iter_stats) / n
            avg_tokens = sum(s['usage']['total_tokens'] for s in iter_stats) / n
            tts_ready_pct = sum(1 for s in iter_stats if s['schema_compliance']['tts_ready']) / n * 100
            
            iteration_metrics.append({
                'iteration': iteration,
                'json_valid_pct': json_valid_pct,
                'avg_duration': avg_duration,
                'avg_tokens': avg_tokens
            })
            
            report_lines.append(f"| {iteration} | {n} | {json_valid_pct:.1f}% | {avg_duration:.2f}s | {avg_tokens:.0f} | {tts_ready_pct:.1f}% |")
        
        # Trend analysis
        if len(iteration_metrics) >= 2:
            report_lines.append("")
            first = iteration_metrics[0]
            last = iteration_metrics[-1]
            json_change = last['json_valid_pct'] - first['json_valid_pct']
            duration_change = last['avg_duration'] - first['avg_duration']
            
            json_trend = "📈 Improving" if json_change > 2 else ("📉 Declining" if json_change < -2 else "➡️ Stable")
            duration_trend = "📈 Faster" if duration_change < -0.5 else ("📉 Slower" if duration_change > 0.5 else "➡️ Stable")
            
            report_lines.append(f"**Trend Analysis (Iter 1 → Iter {iterations[-1]}):**")
            report_lines.append(f"- JSON Validity: {json_trend} ({json_change:+.1f}%)")
            report_lines.append(f"- Duration: {duration_trend} ({duration_change:+.2f}s)")
        
        report_lines.append("")
        
        # ==================== SECTION 13: RESPONSE VARIANCE ANALYSIS ====================
        report_lines.append("## 13. Response Variance Analysis")
        report_lines.append("")
        report_lines.append("Coefficient of Variation (CV) measures consistency. Lower = more reliable.")
        report_lines.append("")
        report_lines.append("| Reliability | CV Range | Description |")
        report_lines.append("|-------------|----------|-------------|")
        report_lines.append("| ✅ High | < 10% | Very consistent responses |")
        report_lines.append("| ⚠️ Medium | 10-25% | Moderate variance |")
        report_lines.append("| ❌ Low | > 25% | High variance, investigate |")
        report_lines.append("")
        
        report_lines.append("### High Variance Combinations (CV > 25%)")
        report_lines.append("")
        report_lines.append("| Model | Prompt | Duration CV | Token CV | Reliability |")
        report_lines.append("|-------|--------|-------------|----------|-------------|")
        
        high_variance_count = 0
        for (model, prompt), data in sorted(consistency_data.items(), key=lambda x: x[1]['duration_cv'], reverse=True):
            if data['duration_cv'] > 25 or data['token_cv'] > 25:
                reliability = "❌ Low"
                model_short = model.split('/')[-1][:25]
                report_lines.append(f"| {model_short} | {prompt} | {data['duration_cv']:.1f}% | {data['token_cv']:.1f}% | {reliability} |")
                high_variance_count += 1
                if high_variance_count >= 15:
                    break
        
        if high_variance_count == 0:
            report_lines.append("| *No high variance combinations detected* | | | | ✅ |")
        
        report_lines.append("")
        
        # ==================== SECTION 14: STUDY REPRODUCIBILITY SUMMARY ====================
        report_lines.append("## 14. Study Reproducibility Summary")
        report_lines.append("")
        report_lines.append("Overall quality score for academic publication readiness.")
        report_lines.append("")
        
        repro_score = calculate_reproducibility_score(all_stats, consistency_data)
        
        report_lines.append("| Metric | Value | Weight |")
        report_lines.append("|--------|-------|--------|")
        report_lines.append(f"| JSON Consistency | {repro_score['avg_json_consistency']:.1f}% | 50% |")
        report_lines.append(f"| Duration CV (inverted) | {repro_score['avg_duration_cv']:.1f}% | 25% |")
        report_lines.append(f"| Token CV (inverted) | {repro_score['avg_token_cv']:.1f}% | 25% |")
        report_lines.append("|--------|-------|--------|")
        report_lines.append(f"| **Reproducibility Score** | **{repro_score['score']:.1f}/100** | {repro_score['interpretation']} |")
        report_lines.append("")
        
        report_lines.append("### Score Interpretation")
        report_lines.append("")
        report_lines.append("| Score Range | Quality | Recommendation |")
        report_lines.append("|-------------|---------|----------------|")
        report_lines.append("| 90-100 | ✅ Excellent | Ready for publication |")
        report_lines.append("| 75-89 | ✅ Good | Publishable with notes |")
        report_lines.append("| 60-74 | ⚠️ Moderate | Address variance issues |")
        report_lines.append("| < 60 | ❌ Low | Investigate root causes |")
        report_lines.append("")
        
    else:
        report_lines.append("## 11-14. Cross-Iteration Analytics")
        report_lines.append("")
        report_lines.append("*Cross-iteration analytics require ≥2 iterations. Current run: 1 iteration.*")
        report_lines.append("")
        report_lines.append("**Recommendation**: Run with multiple iterations to enable:")
        report_lines.append("- Consistency analysis")
        report_lines.append("- Variance analysis")
        report_lines.append("- Reproducibility scoring")
        report_lines.append("")
    
    # ==================== MODEL ARCHITECTURE ANALYSIS ====================
    # Classify models by architecture (reasoning vs standard, vision vs text)
    model_arch = classify_models_by_architecture(all_stats)
    
    # ==================== SECTION 15: 4-QUADRANT MODEL CLASSIFICATION ====================
    report_lines.append("## 15. Model Architecture Classification (4-Quadrant Analysis)")
    report_lines.append("")
    report_lines.append("Empirical classification based on actual test run data.")
    report_lines.append("Models are classified into 4 quadrants based on two dimensions:")
    report_lines.append("- **Reasoning**: Model produced reasoning tokens (chain-of-thought)")
    report_lines.append("- **Input Type**: Vision (image+text) or Text-Only")
    report_lines.append("")
    
    # Count quadrants
    quadrant_models = {
        'reasoning_vision': [],
        'reasoning_text': [],
        'standard_vision': [],
        'standard_text': []
    }
    
    for model in TARGET_MODELS:
        if model not in model_arch:
            continue
        mc = model_arch[model]
        quadrant_models[mc['quadrant']].append(model)
    
    # Quadrant Distribution Summary
    report_lines.append("### 15.1 Quadrant Distribution")
    report_lines.append("")
    report_lines.append("| Quadrant | Description | Models | Count |")
    report_lines.append("|----------|-------------|--------|-------|")
    report_lines.append(f"| Q1 | 🧠🔍 Reasoning + Vision | {len(quadrant_models['reasoning_vision'])} | Best UCL support expected |")
    report_lines.append(f"| Q2 | 🧠📝 Reasoning + Text | {len(quadrant_models['reasoning_text'])} | Strong reasoning, no image |")
    report_lines.append(f"| Q3 | 📊🔍 Standard + Vision | {len(quadrant_models['standard_vision'])} | Direct image, simpler logic |")
    report_lines.append(f"| Q4 | 📊📝 Standard + Text | {len(quadrant_models['standard_text'])} | May need simpler UCL |")
    report_lines.append("")
    
    # Detailed Model Table with Quadrant
    report_lines.append("### 15.2 Per-Model Classification")
    report_lines.append("")
    report_lines.append("| Model | Quadrant | Reasoning Tokens | Vision Rate | JSON % | Schema % |")
    report_lines.append("|-------|----------|------------------|-------------|--------|----------|")
    
    for model in TARGET_MODELS:
        if model not in model_arch:
            continue
        mc = model_arch[model]
        model_short = model.split('/')[-1][:28]
        
        report_lines.append(f"| {model_short} | {mc['quadrant_label']} | {mc['avg_reasoning_tokens']:.0f} | {mc['vision_rate']:.0f}% | {mc['json_valid_pct']:.1f}% | {mc['schema_pct_avg']:.1f}% |")
    
    report_lines.append("")
    
    # ==================== SECTION 16: UCL PERFORMANCE BY QUADRANT ====================
    report_lines.append("## 16. UCL Performance by Model Quadrant")
    report_lines.append("")
    report_lines.append("Compares UCL effectiveness across all 4 model architecture combinations.")
    report_lines.append("*Critical for understanding how to optimize UCL per model type.*")
    report_lines.append("")
    
    # 4-Quadrant Performance Comparison
    report_lines.append("### 16.1 Overall Performance by Quadrant")
    report_lines.append("")
    report_lines.append("| Quadrant | Description | Models | Runs | JSON % | Schema % | TTS % |")
    report_lines.append("|----------|-------------|--------|------|--------|----------|-------|")
    
    quadrant_order = [
        ('reasoning_vision', '🧠🔍 Reasoning+Vision'),
        ('reasoning_text', '🧠📝 Reasoning+Text'),
        ('standard_vision', '📊🔍 Standard+Vision'),
        ('standard_text', '📊📝 Standard+Text')
    ]
    
    quadrant_stats = {}
    for quad_key, quad_label in quadrant_order:
        quad_models = quadrant_models[quad_key]
        if not quad_models:
            quadrant_stats[quad_key] = None
            continue
        
        quad_stats_list = [s for s in all_stats if s['model'] in quad_models]
        n = len(quad_stats_list)
        if n == 0:
            quadrant_stats[quad_key] = None
            continue
        
        json_pct = sum(1 for s in quad_stats_list if s['schema_compliance']['is_valid_json']) / n * 100
        schema_pct = sum(s['schema_compliance']['schema_compliance_pct'] for s in quad_stats_list) / n
        tts_pct = sum(1 for s in quad_stats_list if s['schema_compliance']['tts_ready']) / n * 100
        
        quadrant_stats[quad_key] = {
            'json_pct': json_pct,
            'schema_pct': schema_pct,
            'tts_pct': tts_pct,
            'n': n
        }
        
        report_lines.append(f"| Q{['reasoning_vision', 'reasoning_text', 'standard_vision', 'standard_text'].index(quad_key)+1} | {quad_label} | {len(quad_models)} | {n} | {json_pct:.1f}% | {schema_pct:.1f}% | {tts_pct:.1f}% |")
    
    report_lines.append("")
    
    # UCL Version Performance per Quadrant
    report_lines.append("### 16.2 Best UCL Version per Quadrant")
    report_lines.append("")
    report_lines.append("Identifies which UCL version works best for each model architecture.")
    report_lines.append("")
    
    ucl_stats = [s for s in all_stats if s['prompt_category'] == 'ucl']
    ucl_versions = ['ucl_v1', 'ucl_v2', 'ucl_v3', 'ucl_v4', 'ucl_v4.1']
    
    report_lines.append("| Quadrant | Best UCL | JSON % | Schema % | TTS % | Score |")
    report_lines.append("|----------|----------|--------|----------|-------|-------|")
    
    best_ucl_per_quadrant = {}
    
    for quad_key, quad_label in quadrant_order:
        quad_models = quadrant_models[quad_key]
        if not quad_models:
            continue
        
        best_version = None
        best_score = 0
        best_metrics = {}
        
        for version in ucl_versions:
            version_stats = [s for s in ucl_stats if s['prompt_label'] == version and s['model'] in quad_models]
            n = len(version_stats)
            if n == 0:
                continue
            
            json_pct = sum(1 for s in version_stats if s['schema_compliance']['is_valid_json']) / n * 100
            schema_pct = sum(s['schema_compliance']['schema_compliance_pct'] for s in version_stats) / n
            tts_pct = sum(1 for s in version_stats if s['schema_compliance']['tts_ready']) / n * 100
            
            score = json_pct * 0.5 + schema_pct * 0.3 + tts_pct * 0.2
            if score > best_score:
                best_score = score
                best_version = version
                best_metrics = {'json': json_pct, 'schema': schema_pct, 'tts': tts_pct}
        
        if best_version:
            best_ucl_per_quadrant[quad_key] = best_version
            report_lines.append(f"| {quad_label} | **{best_version}** | {best_metrics['json']:.1f}% | {best_metrics['schema']:.1f}% | {best_metrics['tts']:.1f}% | {best_score:.1f} |")
    
    report_lines.append("")
    
    # ==================== SECTION 17: UCL ARCHITECTURE OPTIMIZATION GUIDE ====================
    report_lines.append("## 17. UCL Architecture Optimization Guide")
    report_lines.append("")
    report_lines.append("**UCL (Universal Conditional Logic)** is a structured prompt architecture using:")
    report_lines.append("- `{{concept:variable:domain}}` - Universal expressions")
    report_lines.append("- `^^CONDITION^^...^^/CONDITION^^` - Conditional logic")
    report_lines.append("- `<<REPEAT>>...</REPEAT>>` - Loop structures")
    report_lines.append("- `[[LLM:...]]` - Meta-instructions")
    report_lines.append("")
    report_lines.append("**Key Insight**: UCL may need to be tweaked for different model architectures.")
    report_lines.append("")
    
    report_lines.append("### 17.1 Per-Quadrant Observations")
    report_lines.append("")
    
    # Q1: Reasoning + Vision
    report_lines.append("#### Q1: 🧠🔍 Reasoning + Vision Models")
    if quadrant_models['reasoning_vision']:
        report_lines.append(f"- **Models**: {', '.join([m.split('/')[-1] for m in quadrant_models['reasoning_vision']])}")
        if 'reasoning_vision' in best_ucl_per_quadrant:
            report_lines.append(f"- **Best UCL Version (observed)**: `{best_ucl_per_quadrant['reasoning_vision']}`")
    else:
        report_lines.append("- *No models in this quadrant*")
    report_lines.append("")
    
    # Q2: Reasoning + Text
    report_lines.append("#### Q2: 🧠📝 Reasoning + Text-Only Models")
    if quadrant_models['reasoning_text']:
        report_lines.append(f"- **Models**: {', '.join([m.split('/')[-1] for m in quadrant_models['reasoning_text']])}")
        if 'reasoning_text' in best_ucl_per_quadrant:
            report_lines.append(f"- **Best UCL Version (observed)**: `{best_ucl_per_quadrant['reasoning_text']}`")
    else:
        report_lines.append("- *No models in this quadrant*")
    report_lines.append("")
    
    # Q3: Standard + Vision
    report_lines.append("#### Q3: 📊🔍 Standard + Vision Models")
    if quadrant_models['standard_vision']:
        report_lines.append(f"- **Models**: {', '.join([m.split('/')[-1] for m in quadrant_models['standard_vision']])}")
        if 'standard_vision' in best_ucl_per_quadrant:
            report_lines.append(f"- **Best UCL Version (observed)**: `{best_ucl_per_quadrant['standard_vision']}`")
    else:
        report_lines.append("- *No models in this quadrant*")
    report_lines.append("")
    
    # Q4: Standard + Text
    report_lines.append("#### Q4: 📊📝 Standard + Text-Only Models")
    if quadrant_models['standard_text']:
        report_lines.append(f"- **Models**: {', '.join([m.split('/')[-1] for m in quadrant_models['standard_text']])}")
        if 'standard_text' in best_ucl_per_quadrant:
            report_lines.append(f"- **Best UCL Version (observed)**: `{best_ucl_per_quadrant['standard_text']}`")
    else:
        report_lines.append("- *No models in this quadrant*")
    report_lines.append("")
    
    # Key Findings Summary
    report_lines.append("### 17.2 Key Findings")
    report_lines.append("")
    
    # Compare quadrants
    if quadrant_stats.get('reasoning_vision') and quadrant_stats.get('standard_text'):
        q1_json = quadrant_stats['reasoning_vision']['json_pct']
        q4_json = quadrant_stats['standard_text']['json_pct']
        diff = q1_json - q4_json
        if diff > 0:
            report_lines.append(f"- **Q1 vs Q4 Gap**: Reasoning+Vision models outperform Standard+Text by **{diff:.1f}%** on JSON validity")
        
    if quadrant_stats.get('reasoning_vision') and quadrant_stats.get('reasoning_text'):
        q1_json = quadrant_stats['reasoning_vision']['json_pct']
        q2_json = quadrant_stats['reasoning_text']['json_pct']
        diff = q1_json - q2_json
        if abs(diff) > 2:
            if diff > 0:
                report_lines.append(f"- **Vision Impact on Reasoning**: Vision adds {diff:.1f}% JSON validity for reasoning models")
            else:
                report_lines.append(f"- **Vision Impact on Reasoning**: Text-only reasoning models perform {-diff:.1f}% better")
    
    # Best overall quadrant
    best_quadrant = None
    best_quadrant_score = 0
    for quad_key, stats in quadrant_stats.items():
        if stats:
            score = stats['json_pct']
            if score > best_quadrant_score:
                best_quadrant_score = score
                best_quadrant = quad_key
    
    if best_quadrant:
        quad_labels = {
            'reasoning_vision': '🧠🔍 Reasoning + Vision',
            'reasoning_text': '🧠📝 Reasoning + Text',
            'standard_vision': '📊🔍 Standard + Vision',
            'standard_text': '📊📝 Standard + Text'
        }
        report_lines.append(f"- **Best Quadrant for UCL**: {quad_labels[best_quadrant]} ({best_quadrant_score:.1f}% JSON validity)")
    
    report_lines.append("")
    
    # LIMITATIONS SECTION
    report_lines.append("### 17.3 Limitations & Future Work")
    report_lines.append("")
    report_lines.append("> **⚠️ IMPORTANT LIMITATIONS**")
    report_lines.append("")
    report_lines.append("The following factors may influence UCL performance but require additional research:")
    report_lines.append("")
    report_lines.append("1. **Insufficient Data for Per-Quadrant UCL Modifications**")
    report_lines.append("   - Current testing does not provide sufficient evidence to recommend specific UCL prompt modifications per quadrant")
    report_lines.append("   - Observed performance differences may be due to factors other than quadrant classification")
    report_lines.append("   - Further controlled experiments needed before making optimization recommendations")
    report_lines.append("")
    report_lines.append("2. **Underlying Model Architecture (PyTorch/Implementation)**")
    report_lines.append("   - The literal PyTorch architecture and training data of each model could play a significant role")
    report_lines.append("   - Models within the same quadrant may have vastly different internal implementations")
    report_lines.append("   - Transformer variants (encoder-decoder, decoder-only, MoE, etc.) may process UCL differently")
    report_lines.append("   - Tokenization strategies and attention mechanisms vary between models")
    report_lines.append("")
    report_lines.append("3. **Model Family Differences**")
    report_lines.append("   - Even with same quadrant classification, model families (OpenAI, Anthropic, Google, Meta, etc.) may have unique response patterns")
    report_lines.append("   - Training methodologies (RLHF, DPO, SFT, etc.) could affect UCL interpretation")
    report_lines.append("")
    report_lines.append("4. **Sample Size Considerations**")
    report_lines.append("   - Results are based on limited test iterations")
    report_lines.append("   - Statistical significance of quadrant differences should be validated with larger samples")
    report_lines.append("")
    report_lines.append("**Recommended Future Work:**")
    report_lines.append("- Conduct targeted experiments with UCL prompt variations per quadrant")
    report_lines.append("- Control for model family and architecture when comparing quadrants")
    report_lines.append("- Increase sample size for statistical significance")
    report_lines.append("- Test on diverse problem domains beyond math/TTS")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("*This analysis provides observational data. UCL optimization recommendations require additional controlled experimentation.*")
    report_lines.append("")
    
    with open("TEST_SUITE_REPORT.md", "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    
    print("\n" + "="*60)
    print(" COMPREHENSIVE REPORT GENERATED: TEST_SUITE_REPORT.md")
    print("="*60)

def export_to_csv(all_stats):
    """Export all stats to CSV for external analysis."""
    if not all_stats:
        return
    
    fieldnames = [
        'model', 'prompt_label', 'prompt_category', 'iteration', 'input_mode', 'timestamp',
        'model_capability', 'vision_attempted', 'vision_fallback',
        'prompt_tokens', 'completion_tokens', 'total_tokens', 'reasoning_tokens', 'duration_seconds',
        'is_valid_json', 'schema_compliance_pct', 'has_main_answer', 'has_scratchwork', 'tts_ready',
        'char_count', 'word_count', 'line_count', 'tts_keyword_count', 'tts_keyword_density',
        'token_efficiency_ratio', 'reasoning_overhead_pct', 'tokens_per_second', 'cost_proxy'
    ]
    
    with open("TEST_SUITE_DATA.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for stat in all_stats:
            row = {
                'model': stat['model'],
                'prompt_label': stat['prompt_label'],
                'prompt_category': stat['prompt_category'],
                'iteration': stat['iteration'],
                'input_mode': stat['input_mode'],
                'timestamp': stat['timestamp'],
                'model_capability': stat.get('model_capability', 'unknown'),
                'vision_attempted': stat.get('vision_attempted', False),
                'vision_fallback': stat.get('vision_fallback', False),
                'prompt_tokens': stat['usage']['prompt_tokens'],
                'completion_tokens': stat['usage']['completion_tokens'],
                'total_tokens': stat['usage']['total_tokens'],
                'reasoning_tokens': stat['usage']['reasoning_tokens'],
                'duration_seconds': stat['usage']['duration_seconds'],
                'is_valid_json': stat['schema_compliance']['is_valid_json'],
                'schema_compliance_pct': stat['schema_compliance']['schema_compliance_pct'],
                'has_main_answer': stat['schema_compliance']['has_main_answer'],
                'has_scratchwork': stat['schema_compliance']['has_scratchwork'],
                'tts_ready': stat['schema_compliance']['tts_ready'],
                'char_count': stat['content_metrics']['char_count'],
                'word_count': stat['content_metrics']['word_count'],
                'line_count': stat['content_metrics']['line_count'],
                'tts_keyword_count': stat['content_metrics']['tts_keyword_count'],
                'tts_keyword_density': stat['content_metrics']['tts_keyword_density'],
                'token_efficiency_ratio': stat['efficiency_metrics']['token_efficiency_ratio'],
                'reasoning_overhead_pct': stat['efficiency_metrics']['reasoning_overhead_pct'],
                'tokens_per_second': stat['efficiency_metrics']['tokens_per_second'],
                'cost_proxy': stat['efficiency_metrics']['cost_proxy']
            }
            writer.writerow(row)
    
    print(" CSV DATA EXPORTED: TEST_SUITE_DATA.csv")

# ================= MAIN EXECUTION =================

def main():
    # 1. Prepare the image
    if not os.path.exists(IMAGE_FILENAME):
        print(f"CRITICAL ERROR: Image file '{IMAGE_FILENAME}' not found.")
        return

    base64_image = encode_image(IMAGE_FILENAME)
    
    # Pre-load math problem text
    math_problem_text = read_text_file(TEXT_PROBLEM_FILE)
    if not math_problem_text:
        print(f"CRITICAL ERROR: Text problem file '{TEXT_PROBLEM_FILE}' not found.")
        return
    
    # Ask for Iterations
    try:
        num_iterations_input = input("Enter number of iterations (default 1): ").strip()
        num_iterations = int(num_iterations_input) if num_iterations_input else 1
    except ValueError:
        print("Invalid input, defaulting to 1 iteration.")
        num_iterations = 1
    
    total_calls = len(TARGET_MODELS) * len(PROMPT_CONFIGS) * num_iterations
    print(f"\n{'='*60}")
    print(f" UCL ARCHITECTURE VALIDATION - COMPREHENSIVE TEST SUITE")
    print(f"{'='*60}")
    print(f" Models: {len(TARGET_MODELS)}")
    print(f" Prompt Conditions: {len(PROMPT_CONFIGS)} (5 UCL + 1 Baseline + 1 Control)")
    print(f" Iterations: {num_iterations}")
    print(f" Total API Calls: {total_calls}")
    print(f"{'='*60}")
    print(f" Analytics Enabled:")
    print(f"   - JSON Schema Compliance Analysis")
    print(f"   - TTS Format Detection")
    print(f"   - Content Metrics (chars, words, lines)")
    print(f"   - Efficiency Metrics (token ratios, speed)")
    print(f"   - UCL Version Evolution Tracking")
    print(f"   - Baseline Gap Analysis")
    print(f"{'='*60}\n")

    all_run_stats = []
    skipped_count = 0
    completed_count = 0

    # 2. Outer Loop: Iterations
    for iteration in range(1, num_iterations + 1):
        print(f"\n>>> STARTING ITERATION {iteration}/{num_iterations} <<<\n")
        
        # 3. Loop through Models
        for model in TARGET_MODELS:
            print(f"\nProcessing Model: {model} (Iter {iteration})")
            
            # 4. Loop through Prompt Configurations
            for prompt_config in PROMPT_CONFIGS:
                prompt_file = prompt_config["file"]
                prompt_label = prompt_config["label"]
                prompt_category = prompt_config["category"]
                
                # Handle prompt text loading
                if prompt_file:
                    prompt_text = read_text_file(prompt_file)
                    if not prompt_text:
                        print(f"  -> Skipping {prompt_label}: Could not load prompt file")
                        continue
                else:
                    prompt_text = None

                # CHECK FOR RESUME - Skip if output already exists
                output_filename = get_output_filename(model, prompt_label, iteration)
                if os.path.exists(output_filename):
                    print(f"  -> {prompt_label} [SKIPPED - already exists]")
                    skipped_count += 1
                    continue

                print(f"  -> Testing [{prompt_category}] {prompt_label} ...")

                # Get model capability for smart routing
                model_capability = get_model_capability(model)
                input_mode = None
                response = None
                vision_attempted = False
                vision_fallback = False
                
                # 5. CAPABILITY-AWARE API ROUTING
                if model_capability == "text_only":
                    # TEXT-ONLY MODEL: Skip vision entirely, go straight to text
                    print(f"     [TEXT-ONLY MODEL - Skipping vision attempt]")
                    try:
                        start_time = time.time()
                        
                        if prompt_text is not None:
                            combined_prompt = prompt_text + "\n\n---\n\n**MATH PROBLEM:**\n" + math_problem_text
                        else:
                            combined_prompt = "Solve this problem:\n\n" + math_problem_text
                        
                        response = client.chat.completions.create(
                            model=model,
                            messages=[{"role": "user", "content": combined_prompt}],
                            extra_body={
                                "include_reasoning": True,  # Legacy parameter
                                "reasoning": {"enabled": True}  # Newer OpenRouter parameter
                            }
                        )
                        end_time = time.time()
                        input_mode = "text"
                        print(f"     [TEXT MODE SUCCESS]")
                        
                    except Exception as text_error:
                        print(f"  !! TEXT MODE FAILED for {model}: {text_error}")
                        continue
                else:
                    # VISION or AUTO MODEL: Try vision first, fallback to text on error
                    vision_attempted = True
                    try:
                        start_time = time.time()
                        
                        if prompt_text is not None:
                            messages = [
                                {
                                    "role": "user",
                                    "content": [
                                        {"type": "text", "text": prompt_text},
                                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                                    ]
                                }
                            ]
                        else:
                            messages = [
                                {
                                    "role": "user",
                                    "content": [
                                        {"type": "text", "text": "Solve this problem."},
                                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                                    ]
                                }
                            ]
                        
                        response = client.chat.completions.create(
                            model=model,
                            messages=messages,
                            extra_body={
                                "include_reasoning": True,  # Legacy parameter
                                "reasoning": {"enabled": True}  # Newer OpenRouter parameter
                            }
                        )
                        end_time = time.time()
                        input_mode = "vision"
                        print(f"     [VISION MODE SUCCESS]")
                        
                    except Exception as vision_error:
                        print(f"     [VISION FAILED: {str(vision_error)[:50]}...]")
                        print(f"     [FALLING BACK TO TEXT MODE...]")
                        vision_fallback = True
                        
                        # Fallback to text mode
                        try:
                            start_time = time.time()
                            
                            if prompt_text is not None:
                                combined_prompt = prompt_text + "\n\n---\n\n**MATH PROBLEM:**\n" + math_problem_text
                            else:
                                combined_prompt = "Solve this problem:\n\n" + math_problem_text
                            
                            response = client.chat.completions.create(
                                model=model,
                                messages=[{"role": "user", "content": combined_prompt}],
                                extra_body={
                                    "include_reasoning": True,  # Legacy parameter
                                    "reasoning": {"enabled": True}  # Newer OpenRouter parameter
                                }
                            )
                            end_time = time.time()
                            input_mode = "text"
                            print(f"     [TEXT FALLBACK SUCCESS]")
                            
                        except Exception as text_error:
                            print(f"  !! BOTH MODES FAILED for {model}: {text_error}")
                            continue
                
                duration = end_time - start_time

                # 6. Extract Data with null checks
                if not response.choices:
                    print(f"  !! Empty response from {model}, skipping...")
                    continue
                    
                choice = response.choices[0]
                message = choice.message
                
                if not message or not message.content:
                    print(f"  !! No content in response from {model}, skipping...")
                    continue
                
                content = message.content
                
                # Extract Reasoning
                reasoning = getattr(message, 'reasoning', None)
                if not reasoning and hasattr(choice, 'model_extra'): 
                     reasoning = choice.model_extra.get('reasoning')
                
                reasoning_tokens = 0
                if hasattr(response, 'usage') and response.usage:
                    if hasattr(response.usage, 'completion_tokens_details') and response.usage.completion_tokens_details:
                         reasoning_tokens = getattr(response.usage.completion_tokens_details, 'reasoning_tokens', 0)

                # 7. Calculate ALL Analytics
                usage_stats = {
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0,
                    "reasoning_tokens": reasoning_tokens or 0,
                    "duration_seconds": round(duration, 2),
                    "input_mode": input_mode,
                    "model_capability": model_capability,
                    "vision_attempted": vision_attempted,
                    "vision_fallback": vision_fallback
                }
                
                schema_metrics = calculate_json_schema_compliance(content)
                content_metrics = calculate_content_metrics(content)
                efficiency_metrics = calculate_efficiency_metrics(usage_stats)

                # 8. Save Data
                save_output(model, prompt_label, prompt_category, content, reasoning, 
                           usage_stats, schema_metrics, content_metrics, efficiency_metrics, iteration)
                
                # 9. Add to Global Stats
                all_run_stats.append({
                    "model": model,
                    "prompt_label": prompt_label,
                    "prompt_category": prompt_category,
                    "iteration": iteration,
                    "input_mode": input_mode,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "usage": usage_stats,
                    "schema_compliance": {
                        "is_valid_json": schema_metrics['is_valid_json'],
                        "schema_compliance_pct": schema_metrics['schema_compliance_pct'],
                        "has_main_answer": schema_metrics['has_main_answer'],
                        "has_scratchwork": schema_metrics['has_scratchwork'],
                        "tts_ready": schema_metrics['tts_ready']
                    },
                    "content_metrics": content_metrics,
                    "efficiency_metrics": efficiency_metrics,
                    "model_capability": model_capability,
                    "vision_attempted": vision_attempted,
                    "vision_fallback": vision_fallback
                })
                
                completed_count += 1

            # End of prompt loop
        # End of model loop
        print(f"\nDONE: Iteration {iteration} complete.")
        print("-" * 50)
        
    # End of Iteration Loop
    print("\n" + "="*60)
    print(" ALL ITERATIONS COMPLETE")
    print(f" Skipped (already exists): {skipped_count}")
    print(f" Completed this run: {completed_count}")
    print("="*60)
    
    if all_run_stats:
        generate_comprehensive_report(all_run_stats)
        export_to_csv(all_run_stats)
        
        print("\n" + "="*60)
        print(" OUTPUTS GENERATED:")
        print("   - TEST_SUITE_REPORT.md (Comprehensive Analytics)")
        print("   - TEST_SUITE_DATA.csv (Raw Data for External Analysis)")
        print("   - Individual *_OUTPUT.txt and *_META.json files")
        print("="*60 + "\n")
    else:
        print("\n Note: No new data collected, skipping report generation.")

if __name__ == "__main__":
    main()