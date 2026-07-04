# UCL Warm-Up Spike Test — Visual Walkthrough

> **Purpose**: Step-by-step visual guide for running the spike test that validates the 8-message warm-up hypothesis  
> **Audience**: Developers evaluating the UCL Preparation Pipeline architecture  
> **Time to run**: 5-10 minutes (excluding API calls)

---

## Quick Start (2 Minutes)

If you just want to run the test without understanding the architecture:

```bash
# 1. Start HTTP server
cd C:\Users\antmi\Universal-Conditional-Logic\UCL_LINTER\ucl-builder
python -m http.server 8766

# 2. Open in browser
start http://localhost:8766/spike-test.html

# 3. Configure settings
#    - Endpoint: http://localhost:13305 (Lemonade) or https://openrouter.ai/api/v1
#    - API Key: your-key-here
#    - Model: your-model-here

# 4. Click "🔥 Run Warm-Up Test" — wait for 8 steps to complete

# 5. Click "⚡ Run Single-Shot Test" — baseline comparison

# 6. Compare outputs and rate quality (1-5 scale)

# 7. Make decision: Proceed with pipeline OR redesign approach
```

---

## Detailed Walkthrough

### Part 1: Opening the Spike Test

**What you see**:
![Initial Page Load](spike-test-01-initial.png)

**What it means**:
- **Header**: "🔬 UCL Warm-Up Spike Test" — this is a validation tool, not production code
- **Subtitle**: "Validate the 8-message warm-up hypothesis for UCL generation" — the goal is to answer: does warm-up improve output quality?
- **4 Main Sections**:
  1. **⚙️ Settings** — configure API endpoint, key, model
  2. **📚 Warm-Up Messages (8 Steps)** — the 8 message pairs that prime the LLM
  3. **🧪 Test Configuration** — run buttons and test prompt
  4. **📊 Results Comparison** — side-by-side output comparison

**Key Insight**: This page is self-contained (no external dependencies). All CSS/JS is inline. Just open the HTML file.

---

### Part 2: Configuring Settings

**What you see**:
![Settings Panel](spike-test-02-settings.png)

**What it means**:
- **Endpoint**: URL of your LLM API
  - **Lemonade** (local): `http://localhost:13305`
  - **OpenRouter**: `https://openrouter.ai/api/v1`
- **API Key**: Your authentication key
  - Lemonade: may be empty if no auth required
  - OpenRouter: `sk-or-v1-...`
- **Model**: Which model to use
  - Lemonade: `meta-llama-3.1-8b-instruct` or similar
  - OpenRouter: `anthropic/claude-3.5-sonnet` or similar

**Configuration Tips**:
- Use the **same model** for both warm-up and single-shot tests (fair comparison)
- Start with a model that has **≥16K context window** (warm-up consumes ~9K tokens)
- If using Lemonade, ensure the server is running on port 13305

**Expected Behavior**: Settings are saved to `localStorage` — they persist across page reloads.

---

### Part 3: Understanding the Warm-Up Messages

**What you see**:
![Warm-Up Messages](spike-test-03-warmup-messages.png)

**What it means**:
The 8 warm-up message pairs are structured as a **progressive teaching sequence**:

| Steps | Purpose | Content |
|-------|---------|---------|
| **1-2** | UCL Syntax | Teaches ^^CONDITION^^, [[TOOL:]], [[PARAM:]] syntax |
| **3-4** | Tool Definitions | Shows filesystem domain tools (read_file, write_file, exec_command) |
| **5-6** | Conversion Examples | Demonstrates natural language → UCL conversion |
| **7-8** | Rules & Patterns | 7 conversion heuristics + output format requirements |

**Each step is a message pair**:
```javascript
{
  step: 1,
  user: "I'll teach you UCL syntax. Acknowledge when ready.",
  assistant: "Ready to learn UCL syntax."
}
```

**Why 8 messages?**
- Provides enough context for the LLM to understand UCL patterns
- Mimics a "training conversation" before the actual task
- Accumulates context across all 8 steps (16 messages total by step 8)

**Collapsible Sections**: Click the ▼ arrows to expand/collapse each step. This reduces visual clutter.

**Key Insight**: The warm-up messages are **hardcoded** in this spike test. In the full pipeline, they'll be dynamically generated based on discovered tools.

---

### Part 4: Test Configuration

**What you see**:
![Test Configuration](spike-test-04-test-config.png)

**What it means**:
- **Test Prompt**: The natural language prompt to convert to UCL
  - Default: "Create a system prompt for a coding assistant that can read files, write files, and execute shell commands."
  - You can edit this to test different scenarios
- **🔥 Run Warm-Up Test**: Executes the 8-message warm-up sequence, then sends the test prompt
  - Makes **9 API calls** (8 warm-up + 1 generation)
  - Shows progress bar and execution log
  - Displays token count and output
- **⚡ Run Single-Shot Test**: Executes the existing Optimize flow (baseline)
  - Makes **1 API call** (system prompt + user message)
  - Faster, but may produce lower-quality UCL
- **Status Indicator**: Shows "Ready to start" → "Running..." → "Complete"

**Execution Flow**:
```
Warm-Up Test:
  Step 1: Send message pair 1 → receive response
  Step 2: Send message pair 2 → receive response
  ...
  Step 8: Send message pair 8 → receive response
  Step 9: Send test prompt → receive UCL output

Single-Shot Test:
  Step 1: Send system prompt + test prompt → receive UCL output
```

**Expected Duration**:
- Warm-up: 30-60 seconds (8 API calls)
- Single-shot: 5-15 seconds (1 API call)

---

### Part 5: Results Comparison

**What you see**:
![Results Section](spike-test-05-results-section.png)

**What it means**:
After running both tests, you'll see:

**Warm-Up Output**:
- Full UCL-formatted output from the warm-up approach
- Token count (from API `usage.total_tokens`)
- Output length (characters)

**Single-Shot Output**:
- Full UCL-formatted output from the single-shot approach
- Token count
- Output length

**Metrics Dashboard**:
| Metric | Warm-Up | Single-Shot | Winner |
|--------|---------|-------------|--------|
| **Token Count** | ~12,000 | ~2,000 | Single-shot (6x less) |
| **Output Length** | ~1,500 chars | ~1,200 chars | Warm-up (25% longer) |
| **^^CONDITION^^ count** | 3 | 1 | Warm-up (3x more) |
| **[[TOOL:]] count** | 6 | 4 | Warm-up (50% more) |
| **[[PARAM:]] count** | 12 | 8 | Warm-up (50% more) |
| **Manual Quality (1-5)** | ? | ? | You decide |

**Manual Quality Rating**:
- **5**: Perfect UCL syntax, all tools correctly defined, conditions make sense
- **4**: Good UCL syntax, minor issues (missing optional params, weak keywords)
- **3**: Acceptable UCL syntax, some tools missing or incorrectly defined
- **2**: Poor UCL syntax, many errors, hard to parse
- **1**: Not valid UCL, just natural language or broken syntax

**Decision Framework**:
- ✅ **Warm-up wins** if: Quality rating ≥ 4 AND token overhead acceptable
- ⚠️ **Marginal** if: Quality rating 3-4 OR token overhead too high
- ❌ **Single-shot wins** if: Quality rating ≤ 3 OR token overhead unacceptable

---

### Part 6: Execution Log

**What you see**:
The execution log shows each API call with timestamps and status:

```
[00:00:01] Starting warm-up test...
[00:00:02] Step 1: Sending message pair 1/8
[00:00:03] Step 1: ✅ Received response (150 tokens)
[00:00:04] Step 2: Sending message pair 2/8
[00:00:05] Step 2: ✅ Received response (180 tokens)
...
[00:00:25] Step 8: ✅ Received response (220 tokens)
[00:00:26] Step 9: Sending test prompt
[00:00:30] Step 9: ✅ Received UCL output (1,500 chars)
[00:00:31] Warm-up test complete. Total tokens: 12,340
```

**Why it matters**:
- Shows progress (helpful for long warm-up sequences)
- Identifies which step failed (if any)
- Tracks token consumption per step
- Useful for debugging API errors

---

## Troubleshooting

### Issue 1: "Failed to fetch" Error

**Symptom**: Console shows `TypeError: Failed to fetch`

**Causes**:
- Lemonade server not running
- Wrong endpoint URL
- CORS policy blocking browser requests

**Solutions**:
```bash
# Check if Lemonade is running
curl http://localhost:13305/health

# Start Lemonade if needed
lemonade-server start

# For CORS issues, use a proxy or run with --disable-web-security
chrome.exe --disable-web-security --user-data-dir="C:\chrome-debug"
```

---

### Issue 2: "401 Unauthorized" Error

**Symptom**: Console shows `401 Unauthorized`

**Causes**:
- Missing or invalid API key
- API key doesn't have access to the model

**Solutions**:
- Verify API key in settings
- Check model name matches your subscription
- For OpenRouter, ensure billing is set up

---

### Issue 3: "Context window exceeded" Error

**Symptom**: API returns error about token limit

**Causes**:
- Model has <16K context window
- Warm-up messages too long

**Solutions**:
- Use a model with ≥16K context (e.g., Claude 3.5 Sonnet, GPT-4)
- Reduce warm-up message length (edit `spike-test.html`)
- Skip steps 5-6 (examples) to save ~2K tokens

---

### Issue 4: Warm-Up Output is Garbage

**Symptom**: Output contains no UCL syntax, just natural language

**Causes**:
- Model doesn't understand UCL (too small)
- Warm-up messages not clear enough
- Model temperature too high

**Solutions**:
- Use a larger model (≥8B parameters)
- Lower temperature to 0.3-0.5
- Review warm-up messages — ensure they explicitly teach UCL syntax

---

### Issue 5: Single-Shot Output is Better Than Warm-Up

**Symptom**: Single-shot has higher quality rating

**Causes**:
- Warm-up overhead confuses the model
- Single-shot system prompt is already well-tuned
- Test prompt is too simple (warm-up doesn't help)

**Solutions**:
- Try a more complex test prompt (multiple tools, nested conditions)
- Review warm-up messages — ensure they're teaching, not confusing
- Accept that warm-up may not help for simple cases

---

## Interpreting Results

### Scenario A: Warm-Up Clearly Wins

**Evidence**:
- Quality rating: Warm-up 5, Single-shot 3
- Token overhead: 12K vs 2K (6x more, but acceptable)
- UCL syntax counts: Warm-up has 3x more ^^CONDITION^^ blocks

**Decision**: ✅ **Proceed with full pipeline implementation**

**Next Steps**:
1. Start Phase 1: Build source parser (regex-based)
2. Build warm-up engine (8-message sequence)
3. Build pipeline orchestrator (state machine)
4. Integrate with Builder UI (pipeline tab)

**Timeline**: 33-50 hours over 2.5 weeks

---

### Scenario B: Marginal Improvement

**Evidence**:
- Quality rating: Warm-up 4, Single-shot 3
- Token overhead: 12K vs 2K (6x more, borderline)
- UCL syntax counts: Warm-up has 50% more syntax elements

**Decision**: ⚠️ **Optimize warm-up sequence**

**Next Steps**:
1. Reduce 8 messages → 4-5 messages (skip examples)
2. Compress warm-up content (summarize instead of full examples)
3. Re-test with optimized sequence
4. If still marginal, consider single-shot with better system prompt

**Timeline**: 1 week for optimization + re-test

---

### Scenario C: No Improvement or Worse

**Evidence**:
- Quality rating: Warm-up 3, Single-shot 4
- Token overhead: 12K vs 2K (6x more, unacceptable)
- UCL syntax counts: Similar or worse

**Decision**: ❌ **Redesign approach**

**Next Steps**:
1. Abandon 8-message warm-up hypothesis
2. Investigate single-shot improvements:
   - Better system prompt with UCL examples
   - Few-shot learning in system prompt
   - Retrieval-augmented generation (RAG) for tool definitions
3. Re-test with new approach

**Timeline**: 1-2 weeks for redesign + re-test

---

## Decision Matrix

Use this matrix to make your go/no-go decision:

| Quality Rating | Token Overhead | Decision | Action |
|---------------|----------------|----------|--------|
| 5 | <15K | ✅ **Strong GO** | Proceed with pipeline |
| 4 | <15K | ✅ **GO** | Proceed with pipeline |
| 5 | 15-20K | ⚠️ **Conditional GO** | Optimize warm-up first |
| 4 | 15-20K | ⚠️ **Marginal** | Optimize warm-up or redesign |
| 3 | <15K | ⚠️ **Marginal** | Optimize warm-up or redesign |
| 3 | 15-20K | ❌ **NO GO** | Redesign approach |
| ≤2 | Any | ❌ **NO GO** | Redesign approach |

**Token Overhead Thresholds**:
- **<15K**: Acceptable (fits in 16K context models)
- **15-20K**: Borderline (needs 32K context models)
- **>20K**: Unacceptable (too expensive, slow)

---

## What to Document

After running the test, fill in `spike-test-results.md`:

```markdown
## Test Configuration
- **Model**: [model name]
- **Endpoint**: [endpoint URL]
- **Test Prompt**: [prompt text]

## Results

### Warm-Up Test
- **Token Count**: [number]
- **Output Length**: [chars]
- **^^CONDITION^^ Count**: [number]
- **[[TOOL:]] Count**: [number]
- **[[PARAM:]] Count**: [number]
- **Quality Rating**: [1-5]

### Single-Shot Test
- **Token Count**: [number]
- **Output Length**: [chars]
- **^^CONDITION^^ Count**: [number]
- **[[TOOL:]] Count**: [number]
- **[[PARAM:]] Count**: [number]
- **Quality Rating**: [1-5]

## Analysis
[What did you observe? Which approach produced better UCL? Why?]

## Decision
[Proceed / Optimize / Redesign]

## Rationale
[Why did you make this decision? What evidence supports it?]
```

---

## Next Steps

Based on your decision:

### If GO → Build the Pipeline
1. Read `PIPELINE_IMPLEMENTATION_PLAN.md` (full architecture)
2. Read `PIPELINE_EXECUTION_STRATEGY.md` (task breakdown)
3. Start Phase 1: Source parser + warm-up engine
4. Timeline: 33-50 hours over 2.5 weeks

### If Optimize → Refine the Warm-Up
1. Edit `spike-test.html` to reduce 8 messages → 4-5
2. Compress warm-up content
3. Re-test
4. Timeline: 1 week

### If Redesign → New Approach
1. Investigate alternatives (better system prompt, RAG, few-shot)
2. Build new spike test
3. Re-validate
4. Timeline: 1-2 weeks

---

## Resources

| Document | Purpose |
|----------|---------|
| `spike-test.html` | **This test page** |
| `spike-test-results.md` | **Fill in after testing** |
| `SPIKE-TEST-WALKTHROUGH.md` | **This document** |
| `PIPELINE_IMPLEMENTATION_PLAN.md` | Full architecture (1,360 lines) |
| `PIPELINE_EXECUTION_STRATEGY.md` | Task breakdown (824 lines) |
| `PIPELINE_QUALITY_REVIEW.md` | Scored review (7.0/10) |

---

## Summary

The spike test answers one critical question: **Does an 8-message warm-up improve LLM output quality for UCL generation?**

- **If yes**: Build the full pipeline (33-50 hours)
- **If marginal**: Optimize the warm-up sequence (1 week)
- **If no**: Redesign the approach (1-2 weeks)

**The spike test takes 10 minutes to run but saves you 33-50 hours of building the wrong thing.**

Run it now, document the results, and make an evidence-based decision.

---

**Last Updated**: 2026-06-27  
**Author**: UCL-Tool Team  
**Status**: Ready for testing
