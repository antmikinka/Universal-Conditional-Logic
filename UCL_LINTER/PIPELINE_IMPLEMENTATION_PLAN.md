# UCL Preparation Pipeline — Implementation Plan

> **Status**: Draft v1.0  
> **Created**: 2026-06-26  
> **Author**: Architecture analysis via Clear Thought MCP tools  
> **Purpose**: Implementation blueprint for a conversational UCL preparation pipeline

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Architecture Overview](#2-architecture-overview)
3. [Module Breakdown](#3-module-breakdown)
4. [Data Flow Diagrams](#4-data-flow-diagrams)
5. [API Design](#5-api-design)
6. [Integration Points](#6-integration-points)
7. [UI Design](#7-ui-design)
8. [Testing Strategy](#8-testing-strategy)
9. [Phased Implementation](#9-phased-implementation)
10. [Risk Assessment](#10-risk-assessment)
11. [Open Questions](#11-open-questions)

---

## 1. Executive Summary

### What We're Building

A **UCL Preparation Pipeline** that allows an LLM to "prepare" to generate UCL prompts by:

1. **Discovering tools dynamically** from source code files (.ts, .py, .js)
2. **Preloading context** via an 8-message warm-up conversation sequence
3. **Generating UCL-formatted output** from natural language prompts

### What This Is NOT

- NOT a modification of the existing `tools/*.json` files (those are UI-only for the Builder)
- NOT a tool-calling system
- NOT a replacement for the existing single-shot "Optimize" flow

### Critical Distinction

| Existing "Optimize" Flow | New "Preparation Pipeline" |
|---|---|
| Single API call (system + user) | 8-message warm-up + generation call |
| Pre-defined tools from JSON files | Tools discovered from source code |
| LLM already "knows" UCL syntax | LLM learns UCL via warm-up sequence |
| Output: optimized prompt text | Output: UCL-formatted system prompt |

---

## 2. Architecture Overview

### High-Level System Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      UCL PREPARATION PIPELINE                           │
│                                                                         │
│  ┌──────────────┐     ┌───────────────────┐     ┌────────────────────┐ │
│  │   SOURCE      │     │   WARM-UP         │     │   UCL GENERATION   │ │
│  │   DISCOVERY   │────>│   ENGINE          │────>│                    │ │
│  │               │     │                   │     │                    │ │
│  │  .ts files    │     │  Msg 1-2: Syntax  │     │  User Prompt ──┐   │ │
│  │  .py files    │     │  Msg 3-4: Tools   │     │                 │   │ │
│  │  .js files    │     │  Msg 5-6: Examples│     │  ┌──────────────┘   │ │
│  │               │     │  Msg 7-8: Rules   │     │  │                  │ │
│  └───────┬───────┘     └─────────┬─────────┘     │  ▼                  │ │
│          │                       │                │  LLM Response       │ │
│          ▼                       ▼                │  (UCL format)       │ │
│  DiscoveredTool[]        conversation[]          │                    │ │
│  ─ name                  ─ 8 msg pairs           └─────────┬──────────┘ │
│  ─ params                (user + assistant)                 │            │
│  ─ keywords                                              ▼            │
│  ─ source                                              UCL Output      │
└─────────────────────────────────────────────────────────────────────────┘
```

### Three Phases

| Phase | Name | Input | Output | Duration |
|-------|------|-------|--------|----------|
| 1 | Source Discovery | File paths (.ts, .py, .js) | `DiscoveredTool[]` | Instant (local parsing) |
| 2 | Context Preloading | DiscoveredTools + UCL_SYNTAX | `conversation[]` (16 messages) | ~30-60s (8 LLM calls) |
| 3 | UCL Generation | User prompt + conversation | UCL-formatted text | ~5-15s (1 LLM call) |

---

## 3. Module Breakdown

### 3.1 File Structure

```
UCL_LINTER/ucl-builder/
├── pipeline/                          # NEW MODULE
│   ├── source-parser.js               # Tool discovery from source files
│   ├── warmup-engine.js               # 8-message warm-up sequence manager
│   ├── warmup-templates.js            # Content templates for each warm-up step
│   ├── pipeline-orchestrator.js       # Coordinates all 3 phases
│   └── pipeline-ui.js                # Pipeline tab UI logic
│
├── tools/                             # EXISTING (unchanged)
│   ├── index.json
│   ├── filesystem.json
│   └── ... (9 domain files)
│
├── app.js                             # MODIFIED: Add Pipeline tab init
├── index.html                         # MODIFIED: Add Pipeline tab
├── tools-data.js                      # EXISTING (unchanged)
├── ucl-data.js                        # EXISTING (unchanged, used by warmup)
├── lemonade-api.js                    # EXISTING (reused as-is)
└── tools.js                           # EXISTING (unchanged)
```

### 3.2 Module Responsibilities

#### `source-parser.js` — Tool Discovery

**Purpose**: Read source files and extract tool/API definitions.

**Strategy**: Regex-based parsing for all languages (Phase 1). TypeScript Compiler API AST parsing (Phase 2 enhancement).

| Language | Parsing Method | What It Extracts |
|----------|---------------|-----------------|
| TypeScript (.ts) | Regex patterns | function signatures, exports, interfaces, JSDoc |
| Python (.py) | Regex patterns | def/class, type hints, docstrings, decorators |
| JavaScript (.js) | Regex patterns | function declarations, exports, arrow functions, JSDoc |

**Key Design Decision**: Regex-first avoids the ~60MB TypeScript Compiler API dependency. Validates the pipeline concept before adding AST complexity.

#### `warmup-engine.js` — Warm-Up Sequence Manager

**Purpose**: Execute the 8-message warm-up conversation with the LLM.

**Core Behavior**:
- Maintains a growing `messages[]` array (conversation history)
- Each step: send user message → receive assistant response → append both to history
- Tracks token usage across all 8 steps
- Reports progress via callback

**The 8 Messages**:

| Step | Phase | User Message Content | Expected Assistant Response |
|------|-------|---------------------|---------------------------|
| 1 | Syntax | UCL syntax reference (core directives) | Acknowledges + summarizes key directives |
| 2 | Syntax | Conditional logic patterns (^^CONDITION^^) | Confirms understanding of conditions |
| 3 | Tools | Discovered tool definitions formatted as UCL | Lists tools and their activation keywords |
| 4 | Tools | Tool parameter patterns and return types | Confirms parameter handling capability |
| 5 | Examples | 3-5 examples of natural → UCL conversions | Identifies patterns in the examples |
| 6 | Examples | Edge cases and complex conversion examples | Notes edge case handling strategies |
| 7 | Rules | Conversion rules (mapping heuristics) | Summarizes the conversion approach |
| 8 | Rules | Output format requirements + quality criteria | Confirms readiness to convert |

#### `warmup-templates.js` — Message Content Templates

**Purpose**: Generate the actual text content for each warm-up message.

**Key Functions**:
- `buildSyntaxMessage()` — Formats UCL_SYNTAX data into warm-up text
- `buildToolsMessage(discoveredTools)` — Formats discovered tools into UCL notation
- `buildExamplesMessage()` — Returns curated UCL conversion examples
- `buildRulesMessage()` — Returns conversion rules and heuristics
- `buildSystemInstruction()` — Base instruction for each warm-up step

#### `pipeline-orchestrator.js` — Phase Coordinator

**Purpose**: Coordinate the full pipeline flow: discovery → warmup → generation.

**State Machine**:

```
 idle ──[prepare]──> discovering ──[tools found]──> warmup ──[complete]──> ready
   ▲                                                        │
   │                                                    [generate]
   │                                                        │
   └──────────────────────────────────────────────── generating ──> complete
```

#### `pipeline-ui.js` — UI Logic

**Purpose**: Pipeline tab rendering, user interactions, progress display.

---

## 4. Data Flow Diagrams

### 4.1 Source Discovery Flow

```
User selects source files
         │
         ▼
┌─────────────────────┐
│  File Filter         │  Filter by extension (.ts, .py, .js)
│  ─────────           │  Exclude node_modules, .git, dist
│  Include patterns    │
│  Exclude patterns    │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Route by Extension  │
│  ─────────────────   │
│  .ts → tsParser      │
│  .py → pyParser      │
│  .js → jsParser      │
└─────────┬───────────┘
          │
    ┌─────┼─────┐
    ▼     ▼     ▼
┌──────┐┌──────┐┌──────┐
│ .ts  ││ .py  ││ .js  │     Each parser:
│parse ││parse ││parse │     1. Read file content
│File  ││File  ││File  │     2. Apply regex patterns
└──┬───┘└──┬───┘└──┬───┘     3. Extract signatures
   │       │       │          4. Build DiscoveredTool objects
   └───────┼───────┘
           │
           ▼
┌─────────────────────┐
│  Merge & Deduplicate │  Remove duplicates by name+file
│  ──────────────────  │
│  Auto-generate       │  Generate keywords from names
│  keywords            │
└─────────┬───────────┘
          │
          ▼
    DiscoveredTool[]
```

### 4.2 Warm-Up Sequence Flow

```
┌──────────────────────────────────────────────────────────────────┐
│                    WARM-UP ENGINE                                │
│                                                                  │
│  messages = []  (conversation history)                           │
│                                                                  │
│  Step 1 ──────────────────────────────────────────────           │
│  ┌─────────────────────────────┐                                │
│  │ messages.push({              │                                │
│  │   role: 'user',              │                                │
│  │   content: syntaxRef         │──── fetch('/v1/chat/completions')
│  │ })                           │         │                      │
│  └─────────────────────────────┘         │                      │
│                                          ▼                      │
│  ┌─────────────────────────────┐                                │
│  │ messages.push({              │                                │
│  │   role: 'assistant',         │◄──── response.choices[0]      │
│  │   content: llmResponse       │                                │
│  │ })                           │                                │
│  └─────────────────────────────┘                                │
│                                                                  │
│  Step 2 ─── (same pattern with conditional logic content)        │
│  Step 3 ─── (discovered tools formatted as UCL)                  │
│  ...                                                             │
│  Step 8 ─── (conversion rules + readiness confirmation)          │
│                                                                  │
│  After Step 8:                                                   │
│  ┌──────────────────────────────────────────────┐               │
│  │ messages = [                                  │               │
│  │   {role:'user',      content:'...syntax...'}, │               │
│  │   {role:'assistant', content:'...ack...'},     │               │
│  │   {role:'user',      content:'...syntax2...'}, │               │
│  │   {role:'assistant', content:'...ack...'},     │               │
│  │   ... (16 messages total)                     │               │
│  │ ]                                             │               │
│  └──────────────────────────────────────────────┘               │
│           │                                                      │
│           ▼                                                      │
│    conversation[] ──── passed to Generation phase                │
└──────────────────────────────────────────────────────────────────┘
```

### 4.3 Generation Flow

```
User enters prompt to convert
         │
         ▼
┌───────────────────────────────┐
│  Build generation request     │
│  ─────────────────────────    │
│  messages = [                 │
│    ...conversation (16 msgs), │  ← from warm-up
│    {                          │
│      role: 'user',            │
│      content: userPrompt      │  ← the prompt to convert
│    }                          │
│  ]                            │
└──────────┬────────────────────┘
           │
           ▼
┌───────────────────────────────┐
│  POST /v1/chat/completions    │  ← same API as existing optimize
│  {                            │
│    model: selectedModel,      │
│    messages: messages,        │
│    temperature: 0.7,          │
│    max_tokens: 4000           │
│  }                            │
└──────────┬────────────────────┘
           │
           ▼
┌───────────────────────────────┐
│  Extract response             │
│  ─────────────────────────    │
│  choices[0].message.content   │
│  = UCL-formatted output       │
│                               │
│  Example output:              │
│  [[CRITICAL: Agent system]]   │
│  ^^CONDITION: tool, call^^    │
│    [[TOOL: execute_task]]     │
│    [[PARAM: name:string]]     │
│  ^^/CONDITION^^               │
└───────────────────────────────┘
```

### 4.4 Integration with Existing System

```
┌────────────────────────────────────────────────────────────────┐
│  EXISTING SYSTEM (unchanged)                                   │
│                                                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐ │
│  │ tools/*.json │  │ ucl-data.js  │  │ lemonade-api.js      │ │
│  │ (71 tools,   │  │ (UCL_SYNTAX) │  │ (LLM communication)  │ │
│  │  UI-only)    │  │              │  │                      │ │
│  └──────────────┘  └──────┬───────┘  └──────────┬───────────┘ │
│                           │                      │             │
├───────────────────────────┼──────────────────────┼─────────────┤
│  NEW PIPELINE (uses existing, adds new)          │             │
│                           │                      │             │
│  ┌──────────────┐         │                      │             │
│  │ source-      │         │                      │             │
│  │ parser.js    │         │                      │             │
│  └──────┬───────┘         │                      │             │
│         │                 │                      │             │
│         ▼                 │                      │             │
│  ┌──────────────┐         │                      │             │
│  │ warmup-      │◄────────┘                      │             │
│  │ engine.js    │                                │             │
│  │              │◄───────────────────────────────┘             │
│  └──────┬───────┘                                             │
│         │                                                     │
│         ▼                                                     │
│  ┌──────────────┐                                             │
│  │ pipeline-    │                                             │
│  │ orchestrator │                                             │
│  └──────┬───────┘                                             │
│         │                                                     │
│         ▼                                                     │
│  ┌──────────────┐                                             │
│  │ pipeline-ui  │  ← New tab in existing Builder UI           │
│  └──────────────┘                                             │
└────────────────────────────────────────────────────────────────┘
```

---

## 5. API Design

### 5.1 Source Parser

```javascript
/**
 * source-parser.js — Tool discovery from source files
 */

const SourceParser = {
    /**
     * Discover tools from an array of file paths or directory paths.
     *
     * @param {string[]} paths - File or directory paths to scan
     * @param {object} options
     * @param {string[]} options.includePatterns - Glob patterns to include (e.g., ['**\/*.ts'])
     * @param {string[]} options.excludePatterns - Glob patterns to exclude (e.g., ['node_modules/**'])
     * @param {number} options.maxDepth - Max directory recursion depth (default: 5)
     * @param {boolean} options.extractDescriptions - Parse JSDoc/docstrings (default: true)
     * @returns {Promise<DiscoveredTool[]>}
     */
    async discoverTools(paths, options = {}) { ... },

    /**
     * Parse a single file and extract tool definitions.
     *
     * @param {string} filePath - Absolute path to source file
     * @returns {Promise<DiscoveredTool[]>}
     */
    async parseFile(filePath) { ... },

    /**
     * Parse TypeScript source (regex-based).
     * @private
     */
    _parseTypeScript(content, filePath) { ... },

    /**
     * Parse Python source (regex-based).
     * @private
     */
    _parsePython(content, filePath) { ... },

    /**
     * Parse JavaScript source (regex-based).
     * @private
     */
    _parseJavaScript(content, filePath) { ... },

    /**
     * Auto-generate keywords from tool name and description.
     * @private
     */
    _generateKeywords(tool) { ... },

    /**
     * Get supported file extensions.
     * @returns {string[]}
     */
    getSupportedExtensions() {
        return ['.ts', '.py', '.js'];
    }
};
```

### 5.2 DiscoveredTool Data Shape

```javascript
/**
 * @typedef {object} DiscoveredTool
 * @property {string} name          - Function/tool name (e.g., "read_file")
 * @property {string} description   - Extracted from JSDoc/docstring or inferred
 * @property {DiscoveredParam[]} parameters - Function parameters
 * @property {string} [returnType]  - Return type annotation if present
 * @property {string[]} keywords    - Auto-generated activation keywords
 * @property {object} source        - Where this tool was found
 * @property {string} source.file   - File path
 * @property {number} source.line   - Line number
 * @property {string} source.language - 'ts' | 'py' | 'js'
 * @property {string} rawSignature  - Original function signature text
 */

/**
 * @typedef {object} DiscoveredParam
 * @property {string} name          - Parameter name
 * @property {string} type          - Type annotation (or 'any' if unknown)
 * @property {boolean} required     - Whether the parameter is required
 * @property {string} [description] - Parameter description from JSDoc
 * @property {string} [default]     - Default value if present
 */
```

### 5.3 Warmup Engine

```javascript
/**
 * warmup-engine.js — 8-message warm-up sequence manager
 */

const WarmupEngine = {
    /**
     * @param {object} config
     * @param {DiscoveredTool[]} config.discoveredTools - Tools from source discovery
     * @param {string} config.model - Target LLM model
     * @param {string} config.provider - 'lemonade' | 'openrouter' | 'openai_compat'
     * @param {string} config.endpoint - API endpoint URL
     * @param {string} [config.apiKey] - API key (for OpenRouter)
     */
    init(config) { ... },

    /**
     * Build the 8 warm-up message templates.
     * Called after init, before executeAll.
     *
     * @returns {WarmupStep[]}
     */
    buildWarmupSequence() { ... },

    /**
     * Execute a single warm-up step.
     * Sends user message, receives assistant response.
     *
     * @param {number} stepNumber - Step 1-8
     * @param {function} [onProgress] - Progress callback
     * @returns {Promise<WarmupStep>}
     */
    async executeStep(stepNumber, onProgress) { ... },

    /**
     * Execute all 8 warm-up steps sequentially.
     *
     * @param {function} [onProgress] - Called after each step with (step, totalSteps, stepData)
     * @returns {Promise<WarmupState>}
     */
    async executeAll(onProgress) { ... },

    /**
     * Get the full conversation history for passing to generation.
     *
     * @returns {{role: string, content: string}[]}
     */
    getConversationHistory() { ... },

    /**
     * Get current warm-up state.
     * @returns {WarmupState}
     */
    getState() { ... },

    /**
     * Reset warm-up state.
     */
    reset() { ... },

    /**
     * Estimate total tokens in conversation history.
     * @returns {number}
     */
    estimateTokens() { ... }
};
```

### 5.4 Warmup State

```javascript
/**
 * @typedef {object} WarmupStep
 * @property {number} stepNumber       - 1 through 8
 * @property {string} phase            - 'syntax' | 'tools' | 'examples' | 'rules'
 * @property {string} phaseLabel       - Human-readable label
 * @property {string} userMessage      - Content of the user message
 * @property {string} [assistantResponse] - LLM's response (after execution)
 * @property {'pending'|'sending'|'confirmed'} status
 * @property {number} [tokenCount]     - Estimated tokens for this step
 */

/**
 * @typedef {object} WarmupState
 * @property {WarmupStep[]} steps      - All 8 warm-up steps
 * @property {object[]} conversation   - Full message history [{role, content}]
 * @property {number} totalTokensUsed  - Cumulative token estimate
 * @property {boolean} isComplete      - Whether all 8 steps executed
 * @property {number} currentStep      - Current step number (0 if not started)
 * @property {number} startedAt        - Timestamp
 * @property {number} [completedAt]    - Timestamp
 */
```

### 5.5 Pipeline Orchestrator

```javascript
/**
 * pipeline-orchestrator.js — Coordinates the full pipeline
 */

const PipelineOrchestrator = {
    /**
     * @param {object} config
     * @param {string[]} config.sourcePaths    - Files/directories to scan
     * @param {object} config.parserOptions    - Options for SourceParser
     * @param {string} config.model            - Target LLM model
     * @param {string} config.provider         - Provider type
     * @param {string} config.endpoint         - API endpoint
     * @param {string} [config.apiKey]         - API key
     */
    init(config) { ... },

    /**
     * Phase 1: Discover tools from source files.
     * @returns {Promise<DiscoveredTool[]>}
     */
    async discover() { ... },

    /**
     * Phase 2: Run the 8-message warm-up.
     * @param {function} [onProgress] - Progress callback
     * @returns {Promise<WarmupState>}
     */
    async warmup(onProgress) { ... },

    /**
     * Phase 3: Generate UCL from user prompt.
     * @param {string} userPrompt - Natural language prompt to convert
     * @returns {Promise<GenerationResult>}
     */
    async generate(userPrompt) { ... },

    /**
     * Full pipeline: discover → warmup → ready for generation.
     * @param {function} [onProgress] - Phase-level progress callback
     * @returns {Promise<void>}
     */
    async prepare(onProgress) { ... },

    /**
     * Get current pipeline state.
     * @returns {PipelineState}
     */
    getState() { ... },

    /**
     * Reset entire pipeline.
     */
    reset() { ... }
};

/**
 * @typedef {object} GenerationResult
 * @property {string} uclOutput      - The UCL-formatted text
 * @property {number} tokensUsed     - Tokens consumed by generation call
 * @property {number} duration       - Time in milliseconds
 * @property {string} model          - Model used
 * @property {number} totalPipelineTokens - Total tokens across warmup + generation
 */

/**
 * @typedef {object} PipelineState
 * @property {'idle'|'discovering'|'warmup'|'ready'|'generating'|'complete'} phase
 * @property {DiscoveredTool[]} discoveredTools
 * @property {WarmupState|null} warmupState
 * @property {GenerationResult|null} lastGeneration
 * @property {number} totalTokensUsed
 */
```

### 5.6 LLM Communication (Reused)

The warm-up and generation steps use the **same fetch pattern** as the existing `optimizeWithUCL()` in `app.js`. No changes needed to `lemonade-api.js`.

```javascript
// Pattern (already used in app.js, lines 760-808):
const response = await fetch(endpoint, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        model: model,
        messages: messages,  // ← Growing array during warm-up
        temperature: 0.7,
        max_tokens: 4000
    })
});
```

**Key**: During warm-up, `messages` grows from 2 messages (step 1) to 16 messages (step 8). The LLM sees the full conversation each time, which is what "primes" it.

---

## 6. Integration Points

### 6.1 What Is Reused (No Changes)

| Component | File | How It's Used |
|-----------|------|---------------|
| UCL Syntax Reference | `ucl-data.js` | Warmup Messages 1-2 pull directly from `UCL_SYNTAX` |
| LemonadeClient | `lemonade-api.js` | Same fetch pattern for warm-up and generation calls |
| App Settings | `app.js` AppState | Pipeline reuses model, endpoint, API key settings |
| UCL Formatter patterns | `ucl-tool-formatter.ts` | Logic for formatting tools as UCL directives (ported to JS) |

### 6.2 What Is Modified

| Component | File | Change |
|-----------|------|--------|
| Main App Init | `app.js` | Add `initPipelineUI()` call in DOMContentLoaded |
| HTML Layout | `index.html` | Add Pipeline tab button and panel container |

### 6.3 What Is NOT Touched

| Component | Reason |
|-----------|--------|
| `tools/*.json` | UI-only tool definitions; pipeline discovers tools independently |
| `tool-loader.js` | Loads JSON tool files for the UI; pipeline doesn't use it |
| `tools.js` | UI state/rendering for tool panel; pipeline has its own UI |
| `tools-data.js` | Hardcoded fallback data; pipeline uses source-parsed data |

### 6.4 Data Dependencies

```
ucl-data.js ──────────> warmup-templates.js
(UCL_SYNTAX)            (builds syntax messages)

source-parser.js ─────> warmup-templates.js
(DiscoveredTool[])      (builds tool messages)

warmup-engine.js ─────> pipeline-orchestrator.js
(conversation[])        (passes to generation)

AppState ─────────────> pipeline-orchestrator.js
(model, endpoint)       (uses for LLM calls)
```

---

## 7. UI Design

### 7.1 Tab Layout

Add a third tab to the existing sidebar tab bar:

```
┌──────────┬──────────┬──────────┐
│ ⚡ UCL   │ 🔧 Tools │ 🔄 Pipe  │
│  Syntax  │          │  line    │
└──────────┴──────────┴──────────┘
```

### 7.2 Pipeline Panel Layout

When the Pipeline tab is active, the sidebar shows:

```
┌─────────────────────────────┐
│ PIPELINE STATUS             │
│ ─────────────────────────── │
│ Phase: ● Ready              │
│ Tools: 12 discovered        │
│ Warm-up: 8/8 complete       │
│ Tokens: ~2,840 used         │
│                             │
│ ─────────────────────────── │
│ SOURCE FILES                │
│ ─────────────────────────── │
│ 📁 /src/tools/              │
│   agent.ts          (5)     │
│   search.ts         (3)     │
│   formatter.ts      (4)     │
│                             │
│ [+ Add Files]  [Re-scan]    │
│                             │
│ ─────────────────────────── │
│ WARM-UP STEPS               │
│ ─────────────────────────── │
│ ✅ 1. UCL Syntax Basics     │
│ ✅ 2. Conditional Logic      │
│ ✅ 3. Tool Definitions       │
│ ✅ 4. Tool Parameters        │
│ ✅ 5. Conversion Examples    │
│ ✅ 6. Edge Case Examples     │
│ ✅ 7. Conversion Rules       │
│ ✅ 8. Readiness Check        │
│                             │
│ [Re-run Warm-up]            │
└─────────────────────────────┘
```

### 7.3 Main Content Area — Pipeline Tab

```
┌──────────────────────────────────────────────────────────────────┐
│ Pipeline: UCL Preparation & Generation                           │
│ ────────────────────────────────────────────────────────────────  │
│                                                                   │
│ ┌─── Step 1: Source Discovery ─────────────────────────────────┐ │
│ │                                                               │ │
│ │  Source Paths:                                                │ │
│ │  ┌───────────────────────────────────────────────┐           │ │
│ │  │ C:\Users\antmi\project\src                     │           │ │
│ │  └───────────────────────────────────────────────┘           │ │
│ │  [+ Add Path]  [Exclude: node_modules, dist]                 │ │
│ │                                                               │ │
│ │  [Scan for Tools ▶]                                          │ │
│ │                                                               │ │
│ │  Results: 12 tools discovered across 3 files                 │ │
│ │  ┌─────────────────────────────────────────────────────┐     │ │
│ │  │ agent.ts:                                             │     │ │
│ │  │   • execute_task(name, params) → Promise<result>     │     │ │
│ │  │   • list_capabilities() → string[]                   │     │ │
│ │  │   • configure_agent(settings) → void                 │     │ │
│ │  │ search.ts:                                            │     │ │
│ │  │   • web_search(query, max_results) → SearchResult[] │     │ │
│ │  │   • ...                                               │     │ │
│ │  └─────────────────────────────────────────────────────┘     │ │
│ └───────────────────────────────────────────────────────────────┘ │
│                                                                   │
│ ┌─── Step 2: Context Preloading ───────────────────────────────┐ │
│ │                                                               │ │
│ │  [Start Warm-Up ▶]                                           │ │
│ │                                                               │ │
│ │  Progress: ████████████████░░░░░░░░  5/8 steps               │ │
│ │                                                               │ │
│ │  ✅ Msg 1-2: UCL Syntax loaded (420 tokens)                  │ │
│ │  ✅ Msg 3-4: 12 tool definitions loaded (890 tokens)         │ │
│ │  ✅ Msg 5-6: 6 examples loaded (640 tokens)                  │ │
│ │  ◉️ Msg 7:   Conversion rules... (running)                    │ │
│ │  ○ Msg 8:   Readiness check (pending)                        │ │
│ │                                                               │ │
│ └───────────────────────────────────────────────────────────────┘ │
│                                                                   │
│ ┌─── Step 3: Generate UCL ─────────────────────────────────────┐ │
│ │                                                               │ │
│ │  Enter prompt to convert to UCL:                              │ │
│ │  ┌─────────────────────────────────────────────────────┐     │ │
│ │  │ Build me an AI agent that can search the web,       │     │ │
│ │  │ read files from disk, and execute shell commands.   │     │ │
│ │  │ It should adapt its behavior based on the user's    │     │ │
│ │  │ expertise level.                                     │     │ │
│ │  └─────────────────────────────────────────────────────┘     │ │
│ │                                                               │ │
│ │  [Generate UCL ▶]                                            │ │
│ │                                                               │ │
│ │  ── Output ──────────────────────────────────────────────     │ │
│ │  ┌─────────────────────────────────────────────────────┐     │ │
│ │  │ [[CRITICAL: You are an adaptive AI agent]]          │     │ │
│ │  │                                                     │     │ │
│ │  │ ^^CONDITION: search, web, query^^                   │     │ │
│ │  │   [[TOOL: web_search]]                              │     │ │
│ │  │   [[PARAM: query:string:required]]                  │     │ │
│ │  │   [[PARAM: max_results:integer]]                    │     │ │
│ │  │ ^^/CONDITION^^                                      │     │ │
│ │  │                                                     │     │ │
│ │  │ ^^CONDITION: file, read, path^^                     │     │ │
│ │  │   [[TOOL: read_file]]                               │     │ │
│ │  │   [[PARAM: path:string:required]]                   │     │ │
│ │  │ ^^/CONDITION^^                                      │     │ │
│ │  │ ...                                                 │     │ │
│ │  └─────────────────────────────────────────────────────┘     │ │
│ │                                                               │ │
│ │  [Copy] [Download] [Swap]                                    │ │
│ │                                                               │ │
│ └───────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
```

---

## 8. Testing Strategy

### 8.1 Unit Tests — Source Parser

| Test Case | Input | Expected Output |
|-----------|-------|----------------|
| TS: exported function | `export function read_file(path: string): string` | DiscoveredTool with name='read_file', params=[{name:'path',type:'string',required:true}] |
| TS: interface extraction | `interface Config { name: string; value?: number }` | Parameters with optional field marked required=false |
| TS: JSDoc parsing | `/** Read a file */ \n export function...` | description='Read a file' |
| PY: def with type hints | `def search(query: str, limit: int = 10) -> list:` | DiscoveredTool with typed params and return type |
| PY: docstring extraction | Triple-quoted docstring | description extracted correctly |
| JS: arrow function export | `export const handler = (req, res) => {...}` | name='handler', params from destructuring |
| JS: JSDoc params | `@param {string} path - File path` | Parameter description extracted |
| Empty file | Empty string | Returns empty array |
| No exports | File with only internal functions | Returns empty array |
| Multiple files | Array of 3 file paths | Merged, deduplicated results |

### 8.2 Unit Tests — Warmup Engine

| Test Case | Validation |
|-----------|-----------|
| buildWarmupSequence() returns 8 steps | steps.length === 8 |
| Steps are in correct phase order | [syntax, syntax, tools, tools, examples, examples, rules, rules] |
| Each step has non-empty userMessage | steps.every(s => s.userMessage.length > 0) |
| executeStep() adds 2 messages to history | conversation.length increases by 2 |
| executeAll() completes all 8 steps | state.isComplete === true, conversation.length === 16 |
| Token estimation is reasonable | totalTokensUsed > 0 and < model's context window |
| reset() clears all state | state.currentStep === 0, conversation.length === 0 |
| Error handling: LLM returns error | Step marked as failed, pipeline doesn't crash |

### 8.3 Integration Tests — Pipeline Orchestrator

| Test Case | Validation |
|-----------|-----------|
| Full pipeline with mock parser | discover() returns tools, warmup() completes, generate() returns UCL |
| Pipeline state transitions | idle → discovering → warmup → ready → generating → complete |
| Re-running warmup after reset | Second warmup works correctly |
| Generate without warmup | Returns error or auto-triggers warmup |
| Token budget enforcement | Pipeline warns when approaching context window limit |

### 8.4 UI Tests (Manual)

| Test Case | Validation |
|-----------|-----------|
| Pipeline tab renders | Tab appears, panel shows correct layout |
| Add source path | File/directory picker works, paths displayed |
| Scan button | Tools discovered and displayed in results list |
| Start warm-up | Progress bar updates, steps show ✅ as they complete |
| Generate UCL | Output appears in output area with valid UCL syntax |
| Copy/Download buttons | Output can be copied or downloaded |
| Error states | Network errors show toast messages, UI doesn't freeze |
| Settings reuse | Changing model in settings affects pipeline calls |

---

## 9. Phased Implementation

### Phase 1: Foundation (Week 1-2)

**Goal**: Source parser + basic warmup engine working end-to-end.

| Task | File | Effort | Priority |
|------|------|--------|----------|
| Create pipeline/ directory | — | 5 min | P0 |
| Build source-parser.js (regex) | `pipeline/source-parser.js` | 4-6 hours | P0 |
| Build warmup-templates.js | `pipeline/warmup-templates.js` | 3-4 hours | P0 |
| Build warmup-engine.js | `pipeline/warmup-engine.js` | 4-6 hours | P0 |
| Build pipeline-orchestrator.js | `pipeline/pipeline-orchestrator.js` | 3-4 hours | P0 |
| Write unit tests for parsers | `pipeline/tests/source-parser.test.js` | 3-4 hours | P1 |
| Write unit tests for warmup | `pipeline/tests/warmup-engine.test.js` | 2-3 hours | P1 |
| Manual end-to-end test | — | 1-2 hours | P0 |

**Deliverable**: CLI-runnable pipeline that discovers tools from .ts files, runs warm-up against a local Lemonade model, and generates UCL output.

### Phase 2: UI Integration (Week 3)

**Goal**: Pipeline tab in the Builder UI.

| Task | File | Effort | Priority |
|------|------|--------|----------|
| Add Pipeline tab HTML | `index.html` | 2-3 hours | P0 |
| Build pipeline-ui.js | `pipeline/pipeline-ui.js` | 6-8 hours | P0 |
| Source file picker UI | `pipeline/pipeline-ui.js` | 3-4 hours | P0 |
| Warm-up progress display | `pipeline/pipeline-ui.js` | 2-3 hours | P0 |
| UCL output display | `pipeline/pipeline-ui.js` | 2-3 hours | P1 |
| Add Python parser | `pipeline/source-parser.js` | 2-3 hours | P1 |
| Add JavaScript parser | `pipeline/source-parser.js` | 2-3 hours | P1 |
| CSS styling for pipeline panel | `index.html` (inline styles) | 3-4 hours | P1 |

**Deliverable**: Working Pipeline tab that users can interact with in the browser.

### Phase 3: Quality & Polish (Week 4)

**Goal**: Error handling, edge cases, performance optimization.

| Task | File | Effort | Priority |
|------|------|--------|----------|
| Token usage tracking & display | `pipeline/warmup-engine.js` | 2-3 hours | P1 |
| Token budget warnings | `pipeline/pipeline-orchestrator.js` | 2-3 hours | P1 |
| Warm-up message compression | `pipeline/warmup-templates.js` | 4-6 hours | P2 |
| Error recovery (retry failed steps) | `pipeline/warmup-engine.js` | 3-4 hours | P1 |
| Pipeline state persistence | `pipeline/pipeline-ui.js` | 2-3 hours | P2 |
| Export/import pipeline configs | `pipeline/pipeline-orchestrator.js` | 3-4 hours | P2 |
| Streaming output support | `pipeline/pipeline-orchestrator.js` | 3-4 hours | P2 |

**Deliverable**: Production-quality pipeline with robust error handling.

### Phase 4: Enhancement (Week 5+)

**Goal**: TypeScript AST parsing, advanced features.

| Task | File | Effort | Priority |
|------|------|--------|----------|
| TypeScript Compiler API integration | `pipeline/source-parser.js` | 6-8 hours | P2 |
| Warm-up template customization | `pipeline/warmup-templates.js` | 4-6 hours | P2 |
| Multi-model warm-up (test across models) | `pipeline/pipeline-orchestrator.js` | 4-6 hours | P3 |
| Quality scoring of UCL output | `pipeline/pipeline-ui.js` | 4-6 hours | P3 |
| Batch generation (multiple prompts) | `pipeline/pipeline-orchestrator.js` | 3-4 hours | P3 |

**Deliverable**: Enhanced accuracy with AST parsing and extensibility features.

---

## 10. Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| **Context window overflow** — 8 warm-up messages exceed model's context | High (for small models) | High | Token estimation before each step; compress messages 1-6 if nearing limit; skip warm-up steps 5-6 for models < 8K context |
| **Regex parsing misses tools** — Complex TypeScript patterns not caught by regex | Medium | Medium | Phase 4 adds AST parsing; for Phase 1, document supported patterns and test against known codebases |
| **Warm-up quality varies by model** — Smaller models may not "learn" effectively | Medium | High | Test warm-up effectiveness across model sizes; provide warm-up template variants optimized for different model capabilities |
| **LLM API rate limits** — 9 calls (8 warmup + 1 generate) in quick succession | Low (local) / Medium (OpenRouter) | Medium | Add configurable delay between warm-up steps; batch where possible; respect 429 responses |
| **User confusion** — Pipeline tab vs existing Optimize button | Low | Low | Clear labeling; help tooltip explaining the difference; default to existing Optimize for simple use cases |
| **Browser file access** — Cannot read arbitrary filesystem paths from browser | High | High | Use `<input type="file" webkitdirectory>` for directory selection, or require files to be served from the app's directory, or use File System Access API where available |

### Critical Risk Detail: Browser File Access

The biggest technical constraint is that **web browsers cannot read arbitrary file paths**. The source parser needs file contents, but the Builder runs in a browser.

**Solutions (ranked by feasibility)**:

1. **File Input + Drag/Drop** (Phase 1): Use `<input type="file" multiple webkitdirectory>` to let users select files/directories. Read contents via `FileReader`. Works in all modern browsers.

2. **File System Access API** (Phase 2): `window.showDirectoryPicker()` gives more control. Supported in Chrome/Edge.

3. **Local server proxy** (Phase 3): If the Builder is served by a local server (e.g., Python http.server), add an API endpoint that reads files from disk. Most flexible but requires server setup.

4. **Pre-process offline** (Alternative): Run source-parser as a Node.js script that outputs a JSON file. User loads this JSON into the Builder.

**Recommendation**: Start with option 1 (file input) for Phase 1. It's the simplest and works immediately.

---

## 11. Open Questions

### Decision Required Before Implementation

| # | Question | Options | Recommendation |
|---|----------|---------|----------------|
| 1 | Should warm-up messages be hardcoded or user-customizable? | A) Hardcoded templates B) User-editable templates C) Both | **C** — Ship with defaults (Phase 1), allow customization (Phase 3) |
| 2 | How should we handle the browser file access limitation? | A) File input B) File System Access API C) Local server proxy D) Offline pre-process | **A** for Phase 1, evaluate **B** for Phase 2 |
| 3 | Should the warm-up conversation be visible to the user? | A) Show all messages B) Show summary C) Hide | **A** — Transparency builds trust; show collapsible message log |
| 4 | What happens when the user changes model after warm-up? | A) Invalidate warm-up B) Re-warm automatically C) Warn user | **C** — Show warning that warm-up was done for a different model |
| 5 | Should the pipeline share the existing "Optimize" button or have its own? | A) Replace Optimize B) Separate button C) Integrate into Optimize with toggle | **B** — Keep Optimize for quick single-shot; Pipeline for thorough preparation |
| 6 | Token budget: what's the maximum warm-up should consume? | A) No limit B) 50% of model context C) Fixed 4K tokens | **B** — Scale with model; 50% leaves room for generation |

---

## Appendix A: Warm-Up Message Templates (Draft)

### Message 1: UCL Syntax Basics

**User message:**
```
You are learning Universal Conditional Logic (UCL) — a mathematical framework
for prompt optimization. Here are the core directives you must know:

[[CRITICAL: content]]     — HIGHEST PRIORITY. +9.3% quality. Always place at top.
[[REQUIRE: requirement]]  — Specify required conditions
[[TRANSFORM: src TO dst]] — Define transformation rules
[[VALIDATE: rule]]        — Specify validation requirements
[[ENFORCE: rule]]         — Strictly enforce a constraint
[[APPLY: rule]]           — Apply specific logic
[[TOOL: tool_name]]       — Reference a tool
[[PARAM: name:type]]      — Define a tool parameter

Key principle: The Over-Specification Paradox — more detail can degrade performance.
Optimal specification threshold S* ≈ 0.509.

Acknowledge that you understand these directives and their purposes.
```

**Expected assistant response** (paraphrased):
```
I understand the UCL directives. [[CRITICAL:]] is the highest priority directive
with a +9.3% quality bonus and should always be placed at the top of prompts.
[etc.]
```

### Message 3: Tool Definitions (Dynamic)

**User message** (generated from discovered tools):
```
Here are the tools you will have access to, discovered from the project source code:

^^CONDITION: read, file, open, load^^
  [[TOOL: read_file]]
  [[PARAM: path:string:required]]
  [[PARAM: encoding:string]]
  // Read file content from disk
^^/CONDITION^^

^^CONDITION: write, save, create^^
  [[TOOL: write_file]]
  [[PARAM: path:string:required]]
  [[PARAM: content:string:required]]
  // Write content to a file
^^/CONDITION^^

... [additional tools] ...

Acknowledge these tools and their activation patterns.
```

### Message 5: Conversion Examples

**User message:**
```
Here are examples of converting natural language prompts into UCL format:

EXAMPLE 1:
Input: "Build me a chatbot that can search the web and answer questions"
Output:
  [[CRITICAL: You are a research chatbot]]
  ^^CONDITION: search, web, query, find^^
    [[TOOL: web_search]]
    [[PARAM: query:string:required]]
    [[REQUIRE: Return top 5 results with URLs]]
  ^^/CONDITION^^
  ^^CONDITION: question, answer, explain^^
    [[APPLY: Synthesize answers from search results]]
    [[REQUIRE: Cite sources]]
  ^^/CONDITION^^

EXAMPLE 2:
Input: "Create a code reviewer that checks Python style"
Output:
  [[CRITICAL: You are a Python code reviewer]]
  ^^CONDITION: review, check, lint, style^^
    [[ENFORCE: PEP 8 compliance]]
    [[VALIDATE: Type hint presence]]
    [[APPLY: Static analysis rules]]
  ^^/CONDITION^^

Study these patterns. Acknowledge what you've learned.
```

---

## Appendix B: Regex Patterns for Source Parsing

### TypeScript Patterns

```javascript
const TS_PATTERNS = {
    // export function name(param: type, ...): returnType
    exportedFunction: /export\s+(?:async\s+)?function\s+(\w+)\s*(?:<[^>]*>)?\s*\(([^)]*)\)(?:\s*:\s*([^\n{]+))?/g,

    // export const name = (param: type): returnType =>
    exportedArrow: /export\s+(?:const|let)\s+(\w+)\s*=\s*(?:async\s+)?\(([^)]*)\)(?:\s*:\s*([^\n=]+))?\s*=>/g,

    // interface Name { prop: type; ... }
    interface: /export\s+interface\s+(\w+)\s*(?:extends\s+[^{]+)?\{([^}]+)\}/g,

    // JSDoc: /** description */
    jsdoc: /\/\*\*\s*\n?\s*\*\s*(.+?)(?:\n|\*\/)/g,

    // @param {type} name - description
    jsdocParam: /@param\s+\{([^}]+)\}\s+(\w+)\s*(?:-\s*(.+))?/g,

    // @returns {type} description
    jsdocReturn: /@returns?\s+\{([^}]+)\}\s*(.*)/g
};
```

### Python Patterns

```javascript
const PY_PATTERNS = {
    // def function_name(param: type, ...) -> return_type:
    functionDef: /(?:async\s+)?def\s+(\w+)\s*\(([^)]*)\)(?:\s*->\s*([^\n:]+))?\s*:/g,

    // class ClassName(BaseClass):
    classDef: /class\s+(\w+)(?:\(([^)]*)\))?\s*:/g,

    // Decorator: @decorator_name
    decorator: /@(\w+(?:\([^)]*\))?)/g,

    // Docstring: """description"""
    docstring: /"""([\s\S]*?)"""|'''([\s\S]*?)'''/g,

    // Type hints in params: name: type = default
    paramTypeHint: /(\w+)(?:\s*:\s*([^=,)]+))?(?:\s*=\s*([^,)]+))?/g
};
```

### JavaScript Patterns

```javascript
const JS_PATTERNS = {
    // function name(params) {
    functionDecl: /(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\(([^)]*)\)/g,

    // const/let/var name = (params) =>
    arrowFunc: /(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?\(([^)]*)\)\s*=>/g,

    // module.exports = { name: ... }
    moduleExport: /module\.exports\s*=\s*\{([^}]+)\}/g,

    // JSDoc (same as TypeScript)
    jsdoc: /\/\*\*\s*\n?\s*\*\s*(.+?)(?:\n|\*\/)/g,
    jsdocParam: /@param\s+\{([^}]+)\}\s+(\w+)\s*(?:-\s*(.+))?/g,
    jsdocReturn: /@returns?\s+\{([^}]+)\}\s*(.*)/g
};
```

---

## Appendix C: Token Budget Planning

### Warm-Up Token Estimates

| Step | Phase | Estimated Input Tokens | Estimated Output Tokens | Cumulative |
|------|-------|----------------------|------------------------|------------|
| 1 | Syntax | ~500 | ~150 | ~650 |
| 2 | Syntax | ~800 (+650 history) | ~150 | ~1,600 |
| 3 | Tools | ~1,200 (+1,600 history) | ~200 | ~3,000 |
| 4 | Tools | ~1,400 (+3,000 history) | ~200 | ~4,600 |
| 5 | Examples | ~1,000 (+4,600 history) | ~200 | ~5,800 |
| 6 | Examples | ~1,200 (+5,800 history) | ~200 | ~7,200 |
| 7 | Rules | ~800 (+7,200 history) | ~200 | ~8,200 |
| 8 | Rules | ~600 (+8,200 history) | ~150 | ~8,950 |

**Total warm-up cost**: ~9,000 tokens in context window by step 8.

### Context Window Requirements

| Model Context | Warm-up Fits? | Generation Room | Notes |
|--------------|--------------|-----------------|-------|
| 4K tokens | NO | — | Need compression; skip steps 5-6 |
| 8K tokens | Barely | ~0 tokens | Too tight; compress or reduce steps |
| 16K tokens | YES | ~7K tokens | Adequate for generation |
| 32K tokens | YES | ~23K tokens | Comfortable |
| 128K tokens | YES | ~119K tokens | No constraints |

### Compression Strategy (for small models)

1. **Combine steps**: Merge messages 1+2, 3+4, 5+6, 7+8 into 4 messages instead of 8
2. **Summarize**: After step 4, ask LLM to summarize what it learned, then continue with summary instead of full history
3. **Skip examples**: Steps 5-6 (examples) are nice-to-have; syntax + tools + rules are essential
4. **System prompt approach**: Put all warm-up content in the system prompt instead of multi-turn (loses "priming" effect but saves tokens)

---

*End of Implementation Plan*
