# UCL Preparation Pipeline — Execution Strategy

> **Status**: Approved for Phase 1 Implementation  
> **Created**: 2026-06-26  
> **Based on**: `PIPELINE_IMPLEMENTATION_PLAN.md` (Architecture Blueprint)  
> **Purpose**: Developer-ready task sequence with dependencies, tests, and integration points

---

## 1. Approach Summary

This document translates the architecture plan into a sequential build order. The strategy is **vertical slicing**: each task produces a testable artifact before moving to the next, so bugs are caught at the module level rather than during integration.

**Key architectural decisions made during strategy development:**

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Tab integration | pipeline-ui.js injects a third sidebar tab after tools.js creates the first two | Mirrors existing pattern; no changes to tools.js |
| Main content area | New `div#pipelineContainer` shown/hidden alongside `editor-container` | Clean separation; existing Optimize flow untouched |
| LLM calls | warmup-engine.js contains its own fetch helper (mirrors optimizeWithUCL pattern) | Avoids modifying app.js; acceptable duplication for Phase 1 |
| File access | `<input type="file" webkitdirectory multiple>` → SourceParser accepts `File[]` | Works in all modern browsers; no server needed |
| Token budget | Build token estimation into warmup-engine from Day 1 (not deferred to Phase 3) | Critical risk; must be validated early |
| app.js changes | Single line: `await initPipelineUI()` added after `initToolsUI()` in DOMContentLoaded | Minimal surface area; pipeline is isolated |

---

## 2. Dependency Graph

```
                    ┌──────────────────┐
                    │ TASK 1: Shared   │
                    │ Types + Dir      │
                    │ (15 min)         │
                    └────────┬─────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼                              ▼
   ┌────────────────────┐        ┌────────────────────┐
   │ TASK 2:            │        │ TASK 3:            │
   │ source-parser.js   │        │ warmup-templates.js│
   │ (4-6 hours)        │        │ (3-4 hours)        │
   └────────┬───────────┘        └────────┬───────────┘
            │                             │
            │    ┌────────────────────────┘
            ▼    ▼
   ┌────────────────────┐
   │ TASK 4:            │
   │ warmup-engine.js   │◄─── Also depends on AppState (existing)
   │ (4-6 hours)        │     and lemonade-api.js fetch pattern
   └────────┬───────────┘
            │
            ▼
   ┌────────────────────┐
   │ TASK 5:            │
   │ pipeline-          │
   │ orchestrator.js    │
   │ (3-4 hours)        │
   └────────┬───────────┘
            │
     ┌──────┴──────┐
     ▼             ▼
┌──────────┐  ┌──────────────┐
│ TASK 6:  │  │ TASK 7:      │
│ E2E      │  │ pipeline-    │
│ smoke    │  │ ui.js + HTML │
│ test     │  │ (8-10 hours) │
│ (1-2 hr) │  └──────────────┘
└──────────┘
     │
     ▼ (parallel track)
┌──────────────────┐    ┌──────────────────────┐
│ TASK 8: Python   │    │ TASK 9: Token budget  │
│ + JS parsers     │    │ enforcement           │
│ (4-6 hours)      │    │ (3-4 hours)           │
└──────────────────┘    └──────────────────────┘
```

**Critical path**: Task 1 → 2 → 4 → 5 → 6 (total: ~14-20 hours)

Tasks 3, 7, 8, 9 can proceed in parallel once their dependencies are met.

---

## 3. Task List (Developer-Ready)

### TASK 1: Create Pipeline Module Skeleton
**File**: `ucl-builder/pipeline/` (new directory)  
**Effort**: 15 minutes  
**Depends on**: Nothing  

**What to do:**
1. Create directory `ucl-builder/pipeline/`
2. Create `pipeline/pipeline-types.js` with JSDoc typedefs for:
   - `DiscoveredParam` — `{name, type, required, description?, default?}`
   - `DiscoveredTool` — `{name, description, parameters, returnType?, keywords, source, rawSignature}`
   - `WarmupStep` — `{stepNumber, phase, phaseLabel, userMessage, assistantResponse?, status, tokenCount?}`
   - `WarmupState` — `{steps, conversation, totalTokensUsed, isComplete, currentStep, startedAt, completedAt?}`
   - `GenerationResult` — `{uclOutput, tokensUsed, duration, model, totalPipelineTokens}`
   - `PipelineState` — `{phase, discoveredTools, warmupState, lastGeneration, totalTokensUsed}`

**Why a separate types file**: All other modules import these types. Centralizing them prevents circular dependencies and makes the data contract explicit.

**Test**: File loads in browser console without errors. Types are documented.

---

### TASK 2: Build Source Parser (TypeScript)
**File**: `ucl-builder/pipeline/source-parser.js`  
**Effort**: 4-6 hours  
**Depends on**: Task 1  

**What to build:**

```
SourceParser object with:
├── discoverToolsFromFiles(files: File[]) → Promise<DiscoveredTool[]>
│   1. Filter files by extension (.ts, .py, .js)
│   2. Read each file via FileReader → text
│   3. Route to language-specific parser by extension
│   4. Merge results, deduplicate by name+file
│   5. Auto-generate keywords from function names
│
├── _parseTypeScript(content: string, fileName: string) → DiscoveredTool[]
│   Regex patterns (from plan Appendix B):
│   - exportedFunction: export function name(params): returnType
│   - exportedArrow: export const name = (params): returnType =>
│   - interface: export interface Name { props }
│   - jsdoc: /** description */
│   - jsdocParam: @param {type} name - description
│   For each match:
│   1. Extract function name
│   2. Parse parameter list (split by comma, extract name:type, detect optional via ?)
│   3. Extract return type
│   4. Extract JSDoc description (look backwards from function position)
│   5. Build DiscoveredTool object
│
├── _parsePython(content, fileName) → DiscoveredTool[]   [STUB for Phase 1]
├── _parseJavaScript(content, fileName) → DiscoveredTool[] [STUB for Phase 1]
│
├── _generateKeywords(tool) → string[]
│   Split name by underscore/camelCase
│   Add type-specific keywords (e.g., "read" for readFile, "file" for readFile)
│   Return lowercase keyword array
│
└── getSupportedExtensions() → ['.ts', '.py', '.js']
```

**Implementation notes:**
- For browser file reading: `const text = await file.text()` (File API, no FileReader needed in modern browsers)
- Deduplication key: `${tool.name}@${tool.source.file}:${tool.source.line}`
- Keyword generation: split `read_file` → `['read', 'file']`, split `readFile` → `['read', 'file']`
- The regex patterns from the plan (Appendix B) are a starting point — test and refine against real files

**Test criteria:**
1. Create a test `.ts` file with known functions:
   ```typescript
   /** Read a file from disk */
   export async function read_file(path: string, encoding?: string): Promise<string> { ... }
   
   /** Execute a task */
   export const executeTask = (name: string, params: Record<string, any>) => { ... }
   
   export interface Config {
     name: string;
     value?: number;
   }
   ```
2. Load it via file input, call `discoverToolsFromFiles()`
3. Verify:
   - `read_file` discovered with params `[{name:'path', type:'string', required:true}, {name:'encoding', type:'string', required:false}]`
   - `executeTask` discovered with params `[{name:'name', type:'string', required:true}, {name:'params', type:'Record<string, any>', required:true}]`
   - `Config` interface discovered (or its properties extracted)
   - Keywords for `read_file`: `['read', 'file']`
   - Source info: `{file: 'test.ts', line: 2, language: 'ts'}`
4. Test empty file → returns `[]`
5. Test file with no exports → returns `[]`

**Test in browser console:**
```javascript
// After loading pipeline scripts via <script> tags:
const input = document.createElement('input');
input.type = 'file'; input.webkitdirectory = true; input.multiple = true;
input.onchange = async (e) => {
    const files = Array.from(e.target.files);
    const tools = await SourceParser.discoverToolsFromFiles(files);
    console.table(tools);
};
input.click();
```

---

### TASK 3: Build Warm-Up Templates
**File**: `ucl-builder/pipeline/warmup-templates.js`  
**Effort**: 3-4 hours  
**Depends on**: Task 1, `ucl-data.js` (existing, for UCL_SYNTAX)  

**What to build:**

```
WarmupTemplates object with:
├── buildSystemInstruction() → string
│   Base instruction: "You are learning UCL to convert prompts..."
│
├── buildStep1_Message() → string
│   Formats UCL_SYNTAX core directives into learning text
│   Pulls from UCL_SYNTAX['Directives'] and UCL_SYNTAX['Getting Started']
│   Ends with: "Acknowledge that you understand these directives."
│
├── buildStep2_Message() → string
│   Conditional logic patterns: ^^CONDITION: keywords^^ ... ^^/CONDITION^^
│   Examples of keyword-based activation
│   Ends with: "Confirm your understanding of conditions."
│
├── buildStep3_Message(discoveredTools: DiscoveredTool[]) → string
│   Formats each tool as UCL:
│   ^^CONDITION: keyword1, keyword2^^
│     [[TOOL: tool_name]]
│     [[PARAM: name:type:required]]
│   ^^/CONDITION^^
│   Ends with: "Acknowledge these tools."
│
├── buildStep4_Message(discoveredTools: DiscoveredTool[]) → string
│   Tool parameter patterns and return types
│   Ends with: "Confirm parameter handling."
│
├── buildStep5_Message() → string
│   3-5 hardcoded examples (from plan Appendix A, Message 5)
│   Natural language → UCL conversion pairs
│   Ends with: "Study these patterns."
│
├── buildStep6_Message() → string
│   Edge cases and complex examples
│   Ends with: "Note edge case strategies."
│
├── buildStep7_Message() → string
│   Conversion rules/heuristics
│   Ends with: "Summarize your approach."
│
├── buildStep8_Message() → string
│   Output format requirements + quality criteria
│   Ends with: "Confirm readiness."
│
└── getStepMetadata(stepNumber) → {phase, phaseLabel}
    Returns phase ('syntax'|'tools'|'examples'|'rules') and label
```

**Implementation notes:**
- Steps 1-2 pull from `UCL_SYNTAX` (global from ucl-data.js) — iterate over categories and extract syntax/description fields
- Steps 3-4 accept `DiscoveredTool[]` and format them into UCL notation
- Steps 5-8 are **hardcoded strings** for Phase 1 (Phase 3 adds customization)
- Each message function returns only the USER message content. The system instruction is separate.
- Target token budget per message: ~500-1200 tokens (see plan Appendix C)

**Test criteria:**
1. `buildStep1_Message()` returns non-empty string containing `[[CRITICAL:`, `[[REQUIRE:`, etc.
2. `buildStep3_Message([{name:'read_file', parameters:[...], keywords:['read','file']}])` returns string containing `^^CONDITION: read, file^^` and `[[TOOL: read_file]]`
3. All 8 message builders return strings > 100 characters
4. Step metadata returns correct phases: [syntax, syntax, tools, tools, examples, examples, rules, rules]

---

### TASK 4: Build Warm-Up Engine
**File**: `ucl-builder/pipeline/warmup-engine.js`  
**Effort**: 4-6 hours  
**Depends on**: Task 2 (for DiscoveredTool type), Task 3 (for templates), AppState (existing)  

**What to build:**

```
WarmupEngine object with:
├── _state: WarmupState (internal)
├── _config: {discoveredTools, model, provider, endpoint, apiKey}
│
├── init(config) → void
│   Store config, reset state, build warmup sequence
│
├── buildWarmupSequence() → WarmupStep[]
│   Call WarmupTemplates for each step
│   Inject discoveredTools into steps 3-4
│   Return 8 WarmupStep objects with status='pending'
│
├── async executeStep(stepNumber, onProgress) → WarmupStep
│   1. Get step template (userMessage)
│   2. Push {role:'user', content:userMessage} to conversation
│   3. Call _callLLM(conversation) → assistantContent
│   4. Push {role:'assistant', content:assistantContent} to conversation
│   5. Update step: status='confirmed', assistantResponse, tokenCount
│   6. Call onProgress(stepNumber, 8, step)
│   7. Return step
│
├── async executeAll(onProgress) → WarmupState
│   For step 1-8: await executeStep(i, onProgress)
│   Mark state.isComplete = true
│   Return state
│
├── getConversationHistory() → {role, content}[]
│   Return _state.conversation (16 messages after full warmup)
│
├── getState() → WarmupState
├── reset() → void (clear state, reset to step 0)
│
├── estimateTokens() → number
│   Simple estimate: totalChars / 4 across all conversation messages
│   (1 token ≈ 4 characters in English)
│
├── async _callLLM(messages) → string
│   SAME PATTERN as optimizeWithUCL() in app.js lines 759-808:
│   - If provider === 'openrouter': fetch openrouter.ai with Authorization header
│   - If provider === 'lemonade': fetch endpoint/v1/chat/completions
│   - If provider === 'openai_compat': fetch endpoint/v1/chat/completions
│   Body: {model, messages, temperature: 0.7, max_tokens: 4000}
│   Return: choices[0].message.content
│   Throw on error (don't catch — let orchestrator handle)
│
└── _estimateStepTokens(step) → number
    Estimate tokens for a single step's messages
```

**Critical implementation details:**

1. **Token estimation from Day 1**: After each step, call `estimateTokens()`. If total exceeds 50% of model's context window (configurable), emit a warning via onProgress. This validates the risk mitigation early.

2. **Error handling per step**: If `_callLLM()` throws, mark the step as `status: 'failed'` and stop execution. Don't silently skip — the warm-up is all-or-nothing for quality.

3. **Progress callback signature**: `onProgress(currentStep, totalSteps, stepData)` — the UI uses this to update the progress bar.

4. **Context window limits by model** (hardcoded lookup table):
   ```javascript
   const CONTEXT_LIMITS = {
       '4K': 4096, '8K': 8192, '16K': 16384, '32K': 32768, '128K': 131072
   };
   // Default assumption: 16K if model unknown
   ```

**Test criteria:**
1. `init({discoveredTools: [], model: 'test', provider: 'lemonade', endpoint: 'http://localhost:13305'})` succeeds
2. `buildWarmupSequence()` returns exactly 8 steps with correct phase order
3. `executeStep(1)` adds 2 messages to conversation (user + assistant)
4. `executeAll()` results in `conversation.length === 16`, `state.isComplete === true`
5. `estimateTokens()` returns reasonable number (>0, < context window)
6. `reset()` clears everything: `state.currentStep === 0`, `conversation.length === 0`
7. **Live test**: Run against local Lemonade server, verify all 8 steps complete without errors

**Milestone: At this point, you can run a complete warm-up conversation from the browser console.**

---

### TASK 5: Build Pipeline Orchestrator
**File**: `ucl-builder/pipeline/pipeline-orchestrator.js`  
**Effort**: 3-4 hours  
**Depends on**: Task 2 (source-parser), Task 4 (warmup-engine)  

**What to build:**

```
PipelineOrchestrator object with:
├── _state: PipelineState (internal)
├── _config: {sourcePaths, parserOptions, model, provider, endpoint, apiKey}
│
├── init(config) → void
│   Store config, set state.phase = 'idle'
│
├── async discover() → DiscoveredTool[]
│   state.phase = 'discovering'
│   Call SourceParser.discoverToolsFromFiles(_files)
│   state.discoveredTools = result
│   state.phase = 'warmup' (if tools found) or 'idle' (if none)
│   Return result
│
├── async warmup(onProgress) → WarmupState
│   Requires: state.discoveredTools.length > 0
│   WarmupEngine.init({discoveredTools, model, provider, endpoint, apiKey})
│   state.warmupState = await WarmupEngine.executeAll(onProgress)
│   state.phase = 'ready'
│   Return state.warmupState
│
├── async generate(userPrompt) → GenerationResult
│   Requires: state.phase === 'ready'
│   state.phase = 'generating'
│   Build messages: [...conversation, {role:'user', content:userPrompt}]
│   Call _callLLM(messages) → uclOutput
│   Build GenerationResult
│   state.lastGeneration = result
│   state.phase = 'complete'
│   Return result
│
├── async prepare(files, onProgress) → void
│   Full pipeline: discover(files) → warmup(onProgress)
│   Single entry point for UI to call
│
├── getState() → PipelineState
├── reset() → void (clear all state, phase = 'idle')
│
└── _callLLM(messages) → Promise<string>
    Same fetch pattern as WarmupEngine._callLLM()
    (Could be extracted to shared helper in Phase 3)
```

**State machine enforcement:**
```
idle ──[discover()]──> discovering ──[tools found]──> warmup ──[complete]──> ready
  ▲                                                          │
  │                                                      [generate()]
  │                                                          │
  └──────────────────── [reset()] ──────────────── generating ──> complete
```

Guard each method with state checks:
- `warmup()` throws if `state.discoveredTools.length === 0`
- `generate()` throws if `state.phase !== 'ready'`
- `discover()` throws if no files provided

**Test criteria:**
1. `init()` → state.phase === 'idle'
2. `discover()` with mock files → state.phase === 'warmup', discoveredTools populated
3. `warmup()` → state.phase === 'ready', warmupState.isComplete === true
4. `generate("build a chatbot")` → returns GenerationResult with uclOutput containing UCL syntax
5. Full `prepare()` flow: idle → discovering → warmup → ready
6. `generate()` before `warmup()` throws error
7. `reset()` returns to idle state

**Milestone: The entire backend pipeline is now functional. A developer can run the full flow from the browser console.**

---

### TASK 6: End-to-End Smoke Test
**Effort**: 1-2 hours  
**Depends on**: Task 5  

**What to do:**
1. Open `index.html` in browser with all pipeline scripts loaded via `<script>` tags (add temporarily for testing)
2. Select a directory of real TypeScript files (e.g., the project's own source)
3. Run `PipelineOrchestrator.discover()` → verify tools found
4. Run `PipelineOrchestrator.warmup()` → verify 8 steps complete against Lemonade
5. Run `PipelineOrchestrator.generate("Build an AI agent that searches the web")` → verify UCL output
6. Check token usage — does it stay under 16K?
7. Check output quality — does the UCL output contain valid syntax?
8. Test error case: what happens if Lemonade server is down?

**Pass/fail criteria:**
| Check | Pass | Fail |
|-------|------|------|
| Tools discovered | ≥1 tool found from real .ts files | 0 tools, regex failures |
| Warm-up completes | All 8 steps, 16 messages | Error on any step |
| UCL output valid | Contains `[[CRITICAL:`, `^^CONDITION:` | Plain text, no UCL syntax |
| Token budget | < 16K tokens total | > model context window |
| Error handling | Graceful error, console message | Unhandled exception, frozen UI |

**If this fails:** Debug at the failing step. Most likely issues:
- Regex doesn't match real code → refine patterns in Task 2
- LLM returns garbage → adjust warm-up templates in Task 3
- Token overflow → add compression logic early

---

### TASK 7: Build Pipeline UI + HTML Integration
**Files**: `pipeline/pipeline-ui.js` (new), `index.html` (modified), `styles.css` (modified)  
**Effort**: 8-10 hours  
**Depends on**: Task 5 (orchestrator must work)  

**HTML changes to `index.html`:**

1. Add pipeline container after editor-container (line ~189):
```html
<!-- Pipeline Container (hidden by default) -->
<div class="pipeline-container" id="pipelineContainer" style="display:none;">
    <!-- Step 1: Source Discovery -->
    <div class="pipeline-section" id="pipelineDiscovery">
        <h3>Step 1: Source Discovery</h3>
        <div class="pipeline-file-area">
            <input type="file" id="pipelineFileInput" webkitdirectory multiple 
                   accept=".ts,.py,.js" style="display:none;">
            <button class="btn btn-primary" id="pipelineAddFilesBtn">
                Add Source Files
            </button>
            <div class="pipeline-file-list" id="pipelineFileList"></div>
            <button class="btn btn-secondary" id="pipelineScanBtn" disabled>
                Scan for Tools
            </button>
            <div class="pipeline-results" id="pipelineDiscoveryResults"></div>
        </div>
    </div>
    
    <!-- Step 2: Context Preloading -->
    <div class="pipeline-section" id="pipelineWarmup">
        <h3>Step 2: Context Preloading</h3>
        <button class="btn btn-primary" id="pipelineWarmupBtn" disabled>
            Start Warm-Up
        </button>
        <div class="pipeline-progress" id="pipelineWarmupProgress">
            <div class="progress-bar-container">
                <div class="progress-bar" id="pipelineProgressBar" style="width:0%"></div>
            </div>
            <span id="pipelineProgressText">0/8 steps</span>
        </div>
        <div class="pipeline-steps-list" id="pipelineStepsList"></div>
    </div>
    
    <!-- Step 3: Generate UCL -->
    <div class="pipeline-section" id="pipelineGeneration">
        <h3>Step 3: Generate UCL</h3>
        <textarea id="pipelinePromptInput" class="prompt-textarea" 
                  placeholder="Enter prompt to convert to UCL..."></textarea>
        <button class="btn btn-primary" id="pipelineGenerateBtn" disabled>
            Generate UCL
        </button>
        <div class="pipeline-output" id="pipelineOutput"></div>
        <div class="pipeline-actions">
            <button class="btn btn-secondary" id="pipelineCopyBtn">Copy</button>
            <button class="btn btn-secondary" id="pipelineDownloadBtn">Download</button>
        </div>
    </div>
</div>
```

2. Add script tags before `</body>`:
```html
<script src="pipeline/pipeline-types.js"></script>
<script src="pipeline/source-parser.js"></script>
<script src="pipeline/warmup-templates.js"></script>
<script src="pipeline/warmup-engine.js"></script>
<script src="pipeline/pipeline-orchestrator.js"></script>
<script src="pipeline/pipeline-ui.js"></script>
```

**JS changes to `app.js` (1 line):**

In DOMContentLoaded, after `await initToolsUI();` (line 37), add:
```javascript
await initPipelineUI();  // Initialize Pipeline tab
```

**pipeline-ui.js must implement:**

```
initPipelineUI() function:
├── 1. Inject third sidebar tab
│   Find .sidebar-tabs (created by tools.js)
│   Append: <button class="sidebar-tab" data-tab="pipeline">
│             <span class="tab-icon">🔄</span>
│             <span class="tab-label">Pipeline</span>
│           </button>
│   Add click handler → switchToPipeline()
│
├── 2. switchToPipeline()
│   - Hide editor-container, show pipelineContainer
│   - Update sidebar title to "Pipeline"
│   - Hide syntax search, syntax categories
│   - Show pipeline sidebar panel (status, tools, warm-up steps)
│   - Mark pipeline tab active, deactivate others
│
├── 3. switchFromPipeline() (called when switching to syntax/tools tabs)
│   - Show editor-container, hide pipelineContainer
│   - Restore sidebar title, syntax panel
│
├── 4. File input handling
│   - "Add Files" button triggers hidden file input
│   - On file selection: display file names + sizes in file list
│   - Enable "Scan for Tools" button
│
├── 5. Scan button → PipelineOrchestrator.discover()
│   - Show spinner during scan
│   - Display results: grouped by file, tool signatures
│   - Enable "Start Warm-Up" button
│
├── 6. Warm-up button → PipelineOrchestrator.warmup()
│   - Show progress bar, update on each step
│   - Display step checklist (✅/⏳/○)
│   - Enable "Generate UCL" button when complete
│
├── 7. Generate button → PipelineOrchestrator.generate()
│   - Show spinner during generation
│   - Display UCL output in output area
│   - Enable Copy/Download buttons
│
├── 8. Pipeline sidebar panel
│   - Phase indicator (idle/discovering/warmup/ready/generating/complete)
│   - Tool count
│   - Warm-up progress (X/8)
│   - Token usage estimate
│   - Re-scan and Re-run Warm-up buttons
│
└── 9. Override switchSidebarTab() behavior
    When pipeline tab is clicked: switchToPipeline()
    When other tabs are clicked: switchFromPipeline() + original behavior
```

**CSS additions to styles.css:**
- `.pipeline-container` — full-width layout matching editor-container dimensions
- `.pipeline-section` — card-style sections with headers (matching existing pane styling)
- `.pipeline-progress` — progress bar styling (matching download-progress pattern in HTML)
- `.pipeline-steps-list` — step checklist with status indicators
- `.pipeline-file-list` — file list with names and sizes
- `.pipeline-output` — monospace output area (matching ucl-output styling)

**Integration test criteria:**
1. Pipeline tab appears in sidebar alongside UCL Syntax and Tools tabs
2. Clicking Pipeline tab: editor disappears, pipeline UI appears
3. Clicking back to UCL Syntax: editor returns, pipeline hides
4. File picker works: select directory, files listed
5. Scan button: tools discovered and displayed
6. Warm-up button: progress bar updates in real-time
7. Generate button: UCL output appears
8. Copy/Download work on pipeline output
9. Existing Optimize flow still works when on UCL Syntax tab (no regression)

---

### TASK 8: Add Python and JavaScript Parsers
**File**: `pipeline/source-parser.js` (extend existing)  
**Effort**: 4-6 hours  
**Depends on**: Task 2 (TypeScript parser as template)  
**Can run in parallel with**: Task 7  

**What to build:**

For Python (`_parsePython`):
- Regex patterns from plan Appendix B: functionDef, classDef, decorator, docstring, paramTypeHint
- Extract: function name, parameters with type hints, return type, docstring
- Handle: `def name(param: type = default) -> return_type:`
- Handle: `@decorator` above functions (mark as decorated)
- Handle: triple-quoted docstrings

For JavaScript (`_parseJavaScript`):
- Regex patterns: functionDecl, arrowFunc, moduleExport, jsdoc, jsdocParam
- Extract: function name, parameters (no type info in JS), JSDoc descriptions
- Handle: `function name(params)`, `const name = (params) =>`, `module.exports = {name}`
- Handle: `@param {type} name - description` in JSDoc

**Test criteria:**
1. Python: `def search(query: str, limit: int = 10) -> list:` → DiscoveredTool with typed params
2. Python: docstring extraction from `"""..."""` blocks
3. JS: `export const handler = (req, res) => {}` → name='handler', params from destructuring
4. JS: JSDoc param extraction
5. Mixed directory with .ts + .py + .js files → all tools discovered, deduplicated

---

### TASK 9: Token Budget Enforcement
**Files**: `pipeline/warmup-engine.js` (extend), `pipeline/pipeline-orchestrator.js` (extend)  
**Effort**: 3-4 hours  
**Depends on**: Task 4 (warmup-engine must exist)  
**Can run in parallel with**: Tasks 7, 8  

**What to build:**

1. **Adaptive step count**: Based on model's context window:
   ```
   if contextWindow >= 16K: run all 8 steps (default)
   if contextWindow >= 8K:  run 6 steps (skip examples: steps 5-6)
   if contextWindow >= 4K:  run 4 steps (compressed: syntax+tools only)
   ```

2. **Token monitoring during warm-up**:
   - After each step, estimate total tokens
   - If > 40% of context window: log warning
   - If > 50% of context window: stop warm-up, proceed with what we have
   - Display token count in UI

3. **Message compression** (for small models):
   - After step 4, ask LLM to summarize what it learned (1 message)
   - Replace steps 1-4 content with summary
   - Continue with steps 5-8 on compressed history
   - Saves ~40% of accumulated tokens

4. **UI token display**:
   - Show token count in pipeline sidebar panel
   - Color-code: green (<30%), yellow (30-50%), red (>50%)
   - Warn user before generation if tokens > 60% of context

**Test criteria:**
1. With 4K context model: only 4 steps run, warm-up still completes
2. Token estimate stays within bounds during 8-step warm-up
3. Compression reduces token count by measurable amount
4. UI shows token usage updating in real-time

---

## 4. Files Modified Summary

| File | Change | Risk | Lines Changed |
|------|--------|------|---------------|
| `index.html` | Add pipeline container div + script tags | Low (additive) | ~40 lines added |
| `app.js` | Add `await initPipelineUI()` after line 37 | Minimal (1 line) | 1 line added |
| `styles.css` | Add pipeline-specific styles | Low (additive) | ~100 lines added |
| `pipeline/pipeline-types.js` | NEW — shared type definitions | None | ~60 lines |
| `pipeline/source-parser.js` | NEW — regex tool discovery | None | ~250 lines |
| `pipeline/warmup-templates.js` | NEW — 8 message builders | None | ~300 lines |
| `pipeline/warmup-engine.js` | NEW — conversation manager | None | ~200 lines |
| `pipeline/pipeline-orchestrator.js` | NEW — state machine | None | ~150 lines |
| `pipeline/pipeline-ui.js` | NEW — tab UI + interactions | None | ~400 lines |

**Total new code**: ~1,360 lines across 6 new files  
**Total changes to existing**: ~141 lines across 3 files (all additive except 1 line in app.js)

---

## 5. Testing Milestones

| Milestone | After Task | What to Test | Pass Criteria |
|-----------|-----------|--------------|---------------|
| **M1: Types compile** | Task 1 | Load pipeline-types.js in console | No errors, types documented |
| **M2: Parser works** | Task 2 | Parse real .ts files | ≥1 tool extracted per file with correct params |
| **M3: Templates render** | Task 3 | Call all 8 message builders | Non-empty strings with correct UCL syntax |
| **M4: Warm-up runs** | Task 4 | executeAll() against Lemonade | 16 messages, all steps confirmed |
| **M5: Full pipeline** | Task 5 | prepare() + generate() | UCL output with valid syntax |
| **M6: Smoke test** | Task 6 | Real files, real model, real prompt | End-to-end success, tokens < 16K |
| **M7: UI working** | Task 7 | Click through pipeline in browser | All buttons work, progress shown, output displayed |
| **M8: Multi-language** | Task 8 | Parse .ts + .py + .js files | All languages produce valid DiscoveredTools |
| **M9: Token safe** | Task 9 | Run with small context model | Adaptive steps, no overflow |

**Regression test after Task 7**: Verify the existing "Optimize with UCL" button still works identically. The pipeline must not break the existing flow.

---

## 6. Risk Mitigation (Concrete Actions)

### Risk 1: Context Window Overflow (HIGH probability for small models)
**Mitigation built into execution:**
- Task 4 includes token estimation from day one (not deferred)
- Task 9 adds adaptive step count based on model context
- If testing (M4) shows overflow with default 8 steps, immediately implement compression before proceeding

### Risk 2: Regex Parsing Misses Complex Patterns (MEDIUM)
**Mitigation:**
- Task 2 test uses real project files (lemonade-api.js, tools.js) — if regex misses exports, patterns are refined immediately
- Task 8 (Phase 4) adds AST parsing as fallback — not blocking MVP
- Document supported patterns in source-parser.js JSDoc

### Risk 3: Warm-Up Doesn't Improve Output Quality (MEDIUM probability, HIGH impact)
**Mitigation:**
- M6 smoke test compares pipeline output vs. existing Optimize output
- If pipeline output is NOT meaningfully better, investigate:
  a. Are warm-up messages too generic? → Make them more specific to discovered tools
  b. Is the LLM too small to "learn"? → Test with larger model
  c. Is 8 steps too many (LLM forgets early messages)? → Try compression
- **Nuclear option**: Fall back to system-prompt approach (all warm-up content in one system message)

### Risk 4: Browser File Access Limitations (HIGH probability)
**Mitigation:**
- `<input type="file" webkitdirectory>` works in Chrome, Edge, Firefox
- Tested pattern: no server needed, reads file contents via `file.text()`
- Fallback: individual file input (non-directory) if webkitdirectory fails on some browsers

### Risk 5: Pipeline Tab Conflicts with Existing Tab System (LOW)
**Mitigation:**
- pipeline-ui.js reads the existing `.sidebar-tabs` element and appends to it
- Uses the same CSS classes (`sidebar-tab`) as tools.js
- switchSidebarTab() in tools.js already handles show/hide — pipeline-ui.js extends this pattern
- Test: verify switching between all 3 tabs doesn't cause visual glitches

---

## 7. Phase 1 Sprint Plan (2-Week Target)

### Week 1: Backend (Tasks 1-5)

| Day | Task | Deliverable |
|-----|------|-------------|
| Mon AM | Task 1 | pipeline/ directory + pipeline-types.js |
| Mon PM | Task 2 (start) | source-parser.js: TypeScript exported function regex |
| Tue | Task 2 (complete) | Full TS parser: functions, arrows, interfaces, JSDoc |
| Wed AM | Task 2 (test) | Parse real .ts files, verify output, refine regex |
| Wed PM | Task 3 (start) | warmup-templates.js: messages 1-2 (syntax) |
| Thu AM | Task 3 (complete) | Messages 3-8 (tools, examples, rules) |
| Thu PM | Task 4 (start) | warmup-engine.js: init, buildWarmupSequence, _callLLM |
| Fri AM | Task 4 (complete) | executeStep, executeAll, token estimation |
| Fri PM | Task 5 | pipeline-orchestrator.js: full state machine |

**Week 1 exit criteria**: M5 milestone passed — full pipeline runnable from console.

### Week 2: Integration + UI (Tasks 6-7)

| Day | Task | Deliverable |
|-----|------|-------------|
| Mon AM | Task 6 | End-to-end smoke test with real Lemonade model |
| Mon PM | Task 6 (fix) | Fix any issues found in smoke test |
| Tue | Task 7 (start) | HTML changes + pipeline-ui.js: tab injection, file picker |
| Wed | Task 7 (mid) | Scan button, discovery results display |
| Thu | Task 7 (mid) | Warm-up progress display, step checklist |
| Fri AM | Task 7 (complete) | Generation UI, output display, copy/download |
| Fri PM | Task 7 (test) | Full UI integration test, regression test on Optimize |

**Week 2 exit criteria**: M7 milestone passed — pipeline tab fully functional in browser.

### Buffer: Tasks 8-9 (Week 3 or as needed)

Python/JS parsers and token enforcement are Phase 1 extensions but not blocking. If Week 1-2 hits delays, these move to Week 3.

---

## 8. Open Questions Requiring Decision Before Implementation

These were raised in the plan and need answers before a developer starts Task 1:

| # | Question | Recommendation | Decision |
|---|----------|---------------|----------|
| 1 | Hardcoded vs. customizable warm-up templates? | Ship hardcoded (Phase 1), customizable (Phase 3) | **Pending** |
| 2 | Should warm-up conversation be visible to user? | Show all messages in collapsible log | **Pending** |
| 3 | Model change after warm-up: invalidate or warn? | Warn user, don't auto-invalidate | **Pending** |
| 4 | Pipeline gets own button or separate tab? | Separate tab (as planned) | **Pending** |
| 5 | Max token budget for warm-up? | 50% of model context window | **Pending** |

**Recommendation**: Accept all recommendations from the plan and proceed. These can be adjusted during implementation without architectural changes.

---

## 9. Definition of Done (Phase 1)

Phase 1 is complete when:

1. [ ] `source-parser.js` extracts tool signatures from .ts files with ≥80% accuracy (tested against 5+ real files)
2. [ ] `warmup-engine.js` completes 8-step warm-up against a local Lemonade model without errors
3. [ ] `pipeline-orchestrator.js` runs the full discover → warmup → generate flow
4. [ ] `pipeline-ui.js` provides a usable Pipeline tab in the Builder UI
5. [ ] Existing "Optimize with UCL" button works identically (no regression)
6. [ ] Token usage stays within context window for models ≥16K context
7. [ ] All new code has JSDoc comments matching the API design in the plan
8. [ ] Smoke test (M6) passes with at least one local model

---

*End of Execution Strategy*
