/**
 * OptimizerUI - UI Rendering and Event Handling for the Optimizer Tab
 *
 * Mirrors the Pipeline panel pattern from tools.js:
 * - Module-level state object (OptimizerState)
 * - Panel HTML builder functions
 * - Event listener attachment
 * - Dynamic import of orchestrator
 *
 * QUALITY FIXES:
 * - dispose() for resource cleanup (HIGH: long-lived resources)
 * - Template editing section (CRITICAL #2)
 * - User feedback mechanism for update phase
 * - Proper state management with cleanup on tab switch
 *
 * Usage:
 *   // Called from tools.js initToolsUI()
 *   await initOptimizerUI();
 *
 * @module optimizer-ui
 */

import { FileLoader } from './file-loader.js';
import {
    DashboardState,
    updateDashboardIteration,
    updateDashboardComplete,
    updateDashboardStatus,
    showRunningState as dashShowRunningState,
    destroyOptimizerDashboard,
} from './optimizer-dashboard.js';

/**
 * Escape HTML special characters to prevent XSS in innerHTML template literals.
 * @param {*} str - Value to escape (converted to string)
 * @returns {string} Safe HTML string
 */
function escapeHtml(str) {
    if (!str) return '';
    return String(str)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
}

// ── Module State (mirrors ToolState / PipelineState pattern) ──────────────

const OptimizerState = {
    /** @type {import('./orchestrator.js').ModelOptimizerOrchestrator|null} */
    orchestrator: null,
    /** @type {import('./storage-manager.js').StorageManager|null} */
    storageManager: null,
    /** @type {string} */
    currentModelId: '',
    /** @type {boolean} */
    isRunning: false,
    /** @type {boolean} */
    initialized: false,
    /** @type {Array<{name: string, content: string}>} */
    loadedFiles: [],
    /** @type {Object[]} */
    versions: [],
    /** @type {Object} Iteration configuration for optimization loop */
    iterationConfig: {
        maxIterations: 5,
        qualityThreshold: 0.85,
        convergenceDelta: 0.02,
        autoIterate: false,
    },
};

// ── Initialization ────────────────────────────────────────────────────────

/**
 * Initialize the Optimizer UI.
 * Called from tools.js initToolsUI() after the optimizer panel is created.
 */
async function initOptimizerUI() {
    if (OptimizerState.initialized) return;

    // Set default model from AppState (if available)
    OptimizerState.currentModelId = (typeof AppState !== 'undefined' && AppState.selectedModel)
        ? AppState.selectedModel
        : 'qwen3-4b-FLM';

    renderConvertPanel();
    renderContextPanel();
    renderOptimizerPanel();
    attachConvertEventListeners();
    attachContextEventListeners();
    attachOptimizerEventListeners();

    OptimizerState.initialized = true;
    console.log('✓ Optimizer UI initialized (Convert + Context + Optimizer)');

    // Optimizer/Convert: mirror main select. Context: always Lemonade-server models.
    setTimeout(() => syncModelDropdowns(), 1000);
    setTimeout(() => syncModelDropdowns(), 3000);
    setTimeout(() => populateContextLemonadeModels(), 500);
    setTimeout(() => populateContextLemonadeModels(), 2500);
}

/**
 * Copy model options from the main modelSelect to optimizer and convert dropdowns.
 * Context select is filled separately from Lemonade server only.
 */
function syncModelDropdowns() {
    const mainSelect = document.getElementById('modelSelect');
    const selects = [
        document.getElementById('optimizerModelSelect'),
        document.getElementById('convertModelSelect'),
    ];

    if (!mainSelect || !mainSelect.options.length) return;

    selects.forEach(select => {
        if (!select) return;
        // Save current value
        const currentVal = select.value;
        // Clear and rebuild
        select.innerHTML = '<option value="">Select a model...</option>';
        for (const option of mainSelect.options) {
            if (option.value === '' || option.disabled) continue;
            const opt = document.createElement('option');
            opt.value = option.value;
            opt.textContent = option.textContent;
            select.appendChild(opt);
        }
        // Restore value or set default
        if (currentVal && select.querySelector(`option[value="${currentVal}"]`)) {
            select.value = currentVal;
        } else if (OptimizerState.currentModelId) {
            select.value = OptimizerState.currentModelId;
        }
    });

    // Context always uses Lemonade models (not OpenRouter list)
    populateContextLemonadeModels();

    updateConvertConvoStatus();
    updateOptimizerPipelineStatus();
}

/**
 * Fill #contextModelSelect from Lemonade server /v1/models (LLM recipes only).
 * Independent of main AppState.provider so Context always hits local Lemonade.
 */
async function populateContextLemonadeModels() {
    const select = document.getElementById('contextModelSelect');
    if (!select) return;

    const prev = select.value || OptimizerState.currentModelId || '';
    const endpoint = getLemonadeEndpoint();
    const statusEl = document.getElementById('contextLemonadeStatus');

    select.innerHTML = '<option value="">Loading Lemonade models…</option>';
    if (statusEl) {
        statusEl.textContent = `Fetching from ${endpoint}…`;
        statusEl.className = 'context-lemonade-status loading';
    }

    const LLM_RECIPES = ['llamacpp', 'flm', 'ryzenai-llm'];
    const NON_LLM_LABELS = ['image', 'edit', 'transcription', 'realtime-transcription', 'embeddings', 'tts', 'audio'];

    try {
        let models = [];

        // Prefer global LemonadeClient if available
        if (typeof lemonadeClient !== 'undefined' && lemonadeClient?.getAvailableModels) {
            try {
                lemonadeClient.endpoint = endpoint.replace(/\/$/, '');
            } catch (_) { /* ignore */ }
            const all = await lemonadeClient.getAvailableModels(false);
            models = (all || []).filter(m => {
                if (m.recipe && !LLM_RECIPES.includes(m.recipe)) return false;
                if (m.labels && m.labels.some(l => NON_LLM_LABELS.includes(l))) return false;
                // If recipe missing, keep if id looks like chat model (exclude obvious non-llm)
                if (!m.recipe) {
                    const id = (m.id || '').toLowerCase();
                    if (/whisper|embed|kokoro|tts|flux|sd-|image/.test(id)) return false;
                }
                return true;
            });
        } else {
            const res = await fetch(`${endpoint.replace(/\/$/, '')}/v1/models`);
            if (!res.ok) throw new Error(`Lemonade HTTP ${res.status}`);
            const data = await res.json();
            const all = data.data || data.models || [];
            models = all.filter(m => {
                const id = (m.id || m.model_name || '').toLowerCase();
                if (/whisper|embed|kokoro|tts|flux|sd-|image|transcription/.test(id)) return false;
                if (m.recipe && !LLM_RECIPES.includes(m.recipe)) return false;
                if (m.labels && m.labels.some(l => NON_LLM_LABELS.includes(l))) return false;
                return true;
            }).map(m => ({ id: m.id || m.model_name, ...m }));
        }

        select.innerHTML = '<option value="">Select Lemonade model…</option>';

        if (!models.length) {
            const opt = document.createElement('option');
            opt.disabled = true;
            opt.textContent = 'No LLM models on Lemonade — install via lemonade CLI';
            select.appendChild(opt);
            if (statusEl) {
                statusEl.textContent = `No chat models at ${endpoint}`;
                statusEl.className = 'context-lemonade-status error';
            }
            return;
        }

        // Prefer loaded models first if we have that info
        models.sort((a, b) => String(a.id).localeCompare(String(b.id), undefined, { numeric: true }));

        const group = document.createElement('optgroup');
        group.label = `Lemonade · ${endpoint}`;
        for (const model of models) {
            const option = document.createElement('option');
            option.value = model.id;
            let label = model.id;
            const bits = [];
            if (model.size) bits.push(`${model.size}GB`);
            if (model.max_context_window) bits.push(`${(model.max_context_window / 1024).toFixed(0)}K ctx`);
            if (model.recipe) bits.push(model.recipe);
            if (bits.length) label += ` (${bits.join(' · ')})`;
            option.textContent = label;
            group.appendChild(option);
        }
        select.appendChild(group);

        if (prev && select.querySelector(`option[value="${prev}"]`)) {
            select.value = prev;
        } else if (OptimizerState.currentModelId && select.querySelector(`option[value="${OptimizerState.currentModelId}"]`)) {
            select.value = OptimizerState.currentModelId;
        }

        if (statusEl) {
            statusEl.textContent = `${models.length} Lemonade models · ${endpoint}`;
            statusEl.className = 'context-lemonade-status ready';
        }
    } catch (err) {
        console.warn('[Context] Lemonade model list failed:', err);
        select.innerHTML = '<option value="">Lemonade unreachable</option>';
        const opt = document.createElement('option');
        opt.disabled = true;
        opt.textContent = err.message || 'Failed to fetch models';
        select.appendChild(opt);
        if (statusEl) {
            statusEl.textContent = `Cannot reach Lemonade at ${endpoint}`;
            statusEl.className = 'context-lemonade-status error';
        }
    }
}

function getLemonadeEndpoint() {
    if (typeof AppState !== 'undefined' && AppState.lemonadeEndpoint) {
        return AppState.lemonadeEndpoint.replace(/\/$/, '');
    }
    return 'http://localhost:13305';
}

/**
 * Keep model id in sync across Optimizer / Convert / Context selects.
 */
function syncAllModelSelects(modelId) {
    if (!modelId) return;
    OptimizerState.currentModelId = modelId;
    ['optimizerModelSelect', 'contextModelSelect', 'convertModelSelect'].forEach(id => {
        const el = document.getElementById(id);
        if (el && el.querySelector(`option[value="${modelId}"]`)) {
            el.value = modelId;
        }
    });
}

/**
 * Sync Context files into OptimizerState.loadedFiles for the pipeline.
 */
function syncLoadedFilesFromContext() {
    if (ContextState.files.length > 0) {
        OptimizerState.loadedFiles = ContextState.files.map(f => ({
            id: f.id,
            name: f.name,
            content: f.content,
            size: f.size,
        }));
    }
}

function updateConvertConvoStatus() {
    const el = document.getElementById('convertConvoStatus');
    if (!el) return;
    const nFiles = ContextState.files.length || OptimizerState.loadedFiles.length;
    const nResp = ContextState.responses.length;
    const nVer = OptimizerState.versions.length;
    if (nResp > 0) {
        el.textContent = `loadedConvo ready: ${nResp} model responses · ${nFiles} files · ${nVer} versions`;
        el.className = 'convert-convo-status ready';
    } else if (nFiles > 0) {
        el.textContent = `${nFiles} context file(s) staged — open Context tab and Send All in Sequence`;
        el.className = 'convert-convo-status warn';
    } else {
        el.textContent = 'No loadedConvo yet — load Context first (sequential file send)';
        el.className = 'convert-convo-status';
    }
}

function updateOptimizerPipelineStatus() {
    const el = document.getElementById('optimizerPipelineStatus');
    if (!el) return;
    const hasModel = !!OptimizerState.currentModelId;
    const hasCtx = (ContextState.responses.length > 0) || (OptimizerState.loadedFiles.length > 0);
    const hasVer = OptimizerState.versions.length > 0;
    const parts = [];
    parts.push(hasModel ? `Model: ${OptimizerState.currentModelId}` : 'No model');
    parts.push(hasCtx ? `Context: ${ContextState.responses.length || OptimizerState.loadedFiles.length} loaded` : 'Context: empty');
    parts.push(hasVer ? `Versions: ${OptimizerState.versions.length}` : 'Versions: none (Convert first)');
    el.textContent = parts.join(' · ');
}

// ── Panel Rendering ───────────────────────────────────────────────────────

/**
 * Build the full optimizer panel HTML.
 * @returns {string}
 */
function buildOptimizerPanelHTML() {
    return `
        <!-- Optimizer Header -->
        <div class="optimizer-header">
            <h3 class="optimizer-title">🎯 Optimizer</h3>
            <div class="optimizer-status" id="optimizerStatus">
                <span class="status-indicator" id="optimizerStatusIndicator">&#x26AA;</span>
                <span class="status-text" id="optimizerStatusText">Ready</span>
            </div>
        </div>

        <!-- Model Selector -->
        <div class="optimizer-section">
            <label class="optimizer-label">Model</label>
            <select id="optimizerModelSelect" class="optimizer-select">
                <option value="">Select a model...</option>
            </select>
            <div class="optimizer-pipeline-status" id="optimizerPipelineStatus">
                Context → Convert → Optimize. Load context &amp; convert a prompt first.
            </div>
        </div>

        <!-- Scoring Legend -->
        <div class="optimizer-section optimizer-scoring-legend">
            <label class="optimizer-label">How Scoring Works</label>
            <div class="score-legend">
                <div class="score-legend-item"><span class="score-w">25%</span> <strong>Syntax</strong> — UCL structure ([[CRITICAL:]], ^^CONDITION^^, closers)</div>
                <div class="score-legend-item"><span class="score-w">40%</span> <strong>Test pass</strong> — Weighted test cases against the model</div>
                <div class="score-legend-item"><span class="score-w">20%</span> <strong>Reasoning</strong> — Depth of extracted thinking log</div>
                <div class="score-legend-item"><span class="score-w">15%</span> <strong>Conciseness</strong> — Penalizes prompt bloat</div>
                <div class="score-legend-note">Composite 0–1; loop stops at threshold or when delta &lt; convergence.</div>
            </div>
        </div>

        <!-- Iteration Config -->
        <div class="optimizer-iter-config">
            <label class="optimizer-label">Iteration Settings</label>
            <div class="optimizer-iter-row">
                <label for="optMaxIter">Max Iterations</label>
                <input type="number" id="optMaxIter" min="1" max="50" value="5">
            </div>
            <div class="optimizer-iter-row">
                <label for="optQualityThresh">Quality Threshold</label>
                <input type="number" id="optQualityThresh" min="0" max="1" step="0.05" value="0.85">
            </div>
            <div class="optimizer-iter-row">
                <label for="optConvDelta">Convergence Delta</label>
                <input type="number" id="optConvDelta" min="0.001" max="0.5" step="0.005" value="0.02">
            </div>
            <div class="optimizer-iter-row">
                <label for="optAutoIter">Auto-Iterate</label>
                <input type="checkbox" id="optAutoIter">
            </div>
        </div>

        <!-- Start / Stop Buttons -->
        <div class="optimizer-actions">
            <button class="optimizer-btn optimizer-btn-start" id="optimizerStartLoopBtn" disabled aria-label="Start optimization loop">
                <span class="btn-icon">&#x26A1;</span> Start Loop
            </button>
            <button class="optimizer-btn optimizer-btn-stop" id="optimizerStopLoopBtn" style="display:none;" aria-label="Stop optimization loop">
                <span class="btn-icon">&#x23F9;</span> Stop
            </button>
        </div>

        <!-- Progress Bar -->
        <div class="optimizer-progress" id="optimizerProgress" style="display:none;">
            <div class="optimizer-progress-bar-container">
                <div class="optimizer-progress-bar" id="optimizerProgressBar" style="width: 0%"></div>
            </div>
            <div class="optimizer-progress-text" id="optimizerProgressText">0%</div>
        </div>

        <!-- Phase Status -->
        <div class="optimizer-phases" id="optimizerPhases" style="display:none;">
            <div class="optimizer-phase-item" id="optimizerPhaseLoading">
                <div class="phase-icon">&#x1F4C2;</div>
                <div class="phase-content">
                    <div class="phase-title">Loading Context</div>
                    <div class="phase-status" id="optimizerPhaseLoadingStatus">Pending</div>
                </div>
            </div>
            <div class="optimizer-phase-item" id="optimizerPhaseConverting">
                <div class="phase-icon">&#x1F4DD;</div>
                <div class="phase-content">
                    <div class="phase-title">Converting</div>
                    <div class="phase-status" id="optimizerPhaseConvertingStatus">Pending</div>
                </div>
            </div>
            <div class="optimizer-phase-item" id="optimizerPhaseTesting">
                <div class="phase-icon">&#x1F9EA;</div>
                <div class="phase-content">
                    <div class="phase-title">Testing</div>
                    <div class="phase-status" id="optimizerPhaseTestingStatus">Pending</div>
                </div>
            </div>
            <div class="optimizer-phase-item" id="optimizerPhaseExtracting">
                <div class="phase-icon">&#x1F4A1;</div>
                <div class="phase-content">
                    <div class="phase-title">Extracting</div>
                    <div class="phase-status" id="optimizerPhaseExtractingStatus">Pending</div>
                </div>
            </div>
            <div class="optimizer-phase-item" id="optimizerPhaseUpdating">
                <div class="phase-icon">&#x1F504;</div>
                <div class="phase-content">
                    <div class="phase-title">Updating</div>
                    <div class="phase-status" id="optimizerPhaseUpdatingStatus">Pending</div>
                </div>
            </div>
        </div>

        <!-- Feedback Input -->
        <div class="optimizer-section" id="optimizerFeedbackSection" style="display:none;">
            <label class="optimizer-label">Feedback</label>
            <textarea id="optimizerFeedbackInput"
                      class="optimizer-textarea"
                      rows="3"
                      placeholder="Describe what to improve..."></textarea>
            <button class="optimizer-btn optimizer-btn-primary" id="optimizerApplyFeedbackBtn">
                Apply Feedback
            </button>
        </div>

        <!-- Version History -->
        <div class="optimizer-section">
            <label class="optimizer-label">Version History</label>
            <div class="optimizer-version-list" id="optimizerVersionList">
                <div class="optimizer-empty">No versions yet</div>
            </div>
        </div>

        <!-- Thinking Log -->
        <div class="optimizer-section" id="optimizerThinkingSection" style="display:none;">
            <div class="optimizer-thinking-header">
                <label class="optimizer-label" id="optimizerThinkingLabel">Thinking Log</label>
                <button class="optimizer-btn optimizer-btn-sm" id="optimizerCloseThinkingBtn">&times;</button>
            </div>
            <div class="optimizer-thinking-log" id="optimizerThinkingLog"></div>
        </div>

        <!-- Error Display -->
        <div class="optimizer-error" id="optimizerError" style="display:none;">
            <div class="error-icon">&#x26A0;&#xFE0F;</div>
            <div class="error-message" id="optimizerErrorMessage"></div>
            <button class="optimizer-btn optimizer-btn-sm" id="optimizerDismissErrorBtn">Dismiss</button>
        </div>
    `;
}

/**
 * Render the optimizer panel into the DOM.
 */
function renderOptimizerPanel() {
    const sidebar = document.getElementById('sidebar');
    if (!sidebar) return;

    let panel = sidebar.querySelector('.sidebar-optimizer-panel');
    if (!panel) {
        panel = document.createElement('div');
        panel.className = 'sidebar-optimizer-panel';
        panel.style.display = 'none';

        // Insert after pipeline panel
        const pipelinePanel = sidebar.querySelector('.sidebar-pipeline-panel');
        if (pipelinePanel) {
            pipelinePanel.after(panel);
        } else {
            sidebar.appendChild(panel);
        }
    }

    panel.innerHTML = buildOptimizerPanelHTML();
}

// ── Convert Panel ─────────────────────────────────────────────────────────

function renderConvertPanel() {
    const sidebar = document.getElementById('sidebar');
    if (!sidebar) return;

    let panel = sidebar.querySelector('.sidebar-convert-panel');
    if (!panel) {
        panel = document.createElement('div');
        panel.className = 'sidebar-convert-panel';
        panel.style.display = 'none';
        const optimizerPanel = sidebar.querySelector('.sidebar-optimizer-panel');
        if (optimizerPanel) {
            optimizerPanel.before(panel);
        } else {
            sidebar.appendChild(panel);
        }
    }

    panel.innerHTML = `
        <div class="convert-panel-header">
            <h3 class="convert-title">🔄 Convert to UCL</h3>
            <p class="convert-subtitle">Uses loadedConvo from Context + base template</p>
        </div>

        <div class="convert-section">
            <label class="convert-label">Model</label>
            <select id="convertModelSelect" class="convert-select optimizer-select">
                <option value="">Select a model...</option>
            </select>
            <div class="convert-convo-status" id="convertConvoStatus">No loadedConvo yet — load Context first</div>
        </div>

        <div class="convert-section">
            <label class="convert-label">Prompt to Convert</label>
            <textarea id="convertUserPrompt"
                      class="convert-textarea"
                      rows="6"
                      placeholder="Enter a natural language prompt to convert to UCL format..."></textarea>
        </div>

        <div class="convert-section">
            <label class="convert-label">Template</label>
            <div class="convert-template-tabs">
                <button class="convert-tmpl-tab active" data-tmpl="convert">Convert (base_prompt)</button>
                <button class="convert-tmpl-tab" data-tmpl="update">Update (base_prompt_update)</button>
            </div>
            <textarea id="convertTemplateEditor"
                      class="convert-textarea convert-textarea-sm"
                      rows="4"
                      placeholder="Template content loads after model is selected..."></textarea>
            <div class="convert-template-actions">
                <button class="convert-btn convert-btn-sm" id="convertResetTemplateBtn">Reset</button>
                <button class="convert-btn convert-btn-sm" id="convertSaveTemplateBtn">Save</button>
            </div>
        </div>

        <div class="convert-section" id="convertFeedbackSection" style="display:none;">
            <label class="convert-label">Update Feedback</label>
            <textarea id="convertFeedbackInput"
                      class="convert-textarea"
                      rows="3"
                      placeholder="What should change in the next version?"></textarea>
        </div>

        <div class="convert-actions">
            <button class="convert-btn convert-btn-primary" id="convertBtn" title="loadedConvo + base_prompt + prompt">
                <span class="btn-icon">▶</span> Convert
            </button>
            <button class="convert-btn" id="convertTestBtn" title="Test latest converted_prompt version">
                <span class="btn-icon">🧪</span> Test
            </button>
            <button class="convert-btn" id="convertUpdateBtn" title="Update with history + thinking">
                <span class="btn-icon">🔄</span> Update
            </button>
        </div>

        <div class="convert-progress" id="convertProgress" style="display:none;">
            <div class="convert-progress-bar-container">
                <div class="convert-progress-bar" id="convertProgressBar" style="width: 0%"></div>
            </div>
            <div class="convert-progress-text" id="convertProgressText">Converting...</div>
        </div>

        <div class="convert-output" id="convertOutput" style="display:none;">
            <label class="convert-label" id="convertOutputLabel">Output</label>
            <div class="convert-output-content" id="convertOutputContent"></div>
            <div class="convert-output-actions">
                <button class="convert-btn convert-btn-sm" id="convertCopyBtn">📋 Copy</button>
                <button class="convert-btn convert-btn-sm" id="convertToEditorBtn">→ Editor</button>
            </div>
        </div>

        <div class="convert-thinking" id="convertThinkingSection" style="display:none;">
            <label class="convert-label" id="convertThinkingLabel">Thinking / Reasoning</label>
            <div class="convert-output-content" id="convertThinkingContent"></div>
        </div>

        <div class="convert-error" id="convertError" style="display:none;">
            <div class="error-icon">⚠️</div>
            <div class="error-message" id="convertErrorMessage"></div>
            <button class="convert-btn convert-btn-sm" id="convertDismissErrorBtn">Dismiss</button>
        </div>
    `;
}

// ── Context Panel ─────────────────────────────────────────────────────────

const ContextState = {
    /** @type {Array<{id: string, name: string, content: string, size: number, order: number, status: string}>} */
    files: [],
    /** @type {Array<{fileId?: string, fileName: string, response: string, timestamp: string, tokens?: number, error?: boolean}>} */
    responses: [],
    isSending: false,
    /** Index currently being sent (for UI highlight) */
    activeSendIndex: -1,
    expandedPreviewId: null,
    /** Final user prompt — sent ONLY after all context docs complete */
    finalPrompt: '',
    finalPromptFileName: '',
    /** pending | ready | sending | done | error | skipped */
    finalPromptStatus: 'pending',
    finalPromptResponse: '',
    finalPromptTokens: 0,
};

function _newContextFileId() {
    return `ctx_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
}

function renumberContextFiles() {
    ContextState.files.forEach((f, i) => {
        f.order = i + 1;
    });
}

function moveContextFile(fromIdx, toIdx) {
    if (ContextState.isSending) return;
    if (fromIdx < 0 || toIdx < 0 || fromIdx >= ContextState.files.length || toIdx >= ContextState.files.length) return;
    const [moved] = ContextState.files.splice(fromIdx, 1);
    ContextState.files.splice(toIdx, 0, moved);
    renumberContextFiles();
    syncLoadedFilesFromContext();
    refreshContextFileList();
    const view = document.querySelector('.context-conversation');
    if (view) window.renderContextConversation?.(view);
}

function getContextFileStatus(file, index) {
    if (ContextState.isSending && ContextState.activeSendIndex === index) return 'sending';
    const resp = ContextState.responses.find(r =>
        (r.fileId && r.fileId === file.id) || r.fileName === file.name
    );
    if (resp?.error) return 'error';
    if (resp) return 'done';
    if (file.status === 'sending') return 'sending';
    if (file.status === 'error') return 'error';
    if (file.status === 'done') return 'done';
    return 'pending';
}

function contextStatusLabel(status) {
    switch (status) {
        case 'sending': return { icon: '⟳', text: 'Sending…', cls: 'is-sending' };
        case 'done':    return { icon: '✓', text: 'Reply saved', cls: 'is-done' };
        case 'error':   return { icon: '!', text: 'Error', cls: 'is-error' };
        default:        return { icon: '○', text: 'Queued', cls: 'is-pending' };
    }
}

function fileTypeIcon(name) {
    const n = (name || '').toLowerCase();
    if (n.endsWith('.md')) return '📝';
    if (n.endsWith('.txt')) return '📄';
    if (n.endsWith('.json')) return '{ }';
    if (n.endsWith('.js') || n.endsWith('.ts')) return '⬡';
    if (n.endsWith('.py')) return '🐍';
    if (n.endsWith('.html') || n.endsWith('.css')) return '🌐';
    if (n.endsWith('.yaml') || n.endsWith('.yml')) return '⚙';
    return '📎';
}

function renderContextPanel() {
    const sidebar = document.getElementById('sidebar');
    if (!sidebar) return;

    let panel = sidebar.querySelector('.sidebar-context-panel');
    if (!panel) {
        panel = document.createElement('div');
        panel.className = 'sidebar-context-panel';
        panel.style.display = 'none';
        sidebar.appendChild(panel);
    }

    panel.innerHTML = `
        <div class="context-panel-header">
            <div>
                <h3 class="context-title">📁 Context Sequence</h3>
                <p class="context-subtitle">Own conversation · one file at a time · order = send order</p>
            </div>
            <span class="context-file-count" id="contextFileCount">0 files</span>
        </div>

        <div class="context-section">
            <div class="context-load-actions">
                <button class="context-btn context-btn-primary" id="contextAddFilesBtn" title="Add files to the sequence">
                    <span class="btn-icon">+</span> Add Files
                </button>
                <button class="context-btn" id="contextLoadFolderBtn" title="Load a folder (order by name)">
                    <span class="btn-icon">📁</span> Load Folder
                </button>
                <button class="context-btn context-btn-danger" id="contextClearAllBtn" style="display:none;">
                    Clear All
                </button>
            </div>
            <input type="file" id="contextFileInput" multiple style="display:none;">
            <input type="file" id="contextFolderInput" webkitdirectory multiple style="display:none;">
        </div>

        <!-- Visual sequence strip: 1 → 2 → 3 … -->
        <div class="context-sequence-strip" id="contextSequenceStrip" style="display:none;" aria-label="Send sequence">
            <div class="context-sequence-strip-label">Send order</div>
            <div class="context-sequence-chips" id="contextSequenceChips"></div>
        </div>

        <div class="context-file-list" id="contextFileList">
            <div class="context-empty" id="contextEmpty">
                <div class="context-empty-icon">📄</div>
                <div class="context-empty-text">No files in sequence</div>
                <div class="context-empty-hint">Add your context docs. They send 1 → 2 → 3… each waits for a reply before the next.</div>
            </div>
        </div>

        <!-- Final prompt: ONLY sent after all context docs -->
        <div class="context-prompt-section" id="contextPromptSection">
            <div class="context-prompt-header">
                <label class="context-label">Your Prompt (final step)</label>
                <span class="context-prompt-badge" id="contextPromptBadge">After docs</span>
            </div>
            <p class="context-prompt-hint">
                Paste a prompt or load a prompt file. It is <strong>not</strong> sent with the docs —
                only after every document has been sent and replied.
            </p>
            <textarea id="contextFinalPrompt"
                      class="context-prompt-textarea"
                      rows="5"
                      placeholder="Enter the prompt to run after context docs load…"></textarea>
            <div class="context-prompt-actions">
                <button type="button" class="context-btn" id="contextLoadPromptFileBtn">
                    📄 Load prompt file
                </button>
                <button type="button" class="context-btn" id="contextClearPromptBtn" title="Clear prompt">
                    Clear
                </button>
                <span class="context-prompt-file" id="contextPromptFileLabel"></span>
            </div>
            <input type="file" id="contextPromptFileInput" accept=".txt,.md,.prompt,.ucl,text/*" style="display:none;">
            <div class="context-prompt-status" id="contextPromptStatus">No prompt yet — optional but recommended</div>
        </div>

        <div class="context-send-section" id="contextSendSection" style="display:none;">
            <div class="context-model-select">
                <div class="context-model-header">
                    <label class="context-label">Lemonade model</label>
                    <button type="button" class="context-btn context-btn-sm" id="contextRefreshModelsBtn" title="Refresh Lemonade models">↻</button>
                </div>
                <select id="contextModelSelect" class="context-select">
                    <option value="">Select Lemonade model…</option>
                </select>
                <div class="context-lemonade-status" id="contextLemonadeStatus">Uses lemonade-server only</div>
            </div>
            <button class="context-btn context-btn-primary context-btn-wide" id="contextSendAllBtn">
                <span class="btn-icon">▶</span> Send Docs → then Prompt
            </button>
            <button class="context-btn context-btn-stop" id="contextStopBtn" style="display:none;">
                <span class="btn-icon">⏹</span> Stop after current
            </button>
            <p class="context-send-hint">
                Order: doc&nbsp;1 → wait → … → doc&nbsp;N → wait → <strong>your prompt</strong> → reply.
                All saved in loadedConvo.
            </p>
        </div>

        <div class="context-progress" id="contextProgress" style="display:none;">
            <div class="context-progress-text" id="contextProgressText">Sending file 1/3...</div>
            <div class="context-progress-bar-container">
                <div class="context-progress-bar" id="contextProgressBar" style="width: 0%"></div>
            </div>
        </div>
    `;
}

// ── Context Event Listeners ───────────────────────────────────────────────

function attachContextEventListeners() {
    const addFilesBtn = document.getElementById('contextAddFilesBtn');
    const loadFolderBtn = document.getElementById('contextLoadFolderBtn');
    const clearAllBtn = document.getElementById('contextClearAllBtn');
    const fileInput = document.getElementById('contextFileInput');
    const folderInput = document.getElementById('contextFolderInput');
    const sendAllBtn = document.getElementById('contextSendAllBtn');
    const stopBtn = document.getElementById('contextStopBtn');
    const modelSelect = document.getElementById('contextModelSelect');
    const refreshModelsBtn = document.getElementById('contextRefreshModelsBtn');
    const finalPromptEl = document.getElementById('contextFinalPrompt');
    const loadPromptBtn = document.getElementById('contextLoadPromptFileBtn');
    const clearPromptBtn = document.getElementById('contextClearPromptBtn');
    const promptFileInput = document.getElementById('contextPromptFileInput');

    if (addFilesBtn) {
        addFilesBtn.addEventListener('click', () => fileInput?.click());
    }
    if (loadFolderBtn) {
        loadFolderBtn.addEventListener('click', () => folderInput?.click());
    }
    if (fileInput) {
        // Multi-select: keep picker order as sequence (user controls order)
        fileInput.addEventListener('change', (e) => {
            handleContextFiles(e.target.files, { sortByName: false });
            e.target.value = '';
        });
    }
    if (folderInput) {
        // Folder: numeric name sort (01_, 02_, …) for stable sequence
        folderInput.addEventListener('change', (e) => {
            handleContextFiles(e.target.files, { sortByName: true });
            e.target.value = '';
        });
    }
    if (clearAllBtn) {
        clearAllBtn.addEventListener('click', async () => {
            ContextState.files = [];
            ContextState.responses = [];
            OptimizerState.loadedFiles = [];
            ContextState.finalPromptResponse = '';
            ContextState.finalPromptStatus = ContextState.finalPrompt?.trim() ? 'ready' : 'pending';
            try {
                if (OptimizerState.currentModelId) {
                    const orchestrator = await getOrchestrator();
                    const cm = orchestrator.getConversationManager();
                    if (cm) await cm.resetAndSave();
                }
            } catch (err) {
                console.warn('[Context] Could not clear loadedConvo:', err.message);
            }
            refreshContextFileList();
            updateContextPromptUI();
            updateConvertConvoStatus();
            updateOptimizerPipelineStatus();
            updateButtonStates();
            const view = document.querySelector('.context-conversation');
            if (view) window.renderContextConversation?.(view);
        });
    }
    if (sendAllBtn) {
        sendAllBtn.addEventListener('click', () => sendContextFilesInSequence());
    }
    if (stopBtn) {
        stopBtn.addEventListener('click', () => {
            ContextState.isSending = false;
            stopBtn.style.display = 'none';
            if (sendAllBtn) sendAllBtn.style.display = '';
        });
    }
    if (modelSelect) {
        modelSelect.addEventListener('change', (e) => {
            if (e.target.value) {
                // Context model = Lemonade model id only
                syncAllModelSelects(e.target.value);
                onModelSelected(e.target.value);
            }
        });
    }
    if (refreshModelsBtn) {
        refreshModelsBtn.addEventListener('click', () => populateContextLemonadeModels());
    }

    // Final prompt text / file
    if (finalPromptEl) {
        finalPromptEl.addEventListener('input', () => {
            ContextState.finalPrompt = finalPromptEl.value;
            if (!ContextState.finalPromptFileName) {
                ContextState.finalPromptStatus = ContextState.finalPrompt.trim() ? 'ready' : 'pending';
            }
            updateContextPromptUI();
        });
    }
    if (loadPromptBtn) {
        loadPromptBtn.addEventListener('click', () => promptFileInput?.click());
    }
    if (clearPromptBtn) {
        clearPromptBtn.addEventListener('click', () => {
            ContextState.finalPrompt = '';
            ContextState.finalPromptFileName = '';
            ContextState.finalPromptStatus = 'pending';
            ContextState.finalPromptResponse = '';
            if (finalPromptEl) finalPromptEl.value = '';
            updateContextPromptUI();
        });
    }
    if (promptFileInput) {
        promptFileInput.addEventListener('change', async (e) => {
            const file = e.target.files?.[0];
            if (!file) return;
            try {
                const text = await file.text();
                ContextState.finalPrompt = text;
                ContextState.finalPromptFileName = file.name;
                ContextState.finalPromptStatus = text.trim() ? 'ready' : 'pending';
                ContextState.finalPromptResponse = '';
                if (finalPromptEl) finalPromptEl.value = text;
                updateContextPromptUI();
                if (typeof showToast === 'function') {
                    showToast(`Prompt file loaded: ${file.name} (sent last, after docs)`, 'success');
                }
            } catch (err) {
                if (typeof showToast === 'function') showToast('Failed to read prompt file: ' + err.message, 'error');
            }
            e.target.value = '';
        });
    }

    // Initial Lemonade model list + prompt UI
    populateContextLemonadeModels();
    updateContextPromptUI();
}

function updateContextPromptUI() {
    const badge = document.getElementById('contextPromptBadge');
    const status = document.getElementById('contextPromptStatus');
    const fileLabel = document.getElementById('contextPromptFileLabel');
    const ta = document.getElementById('contextFinalPrompt');
    const has = !!(ContextState.finalPrompt && ContextState.finalPrompt.trim());

    if (ta && ta.value !== ContextState.finalPrompt) {
        // Don't fight user typing unless file load set state
        if (ContextState.finalPromptFileName || !document.activeElement || document.activeElement !== ta) {
            ta.value = ContextState.finalPrompt || '';
        }
    }

    if (fileLabel) {
        fileLabel.textContent = ContextState.finalPromptFileName
            ? `📎 ${ContextState.finalPromptFileName}`
            : '';
    }

    const st = ContextState.finalPromptStatus;
    if (badge) {
        if (st === 'done') badge.textContent = 'Sent ✓';
        else if (st === 'sending') badge.textContent = 'Sending…';
        else if (st === 'error') badge.textContent = 'Error';
        else if (has) badge.textContent = 'Queued last';
        else badge.textContent = 'After docs';
        badge.className = 'context-prompt-badge' + (has || st === 'done' ? ` is-${st === 'pending' ? 'ready' : st}` : '');
    }

    if (status) {
        if (st === 'done') {
            status.textContent = 'Prompt was sent after all docs · reply in conversation pane';
            status.className = 'context-prompt-status ready';
        } else if (st === 'sending') {
            status.textContent = 'Docs finished — sending your prompt now…';
            status.className = 'context-prompt-status loading';
        } else if (st === 'error') {
            status.textContent = 'Prompt send failed — see conversation pane';
            status.className = 'context-prompt-status error';
        } else if (has) {
            status.textContent = ContextState.finalPromptFileName
                ? `Ready: “${ContextState.finalPromptFileName}” will send only after all docs`
                : 'Ready: typed prompt will send only after all docs';
            status.className = 'context-prompt-status ready';
        } else {
            status.textContent = 'No prompt yet — docs-only sequence is allowed; add prompt to run at the end';
            status.className = 'context-prompt-status';
        }
    }
}

async function handleContextFiles(fileList, options = {}) {
    if (!fileList?.length) return;
    const { sortByName = false } = options;

    const incoming = [];
    for (const file of fileList) {
        // Skip binary files, only accept text
        const textTypes = ['text/', 'application/json', 'application/xml', 'application/javascript'];
        const textExts = ['.txt', '.md', '.js', '.ts', '.py', '.json', '.xml', '.html', '.css', '.yaml', '.yml', '.toml', '.cfg', '.ini', '.sh', '.bat', '.ps1', '.c', '.cpp', '.h', '.java', '.rs', '.go', '.rb'];
        const isText = textTypes.some(t => file.type.startsWith(t)) ||
                       textExts.some(ext => file.name.toLowerCase().endsWith(ext));
        if (!isText) continue;

        try {
            const content = await file.text();
            incoming.push({
                id: _newContextFileId(),
                name: file.name,
                content: content,
                size: file.size,
                order: 0,
                status: 'pending',
            });
        } catch (err) {
            console.warn(`Failed to read file ${file.name}:`, err);
        }
    }

    // Folder loads: numeric name sort helps 01_… 02_…; multi-select keeps picker order
    if (sortByName) {
        incoming.sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true }));
    }

    // Append to sequence (do not reshuffle existing files)
    for (const f of incoming) {
        ContextState.files.push(f);
    }
    renumberContextFiles();

    syncLoadedFilesFromContext();
    refreshContextFileList();
    updateConvertConvoStatus();
    updateOptimizerPipelineStatus();
    updateButtonStates();

    // Live-update conversation pane
    const view = document.querySelector('.context-conversation');
    if (view) window.renderContextConversation?.(view);

    if (typeof showToast === 'function' && incoming.length) {
        showToast(`Added ${incoming.length} file(s) to sequence (${ContextState.files.length} total)`, 'success');
    }
}

function refreshContextSequenceStrip() {
    const strip = document.getElementById('contextSequenceStrip');
    const chips = document.getElementById('contextSequenceChips');
    if (!strip || !chips) return;

    if (ContextState.files.length === 0) {
        strip.style.display = 'none';
        chips.innerHTML = '';
        return;
    }

    strip.style.display = '';
    chips.innerHTML = ContextState.files.map((f, i) => {
        const status = getContextFileStatus(f, i);
        const meta = contextStatusLabel(status);
        const arrow = i < ContextState.files.length - 1
            ? '<span class="context-seq-arrow" aria-hidden="true">→</span>'
            : '';
        return `
            <button type="button" class="context-seq-chip ${meta.cls}" data-scroll-index="${i}" title="${escapeHtml(f.name)}">
                <span class="context-seq-chip-num">${i + 1}</span>
                <span class="context-seq-chip-name">${escapeHtml(f.name.length > 14 ? f.name.slice(0, 12) + '…' : f.name)}</span>
            </button>${arrow}
        `;
    }).join('');

    chips.querySelectorAll('[data-scroll-index]').forEach(btn => {
        btn.addEventListener('click', () => {
            const idx = parseInt(btn.dataset.scrollIndex, 10);
            const card = document.querySelector(`.context-file-card[data-index="${idx}"]`);
            if (card) {
                card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
                card.classList.add('context-file-flash');
                setTimeout(() => card.classList.remove('context-file-flash'), 800);
            }
        });
    });
}

function refreshContextFileList() {
    const listEl = document.getElementById('contextFileList');
    const countEl = document.getElementById('contextFileCount');
    const clearBtn = document.getElementById('contextClearAllBtn');
    const sendSection = document.getElementById('contextSendSection');

    if (!listEl) return;

    const n = ContextState.files.length;
    const done = ContextState.files.filter((f, i) => getContextFileStatus(f, i) === 'done').length;
    if (countEl) countEl.textContent = n ? `${done}/${n} done` : '0 files';
    if (clearBtn) clearBtn.style.display = n > 0 ? '' : 'none';
    if (sendSection) sendSection.style.display = n > 0 ? '' : 'none';

    refreshContextSequenceStrip();

    if (n === 0) {
        listEl.innerHTML = `
            <div class="context-empty" id="contextEmpty">
                <div class="context-empty-icon">📄</div>
                <div class="context-empty-text">No files in sequence</div>
                <div class="context-empty-hint">Add your context docs. They send 1 → 2 → 3… each waits for a reply before the next.</div>
            </div>
        `;
        return;
    }

    const locked = ContextState.isSending;

    listEl.innerHTML = ContextState.files.map((f, i) => {
        const status = getContextFileStatus(f, i);
        const meta = contextStatusLabel(status);
        const resp = ContextState.responses.find(r =>
            (r.fileId && r.fileId === f.id) || r.fileName === f.name
        );
        const previewOpen = ContextState.expandedPreviewId === f.id;
        const preview = (f.content || '').slice(0, 400);
        const truncated = (f.content || '').length > 400;
        const replyPreview = resp
            ? escapeHtml((resp.response || '').slice(0, 160)) + ((resp.response || '').length > 160 ? '…' : '')
            : '';

        return `
            <div class="context-file-card ${meta.cls} ${status === 'sending' ? 'is-active' : ''}"
                 data-index="${i}" data-id="${escapeHtml(f.id)}" draggable="${locked ? 'false' : 'true'}">
                <div class="context-file-card-main">
                    <div class="context-file-drag" title="Drag to reorder">⠿</div>
                    <div class="context-file-order" title="Step ${i + 1} of ${n}">${i + 1}</div>
                    <div class="context-file-type" aria-hidden="true">${fileTypeIcon(f.name)}</div>
                    <div class="context-file-info">
                        <div class="context-file-name" title="${escapeHtml(f.name)}">${escapeHtml(f.name)}</div>
                        <div class="context-file-meta">
                            <span>${formatFileSize(f.size)}</span>
                            <span class="context-file-step">Step ${i + 1}/${n}</span>
                        </div>
                        <div class="context-file-status-pill ${meta.cls}">
                            <span class="context-file-status-icon">${meta.icon}</span>
                            ${meta.text}
                        </div>
                    </div>
                    <div class="context-file-controls">
                        <button type="button" class="context-ctrl-btn" data-move-up="${i}" title="Move earlier in sequence" ${locked || i === 0 ? 'disabled' : ''}>▲</button>
                        <button type="button" class="context-ctrl-btn" data-move-down="${i}" title="Move later in sequence" ${locked || i === n - 1 ? 'disabled' : ''}>▼</button>
                        <button type="button" class="context-ctrl-btn" data-preview="${escapeHtml(f.id)}" title="Preview content">${previewOpen ? '▾' : '▸'}</button>
                        <button type="button" class="context-file-remove" data-index="${i}" title="Remove from sequence" ${locked ? 'disabled' : ''}>×</button>
                    </div>
                </div>
                ${previewOpen ? `
                    <div class="context-file-preview">
                        <div class="context-preview-label">File preview</div>
                        <pre class="context-preview-body">${escapeHtml(preview)}${truncated ? '\n…' : ''}</pre>
                        ${resp ? `
                            <div class="context-preview-label">Model reply (snippet)</div>
                            <pre class="context-preview-body context-preview-reply">${replyPreview || '(empty)'}</pre>
                        ` : `
                            <div class="context-preview-waiting">No reply yet — will appear after this step runs</div>
                        `}
                    </div>
                ` : ''}
                ${i < n - 1 ? '<div class="context-file-connector" aria-hidden="true"><span>↓ next in sequence</span></div>' : ''}
            </div>
        `;
    }).join('');

    // Controls
    listEl.querySelectorAll('[data-move-up]').forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.stopPropagation();
            const idx = parseInt(btn.dataset.moveUp, 10);
            moveContextFile(idx, idx - 1);
        });
    });
    listEl.querySelectorAll('[data-move-down]').forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.stopPropagation();
            const idx = parseInt(btn.dataset.moveDown, 10);
            moveContextFile(idx, idx + 1);
        });
    });
    listEl.querySelectorAll('[data-preview]').forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.stopPropagation();
            const id = btn.dataset.preview;
            ContextState.expandedPreviewId = ContextState.expandedPreviewId === id ? null : id;
            refreshContextFileList();
        });
    });
    listEl.querySelectorAll('.context-file-remove').forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.stopPropagation();
            if (ContextState.isSending) return;
            const idx = parseInt(btn.dataset.index, 10);
            const removed = ContextState.files[idx];
            ContextState.files.splice(idx, 1);
            if (removed) {
                ContextState.responses = ContextState.responses.filter(r =>
                    r.fileId !== removed.id && r.fileName !== removed.name
                );
            }
            renumberContextFiles();
            syncLoadedFilesFromContext();
            refreshContextFileList();
            const view = document.querySelector('.context-conversation');
            if (view) window.renderContextConversation?.(view);
        });
    });

    // Drag reorder
    if (!locked) {
        const items = listEl.querySelectorAll('.context-file-card');
        let dragIdx = null;
        items.forEach(item => {
            item.addEventListener('dragstart', (e) => {
                dragIdx = parseInt(item.dataset.index, 10);
                item.classList.add('dragging');
                e.dataTransfer.effectAllowed = 'move';
            });
            item.addEventListener('dragend', () => {
                item.classList.remove('dragging');
            });
            item.addEventListener('dragover', (e) => {
                e.preventDefault();
                e.dataTransfer.dropEffect = 'move';
                item.classList.add('drag-over');
            });
            item.addEventListener('dragleave', () => item.classList.remove('drag-over'));
            item.addEventListener('drop', (e) => {
                e.preventDefault();
                item.classList.remove('drag-over');
                const dropIdx = parseInt(item.dataset.index, 10);
                if (dragIdx !== null && dragIdx !== dropIdx) {
                    moveContextFile(dragIdx, dropIdx);
                }
                dragIdx = null;
            });
        });
    }
}

function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

/**
 * load_model: for every context file, sendMsg, append to loadedConvo, save.
 * THEN — only after all docs — send final prompt text/file if present.
 * Models: Lemonade-server only.
 */
async function sendContextFilesInSequence() {
    if (ContextState.files.length === 0) return;
    const modelSelect = document.getElementById('contextModelSelect');
    const modelId = modelSelect?.value || OptimizerState.currentModelId;
    if (!modelId) {
        if (typeof showToast === 'function') showToast('Select a Lemonade model first', 'warning');
        else showConvertError('Select a Lemonade model first');
        return;
    }

    // Sync final prompt from textarea (in case user typed without blur)
    const ta = document.getElementById('contextFinalPrompt');
    if (ta) ContextState.finalPrompt = ta.value;

    syncAllModelSelects(modelId);
    syncLoadedFilesFromContext();

    ContextState.isSending = true;
    ContextState.responses = [];
    ContextState.activeSendIndex = 0;
    ContextState.finalPromptResponse = '';
    ContextState.finalPromptTokens = 0;
    if (ContextState.finalPrompt?.trim()) {
        ContextState.finalPromptStatus = 'ready'; // queued until docs done
    } else {
        ContextState.finalPromptStatus = 'skipped';
    }
    ContextState.files.forEach(f => { f.status = 'pending'; });

    const sendBtn = document.getElementById('contextSendAllBtn');
    const stopBtn = document.getElementById('contextStopBtn');
    const progressEl = document.getElementById('contextProgress');
    const progressText = document.getElementById('contextProgressText');
    const progressBar = document.getElementById('contextProgressBar');

    if (sendBtn) sendBtn.style.display = 'none';
    if (stopBtn) stopBtn.style.display = '';
    if (progressEl) progressEl.style.display = '';
    refreshContextFileList();
    updateContextPromptUI();

    // Ensure conversation view is visible in editor
    let contextView = document.querySelector('.context-conversation');
    if (!contextView) {
        const editorContainer = document.querySelector('.editor-container');
        if (editorContainer) {
            contextView = document.createElement('div');
            contextView.className = 'context-conversation';
            editorContainer.appendChild(contextView);
        }
    }
    if (contextView) {
        contextView.style.display = 'flex';
        contextView.classList.add('active');
        const editorHeader = document.querySelector('.editor-header');
        const editorPanes = document.querySelector('.editor-panes');
        const actionBar = document.querySelector('.action-bar');
        const dashboard = document.querySelector('.optimizer-dashboard');
        if (editorHeader) editorHeader.style.display = 'none';
        if (editorPanes) editorPanes.style.display = 'none';
        if (actionBar) actionBar.style.display = 'none';
        if (dashboard) { dashboard.classList.remove('active'); dashboard.style.display = 'none'; }
        window.renderContextConversation?.(contextView);
    }

    try {
        // Force orchestrator onto Lemonade endpoint + selected model
        const orchestrator = await getOrchestrator({ forceLemonade: true, modelId });
        // Snapshot sequence order — strict 1..N (docs only — prompt is separate)
        const files = ContextState.files.map(f => ({
            id: f.id,
            name: f.name,
            content: f.content,
            size: f.size,
        }));

        // Mark first as sending before first API call
        if (ContextState.files[0]) ContextState.files[0].status = 'sending';
        ContextState.activeSendIndex = 0;
        refreshContextFileList();

        const hasFinal = !!(ContextState.finalPrompt && ContextState.finalPrompt.trim());
        const totalPhases = files.length + (hasFinal ? 1 : 0);

        const result = await orchestrator.loadModel(files, {
            mode: 'sequential',
            shouldContinue: () => ContextState.isSending,
            onFileResponse: (file, responseText, index, total, entry) => {
                const meta = ContextState.files[index];
                const isErr = !!entry?.error;
                if (meta) meta.status = isErr ? 'error' : 'done';

                ContextState.responses.push({
                    fileId: meta?.id || file.id,
                    fileName: entry?.fileName || file.name,
                    response: entry?.response || responseText,
                    timestamp: entry?.timestamp || new Date().toISOString(),
                    tokens: entry?.tokens || 0,
                    error: isErr,
                });

                // Next doc step becomes active (if any)
                const next = index + 1;
                ContextState.activeSendIndex = next < total ? next : -1;
                if (next < total && ContextState.files[next] && ContextState.isSending) {
                    ContextState.files[next].status = 'sending';
                }

                if (progressText) {
                    const afterDocs = hasFinal ? ' → then your prompt' : '';
                    progressText.textContent =
                        `Doc ${index + 1}/${total}: ${file.name}` +
                        (next < total && ContextState.isSending
                            ? ` → next doc: ${files[next]?.name || ''}`
                            : (hasFinal && ContextState.isSending ? afterDocs : ' ✓ docs done'));
                }
                if (progressBar) {
                    // Reserve last slice of bar for final prompt if present
                    const base = hasFinal ? (index + 1) / totalPhases : (index + 1) / total;
                    progressBar.style.width = (base * 100) + '%';
                }
                refreshContextFileList();
                const view = document.querySelector('.context-conversation');
                if (view) window.renderContextConversation?.(view);
            },
        });

        // ── FINAL STEP: user prompt ONLY after all documents ──────────────
        if (ContextState.isSending && hasFinal) {
            ContextState.finalPromptStatus = 'sending';
            ContextState.activeSendIndex = -1;
            updateContextPromptUI();
            if (progressText) {
                progressText.textContent = ContextState.finalPromptFileName
                    ? `Docs done — sending prompt file: ${ContextState.finalPromptFileName}`
                    : 'Docs done — sending your final prompt…';
            }
            if (progressBar) {
                progressBar.style.width = ((files.length / totalPhases) * 100 + 5) + '%';
            }
            const viewMid = document.querySelector('.context-conversation');
            if (viewMid) window.renderContextConversation?.(viewMid);

            try {
                const cm = orchestrator.getConversationManager();
                if (!cm) throw new Error('No conversation manager for final prompt');

                const promptText = ContextState.finalPrompt.trim();
                const userPayload = ContextState.finalPromptFileName
                    ? `--- Final prompt file: ${ContextState.finalPromptFileName} ---\n${promptText}`
                    : promptText;

                const messages = [
                    ...cm.getLoadedConvo(),
                    { role: 'user', content: userPayload },
                ];

                const { content, usage } = await cm.sendToModel(messages, {
                    temperature: 0.7,
                    maxTokens: 4096,
                });

                await cm.appendExchange(userPayload, content, {
                    phase: 'final_prompt',
                });

                ContextState.finalPromptResponse = content;
                ContextState.finalPromptTokens = usage?.total || 0;
                ContextState.finalPromptStatus = 'done';

                if (progressText) {
                    progressText.textContent =
                        `Complete — ${result.filesLoaded} docs + final prompt in loadedConvo`;
                }
                if (progressBar) progressBar.style.width = '100%';

                if (typeof showToast === 'function') {
                    showToast('Docs + final prompt saved to loadedConvo', 'success');
                }
            } catch (promptErr) {
                console.error('[Context] Final prompt failed:', promptErr);
                ContextState.finalPromptStatus = 'error';
                ContextState.finalPromptResponse = `Error: ${promptErr.message}`;
                if (progressText) progressText.textContent = `Docs OK — prompt failed: ${promptErr.message}`;
                if (typeof showToast === 'function') {
                    showToast('Docs loaded but final prompt failed: ' + promptErr.message, 'error');
                }
            }
        } else if (progressBar) {
            progressBar.style.width = '100%';
            if (progressText) {
                progressText.textContent =
                    `Docs complete — ${result.filesLoaded}/${files.length} in loadedConvo` +
                    (result.partialFailure ? ' (partial failures)' : '') +
                    (hasFinal ? '' : ' · no final prompt set');
            }
            if (typeof showToast === 'function') {
                showToast(`Sequence saved: ${result.filesLoaded} docs in loadedConvo`, 'success');
            }
        }
    } catch (err) {
        console.error('[Context] Sequential load failed:', err);
        if (progressText) progressText.textContent = `Error: ${err.message}`;
        if (typeof showToast === 'function') showToast('Context load failed: ' + err.message, 'error');
        else showConvertError(err.message);
    } finally {
        ContextState.isSending = false;
        ContextState.activeSendIndex = -1;
        if (sendBtn) sendBtn.style.display = '';
        if (stopBtn) stopBtn.style.display = 'none';
        refreshContextFileList();
        updateContextPromptUI();
        updateConvertConvoStatus();
        updateOptimizerPipelineStatus();
        updateButtonStates();
        const view = document.querySelector('.context-conversation');
        if (view) window.renderContextConversation?.(view);
    }
}

// ── Convert Event Listeners ───────────────────────────────────────────────

let _convertTemplateTab = 'convert';

function attachConvertEventListeners() {
    const convertBtn = document.getElementById('convertBtn');
    const testBtn = document.getElementById('convertTestBtn');
    const updateBtn = document.getElementById('convertUpdateBtn');
    const copyBtn = document.getElementById('convertCopyBtn');
    const toEditorBtn = document.getElementById('convertToEditorBtn');
    const dismissBtn = document.getElementById('convertDismissErrorBtn');
    const resetTmplBtn = document.getElementById('convertResetTemplateBtn');
    const saveTmplBtn = document.getElementById('convertSaveTemplateBtn');
    const modelSelect = document.getElementById('convertModelSelect');
    const tmplTabs = document.querySelectorAll('.convert-tmpl-tab');

    if (convertBtn) convertBtn.addEventListener('click', () => runConversion());
    if (testBtn) testBtn.addEventListener('click', () => runConvertTest());
    if (updateBtn) updateBtn.addEventListener('click', () => runConvertUpdate());

    if (modelSelect) {
        modelSelect.addEventListener('change', async (e) => {
            const id = e.target.value;
            if (!id) return;
            syncAllModelSelects(id);
            await onModelSelected(id);
            await loadConvertTemplateForEdit(_convertTemplateTab);
            updateConvertConvoStatus();
        });
    }

    if (copyBtn) {
        copyBtn.addEventListener('click', () => {
            const output = document.getElementById('convertOutputContent');
            if (output) {
                navigator.clipboard.writeText(output.textContent).then(() => {
                    copyBtn.textContent = '✅ Copied!';
                    setTimeout(() => copyBtn.textContent = '📋 Copy', 2000);
                });
            }
        });
    }

    if (toEditorBtn) {
        toEditorBtn.addEventListener('click', () => {
            const content = document.getElementById('convertOutputContent')?.textContent;
            const outputPrompt = document.getElementById('outputPrompt');
            if (content && outputPrompt) {
                outputPrompt.value = content;
                outputPrompt.dispatchEvent(new Event('input'));
                if (typeof showToast === 'function') showToast('Loaded into UCL editor', 'success');
            }
        });
    }

    if (dismissBtn) {
        dismissBtn.addEventListener('click', () => {
            const errEl = document.getElementById('convertError');
            if (errEl) errEl.style.display = 'none';
        });
    }

    if (resetTmplBtn) {
        resetTmplBtn.addEventListener('click', () => onConvertResetTemplate());
    }
    if (saveTmplBtn) {
        saveTmplBtn.addEventListener('click', () => onConvertSaveTemplate());
    }

    // Template tab switching
    tmplTabs.forEach(tab => {
        tab.addEventListener('click', async () => {
            tmplTabs.forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
            _convertTemplateTab = tab.dataset.tmpl || 'convert';
            await loadConvertTemplateForEdit(_convertTemplateTab);
        });
    });
}

async function loadConvertTemplateForEdit(type) {
    const editor = document.getElementById('convertTemplateEditor');
    if (!editor) return;

    try {
        // Prefer live orchestrator storage; fall back to ephemeral StorageManager
        let storage = OptimizerState.orchestrator?.getStorageManager?.();
        if (!storage) {
            const { StorageManager } = await import('./storage-manager.js');
            storage = new StorageManager();
            await storage.init();
            await storage.initDefaultTemplates();
        }

        const templateId = type === 'update' ? 'base_update' : 'base_convert';
        const template = await storage.getTemplate(templateId, OptimizerState.currentModelId);
        editor.value = template?.content || '';
    } catch (err) {
        console.warn('[Convert] Failed to load template:', err);
    }
}

async function onConvertSaveTemplate() {
    const editor = document.getElementById('convertTemplateEditor');
    if (!editor) return;
    try {
        const orchestrator = await getOrchestrator();
        const storage = orchestrator.getStorageManager();
        const kind = _convertTemplateTab === 'update' ? 'update' : 'convert';
        await storage.saveTemplate({
            templateId: `${OptimizerState.currentModelId}:${kind}`,
            name: `${OptimizerState.currentModelId} - ${kind}`,
            content: editor.value,
            scope: 'model',
            modelId: OptimizerState.currentModelId,
        });
        if (typeof showToast === 'function') showToast('Template saved for model', 'success');
    } catch (err) {
        showConvertError(err.message);
    }
}

async function onConvertResetTemplate() {
    const editor = document.getElementById('convertTemplateEditor');
    if (!editor) return;
    try {
        const orchestrator = await getOrchestrator();
        const storage = orchestrator.getStorageManager();
        const templateId = _convertTemplateTab === 'update' ? 'base_update' : 'base_convert';
        const template = await storage.getTemplate(templateId); // global only
        editor.value = template?.content || '';
        if (typeof showToast === 'function') showToast('Template reset to default', 'info');
    } catch (err) {
        showConvertError(err.message);
    }
}

function setConvertProgress(show, pct, text) {
    const progressEl = document.getElementById('convertProgress');
    const progressBar = document.getElementById('convertProgressBar');
    const progressText = document.getElementById('convertProgressText');
    if (progressEl) progressEl.style.display = show ? '' : 'none';
    if (progressBar && pct != null) progressBar.style.width = `${pct}%`;
    if (progressText && text) progressText.textContent = text;
}

function showConvertOutput(label, content) {
    const outputEl = document.getElementById('convertOutput');
    const outputContent = document.getElementById('convertOutputContent');
    const outputLabel = document.getElementById('convertOutputLabel');
    if (outputLabel) outputLabel.textContent = label;
    if (outputContent) outputContent.textContent = content;
    if (outputEl) outputEl.style.display = '';
}

function showConvertThinking(versionNum, thinking) {
    const section = document.getElementById('convertThinkingSection');
    const content = document.getElementById('convertThinkingContent');
    const label = document.getElementById('convertThinkingLabel');
    if (!thinking) {
        if (section) section.style.display = 'none';
        return;
    }
    if (label) label.textContent = `Thinking / Reasoning (v${versionNum})`;
    if (content) content.textContent = thinking;
    if (section) section.style.display = '';
}

/**
 * convertPrompt: loadedConvo + base_prompt + promptToConvert → versioned output
 */
async function runConversion() {
    const prompt = document.getElementById('convertUserPrompt')?.value?.trim();
    if (!prompt) {
        if (typeof showToast === 'function') showToast('Enter a prompt to convert', 'warning');
        return;
    }

    const modelSelect = document.getElementById('convertModelSelect');
    const modelId = modelSelect?.value || OptimizerState.currentModelId;
    if (!modelId) {
        showConvertError('Select a model first');
        return;
    }
    syncAllModelSelects(modelId);

    const errorEl = document.getElementById('convertError');
    if (errorEl) errorEl.style.display = 'none';
    setConvertProgress(true, 20, 'Restoring loadedConvo / converting...');

    try {
        await runOrchestratorPhase(async (orchestrator) => {
            // Prefer Context files; else restore saved loadedConvo
            syncLoadedFilesFromContext();
            if (OptimizerState.loadedFiles.length > 0 && ContextState.responses.length === 0) {
                setConvertProgress(true, 35, 'Loading context into loadedConvo...');
                await orchestrator.loadModel(OptimizerState.loadedFiles, { mode: 'sequential' });
            } else {
                const cm = orchestrator.getConversationManager();
                const restored = cm ? await cm.restore() : false;
                if (!restored && OptimizerState.loadedFiles.length > 0) {
                    await orchestrator.loadModel(OptimizerState.loadedFiles, { mode: 'sequential' });
                }
            }

            // If user edited convert template, push it as model override first
            const editor = document.getElementById('convertTemplateEditor');
            if (editor?.value?.trim() && _convertTemplateTab === 'convert') {
                const storage = orchestrator.getStorageManager();
                await storage.saveTemplate({
                    templateId: `${OptimizerState.currentModelId}:convert`,
                    name: `${OptimizerState.currentModelId} - convert`,
                    content: editor.value,
                    scope: 'model',
                    modelId: OptimizerState.currentModelId,
                });
            }

            setConvertProgress(true, 60, 'Converting with loadedConvo + base_prompt...');
            const result = await orchestrator.convertPrompt(prompt);

            setConvertProgress(true, 100, `Saved converted_prompt_v${result.version.versionNum}`);
            showConvertOutput(
                `converted_prompt_v${result.version.versionNum}`,
                result.convertedText
            );
            showConvertThinking(result.version.versionNum, null);

            // Mirror into main UCL editor
            const outputPrompt = document.getElementById('outputPrompt');
            if (outputPrompt) {
                outputPrompt.value = result.convertedText;
                outputPrompt.dispatchEvent(new Event('input'));
            }

            await refreshVersions();
            updateConvertConvoStatus();
            updateOptimizerPipelineStatus();

            if (typeof showToast === 'function') {
                showToast(`converted_prompt_v${result.version.versionNum} saved`, 'success');
            }
        });
    } catch (err) {
        showConvertError(err.message);
    } finally {
        setTimeout(() => setConvertProgress(false), 1200);
    }
}

/**
 * testPrompt + extractReasoning for latest version
 */
async function runConvertTest() {
    const errorEl = document.getElementById('convertError');
    if (errorEl) errorEl.style.display = 'none';
    setConvertProgress(true, 25, 'Testing latest converted_prompt...');

    try {
        await runOrchestratorPhase(async (orchestrator) => {
            // Restore history if needed
            const cm = orchestrator.getConversationManager();
            if (cm) await cm.restore();

            const vm = orchestrator.getVersionManager();
            const latest = await vm.getLatestVersion();
            if (!latest) throw new Error('No versions to test. Run Convert first.');

            setConvertProgress(true, 50, `Testing v${latest.versionNum}...`);
            const result = await orchestrator.testPrompt(latest.versionNum);

            setConvertProgress(true, 75, 'Extracting reasoning...');
            const extraction = await orchestrator.extractReasoning(result.output, latest.versionNum);

            setConvertProgress(true, 100, 'Test complete');
            showConvertOutput(`Test output (v${latest.versionNum})`, result.output);
            showConvertThinking(latest.versionNum, extraction.thinking || '(No thinking block detected)');

            if (extraction.thinking) {
                showThinkingLog(extraction.thinking, latest.versionNum);
            }

            await refreshVersions();
            if (typeof showToast === 'function') showToast('Test + extract complete', 'success');
        });
    } catch (err) {
        showConvertError(err.message);
    } finally {
        setTimeout(() => setConvertProgress(false), 1200);
    }
}

/**
 * updatePrompt with historicalConvo + thinking; history saved for future mods
 */
async function runConvertUpdate() {
    const feedbackSection = document.getElementById('convertFeedbackSection');
    const feedbackInput = document.getElementById('convertFeedbackInput');

    // First click reveals feedback; second applies if filled
    if (feedbackSection && feedbackSection.style.display === 'none') {
        feedbackSection.style.display = '';
        if (typeof showToast === 'function') {
            showToast('Enter feedback, then click Update again', 'info');
        }
        return;
    }

    const feedback = feedbackInput?.value?.trim();
    if (!feedback) {
        if (typeof showToast === 'function') showToast('Enter update feedback first', 'warning');
        return;
    }

    const errorEl = document.getElementById('convertError');
    if (errorEl) errorEl.style.display = 'none';
    setConvertProgress(true, 30, 'Updating with history + thinking...');

    try {
        await runOrchestratorPhase(async (orchestrator) => {
            const cm = orchestrator.getConversationManager();
            if (cm) await cm.restore();

            const vm = orchestrator.getVersionManager();
            const latest = await vm.getLatestVersion();
            if (!latest) throw new Error('No versions to update. Convert first.');

            // Push update template override if user is on update tab
            const editor = document.getElementById('convertTemplateEditor');
            if (editor?.value?.trim() && _convertTemplateTab === 'update') {
                const storage = orchestrator.getStorageManager();
                await storage.saveTemplate({
                    templateId: `${OptimizerState.currentModelId}:update`,
                    name: `${OptimizerState.currentModelId} - update`,
                    content: editor.value,
                    scope: 'model',
                    modelId: OptimizerState.currentModelId,
                });
            }

            setConvertProgress(true, 60, `Updating from v${latest.versionNum}...`);
            const newVersion = await orchestrator.updatePrompt(feedback, latest.versionNum);

            setConvertProgress(true, 100, `Saved converted_prompt_v${newVersion.versionNum}`);
            showConvertOutput(
                `converted_prompt_v${newVersion.versionNum} (from v${latest.versionNum})`,
                newVersion.promptText
            );

            const outputPrompt = document.getElementById('outputPrompt');
            if (outputPrompt) {
                outputPrompt.value = newVersion.promptText;
                outputPrompt.dispatchEvent(new Event('input'));
            }

            if (feedbackInput) feedbackInput.value = '';
            if (feedbackSection) feedbackSection.style.display = 'none';

            await refreshVersions();
            updateConvertConvoStatus();
            updateOptimizerPipelineStatus();

            if (typeof showToast === 'function') {
                showToast(`Updated → v${newVersion.versionNum} (history saved)`, 'success');
            }
        });
    } catch (err) {
        showConvertError(err.message);
    } finally {
        setTimeout(() => setConvertProgress(false), 1200);
    }
}

function showConvertError(msg) {
    const errorEl = document.getElementById('convertError');
    const errorMsg = document.getElementById('convertErrorMessage');
    if (errorEl) errorEl.style.display = '';
    if (errorMsg) errorMsg.textContent = msg;
    if (typeof showToast === 'function') showToast(msg, 'error');
}

// ── Context Conversation View (rendered in editor area) ───────────────────

function renderContextConversation(container) {
    if (!container) return;

    if (ContextState.files.length === 0) {
        container.innerHTML = `
            <div class="conv-empty">
                <div class="conv-empty-icon">📁</div>
                <h3>Context Sequence Conversation</h3>
                <p>Add files in the <strong>Context</strong> tab. They form one conversation sent strictly in order:</p>
                <p><strong>file 1 → wait → reply → file 2 → wait → reply → …</strong></p>
                <p>When you have ~7 docs, each step builds on the previous replies (loadedConvo).</p>
            </div>
        `;
        return;
    }

    const done = ContextState.files.filter((f, i) => getContextFileStatus(f, i) === 'done').length;
    const total = ContextState.files.length;
    const hasFinal = !!(ContextState.finalPrompt && ContextState.finalPrompt.trim());
    const finalSt = ContextState.finalPromptStatus;
    const totalSteps = total + (hasFinal ? 1 : 0);

    let html = `
        <div class="conv-header">
            <div>
                <h2>📁 Context Sequence Conversation</h2>
                <p class="conv-header-sub">Lemonade · docs first${hasFinal ? ' · prompt last' : ''} · ${done}/${total} docs</p>
            </div>
            <div class="conv-stats">
                <span>${total} docs</span>
                <span>${done} replies</span>
                ${hasFinal ? `<span class="conv-stat-prompt">+ prompt</span>` : ''}
                <span>${ContextState.responses.reduce((sum, r) => sum + (r.tokens || 0), 0) + (ContextState.finalPromptTokens || 0)} tokens</span>
            </div>
        </div>
        <div class="conv-sequence-bar">
            ${ContextState.files.map((f, i) => {
                const st = getContextFileStatus(f, i);
                return `<span class="conv-seq-dot ${st}" title="Doc ${i + 1}: ${escapeHtml(f.name)}">${i + 1}</span><span class="conv-seq-line"></span>`;
            }).join('')}
            ${hasFinal
                ? `<span class="conv-seq-dot final ${finalSt === 'done' ? 'done' : finalSt === 'sending' ? 'sending' : finalSt === 'error' ? 'error' : 'pending'}" title="Final prompt">P</span>`
                : ''}
        </div>
        <div class="conv-entries">
    `;

    ContextState.files.forEach((file, i) => {
        const status = getContextFileStatus(file, i);
        const resp = ContextState.responses.find(r =>
            (r.fileId && r.fileId === file.id) || r.fileName === file.name
        );
        html += `
            <div class="conv-entry conv-status-${status}" id="conv-step-${i}">
                <div class="conv-entry-file">
                    <div class="conv-entry-badge">Doc ${i + 1}/${total}</div>
                    <div class="conv-entry-name">${fileTypeIcon(file.name)} ${escapeHtml(file.name)}
                        <span class="conv-entry-size">(${formatFileSize(file.size)})</span>
                    </div>
                    <div class="conv-entry-status-line">${contextStatusLabel(status).text}</div>
                    <details class="conv-entry-content">
                        <summary>View file content</summary>
                        <pre>${escapeHtml((file.content || '').substring(0, 2000))}${(file.content || '').length > 2000 ? '\n... (truncated)' : ''}</pre>
                    </details>
                </div>
                <div class="conv-entry-arrow">${status === 'done' ? '✓' : status === 'sending' ? '…' : '→'}</div>
                <div class="conv-entry-response ${resp ? (resp.error ? 'conv-entry-error' : '') : 'conv-entry-pending'}">
                    ${resp
                        ? `<pre>${escapeHtml(resp.response)}</pre><div class="conv-entry-meta">${resp.tokens || 0} tokens · ${escapeHtml(resp.timestamp || '')}</div>`
                        : status === 'sending'
                            ? '<div class="conv-entry-waiting">⟳ Waiting for model reply…</div>'
                            : '<div class="conv-entry-waiting">○ Queued — earlier docs must finish first</div>'
                    }
                </div>
            </div>
        `;
    });

    // Final prompt block — only after docs conceptually
    if (hasFinal) {
        const promptLabel = ContextState.finalPromptFileName
            ? `Prompt file: ${ContextState.finalPromptFileName}`
            : 'Your typed prompt';
        html += `
            <div class="conv-entry conv-entry-final conv-status-${finalSt === 'done' ? 'done' : finalSt === 'sending' ? 'sending' : 'pending'}" id="conv-step-final">
                <div class="conv-entry-file">
                    <div class="conv-entry-badge conv-badge-prompt">FINAL · after all docs</div>
                    <div class="conv-entry-name">💬 ${escapeHtml(promptLabel)}</div>
                    <div class="conv-entry-status-line">
                        ${finalSt === 'done' ? 'Prompt sent &amp; replied'
                            : finalSt === 'sending' ? 'Sending prompt now…'
                            : finalSt === 'error' ? 'Prompt failed'
                            : done < total ? 'Waiting — all docs must finish first'
                            : 'Ready to send after docs'}
                    </div>
                    <details class="conv-entry-content">
                        <summary>View prompt</summary>
                        <pre>${escapeHtml((ContextState.finalPrompt || '').substring(0, 2000))}${(ContextState.finalPrompt || '').length > 2000 ? '\n... (truncated)' : ''}</pre>
                    </details>
                </div>
                <div class="conv-entry-arrow">${finalSt === 'done' ? '✓' : finalSt === 'sending' ? '…' : '★'}</div>
                <div class="conv-entry-response ${ContextState.finalPromptResponse ? (finalSt === 'error' ? 'conv-entry-error' : '') : 'conv-entry-pending'}">
                    ${ContextState.finalPromptResponse
                        ? `<pre>${escapeHtml(ContextState.finalPromptResponse)}</pre><div class="conv-entry-meta">${ContextState.finalPromptTokens || 0} tokens</div>`
                        : finalSt === 'sending'
                            ? '<div class="conv-entry-waiting">⟳ Waiting for model reply to your prompt…</div>'
                            : '<div class="conv-entry-waiting">★ Not sent yet — runs only after every document</div>'
                    }
                </div>
            </div>
        `;
    }

    html += '</div>';
    container.innerHTML = html;

    // Auto-scroll to active step
    if (ContextState.isSending) {
        const targetId = finalSt === 'sending'
            ? 'conv-step-final'
            : `conv-step-${ContextState.activeSendIndex >= 0 ? ContextState.activeSendIndex : Math.max(0, done - 1)}`;
        const activeEl = document.getElementById(targetId);
        if (activeEl) activeEl.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
}

// Expose globally for tools.js
window.renderContextConversation = renderContextConversation;

// ── Event Listeners ───────────────────────────────────────────────────────

function attachOptimizerEventListeners() {
    // Model selector
    const modelSelect = document.getElementById('optimizerModelSelect');
    if (modelSelect) {
        modelSelect.addEventListener('change', (e) => onModelSelected(e.target.value));
    }

    // New session
    const newSessionBtn = document.getElementById('optimizerNewSessionBtn');
    if (newSessionBtn) {
        newSessionBtn.addEventListener('click', onNewSession);
    }

    // Session actions
    const exportBtn = document.getElementById('optimizerExportBtn');
    if (exportBtn) exportBtn.addEventListener('click', onExportSession);

    const deleteBtn = document.getElementById('optimizerDeleteSessionBtn');
    if (deleteBtn) deleteBtn.addEventListener('click', onDeleteSession);

    // Load context folder
    const loadBtn = document.getElementById('optimizerLoadFolderBtn');
    if (loadBtn) loadBtn.addEventListener('click', onLoadContextFolder);

    // Template tabs
    document.querySelectorAll('.optimizer-tmpl-tab').forEach(tab => {
        tab.addEventListener('click', () => onTemplateTabSwitch(tab.dataset.tmpl));
    });

    const resetTemplateBtn = document.getElementById('optimizerResetTemplateBtn');
    if (resetTemplateBtn) resetTemplateBtn.addEventListener('click', onResetTemplate);

    const saveTemplateBtn = document.getElementById('optimizerSaveTemplateBtn');
    if (saveTemplateBtn) saveTemplateBtn.addEventListener('click', onSaveTemplate);

    // Main actions
    const convertBtn = document.getElementById('optimizerConvertBtn');
    if (convertBtn) convertBtn.addEventListener('click', onConvert);

    const testBtn = document.getElementById('optimizerTestBtn');
    if (testBtn) testBtn.addEventListener('click', onTest);

    const updateBtn = document.getElementById('optimizerUpdateBtn');
    if (updateBtn) updateBtn.addEventListener('click', onUpdate);

    const fullLoopBtn = document.getElementById('optimizerFullLoopBtn');
    if (fullLoopBtn) fullLoopBtn.addEventListener('click', onFullLoop);

    const resetBtn = document.getElementById('optimizerResetBtn');
    if (resetBtn) resetBtn.addEventListener('click', onReset);

    // Feedback
    const applyFeedbackBtn = document.getElementById('optimizerApplyFeedbackBtn');
    if (applyFeedbackBtn) applyFeedbackBtn.addEventListener('click', onApplyFeedback);

    // Thinking log
    const closeThinkingBtn = document.getElementById('optimizerCloseThinkingBtn');
    if (closeThinkingBtn) closeThinkingBtn.addEventListener('click', () => {
        document.getElementById('optimizerThinkingSection').style.display = 'none';
    });

    // Error dismiss
    const dismissErrorBtn = document.getElementById('optimizerDismissErrorBtn');
    if (dismissErrorBtn) dismissErrorBtn.addEventListener('click', () => {
        document.getElementById('optimizerError').style.display = 'none';
    });

    // Conversation preview toggle
    const toggleConvoBtn = document.getElementById('optimizerToggleConvoBtn');
    if (toggleConvoBtn) toggleConvoBtn.addEventListener('click', () => {
        const content = document.getElementById('optimizerConvoContent');
        if (content) content.style.display = content.style.display === 'none' ? '' : 'none';
    });

    // Iteration config inputs
    const maxIterInput = document.getElementById('optMaxIter');
    if (maxIterInput) maxIterInput.addEventListener('change', (e) => {
        OptimizerState.iterationConfig.maxIterations = parseInt(e.target.value) || 5;
    });

    const qualityThreshInput = document.getElementById('optQualityThresh');
    if (qualityThreshInput) qualityThreshInput.addEventListener('change', (e) => {
        const newThreshold = parseFloat(e.target.value) || 0.85;
        OptimizerState.iterationConfig.qualityThreshold = newThreshold;
        DashboardState.qualityThreshold = newThreshold;  // Sync to shared dashboard state
        // Re-render chart so the threshold line updates to match
        if (typeof window.renderScoreChart === 'function') {
            window.renderScoreChart();
        }
    });

    const convDeltaInput = document.getElementById('optConvDelta');
    if (convDeltaInput) convDeltaInput.addEventListener('change', (e) => {
        OptimizerState.iterationConfig.convergenceDelta = parseFloat(e.target.value) || 0.02;
    });

    const autoIterCheck = document.getElementById('optAutoIter');
    if (autoIterCheck) autoIterCheck.addEventListener('change', (e) => {
        OptimizerState.iterationConfig.autoIterate = e.target.checked;
    });

    // Start/Stop optimization loop buttons
    const startLoopBtn = document.getElementById('optimizerStartLoopBtn');
    if (startLoopBtn) startLoopBtn.addEventListener('click', onStartOptimization);

    const stopLoopBtn = document.getElementById('optimizerStopLoopBtn');
    if (stopLoopBtn) stopLoopBtn.addEventListener('click', onStopOptimization);
}

// ── Event Handlers ────────────────────────────────────────────────────────

async function onModelSelected(modelId) {
    if (!modelId) return;
    const previousModel = OptimizerState.currentModelId;
    syncAllModelSelects(modelId);

    // Dispose old orchestrator if exists (model identity changed)
    if (OptimizerState.orchestrator && OptimizerState.orchestrator.modelId !== modelId) {
        OptimizerState.orchestrator.dispose();
        OptimizerState.orchestrator = null;
        // Clear prior model's context view (files may not apply to new model)
        if (previousModel && previousModel !== modelId) {
            ContextState.files = [];
            ContextState.responses = [];
            OptimizerState.loadedFiles = [];
            OptimizerState.versions = [];
            refreshContextFileList();
            updateVersionListUI([]);
        }
    }

    // Update session actions visibility
    const actions = document.getElementById('optimizerSessionActions');
    if (actions) actions.style.display = modelId ? '' : 'none';

    // Create orchestrator so we can restore loadedConvo + versions for this model
    try {
        const orchestrator = await getOrchestrator();
        const cm = orchestrator.getConversationManager();
        const restored = cm ? await cm.restore() : false;
        if (restored && cm) {
            const state = cm.getState();
            // Surface restored file names as staged context if empty
            if (ContextState.files.length === 0 && state.loadedFileNames?.length) {
                ContextState.responses = state.loadedFileNames.map(name => ({
                    fileName: name,
                    response: '(restored from loadedConvo — re-send files to refresh responses)',
                    timestamp: new Date().toISOString(),
                    tokens: 0,
                    restored: true,
                }));
            }
        }
        await refreshVersions();
        await loadTemplateForEdit('convert');
        await loadConvertTemplateForEdit(_convertTemplateTab);
    } catch (err) {
        console.warn('[OptimizerUI] Model init:', err.message);
    }

    updateConvertConvoStatus();
    updateOptimizerPipelineStatus();
    updateButtonStates();
}

function onNewSession() {
    const modelId = prompt('Enter model ID (e.g. qwen3-4b-FLM):');
    if (!modelId?.trim()) return;

    OptimizerState.currentModelId = modelId.trim();

    // Add to selector if not present
    const select = document.getElementById('optimizerModelSelect');
    if (select && !select.querySelector(`option[value="${modelId}"]`)) {
        const opt = document.createElement('option');
        opt.value = modelId;
        opt.textContent = modelId;
        opt.selected = true;
        select.appendChild(opt);
    }

    onModelSelected(modelId.trim());
}

async function onLoadContextFolder() {
    try {
        let files;
        if (FileLoader.isFileSystemAccessSupported()) {
            files = await FileLoader.pickFolder();
        } else {
            // Fallback
            const input = FileLoader.createFallbackInput();
            document.body.appendChild(input);
            input.click();

            files = await new Promise((resolve, reject) => {
                input.addEventListener('change', async () => {
                    try {
                        const result = await FileLoader.readFromInput(input);
                        resolve(result);
                    } catch (e) {
                        reject(e);
                    } finally {
                        input.remove();
                    }
                });
            });
        }

        OptimizerState.loadedFiles = files;
        updateFileListUI(files);

        // Enable convert button
        const convertBtn = document.getElementById('optimizerConvertBtn');
        if (convertBtn) convertBtn.disabled = false;

        const fullLoopBtn = document.getElementById('optimizerFullLoopBtn');
        if (fullLoopBtn) fullLoopBtn.disabled = false;

        if (typeof showToast === 'function') {
            showToast(`Loaded ${files.length} context files`, 'success');
        }

    } catch (error) {
        if (error.message.includes('cancelled')) return; // User cancelled
        showOptimizerError(error.message);
    }
}

async function onConvert() {
    const promptInput = document.getElementById('optimizerPromptInput');
    const userPrompt = promptInput?.value?.trim();
    if (!userPrompt) {
        if (typeof showToast === 'function') showToast('Enter a prompt first', 'warning');
        return;
    }

    await runOrchestratorPhase(async (orchestrator) => {
        // Load context first if not already loaded
        if (OptimizerState.loadedFiles.length > 0) {
            await orchestrator.loadModel(OptimizerState.loadedFiles);
        } else {
            // Try to restore from storage
            const cm = orchestrator.getConversationManager();
            const restored = cm ? await cm.restore() : false;
            if (!restored) {
                throw new Error('No context loaded. Load context files first.');
            }
        }

        const result = await orchestrator.convertPrompt(userPrompt);
        await refreshVersions();
        updateButtonStates();

        if (typeof showToast === 'function') {
            showToast(`v${result.version.versionNum} created`, 'success');
        }
    });
}

async function onTest() {
    await runOrchestratorPhase(async (orchestrator) => {
        const vm = orchestrator.getVersionManager();
        const latest = await vm.getLatestVersion();
        if (!latest) throw new Error('No versions to test. Convert a prompt first.');

        const result = await orchestrator.testPrompt(latest.versionNum);

        // Auto-extract reasoning
        const extraction = await orchestrator.extractReasoning(result.output, latest.versionNum);
        if (extraction.thinking) {
            showThinkingLog(extraction.thinking, latest.versionNum);
        }

        if (typeof showToast === 'function') {
            showToast('Test complete', 'success');
        }
    });
}

function onUpdate() {
    // Show feedback section (user feedback mechanism)
    const feedbackSection = document.getElementById('optimizerFeedbackSection');
    if (feedbackSection) feedbackSection.style.display = '';
}

async function onApplyFeedback() {
    const feedbackInput = document.getElementById('optimizerFeedbackInput');
    const feedback = feedbackInput?.value?.trim();
    if (!feedback) {
        if (typeof showToast === 'function') showToast('Enter feedback first', 'warning');
        return;
    }

    await runOrchestratorPhase(async (orchestrator) => {
        const vm = orchestrator.getVersionManager();
        const latest = await vm.getLatestVersion();
        if (!latest) throw new Error('No versions to update from.');

        await orchestrator.updatePrompt(feedback, latest.versionNum);
        await refreshVersions();
        updateButtonStates();

        // Hide feedback section
        const feedbackSection = document.getElementById('optimizerFeedbackSection');
        if (feedbackSection) feedbackSection.style.display = 'none';
        if (feedbackInput) feedbackInput.value = '';

        if (typeof showToast === 'function') {
            showToast('Updated successfully', 'success');
        }
    });
}

async function onFullLoop() {
    const promptInput = document.getElementById('optimizerPromptInput');
    const userPrompt = promptInput?.value?.trim();
    if (!userPrompt) {
        if (typeof showToast === 'function') showToast('Enter a prompt first', 'warning');
        return;
    }

    await runOrchestratorPhase(async (orchestrator) => {
        await orchestrator.execute({
            contextFiles: OptimizerState.loadedFiles,
            userPrompt,
            skipContextLoading: OptimizerState.loadedFiles.length === 0,
            runFullLoop: true,
        });

        await refreshVersions();
        updateButtonStates();

        if (typeof showToast === 'function') {
            showToast('Full loop complete!', 'success');
        }
    });
}

// ── Optimization Loop Start/Stop ──────────────────────────────────────────

/**
 * Start the optimization loop (called from sidebar or dashboard button).
 */
async function onStartOptimization() {
    if (OptimizerState.isRunning) {
        if (typeof showToast === 'function') showToast('Already running', 'warning');
        return;
    }

    try {
        OptimizerState.isRunning = true;
        updateRunningUI(true);
        updateLoopButtonVisibility(true);
        dashShowRunningState(true);
        hideOptimizerError();
        updateDashboardStatus('running', 'Starting optimization...');

        const orchestrator = await getOrchestrator();

        // Ensure we have a UCL prompt (either from existing version or user input)
        if (!orchestrator.uclPrompt) {
            const vm = orchestrator.getVersionManager();
            if (vm) {
                const latest = await vm.getLatestVersion();
                if (latest) {
                    orchestrator.uclPrompt = latest.promptText;
                }
            }
        }

        if (!orchestrator.uclPrompt) {
            throw new Error('No UCL prompt available. Convert a prompt first.');
        }

        // Update iteration config from UI inputs
        orchestrator.iterationConfig = {
            ...orchestrator.iterationConfig,
            ...OptimizerState.iterationConfig,
        };

        // Sync quality threshold to DashboardState for chart threshold line
        DashboardState.qualityThreshold = OptimizerState.iterationConfig.qualityThreshold;

        // Run the optimization loop
        const result = await orchestrator.runOptimizationLoop({
            testInput: null,
        });

        // Final dashboard update
        updateDashboardComplete(result);
        await refreshVersions();
        updateButtonStates();

        if (typeof showToast === 'function') {
            showToast(`Optimization complete! Best: ${(result.bestQualityScore * 100).toFixed(1)}%`, 'success');
        }

    } catch (error) {
        console.error('[OptimizerUI] Optimization loop failed:', error);
        showOptimizerError(error.message);
        updateDashboardStatus('error', error.message);
        if (typeof showToast === 'function') showToast('Optimization failed: ' + error.message, 'error');
    } finally {
        OptimizerState.isRunning = false;
        updateRunningUI(false);
        updateLoopButtonVisibility(false);
        dashShowRunningState(false);
    }
}

/**
 * Stop the optimization loop (called from sidebar or dashboard button).
 */
function onStopOptimization() {
    if (!OptimizerState.isRunning) return;

    if (OptimizerState.orchestrator) {
        OptimizerState.orchestrator.cancelOptimization();
    }

    OptimizerState.isRunning = false;
    updateRunningUI(false);
    updateLoopButtonVisibility(false);
    dashShowRunningState(false);
    updateDashboardStatus('idle', 'Cancelled');

    if (typeof showToast === 'function') showToast('Optimization stopped', 'info');
}

/**
 * Toggle visibility of start/stop loop buttons.
 * @param {boolean} isRunning
 */
function updateLoopButtonVisibility(isRunning) {
    const startBtn = document.getElementById('optimizerStartLoopBtn');
    const stopBtn = document.getElementById('optimizerStopLoopBtn');
    if (startBtn) startBtn.style.display = isRunning ? 'none' : '';
    if (stopBtn) stopBtn.style.display = isRunning ? '' : 'none';
}

function onReset() {
    if (OptimizerState.orchestrator) {
        OptimizerState.orchestrator.reset().catch(err => {
            console.warn('[OptimizerUI] Reset error:', err.message);
        });
    }
    OptimizerState.loadedFiles = [];
    OptimizerState.versions = [];

    // Reset UI
    const fileList = document.getElementById('optimizerFileList');
    if (fileList) fileList.style.display = 'none';

    const convoPreview = document.getElementById('optimizerConvoPreview');
    if (convoPreview) convoPreview.style.display = 'none';

    const feedbackSection = document.getElementById('optimizerFeedbackSection');
    if (feedbackSection) feedbackSection.style.display = 'none';

    const thinkingSection = document.getElementById('optimizerThinkingSection');
    if (thinkingSection) thinkingSection.style.display = 'none';

    const progress = document.getElementById('optimizerProgress');
    if (progress) progress.style.display = 'none';

    const phases = document.getElementById('optimizerPhases');
    if (phases) phases.style.display = 'none';

    updateButtonStates();
    updateVersionListUI([]);
    updateOptimizerPhaseUI('IDLE');

    if (typeof showToast === 'function') showToast('Optimizer reset', 'info');
}

// ── Template Management (CRITICAL FIX #2) ─────────────────────────────────

let _currentTemplateTab = 'convert';

async function onTemplateTabSwitch(tab) {
    _currentTemplateTab = tab;
    document.querySelectorAll('.optimizer-tmpl-tab').forEach(t => {
        t.classList.toggle('active', t.dataset.tmpl === tab);
    });
    await loadTemplateForEdit(tab);
}

async function loadTemplateForEdit(type) {
    const editor = document.getElementById('optimizerTemplateEditor');
    if (!editor || !OptimizerState.orchestrator) return;

    const storage = OptimizerState.orchestrator.getStorageManager();
    if (!storage) return;

    const templateId = type === 'convert' ? 'base_convert' : 'base_update';
    const template = await storage.getTemplate(templateId, OptimizerState.currentModelId);
    editor.value = template?.content || '';
}

async function onSaveTemplate() {
    const editor = document.getElementById('optimizerTemplateEditor');
    if (!editor || !OptimizerState.orchestrator) return;

    const storage = OptimizerState.orchestrator.getStorageManager();
    if (!storage) return;

    const templateId = _currentTemplateTab === 'convert' ? 'base_convert' : 'base_update';

    // Save as model-specific override
    await storage.saveTemplate({
        templateId: `${OptimizerState.currentModelId}:${_currentTemplateTab}`,
        name: `${OptimizerState.currentModelId} - ${_currentTemplateTab}`,
        content: editor.value,
        scope: 'model',
        modelId: OptimizerState.currentModelId,
    });

    if (typeof showToast === 'function') showToast('Template saved', 'success');
}

async function onResetTemplate() {
    const editor = document.getElementById('optimizerTemplateEditor');
    if (!editor || !OptimizerState.orchestrator) return;

    const storage = OptimizerState.orchestrator.getStorageManager();
    if (!storage) return;

    const templateId = _currentTemplateTab === 'convert' ? 'base_convert' : 'base_update';
    const template = await storage.getTemplate(templateId); // Global only
    editor.value = template?.content || '';

    if (typeof showToast === 'function') showToast('Template reset to default', 'info');
}

// ── Session Management ────────────────────────────────────────────────────

async function onExportSession() {
    if (!OptimizerState.orchestrator) return;
    const storage = OptimizerState.orchestrator.getStorageManager();
    if (!storage) return;

    try {
        const data = await storage.exportSession(OptimizerState.currentModelId);
        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `optimizer-${OptimizerState.currentModelId}-${Date.now()}.json`;
        a.click();
        URL.revokeObjectURL(url);
        if (typeof showToast === 'function') showToast('Session exported', 'success');
    } catch (error) {
        showOptimizerError(error.message);
    }
}

async function onDeleteSession() {
    if (!confirm(`Delete all data for ${OptimizerState.currentModelId}?`)) return;

    if (!OptimizerState.orchestrator) return;
    const storage = OptimizerState.orchestrator.getStorageManager();
    if (!storage) return;

    await storage.deleteSession(OptimizerState.currentModelId);
    onReset();
    if (typeof showToast === 'function') showToast('Session deleted', 'info');
}

// ── Orchestrator Helpers ──────────────────────────────────────────────────

/**
 * Get or create the orchestrator instance.
 * @param {Object} [opts]
 * @param {boolean} [opts.forceLemonade] - Always use lemonade endpoint (Context tab)
 * @param {string} [opts.modelId] - Override model id
 */
async function getOrchestrator(opts = {}) {
    if (opts.modelId) {
        OptimizerState.currentModelId = opts.modelId;
    }
    if (!OptimizerState.currentModelId) {
        throw new Error('Select a model first');
    }

    const endpoint = opts.forceLemonade
        ? getLemonadeEndpoint()
        : ((typeof AppState !== 'undefined' && (AppState.lemonadeEndpoint || AppState.endpoint))
            ? (AppState.lemonadeEndpoint || AppState.endpoint)
            : getLemonadeEndpoint());

    // Recreate if missing, model changed, or endpoint changed
    const existing = OptimizerState.orchestrator;
    const needNew = !existing
        || existing.modelId !== OptimizerState.currentModelId
        || (existing._llmClient && existing._llmClient.endpoint !== endpoint.replace(/\/$/, '')
            && existing._llmClient.endpoint !== endpoint);

    if (needNew) {
        if (existing) {
            try { existing.dispose(); } catch (_) { /* ignore */ }
        }
        const { ModelOptimizerOrchestrator } = await import('./orchestrator.js');

        OptimizerState.orchestrator = new ModelOptimizerOrchestrator({
            modelId: OptimizerState.currentModelId,
            endpoint,
            model: OptimizerState.currentModelId,
            apiKey: (typeof AppState !== 'undefined' ? (AppState.lemonadeApiKey || AppState.apiKey) : '') || '',
            optimization: {
                ...OptimizerState.iterationConfig,
                convergencePatience: 2,
                tokenBudget: 0,
            },
            onStateChange: (newState, prevState, event) => {
                console.log(`[OptimizerUI] ${prevState} → ${newState} (${event})`);
                updateOptimizerPhaseUI(newState);
            },
            onProgress: (phase, step, total, detail) => {
                updatePhaseProgressUI(phase, step, total, detail);
            },
            onError: (phase, error) => {
                showOptimizerError(`[${phase}] ${error.message}`);
            },
            onVersionCreated: (version) => {
                refreshVersions();
            },
            onIteration: (result) => {
                console.log(`[OptimizerUI] Iteration ${result.iterationNum} complete, score: ${result.qualityScore?.composite}`);
                updateDashboardIteration(result);
            },
            onOptimizationComplete: (result) => {
                console.log(`[OptimizerUI] Optimization complete: ${result.stopReason}`);
                updateDashboardComplete(result);
                OptimizerState.isRunning = false;
                updateRunningUI(false);
                updateLoopButtonVisibility(false);
                dashShowRunningState(false);
            },
        });
    }

    return OptimizerState.orchestrator;
}

/**
 * Run an orchestrator phase with proper UI state management.
 */
async function runOrchestratorPhase(phaseFn) {
    if (OptimizerState.isRunning) {
        if (typeof showToast === 'function') showToast('Already running', 'warning');
        return;
    }

    try {
        OptimizerState.isRunning = true;
        updateRunningUI(true);
        hideOptimizerError();

        const orchestrator = await getOrchestrator();
        await phaseFn(orchestrator);

    } catch (error) {
        console.error('[OptimizerUI] Phase failed:', error);
        showOptimizerError(error.message);
        if (typeof showToast === 'function') showToast('Operation failed: ' + error.message, 'error');
    } finally {
        OptimizerState.isRunning = false;
        updateRunningUI(false);
    }
}

// ── UI Update Helpers ─────────────────────────────────────────────────────

function updateFileListUI(files) {
    const fileList = document.getElementById('optimizerFileList');
    const convoPreview = document.getElementById('optimizerConvoPreview');
    if (!fileList) return;

    const totalTokens = FileLoader.estimateTotalTokens(files);

    fileList.innerHTML = files.map(f => `
        <div class="optimizer-file-item">
            <span class="file-icon">&#x2705;</span>
            <span class="file-name">${escapeHtml(f.name)}</span>
            <span class="file-size">${escapeHtml(FileLoader.formatFileSize(f.size))}</span>
        </div>
    `).join('') + `
        <div class="optimizer-file-summary">
            ${files.length} files &middot; ~${escapeHtml(totalTokens.toLocaleString())} tokens
        </div>
    `;
    fileList.style.display = '';

    if (convoPreview) convoPreview.style.display = '';
}

async function refreshVersions() {
    if (!OptimizerState.orchestrator) return;
    const vm = OptimizerState.orchestrator.getVersionManager();
    if (!vm) return;

    const versions = await vm.getAllVersions();
    OptimizerState.versions = versions;
    updateVersionListUI(versions);
    updateButtonStates();
}

function updateVersionListUI(versions) {
    const list = document.getElementById('optimizerVersionList');
    if (!list) return;

    if (versions.length === 0) {
        list.innerHTML = '<div class="optimizer-empty">No versions yet</div>';
        return;
    }

    // Show newest first
    const sorted = [...versions].sort((a, b) => b.versionNum - a.versionNum);

    list.innerHTML = sorted.map((v, i) => {
        const timeAgo = formatTimeAgo(v.timestamp);
        const isCurrent = i === 0;
        return `
            <div class="optimizer-version-item ${isCurrent ? 'current' : ''}" data-version="${v.versionNum}">
                <span class="version-label">v${v.versionNum}${isCurrent ? ' &larr; current' : ''}</span>
                <span class="version-time">${timeAgo}</span>
                <button class="optimizer-btn optimizer-btn-sm" data-view-version="${v.versionNum}">View</button>
                <button class="optimizer-btn optimizer-btn-sm" data-view-thinking="${v.versionNum}" title="View thinking log">&#x1F4AD;</button>
            </div>
        `;
    }).join('');

    // Attach click handlers for version buttons
    list.querySelectorAll('[data-view-version]').forEach(btn => {
        btn.addEventListener('click', () => viewVersion(parseInt(btn.dataset.viewVersion)));
    });
    list.querySelectorAll('[data-view-thinking]').forEach(btn => {
        btn.addEventListener('click', () => viewThinkingForVersion(parseInt(btn.dataset.viewThinking)));
    });
}

async function viewVersion(versionNum) {
    if (!OptimizerState.orchestrator) return;
    const vm = OptimizerState.orchestrator.getVersionManager();
    if (!vm) return;

    const version = await vm.getVersion(versionNum);
    if (!version) return;

    // Show in output area if available
    const outputPrompt = document.getElementById('outputPrompt');
    if (outputPrompt) {
        outputPrompt.value = version.promptText;
        outputPrompt.dispatchEvent(new Event('input'));
        if (typeof showToast === 'function') showToast(`v${versionNum} loaded into editor`, 'success');
    }
}

async function viewThinkingForVersion(versionNum) {
    if (!OptimizerState.orchestrator) return;
    const vm = OptimizerState.orchestrator.getVersionManager();
    if (!vm) return;

    const thinking = await vm.getThinkingLogForVersion(versionNum);
    if (thinking) {
        showThinkingLog(thinking, versionNum);
    } else {
        if (typeof showToast === 'function') showToast(`No thinking log for v${versionNum}`, 'info');
    }
}

function showThinkingLog(thinking, versionNum) {
    const section = document.getElementById('optimizerThinkingSection');
    const log = document.getElementById('optimizerThinkingLog');
    const label = document.getElementById('optimizerThinkingLabel');

    if (section) section.style.display = '';
    if (log) log.textContent = thinking;
    if (label) label.textContent = `Thinking Log (v${versionNum})`;
}

function updateButtonStates() {
    const hasFiles = OptimizerState.loadedFiles.length > 0 || ContextState.files.length > 0;
    const hasConvo = ContextState.responses.length > 0;
    const hasVersions = OptimizerState.versions.length > 0;
    const hasModel = !!OptimizerState.currentModelId;
    const running = OptimizerState.isRunning;

    // Legacy optimizer action buttons (may not exist after tab split)
    setDisabled('optimizerConvertBtn', (!hasFiles && !hasConvo) || running);
    setDisabled('optimizerTestBtn', !hasVersions || running);
    setDisabled('optimizerUpdateBtn', !hasVersions || running);
    setDisabled('optimizerFullLoopBtn', (!hasFiles && !hasConvo) || running);

    // Convert tab buttons
    setDisabled('convertBtn', !hasModel || running);
    setDisabled('convertTestBtn', !hasVersions || running);
    setDisabled('convertUpdateBtn', !hasVersions || running);

    // Start loop needs model + a converted UCL version (from Convert tab)
    setDisabled('optimizerStartLoopBtn', !hasVersions || running || !hasModel);

    updateOptimizerPipelineStatus();
}

function setDisabled(id, disabled) {
    const el = document.getElementById(id);
    if (el) el.disabled = disabled;
}

function updateRunningUI(isRunning) {
    updateButtonStates();

    const progress = document.getElementById('optimizerProgress');
    if (progress) progress.style.display = isRunning ? '' : 'none';

    const phases = document.getElementById('optimizerPhases');
    if (phases) phases.style.display = isRunning ? '' : 'none';
}

function updateOptimizerPhaseUI(state) {
    const statusMap = {
        'IDLE':            { icon: '\u26AA', text: 'Ready' },
        'LOADING_CONTEXT': { icon: '\uD83D\uDCC2', text: 'Loading context...' },
        'CONVERTING':      { icon: '\uD83D\uDCDD', text: 'Converting...' },
        'TESTING':         { icon: '\uD83E\uDDEA', text: 'Testing...' },
        'EXTRACTING':      { icon: '\uD83D\uDCA1', text: 'Extracting reasoning...' },
        'UPDATING':        { icon: '\uD83D\uDD04', text: 'Updating...' },
        'COMPLETE':        { icon: '\u2705', text: 'Complete' },
        'ERROR':           { icon: '\u274C', text: 'Error' },
    };

    const status = statusMap[state] || { icon: '\u26AA', text: state };
    const indicator = document.getElementById('optimizerStatusIndicator');
    const text = document.getElementById('optimizerStatusText');

    if (indicator) indicator.textContent = status.icon;
    if (text) text.textContent = status.text;

    // Update phase items
    const phaseMap = {
        'LOADING_CONTEXT': 'optimizerPhaseLoading',
        'CONVERTING': 'optimizerPhaseConverting',
        'TESTING': 'optimizerPhaseTesting',
        'EXTRACTING': 'optimizerPhaseExtracting',
        'UPDATING': 'optimizerPhaseUpdating',
    };

    // Reset all phases
    document.querySelectorAll('.optimizer-phase-item').forEach(el => {
        el.className = 'optimizer-phase-item';
    });

    // Mark completed and active phases
    const phaseOrder = ['LOADING_CONTEXT', 'CONVERTING', 'TESTING', 'EXTRACTING', 'UPDATING'];
    const currentIdx = phaseOrder.indexOf(state);

    for (let i = 0; i < phaseOrder.length; i++) {
        const elId = phaseMap[phaseOrder[i]];
        const el = document.getElementById(elId);
        if (!el) continue;

        if (state === 'COMPLETE') {
            el.className = 'optimizer-phase-item complete';
        } else if (i < currentIdx) {
            el.className = 'optimizer-phase-item complete';
        } else if (i === currentIdx) {
            el.className = 'optimizer-phase-item active';
        }
    }
}

function updatePhaseProgressUI(phase, step, total, detail) {
    const progressBar = document.getElementById('optimizerProgressBar');
    const progressText = document.getElementById('optimizerProgressText');

    const pct = total > 0 ? Math.round((step / total) * 100) : 0;
    if (progressBar) progressBar.style.width = `${pct}%`;
    if (progressText) progressText.textContent = `${pct}%`;

    // Update phase status text
    const phaseStatusMap = {
        'LOADING_CONTEXT': 'optimizerPhaseLoadingStatus',
        'CONVERTING': 'optimizerPhaseConvertingStatus',
        'TESTING': 'optimizerPhaseTestingStatus',
        'EXTRACTING': 'optimizerPhaseExtractingStatus',
        'UPDATING': 'optimizerPhaseUpdatingStatus',
    };

    const statusEl = document.getElementById(phaseStatusMap[phase]);
    if (statusEl) statusEl.textContent = detail || `Step ${step}/${total}`;
}

function showOptimizerError(message) {
    const error = document.getElementById('optimizerError');
    const errorMsg = document.getElementById('optimizerErrorMessage');
    if (errorMsg) errorMsg.textContent = message;
    if (error) error.style.display = '';
}

function hideOptimizerError() {
    const error = document.getElementById('optimizerError');
    if (error) error.style.display = 'none';
}

// ── Utilities ─────────────────────────────────────────────────────────────

function formatTimeAgo(timestamp) {
    const seconds = Math.floor((Date.now() - timestamp) / 1000);
    if (seconds < 60) return 'just now';
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
    if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
    return `${Math.floor(seconds / 86400)}d ago`;
}

/**
 * Dispose all optimizer resources.
 * QUALITY FIX: Cleanup long-lived resources.
 */
function disposeOptimizerUI() {
    if (OptimizerState.orchestrator) {
        OptimizerState.orchestrator.dispose();
        OptimizerState.orchestrator = null;
    }
    OptimizerState.storageManager = null;
    OptimizerState.initialized = false;
    destroyOptimizerDashboard();
}

// ── Exports ────────────────────────────────────────────────────────────────
export {
    initOptimizerUI,
    disposeOptimizerUI,
    OptimizerState,
    onStartOptimization,
    onStopOptimization,
};

// Make start/stop available globally for dashboard button handlers
if (typeof window !== 'undefined') {
    window.onStartOptimization = onStartOptimization;
    window.onStopOptimization = onStopOptimization;
}
