// ============================================================================
// UCL TOOL UI — STATE MANAGEMENT, RENDERING & INTERACTION
// Handles the Tools tab in the sidebar: domain selection, preview, metrics
// ============================================================================

// ============================================================================
// TOOL STATE
// ============================================================================
const ToolState = {
    selectedDomains: [],       // Array of domain keys: ['filesystem', 'web', ...]
    includePlaceholder: false, // {{UCL_TOOLS}} toggle
    expandedDomains: [],       // Domain keys with expanded tool lists
    searchQuery: '',           // Filter tools by search
    initialized: false
};

const ToolImportState = {
    last: null,
    busy: false
};

// ============================================================================
// DATA ACCESS HELPERS (ToolLoader with fallback to hardcoded data)
// ============================================================================
function getToolDomains() {
    // Use ToolLoader if loaded, otherwise fall back to hardcoded data
    return ToolLoader.isLoaded() ? ToolLoader.getAllDomains() : UCL_TOOL_DOMAINS;
}

function getDomainKeys() {
    return ToolLoader.isLoaded() ? ToolLoader.getDomainNames() : Object.keys(UCL_TOOL_DOMAINS);
}

function getDomain(domainKey) {
    return ToolLoader.isLoaded() ? ToolLoader.getDomain(domainKey) : UCL_TOOL_DOMAINS[domainKey];
}

function getDomainToolCount(domainKey) {
    if (ToolLoader.isLoaded()) {
        return ToolLoader.getDomainToolCount(domainKey);
    }
    const domain = UCL_TOOL_DOMAINS[domainKey];
    return domain ? Object.keys(domain.tools).length : 0;
}

function getTotalToolCount() {
    return ToolLoader.isLoaded() ? ToolLoader.getTotalToolCount() : UCL_TOOLS_METADATA.totalTools;
}

function getTotalDomainCount() {
    return ToolLoader.isLoaded() ? ToolLoader.getDomainNames().length : UCL_TOOLS_METADATA.totalDomains;
}

function getSelectedToolCount(selectedDomains) {
    return selectedDomains.reduce((sum, key) => sum + getDomainToolCount(key), 0);
}

// ============================================================================
// INITIALIZATION
// ============================================================================
async function initToolsUI() {
    loadToolSettings();
    renderSidebarTabs();

    // Try dynamic loading first, fall back to hardcoded data
    try {
        await ToolLoader.loadAll();
        console.log(`✓ Loaded ${ToolLoader.getTotalToolCount()} tools dynamically from tools/ folder`);
    } catch (error) {
        console.warn('Dynamic tool loading failed, using fallback data:', error.message);
    }

    renderToolPanel();
    updateToolMetrics();
    attachToolEventListeners();
    ToolState.initialized = true;

    const totalTools = ToolLoader.isLoaded() ? ToolLoader.getTotalToolCount() : UCL_TOOLS_METADATA.totalTools;
    const totalDomains = ToolLoader.isLoaded() ? ToolLoader.getDomainNames().length : UCL_TOOLS_METADATA.totalDomains;
    console.log('✓ Tool UI initialized (' + totalTools + ' tools across ' + totalDomains + ' domains)');
}

// ============================================================================
// SETTINGS PERSISTENCE
// ============================================================================
function loadToolSettings() {
    try {
        const saved = localStorage.getItem('ucl_tool_domains');
        if (saved) {
            ToolState.selectedDomains = JSON.parse(saved);
        }
        ToolState.includePlaceholder = localStorage.getItem('ucl_tool_placeholder') === 'true';
    } catch (e) {
        console.error('Error loading tool settings:', e);
        ToolState.selectedDomains = [];
        ToolState.includePlaceholder = false;
    }
}

function saveToolSettings() {
    try {
        localStorage.setItem('ucl_tool_domains', JSON.stringify(ToolState.selectedDomains));
        localStorage.setItem('ucl_tool_placeholder', ToolState.includePlaceholder);
    } catch (e) {
        console.error('Error saving tool settings:', e);
    }
}

// ============================================================================
// SIDEBAR TAB SYSTEM
// ============================================================================
function renderSidebarTabs() {
    const sidebar = document.getElementById('sidebar');
    if (!sidebar) return;

    // Find the sidebar-header and insert tabs after it
    const sidebarHeader = sidebar.querySelector('.sidebar-header');
    if (!sidebarHeader) return;

    // Remove existing tabs if any
    const existingTabs = sidebar.querySelector('.sidebar-tabs');
    if (existingTabs) existingTabs.remove();

    // Tab definitions
    const TABS = [
        { id: 'syntax',    icon: '⚡', label: 'UCL Syntax' },
        { id: 'tools',     icon: '🔧', label: 'Tools', badge: 'toolTabBadge' },
        { id: 'pipeline',  icon: '🔀', label: 'Pipeline' },
        { id: 'convert',   icon: '🔄', label: 'Convert' },
        { id: 'optimizer', icon: '🎯', label: 'Optimizer' },
        { id: 'compress',  icon: '📉', label: 'Compress' },
        { id: 'context',   icon: '📁', label: 'Context' },
    ];

    // Create tab bar with all tabs + scroll indicator
    const tabBar = document.createElement('div');
    tabBar.className = 'sidebar-tabs';

    // Build tab buttons
    const btnRow = document.createElement('div');
    btnRow.className = 'sidebar-tab-btn-row';
    TABS.forEach((t, i) => {
        const btn = document.createElement('button');
        btn.className = 'sidebar-tab' + (i === 0 ? ' active' : '');
        btn.dataset.tab = t.id;
        btn.innerHTML = `
            <span class="tab-icon">${t.icon}</span>
            <span class="tab-label">${t.label}</span>
            ${t.badge ? `<span class="tab-badge" id="${t.badge}">0</span>` : ''}
        `;
        btnRow.appendChild(btn);
    });

    // Build scroll indicator track
    const track = document.createElement('div');
    track.className = 'sidebar-tab-scroll-track';
    track.setAttribute('aria-hidden', 'true');

    const dotsRow = document.createElement('div');
    dotsRow.className = 'sidebar-tab-scroll-dots';
    TABS.forEach(t => {
        const dot = document.createElement('span');
        dot.className = 'scroll-dot' + (t.id === 'syntax' ? ' active' : '');
        dot.dataset.dot = t.id;
        dotsRow.appendChild(dot);
    });

    const pill = document.createElement('div');
    pill.className = 'sidebar-tab-scroll-pill';

    const label = document.createElement('div');
    label.className = 'sidebar-tab-scroll-label';
    label.textContent = TABS[0].icon;

    track.appendChild(dotsRow);
    track.appendChild(pill);
    track.appendChild(label);

    tabBar.appendChild(btnRow);
    tabBar.appendChild(track);

    // Insert after sidebar-header
    sidebarHeader.after(tabBar);

    // Tab click handlers
    tabBar.querySelectorAll('.sidebar-tab').forEach(tab => {
        tab.addEventListener('click', () => switchSidebarTab(tab.dataset.tab));
    });

    // Scroll indicator dot clicks
    tabBar.querySelectorAll('.scroll-dot').forEach(dot => {
        dot.addEventListener('click', () => switchSidebarTab(dot.dataset.dot));
    });

    // Scroll wheel to switch tabs + animate indicator
    tabBar.addEventListener('wheel', (e) => {
        e.preventDefault();
        const currentTab = tabBar.querySelector('.sidebar-tab.active');
        if (!currentTab) return;
        const tabOrder = TABS.map(t => t.id);
        const currentIndex = tabOrder.indexOf(currentTab.dataset.tab);
        if (currentIndex === -1) return;
        const direction = e.deltaY > 0 ? 1 : -1;
        const newIndex = Math.max(0, Math.min(tabOrder.length - 1, currentIndex + direction));
        if (newIndex !== currentIndex) {
            switchSidebarTab(tabOrder[newIndex]);
        }
    }, { passive: false });

    // Initialize pill position
    _updateScrollIndicator('syntax');
}

/**
 * Animate the scroll indicator pill to match the active tab.
 */
function _updateScrollIndicator(tabName) {
    const TABS = [
        { id: 'syntax',    icon: '⚡' },
        { id: 'tools',     icon: '🔧' },
        { id: 'pipeline',  icon: '🔀' },
        { id: 'convert',   icon: '🔄' },
        { id: 'optimizer', icon: '🎯' },
        { id: 'context',   icon: '📁' },
    ];
    const index = TABS.findIndex(t => t.id === tabName);
    if (index === -1) return;

    const track = document.querySelector('.sidebar-tab-scroll-track');
    const pill = track?.querySelector('.sidebar-tab-scroll-pill');
    const label = track?.querySelector('.sidebar-tab-scroll-label');
    const dots = track?.querySelectorAll('.scroll-dot');

    if (!pill || !track) return;

    // Each segment = 1/6 of track width
    const segmentWidth = 100 / TABS.length;
    const pillLeft = index * segmentWidth;
    pill.style.left = pillLeft + '%';
    pill.style.width = segmentWidth + '%';

    // Update label icon
    if (label) {
        label.textContent = TABS[index].icon;
        label.style.left = (pillLeft + segmentWidth / 2) + '%';
    }

    // Update dot active states
    if (dots) {
        dots.forEach(dot => {
            dot.classList.toggle('active', dot.dataset.dot === tabName);
        });
    }

    // Scroll the active tab button into view
    const activeBtn = document.querySelector(`.sidebar-tab[data-tab="${tabName}"]`);
    if (activeBtn) {
        activeBtn.scrollIntoView({ behavior: 'smooth', block: 'nearest', inline: 'center' });
    }
}

function switchSidebarTab(tabName) {
    // Update tab buttons
    document.querySelectorAll('.sidebar-tab').forEach(tab => {
        tab.classList.toggle('active', tab.dataset.tab === tabName);
    });

    // Show/hide sidebar content panels for all tabs
    const panels = {
        syntax:    '.sidebar-syntax-panel',
        tools:     '.sidebar-tools-panel',
        pipeline:  '.sidebar-pipeline-panel',
        convert:   '.sidebar-convert-panel',
        optimizer: '.sidebar-optimizer-panel',
        context:   '.sidebar-context-panel',
    };

    Object.entries(panels).forEach(([tab, selector]) => {
        const panel = document.querySelector(selector);
        if (panel) panel.style.display = tabName === tab ? '' : 'none';
    });

    // Update search placeholder based on active tab
    const searchPlaceholders = {
        syntax: 'Search syntax...',
        tools: 'Search tools...',
        pipeline: 'Search pipeline...',
        convert: 'Search convert options...',
        optimizer: 'Search optimizer...',
        compress: 'Name list compressor...',
        context: 'Search context files...'
    };
    const searchInput = document.getElementById('syntaxSearch');
    if (searchInput) {
        searchInput.placeholder = searchPlaceholders[tabName] || 'Search...';
        searchInput.value = '';
        ToolState.searchQuery = '';
    }

    // Toggle editor area based on active tab
    const editorHeader = document.querySelector('.editor-header');
    const editorPanes = document.querySelector('.editor-panes');
    const actionBar = document.querySelector('.action-bar');
    let dashboard = document.querySelector('.optimizer-dashboard');
    let contextView = document.querySelector('.context-conversation');
    let compressorFrame = document.querySelector('.ucl-compressor-frame');

    if (tabName === 'optimizer') {
        // Show optimizer dashboard in editor area
        if (editorHeader) editorHeader.style.display = 'none';
        if (editorPanes) editorPanes.style.display = 'none';
        if (actionBar) actionBar.style.display = 'none';
        if (contextView) {
            contextView.classList.remove('active');
            contextView.style.display = 'none';
        }
        if (compressorFrame) {
            compressorFrame.classList.remove('active');
        }

        if (!dashboard) {
            const editorContainer = document.querySelector('.editor-container');
            if (editorContainer) {
                dashboard = document.createElement('div');
                dashboard.className = 'optimizer-dashboard';
                editorContainer.appendChild(dashboard);
            }
        }
        if (dashboard) {
            dashboard.classList.add('active');
            dashboard.style.display = '';
            if (typeof initOptimizerDashboard === 'function') {
                initOptimizerDashboard(dashboard);
            }
        }
    } else if (tabName === 'context') {
        // Show context conversation view in editor area
        if (editorHeader) editorHeader.style.display = 'none';
        if (editorPanes) editorPanes.style.display = 'none';
        if (actionBar) actionBar.style.display = 'none';
        if (dashboard) { dashboard.classList.remove('active'); dashboard.style.display = 'none'; }
        if (compressorFrame) compressorFrame.classList.remove('active');

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
            if (typeof window.renderContextConversation === 'function') {
                window.renderContextConversation(contextView);
            }
        }
    } else if (tabName === 'compress') {
        if (editorHeader) editorHeader.style.display = 'none';
        if (editorPanes) editorPanes.style.display = 'none';
        if (actionBar) actionBar.style.display = 'none';
        if (dashboard) { dashboard.classList.remove('active'); dashboard.style.display = 'none'; }
        if (contextView) {
            contextView.classList.remove('active');
            contextView.style.display = 'none';
        }
        if (!compressorFrame) {
            const editorContainer = document.querySelector('.editor-container');
            if (editorContainer) {
                compressorFrame = document.createElement('iframe');
                compressorFrame.className = 'ucl-compressor-frame';
                compressorFrame.title = 'Name list compressor';
                compressorFrame.src = 'compressor.html';
                editorContainer.appendChild(compressorFrame);
            }
        }
        if (compressorFrame) compressorFrame.classList.add('active');
    } else {
        // Show standard editor (for convert, syntax, tools, pipeline)
        if (editorHeader) editorHeader.style.display = '';
        if (editorPanes) editorPanes.style.display = '';
        if (actionBar) actionBar.style.display = '';
        if (dashboard) { dashboard.classList.remove('active'); dashboard.style.display = 'none'; }
        if (contextView) {
            contextView.classList.remove('active');
            contextView.style.display = 'none';
        }
        if (compressorFrame) compressorFrame.classList.remove('active');
    }

    // Animate the scroll indicator pill
    if (typeof _updateScrollIndicator === 'function') {
        _updateScrollIndicator(tabName);
    }
}

// ============================================================================
// TOOL PANEL RENDERING
// ============================================================================
function renderToolPanel() {
    const sidebar = document.getElementById('sidebar');
    if (!sidebar) return;

    // Mark only syntax categories as the syntax panel (search stays visible for both tabs)
    const syntaxCategories = document.getElementById('syntaxCategories');
    if (syntaxCategories) syntaxCategories.classList.add('sidebar-syntax-panel');

    // Create tools panel
    let toolsPanel = sidebar.querySelector('.sidebar-tools-panel');
    if (!toolsPanel) {
        toolsPanel = document.createElement('div');
        toolsPanel.className = 'sidebar-tools-panel';
        toolsPanel.style.display = 'none';

        // Insert after syntax categories
        if (syntaxCategories) {
            syntaxCategories.after(toolsPanel);
        } else {
            sidebar.appendChild(toolsPanel);
        }
    }

    // Import form stays put. Catalog HTML is replaced on refresh.
    if (!toolsPanel.querySelector('.tool-import')) {
        const importHost = document.createElement('div');
        importHost.className = 'tool-import';
        importHost.innerHTML = buildToolImportHTML();
        const catalog = document.createElement('div');
        catalog.className = 'tool-catalog';
        toolsPanel.appendChild(importHost);
        toolsPanel.appendChild(catalog);
        attachToolImportListeners();
    }

    const catalog = toolsPanel.querySelector('.tool-catalog');
    if (catalog) catalog.innerHTML = buildToolPanelHTML();

    // Create pipeline panel
    let pipelinePanel = sidebar.querySelector('.sidebar-pipeline-panel');
    if (!pipelinePanel) {
        pipelinePanel = document.createElement('div');
        pipelinePanel.className = 'sidebar-pipeline-panel';
        pipelinePanel.style.display = 'none';
        pipelinePanel.innerHTML = buildPipelinePanelHTML();
        // Insert after tools panel
        if (toolsPanel) {
            toolsPanel.after(pipelinePanel);
        } else {
            sidebar.appendChild(pipelinePanel);
        }

        // Attach pipeline event listeners
        attachPipelineEventListeners();
    }

    // Optimizer UI is initialized by the ES module script in index.html
    // (avoid duplicate loading — see index.html <script type="module"> block)
}

function buildToolPanelHTML() {
    const selectedCount = ToolState.selectedDomains.length;
    const totalDomains = getTotalDomainCount();
    const totalTools = getTotalToolCount();

    return `
        <!-- Tool Metrics Bar -->
        <div class="tool-metrics-bar" id="toolMetricsBar">
            <div class="tool-metric">
                <span class="metric-value" id="metricToolCount">0</span>
                <span class="metric-label">/ ${totalTools} tools</span>
            </div>
            <div class="tool-metric">
                <span class="metric-value" id="metricOsReduction">0</span>
                <span class="metric-label">% overhead saved</span>
            </div>
            <div class="tool-metric">
                <span class="metric-value" id="metricDomains">0</span>
                <span class="metric-label">/ ${totalDomains} domains</span>
            </div>
        </div>

        <!-- Quick Actions -->
        <div class="tool-quick-actions">
            <button class="tool-quick-btn" id="toolSelectAllBtn">Select All</button>
            <button class="tool-quick-btn" id="toolClearAllBtn">Clear All</button>
            <label class="tool-placeholder-toggle">
                <input type="checkbox" id="toolPlaceholderCheck" ${ToolState.includePlaceholder ? 'checked' : ''}>
                <span>Include {{UCL_TOOLS}}</span>
            </label>
        </div>

        <!-- Domain Cards -->
        <div class="tool-domains-list" id="toolDomainsList">
            ${buildDomainCardsHTML()}
        </div>

        <!-- Tool Preview (expanded domains) -->
        <div class="tool-preview-section" id="toolPreviewSection">
            ${buildToolPreviewHTML()}
        </div>
    `;
}

// ============================================================================
// PIPELINE PANEL
// ============================================================================
function buildPipelinePanelHTML() {
    return `
        <div class="pipeline-container">
            <!-- Pipeline Header -->
            <div class="pipeline-header">
                <h3 class="pipeline-title">UCL Preparation Pipeline</h3>
                <div class="pipeline-status" id="pipelineStatus">
                    <span class="status-indicator" id="pipelineStatusIndicator">⚪</span>
                    <span class="status-text" id="pipelineStatusText">Ready</span>
                </div>
            </div>

            <!-- Source Directory Configuration -->
            <div class="pipeline-section">
                <label class="pipeline-label">Source Directory</label>
                <div class="pipeline-input-group">
                    <input type="text"
                           id="pipelineSourceDir"
                           class="pipeline-input"
                           placeholder="./src/tools"
                           value="./tools">
                    <button class="pipeline-browse-btn" id="pipelineBrowseBtn" title="Browse">📁</button>
                </div>
                <small class="pipeline-help">Directory containing TypeScript/JavaScript tool definitions</small>
            </div>

            <!-- Test Prompt -->
            <div class="pipeline-section">
                <label class="pipeline-label">Test Prompt</label>
                <textarea id="pipelineTestPrompt"
                          class="pipeline-textarea"
                          placeholder="Enter a natural language prompt to convert to UCL..."
                          rows="4">Create a system prompt for a coding assistant that can read files, write files, and execute shell commands.</textarea>
            </div>

            <!-- Pipeline Phases -->
            <div class="pipeline-phases">
                <div class="phase-item" id="phase1">
                    <div class="phase-icon">🔍</div>
                    <div class="phase-content">
                        <div class="phase-title">Phase 1: Source Discovery</div>
                        <div class="phase-status" id="phase1Status">Pending</div>
                        <div class="phase-detail" id="phase1Detail">Parse source files to discover tools</div>
                    </div>
                </div>

                <div class="phase-item" id="phase2">
                    <div class="phase-icon">🔥</div>
                    <div class="phase-content">
                        <div class="phase-title">Phase 2: Context Preloading</div>
                        <div class="phase-status" id="phase2Status">Pending</div>
                        <div class="phase-detail" id="phase2Detail">8-message warm-up sequence</div>
                    </div>
                </div>

                <div class="phase-item" id="phase3">
                    <div class="phase-icon">🧪</div>
                    <div class="phase-content">
                        <div class="phase-title">Phase 3: UCL Generation</div>
                        <div class="phase-status" id="phase3Status">Pending</div>
                        <div class="phase-detail" id="phase3Detail">Generate UCL from test prompt</div>
                    </div>
                </div>
            </div>

            <!-- Progress Bar -->
            <div class="pipeline-progress" id="pipelineProgress" style="display:none;">
                <div class="progress-bar-container">
                    <div class="progress-bar" id="pipelineProgressBar" style="width: 0%"></div>
                </div>
                <div class="progress-text" id="pipelineProgressText">0%</div>
            </div>

            <!-- Action Buttons -->
            <div class="pipeline-actions">
                <button class="btn btn-primary" id="pipelineRunBtn">
                    <span class="btn-icon">▶</span>
                    Run Pipeline
                </button>
                <button class="btn btn-secondary" id="pipelineResetBtn" disabled>
                    <span class="btn-icon">↺</span>
                    Reset
                </button>
            </div>

            <!-- Results -->
            <div class="pipeline-results" id="pipelineResults" style="display:none;">
                <h4 class="results-title">Results</h4>
                <div class="result-item">
                    <span class="result-label">Tools Discovered:</span>
                    <span class="result-value" id="resultToolsCount">0</span>
                </div>
                <div class="result-item">
                    <span class="result-label">Warm-up Tokens:</span>
                    <span class="result-value" id="resultWarmupTokens">0</span>
                </div>
                <div class="result-item">
                    <span class="result-label">UCL Output:</span>
                    <span class="result-value" id="resultUCLOutput">0 chars</span>
                </div>
                <button class="btn btn-secondary" id="pipelineViewOutputBtn">
                    <span class="btn-icon">👁</span>
                    View UCL Output
                </button>
            </div>

            <!-- Error Display -->
            <div class="pipeline-error" id="pipelineError" style="display:none;">
                <div class="error-icon">⚠️</div>
                <div class="error-message" id="pipelineErrorMessage"></div>
            </div>
        </div>
    `;
}

function attachPipelineEventListeners() {
    const runBtn = document.getElementById('pipelineRunBtn');
    const resetBtn = document.getElementById('pipelineResetBtn');
    const viewOutputBtn = document.getElementById('pipelineViewOutputBtn');

    if (runBtn) {
        runBtn.addEventListener('click', () => runPipeline());
    }

    if (resetBtn) {
        resetBtn.addEventListener('click', () => resetPipeline());
    }

    if (viewOutputBtn) {
        viewOutputBtn.addEventListener('click', () => viewPipelineOutput());
    }
}

// Pipeline state
const PipelineState = {
    orchestrator: null,
    isRunning: false,
    result: null,
};

async function runPipeline() {
    if (PipelineState.isRunning) {
        showToast('Pipeline is already running', 'warning');
        return;
    }

    const sourceDir = document.getElementById('pipelineSourceDir').value || './tools';
    const testPrompt = document.getElementById('pipelineTestPrompt').value;

    if (!testPrompt.trim()) {
        showToast('Please enter a test prompt', 'error');
        return;
    }

    // Initialize pipeline orchestrator
    try {
        const { PipelineOrchestrator } = await import('./pipeline/pipeline-orchestrator.js');

        PipelineState.orchestrator = new PipelineOrchestrator({
            sourceDir: sourceDir,
            endpoint: AppState.lemonadeEndpoint,
            model: AppState.selectedModel || 'qwen3-4b-FLM',
            apiKey: AppState.lemonadeApiKey,
            onStateChange: (newState, oldState, event) => {
                console.log(`Pipeline: ${oldState} → ${newState} (${event})`);
                updatePipelineStatus(newState);
            },
            onProgress: (phase, step, total, title) => {
                updatePhaseProgress(phase, step, total, title);
            },
            onError: (phase, error) => {
                console.error(`Pipeline error in ${phase}:`, error);
                showPipelineError(error.message);
            },
        });

        PipelineState.isRunning = true;
        updatePipelineUI(true);
        resetPhaseStatuses();

        // Execute pipeline
        const result = await PipelineState.orchestrator.execute(testPrompt);
        PipelineState.result = result;

        // Show results
        showPipelineResults(result);
        showToast('Pipeline execution complete!', 'success');

    } catch (error) {
        console.error('Pipeline execution failed:', error);
        showPipelineError(error.message);
        showToast('Pipeline failed: ' + error.message, 'error');
    } finally {
        PipelineState.isRunning = false;
        updatePipelineUI(false);
    }
}

function resetPipeline() {
    if (PipelineState.orchestrator) {
        PipelineState.orchestrator.reset();
    }
    PipelineState.result = null;
    resetPhaseStatuses();
    hidePipelineResults();
    hidePipelineError();
    updatePipelineStatus('IDLE');
    showToast('Pipeline reset', 'info');
}

function updatePipelineUI(isRunning) {
    const runBtn = document.getElementById('pipelineRunBtn');
    const resetBtn = document.getElementById('pipelineResetBtn');
    const progress = document.getElementById('pipelineProgress');

    if (runBtn) {
        runBtn.disabled = isRunning;
        runBtn.innerHTML = isRunning
            ? '<span class="btn-icon">⏳</span> Running...'
            : '<span class="btn-icon">▶</span> Run Pipeline';
    }

    if (resetBtn) {
        resetBtn.disabled = !isRunning && !PipelineState.result;
    }

    if (progress) {
        progress.style.display = isRunning ? '' : 'none';
    }
}

function updatePipelineStatus(state) {
    const indicator = document.getElementById('pipelineStatusIndicator');
    const text = document.getElementById('pipelineStatusText');

    const stateMap = {
        'IDLE': { icon: '⚪', text: 'Ready' },
        'DISCOVERING': { icon: '🔍', text: 'Discovering tools...' },
        'PRELOADING': { icon: '🔥', text: 'Preloading context...' },
        'GENERATING': { icon: '🧪', text: 'Generating UCL...' },
        'COMPLETE': { icon: '✅', text: 'Complete' },
        'ERROR': { icon: '❌', text: 'Error' },
    };

    const status = stateMap[state] || { icon: '⚪', text: state };

    if (indicator) indicator.textContent = status.icon;
    if (text) text.textContent = status.text;
}

function resetPhaseStatuses() {
    for (let i = 1; i <= 3; i++) {
        const status = document.getElementById(`phase${i}Status`);
        const detail = document.getElementById(`phase${i}Detail`);
        const phase = document.getElementById(`phase${i}`);

        if (status) status.textContent = 'Pending';
        if (detail) detail.textContent = getPhaseDefaultDetail(i);
        if (phase) phase.className = 'phase-item';
    }
}

function getPhaseDefaultDetail(phase) {
    const details = {
        1: 'Parse source files to discover tools',
        2: '8-message warm-up sequence',
        3: 'Generate UCL from test prompt',
    };
    return details[phase] || '';
}

function updatePhaseProgress(phase, step, total, title) {
    const phaseNum = phase === 'DISCOVERING' ? 1 : phase === 'PRELOADING' ? 2 : 3;
    const status = document.getElementById(`phase${phaseNum}Status`);
    const detail = document.getElementById(`phase${phaseNum}Detail`);
    const phaseEl = document.getElementById(`phase${phaseNum}`);
    const progress = document.getElementById('pipelineProgressBar');
    const progressText = document.getElementById('pipelineProgressText');

    if (status) status.textContent = `Step ${step}/${total}`;
    if (detail) detail.textContent = title;
    if (phaseEl) phaseEl.className = 'phase-item active';

    // Update overall progress
    const overallProgress = ((step - 1) / total) * 100;
    if (progress) progress.style.width = `${overallProgress}%`;
    if (progressText) progressText.textContent = `${Math.round(overallProgress)}%`;
}

function showPipelineResults(result) {
    const results = document.getElementById('pipelineResults');
    const toolsCount = document.getElementById('resultToolsCount');
    const warmupTokens = document.getElementById('resultWarmupTokens');
    const uclOutput = document.getElementById('resultUCLOutput');

    if (toolsCount) toolsCount.textContent = result.tools.length;
    if (warmupTokens) warmupTokens.textContent = result.warmUp.totalTokens;
    if (uclOutput) uclOutput.textContent = `${result.generation.outputLength} chars`;

    if (results) results.style.display = '';

    // Update progress to 100%
    const progress = document.getElementById('pipelineProgressBar');
    const progressText = document.getElementById('pipelineProgressText');
    if (progress) progress.style.width = '100%';
    if (progressText) progressText.textContent = '100%';
}

function hidePipelineResults() {
    const results = document.getElementById('pipelineResults');
    if (results) results.style.display = 'none';
}

function showPipelineError(message) {
    const error = document.getElementById('pipelineError');
    const errorMessage = document.getElementById('pipelineErrorMessage');

    if (errorMessage) errorMessage.textContent = message;
    if (error) error.style.display = '';
}

function hidePipelineError() {
    const error = document.getElementById('pipelineError');
    if (error) error.style.display = 'none';
}

function viewPipelineOutput() {
    if (!PipelineState.result || !PipelineState.result.generation) {
        showToast('No UCL output available', 'warning');
        return;
    }

    const output = PipelineState.result.generation.output;
    const outputPrompt = document.getElementById('outputPrompt');

    if (outputPrompt) {
        outputPrompt.value = output;
        outputPrompt.dispatchEvent(new Event('input'));
        showToast('UCL output loaded into editor', 'success');
    }
}

function buildDomainCardsHTML() {
    const domains = getDomainKeys();
    let html = '';

    for (const key of domains) {
        const domain = getDomain(key);
        const toolCount = getDomainToolCount(key);
        const isSelected = ToolState.selectedDomains.includes(key);
        const isExpanded = ToolState.expandedDomains.includes(key);
        const categoryLabel = domain.category === 'mcp_server' ? 'MCP' : 'Core';

        html += `
            <div class="tool-domain-card ${isSelected ? 'selected' : ''}" data-domain="${key}">
                <div class="domain-card-header" data-domain-toggle="${key}">
                    <label class="domain-checkbox-label" onclick="event.stopPropagation()">
                        <input type="checkbox"
                               class="domain-checkbox"
                               data-domain-check="${key}"
                               ${isSelected ? 'checked' : ''}>
                    </label>
                    <span class="domain-icon">${domain.icon}</span>
                    <div class="domain-info">
                        <span class="domain-name">${domain.name}</span>
                        <span class="domain-desc">${domain.description}</span>
                    </div>
                    <div class="domain-meta">
                        <span class="domain-tool-count">${toolCount}</span>
                        <span class="domain-category-badge ${domain.category}">${categoryLabel}</span>
                    </div>
                    <span class="domain-expand-toggle ${isExpanded ? 'expanded' : ''}" data-domain-expand="${key}">▶</span>
                </div>
                ${isExpanded ? buildDomainToolListHTML(key) : ''}
            </div>
        `;
    }

    return html;
}

function buildDomainToolListHTML(domainKey) {
    const domain = getDomain(domainKey);
    if (!domain) return '';

    let toolsHtml = '';
    for (const [toolKey, tool] of Object.entries(domain.tools)) {
        const paramsList = tool.parameters.map(p => {
            const req = p.required ? '<span class="param-required">*</span>' : '';
            const typeClass = `param-type-${p.type}`;
            return `<span class="tool-param ${typeClass}">${p.name}: ${p.type}${req}</span>`;
        }).join('');

        const keywordsHtml = tool.keywords.map(k =>
            `<span class="tool-keyword-tag">${k}</span>`
        ).join('');

        toolsHtml += `
            <div class="tool-card" data-tool="${toolKey}" data-domain="${domainKey}">
                <div class="tool-card-name">${tool.name}</div>
                <div class="tool-card-desc">${tool.description}</div>
                <div class="tool-card-params">${paramsList || '<span class="no-params">No parameters</span>'}</div>
                <div class="tool-card-keywords">${keywordsHtml}</div>
                <button class="tool-insert-btn" data-tool-insert="${toolKey}" data-tool-domain="${domainKey}" title="Insert tool syntax">
                    Insert ⚡
                </button>
            </div>
        `;
    }

    return `<div class="domain-tool-list">${toolsHtml}</div>`;
}

function buildToolPreviewHTML() {
    // Show tools from selected domains that are expanded
    // This is handled dynamically by buildDomainToolListHTML when domains expand
    return '';
}

// ============================================================================
// METRICS CALCULATION
// ============================================================================
function updateToolMetrics() {
    const selectedCount = ToolState.selectedDomains.length;
    const toolCount = getSelectedToolCount(ToolState.selectedDomains);
    const totalTools = getTotalToolCount();

    // Overhead reduction scales with tool count (grouping savings)
    // Base: 78% for all tools, proportional for subsets
    const osReduction = totalTools > 0 ? Math.round((toolCount / totalTools) * UCL_TOOLS_METADATA.overheadReduction) : 0;

    // Update DOM
    const metricToolCount = document.getElementById('metricToolCount');
    const metricOsReduction = document.getElementById('metricOsReduction');
    const metricDomains = document.getElementById('metricDomains');
    const toolTabBadge = document.getElementById('toolTabBadge');

    if (metricToolCount) metricToolCount.textContent = toolCount;
    if (metricOsReduction) metricOsReduction.textContent = osReduction;
    if (metricDomains) metricDomains.textContent = selectedCount;
    if (toolTabBadge) {
        toolTabBadge.textContent = selectedCount;
        toolTabBadge.style.display = selectedCount > 0 ? '' : 'none';
    }
}

// ============================================================================
// DOMAIN SELECTION
// ============================================================================
function toggleDomain(domainKey) {
    const idx = ToolState.selectedDomains.indexOf(domainKey);
    if (idx >= 0) {
        ToolState.selectedDomains.splice(idx, 1);
        // Also remove from expanded
        const expIdx = ToolState.expandedDomains.indexOf(domainKey);
        if (expIdx >= 0) ToolState.expandedDomains.splice(expIdx, 1);
    } else {
        ToolState.selectedDomains.push(domainKey);
    }

    saveToolSettings();
    refreshToolPanel();
}

function toggleDomainExpand(domainKey) {
    const idx = ToolState.expandedDomains.indexOf(domainKey);
    if (idx >= 0) {
        ToolState.expandedDomains.splice(idx, 1);
    } else {
        ToolState.expandedDomains.push(domainKey);
    }
    refreshToolPanel();
}

function selectAllDomains() {
    ToolState.selectedDomains = getDomainKeys();
    saveToolSettings();
    refreshToolPanel();
    showToast('All domains selected', 'success');
}

function clearAllDomains() {
    ToolState.selectedDomains = [];
    ToolState.expandedDomains = [];
    saveToolSettings();
    refreshToolPanel();
    showToast('All domains cleared', 'info');
}

function refreshToolPanel() {
    const catalog = document.querySelector('.sidebar-tools-panel .tool-catalog');
    if (catalog) {
        catalog.innerHTML = buildToolPanelHTML();
        attachToolEventListeners();
    }
    updateToolMetrics();
}

function buildToolImportHTML() {
    return `
        <h3 class="pipeline-title">Import tools</h3>
        <p class="tool-import-note">Paste a tool list, pick a JSON file, or name an MCP server. The page asks <code>serve_nocache.py</code> to convert it, then you can drop the UCL into the prompt.</p>
        <div class="convert-section">
            <label class="convert-label" for="toolImportName">Domain name</label>
            <input type="text" id="toolImportName" class="convert-textarea convert-textarea-sm" placeholder="my_tools" spellcheck="false">
        </div>
        <div class="tool-import-row">
            <label><input type="radio" name="toolImportMode" value="json" checked> JSON</label>
            <label><input type="radio" name="toolImportMode" value="mcp"> MCP server</label>
        </div>
        <div id="toolImportJsonSection">
            <div class="convert-section">
                <label class="convert-label" for="toolImportJson">Tool JSON</label>
                <textarea id="toolImportJson" class="convert-textarea" rows="6" spellcheck="false" placeholder='{"tools":[{"name":"ping","description":"Ping a host","inputSchema":{"type":"object","properties":{"host":{"type":"string"}},"required":["host"]}}]}'></textarea>
                <input type="file" id="toolImportFile" accept="application/json,.json">
            </div>
        </div>
        <div id="toolImportMcpSection" hidden>
            <div class="convert-section">
                <label class="convert-label" for="toolImportCommand">Command</label>
                <input type="text" id="toolImportCommand" class="convert-textarea convert-textarea-sm" placeholder="npx" spellcheck="false">
            </div>
            <div class="convert-section">
                <label class="convert-label" for="toolImportArgs">Arguments, one per line</label>
                <textarea id="toolImportArgs" class="convert-textarea convert-textarea-sm" rows="3" spellcheck="false" placeholder="-y&#10;@playwright/mcp"></textarea>
            </div>
            <div class="convert-section">
                <label class="convert-label" for="toolImportFraming">Framing</label>
                <select id="toolImportFraming" class="convert-select optimizer-select">
                    <option value="newline" selected>newline (MCP spec)</option>
                    <option value="content-length">content-length (Loom)</option>
                </select>
            </div>
        </div>
        <label class="tool-import-note">
            <input type="checkbox" id="toolImportSave">
            Save into tools/ so it is still here after reload
        </label>
        <div class="tool-import-actions">
            <button type="button" class="convert-btn convert-btn-primary" id="toolImportBtn">Import</button>
            <button type="button" class="convert-btn" id="toolImportInsertBtn" disabled>Insert UCL</button>
            <button type="button" class="convert-btn" id="toolImportCopyBtn" disabled>Copy UCL</button>
        </div>
        <div class="tool-import-status" id="toolImportStatus"></div>
        <textarea id="toolImportPreview" class="convert-textarea convert-textarea-sm" rows="6" readonly placeholder="UCL for the imported tools shows up here."></textarea>
    `;
}

function attachToolImportListeners() {
    document.querySelectorAll('input[name="toolImportMode"]').forEach(input => {
        input.addEventListener('change', syncToolImportMode);
    });
    document.getElementById('toolImportFile')?.addEventListener('change', onToolImportFile);
    document.getElementById('toolImportBtn')?.addEventListener('click', runToolImport);
    document.getElementById('toolImportInsertBtn')?.addEventListener('click', insertImportedUCL);
    document.getElementById('toolImportCopyBtn')?.addEventListener('click', copyImportedUCL);
    syncToolImportMode();
}

function syncToolImportMode() {
    const selected = document.querySelector('input[name="toolImportMode"]:checked');
    const mode = selected ? selected.value : 'json';
    const jsonSection = document.getElementById('toolImportJsonSection');
    const mcpSection = document.getElementById('toolImportMcpSection');
    if (jsonSection) jsonSection.hidden = mode !== 'json';
    if (mcpSection) mcpSection.hidden = mode !== 'mcp';
}

async function onToolImportFile(event) {
    const file = event.target.files && event.target.files[0];
    if (!file) return;
    const text = await file.text();
    const box = document.getElementById('toolImportJson');
    if (box) box.value = text;
    const name = document.getElementById('toolImportName');
    if (name && !name.value.trim()) {
        const stem = file.name.replace(/\.json$/i, '').replace(/[^A-Za-z0-9_]+/g, '_').replace(/^_+|_+$/g, '');
        name.value = stem && /^[A-Za-z_]/.test(stem) ? stem : ('imported_' + stem);
    }
}

function setToolImportStatus(message, isError) {
    const el = document.getElementById('toolImportStatus');
    if (!el) return;
    el.textContent = message || '';
    el.classList.toggle('error', !!isError);
}

async function runToolImport() {
    if (ToolImportState.busy) return;
    const name = (document.getElementById('toolImportName')?.value || '').trim();
    const mode = document.querySelector('input[name="toolImportMode"]:checked')?.value || 'json';
    const body = {
        name: name,
        save: !!document.getElementById('toolImportSave')?.checked,
    };
    if (mode === 'mcp') {
        body.mcp_command = (document.getElementById('toolImportCommand')?.value || '').trim();
        body.mcp_args = (document.getElementById('toolImportArgs')?.value || '')
            .split(/\r?\n/)
            .map(line => line.trim())
            .filter(Boolean);
        body.framing = document.getElementById('toolImportFraming')?.value || 'newline';
        body.timeout = 30;
    } else {
        body.json_text = document.getElementById('toolImportJson')?.value || '';
    }

    ToolImportState.busy = true;
    setToolImportStatus('Converting…', false);
    try {
        const response = await fetch('/api/convert-tools', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });
        let payload = {};
        try {
            payload = await response.json();
        } catch (parseError) {
            throw new Error('The server did not return JSON. Open this page with python serve_nocache.py.');
        }
        if (!response.ok || !payload.ok) {
            throw new Error(payload.error || ('HTTP ' + response.status));
        }
        ToolImportState.last = payload;
        ToolLoader.registerDomain(payload.website_domain);
        if (!ToolState.selectedDomains.includes(payload.name)) {
            ToolState.selectedDomains.push(payload.name);
        }
        if (!ToolState.expandedDomains.includes(payload.name)) {
            ToolState.expandedDomains.push(payload.name);
        }
        if (typeof saveToolSettings === 'function') saveToolSettings();
        const preview = document.getElementById('toolImportPreview');
        if (preview) preview.value = payload.ucl_text || '';
        document.getElementById('toolImportInsertBtn').disabled = false;
        document.getElementById('toolImportCopyBtn').disabled = false;
        refreshToolPanel();
        const warningText = (payload.warnings || []).length
            ? ' ' + payload.warnings.join(' ')
            : '';
        const savedText = payload.saved ? ' Saved to the tools folder.' : '';
        setToolImportStatus(
            payload.tool_count + ' tools in ' + payload.name + '.' + savedText + warningText,
            false
        );
        if (typeof showToast === 'function') {
            showToast('Imported ' + payload.tool_count + ' tools', 'success');
        }
    } catch (error) {
        const offline = error && error.message && /fetch|NetworkError|Failed to fetch/i.test(error.message);
        setToolImportStatus(
            offline
                ? 'This page cannot convert tools by itself. Start it with python serve_nocache.py and open http://localhost:8765.'
                : error.message,
            true
        );
    } finally {
        ToolImportState.busy = false;
    }
}

function insertImportedUCL() {
    const text = ToolImportState.last && ToolImportState.last.ucl_text;
    if (!text || typeof insertSyntax !== 'function') return;
    insertSyntax(text);
    if (typeof showToast === 'function') showToast('UCL inserted into the prompt', 'success');
}

async function copyImportedUCL() {
    const text = ToolImportState.last && ToolImportState.last.ucl_text;
    if (!text) return;
    if (typeof copyToClipboard === 'function') {
        copyToClipboard(text);
        return;
    }
    await navigator.clipboard.writeText(text);
}

// ============================================================================
// TOOL INSERTION
// ============================================================================
function insertToolSyntax(toolKey, domainKey) {
    const domain = getDomain(domainKey);
    if (!domain || !domain.tools[toolKey]) return;

    const tool = domain.tools[toolKey];

    // Build UCL tool syntax
    const paramsStr = tool.parameters
        .filter(p => p.required)
        .map(p => `[[PARAM:${p.name}:${p.type}]]`)
        .join(' ');

    const syntax = `^^CONDITION: ${tool.keywords.join(', ')}^^
  [[TOOL:${tool.name}]]
  ${paramsStr}
  // ${tool.description}
^^/CONDITION^^`;

    insertSyntax(syntax);
    showToast(`Tool "${tool.name}" inserted`, 'success');
}

function insertUCLToolsPlaceholder() {
    insertSyntax('{{UCL_TOOLS}}');
    showToast('{{UCL_TOOLS}} placeholder inserted', 'info');
}

// ============================================================================
// TOOL SEARCH / FILTER
// ============================================================================
function filterTools(query) {
    ToolState.searchQuery = query.toLowerCase();
    const toolCards = document.querySelectorAll('.tool-card');
    const domainCards = document.querySelectorAll('.tool-domain-card');

    if (!ToolState.searchQuery) {
        // Show all — reset to normal view
        toolCards.forEach(card => card.style.display = '');
        domainCards.forEach(card => card.style.display = '');
        return;
    }

    // Expand all selected domains to show search results
    domainCards.forEach(card => {
        const domainKey = card.dataset.domain;
        const tools = card.querySelectorAll('.tool-card');
        let hasMatch = false;

        tools.forEach(tool => {
            const text = tool.textContent.toLowerCase();
            const matches = text.includes(ToolState.searchQuery);
            tool.style.display = matches ? '' : 'none';
            if (matches) hasMatch = true;
        });

        // Also check domain name/description
        const domainInfo = card.querySelector('.domain-info');
        if (domainInfo && domainInfo.textContent.toLowerCase().includes(ToolState.searchQuery)) {
            hasMatch = true;
            tools.forEach(t => t.style.display = '');
        }

        card.style.display = hasMatch ? '' : 'none';
        if (hasMatch && !ToolState.expandedDomains.includes(domainKey)) {
            ToolState.expandedDomains.push(domainKey);
            // Re-render to show expanded
            refreshToolPanel();
        }
    });
}

// ============================================================================
// SYSTEM PROMPT AUGMENTATION
// ============================================================================
function getToolContextForPrompt() {
    if (ToolState.selectedDomains.length === 0) return '';

    let context = '\n\nSELECTED TOOL DOMAINS:\n';

    for (const domainKey of ToolState.selectedDomains) {
        const domain = getDomain(domainKey);
        if (!domain) continue;

        const toolNames = Object.keys(domain.tools);
        context += `- ${domain.name} (${toolNames.length} tools): ${toolNames.join(', ')}\n`;
    }

    context += '\nInclude relevant [[TOOL:]] references in the optimized prompt.\n';
    context += 'Use keyword-based ^^CONDITION^^ blocks for tool activation.\n';

    if (ToolState.includePlaceholder) {
        context += 'Include {{UCL_TOOLS}} as placeholder for full tool definitions.\n';
    }

    return context;
}

// ============================================================================
// EVENT LISTENERS
// ============================================================================
function attachToolEventListeners() {
    // Domain checkbox toggles
    document.querySelectorAll('.domain-checkbox').forEach(cb => {
        cb.addEventListener('change', (e) => {
            const domainKey = e.target.dataset.domainCheck;
            toggleDomain(domainKey);
        });
    });

    // Domain card header click (toggle expand)
    document.querySelectorAll('.domain-card-header').forEach(header => {
        header.addEventListener('click', (e) => {
            // Don't expand if clicking the checkbox
            if (e.target.closest('.domain-checkbox-label')) return;
            const domainKey = header.dataset.domainToggle;
            if (domainKey) toggleDomainExpand(domainKey);
        });
    });

    // Expand toggle buttons
    document.querySelectorAll('[data-domain-expand]').forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.stopPropagation();
            const domainKey = btn.dataset.domainExpand;
            toggleDomainExpand(domainKey);
        });
    });

    // Select All / Clear All
    const selectAllBtn = document.getElementById('toolSelectAllBtn');
    const clearAllBtn = document.getElementById('toolClearAllBtn');
    if (selectAllBtn) selectAllBtn.addEventListener('click', selectAllDomains);
    if (clearAllBtn) clearAllBtn.addEventListener('click', clearAllDomains);

    // {{UCL_TOOLS}} placeholder toggle
    const placeholderCheck = document.getElementById('toolPlaceholderCheck');
    if (placeholderCheck) {
        placeholderCheck.addEventListener('change', (e) => {
            ToolState.includePlaceholder = e.target.checked;
            saveToolSettings();
            if (e.target.checked) {
                showToast('{{UCL_TOOLS}} will be included in optimization', 'info');
            }
        });
    }

    // Tool insert buttons
    document.querySelectorAll('[data-tool-insert]').forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.stopPropagation();
            const toolKey = btn.dataset.toolInsert;
            const domainKey = btn.dataset.toolDomain;
            insertToolSyntax(toolKey, domainKey);
        });
    });
}
