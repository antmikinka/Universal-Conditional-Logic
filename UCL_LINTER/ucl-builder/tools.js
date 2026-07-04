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

    // Create tab bar with all tabs in one pass (extensible for 3+ tabs)
    const tabBar = document.createElement('div');
    tabBar.className = 'sidebar-tabs';
    tabBar.innerHTML = `
        <button class="sidebar-tab active" data-tab="syntax">
            <span class="tab-icon">⚡</span>
            <span class="tab-label">UCL Syntax</span>
        </button>
        <button class="sidebar-tab" data-tab="tools">
            <span class="tab-icon">🔧</span>
            <span class="tab-label">Tools</span>
            <span class="tab-badge" id="toolTabBadge">0</span>
        </button>
        <button class="sidebar-tab" data-tab="pipeline">
            <span class="tab-icon">🔀</span>
            <span class="tab-label">Pipeline</span>
        </button>
    `;

    // Insert after sidebar-header
    sidebarHeader.after(tabBar);

    // Tab click handlers
    tabBar.querySelectorAll('.sidebar-tab').forEach(tab => {
        tab.addEventListener('click', () => switchSidebarTab(tab.dataset.tab));
    });
}

function switchSidebarTab(tabName) {
    // Update tab buttons
    document.querySelectorAll('.sidebar-tab').forEach(tab => {
        tab.classList.toggle('active', tab.dataset.tab === tabName);
    });

    // Show/hide content panels for all tabs
    const syntaxPanel = document.querySelector('.sidebar-syntax-panel');
    const toolsPanel = document.querySelector('.sidebar-tools-panel');
    const pipelinePanel = document.querySelector('.sidebar-pipeline-panel');

    if (syntaxPanel) syntaxPanel.style.display = tabName === 'syntax' ? '' : 'none';
    if (toolsPanel) toolsPanel.style.display = tabName === 'tools' ? '' : 'none';
    if (pipelinePanel) pipelinePanel.style.display = tabName === 'pipeline' ? '' : 'none';

    // Update search placeholder based on active tab
    const searchPlaceholders = {
        syntax: 'Search syntax...',
        tools: 'Search tools...',
        pipeline: 'Search pipeline...'
    };
    const searchInput = document.getElementById('syntaxSearch');
    if (searchInput) {
        searchInput.placeholder = searchPlaceholders[tabName] || 'Search...';
        searchInput.value = '';
        ToolState.searchQuery = '';
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

    // Build tool panel content
    toolsPanel.innerHTML = buildToolPanelHTML();

    // Create pipeline panel (placeholder for future preparation pipeline UI)
    let pipelinePanel = sidebar.querySelector('.sidebar-pipeline-panel');
    if (!pipelinePanel) {
        pipelinePanel = document.createElement('div');
        pipelinePanel.className = 'sidebar-pipeline-panel';
        pipelinePanel.style.display = 'none';
        pipelinePanel.innerHTML = `
            <div class="pipeline-placeholder">
                <h3>Preparation Pipeline</h3>
                <p>Pipeline configuration will appear here.</p>
            </div>
        `;
        // Insert after tools panel
        if (toolsPanel) {
            toolsPanel.after(pipelinePanel);
        } else {
            sidebar.appendChild(pipelinePanel);
        }
    }
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
    const toolsPanel = document.querySelector('.sidebar-tools-panel');
    if (toolsPanel) {
        toolsPanel.innerHTML = buildToolPanelHTML();
        attachToolEventListeners();
    }
    updateToolMetrics();
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
