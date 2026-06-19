// ============================================================================
// UCL PROMPT BUILDER - MAIN APPLICATION
// Beautiful dark mode web app for UCL prompt optimization with OpenRouter
// ============================================================================

// ============================================================================
// STATE MANAGEMENT
// ============================================================================
const AppState = {
    apiKey: '',
    selectedModel: '',
    customModel: '',
    settings: {
        autoCopy: false,
        syntaxHighlight: true
    },
    initialized: false
};

// ============================================================================
// INITIALIZATION
// ============================================================================
document.addEventListener('DOMContentLoaded', () => {
    console.log('🚀  UCL Prompt Builder Initializing...');

    loadSettings();
    populateModels();
    renderSyntaxCategories();
    attachEventListeners();
    updateCharCounts();

    AppState.initialized = true;
    console.log('✓ UCL Prompt Builder Ready!');
});

// ============================================================================
// SETTINGS MANAGEMENT
// ============================================================================
function loadSettings() {
    try {
        const apiKey = localStorage.getItem('ucl_api_key') || '';
        const model = localStorage.getItem('ucl_model') || '';
        const customModel = localStorage.getItem('ucl_custom_model') || '';
        const autoCopy = localStorage.getItem('ucl_auto_copy') === 'true';
        const syntaxHighlight = localStorage.getItem('ucl_syntax_highlight') !== 'false';

        AppState.apiKey = apiKey;
        AppState.selectedModel = model;
        AppState.customModel = customModel;
        AppState.settings.autoCopy = autoCopy;
        AppState.settings.syntaxHighlight = syntaxHighlight;

        // Update UI
        const apiKeyInput = document.getElementById('apiKeyInput');
        const modelSelect = document.getElementById('modelSelect');
        const customModelInput = document.getElementById('customModelInput');
        const autoCopyCheck = document.getElementById('autoCopyCheck');
        const syntaxHighlightCheck = document.getElementById('syntaxHighlightCheck');

        if (apiKeyInput) apiKeyInput.value = apiKey;
        if (modelSelect) modelSelect.value = model;
        if (customModelInput) customModelInput.value = customModel;
        if (autoCopyCheck) autoCopyCheck.checked = autoCopy;
        if (syntaxHighlightCheck) syntaxHighlightCheck.checked = syntaxHighlight;

        updateApiKeyStatus();
    } catch (error) {
        console.error('Error loading settings:', error);
    }
}

function saveSettings() {
    try {
        const apiKeyInput = document.getElementById('apiKeyInput');
        const modelSelect = document.getElementById('modelSelect');
        const customModelInput = document.getElementById('customModelInput');
        const autoCopyCheck = document.getElementById('autoCopyCheck');
        const syntaxHighlightCheck = document.getElementById('syntaxHighlightCheck');

        AppState.apiKey = apiKeyInput.value.trim();
        AppState.selectedModel = modelSelect.value;
        AppState.customModel = customModelInput.value.trim();
        AppState.settings.autoCopy = autoCopyCheck.checked;
        AppState.settings.syntaxHighlight = syntaxHighlightCheck.checked;

        localStorage.setItem('ucl_api_key', AppState.apiKey);
        localStorage.setItem('ucl_model', AppState.selectedModel);
        localStorage.setItem('ucl_custom_model', AppState.customModel);
        localStorage.setItem('ucl_auto_copy', AppState.settings.autoCopy);
        localStorage.setItem('ucl_syntax_highlight', AppState.settings.syntaxHighlight);

        updateApiKeyStatus();
        closeModal('settingsModal');
        showToast('Settings saved successfully', 'success');
    } catch (error) {
        console.error('Error saving settings:', error);
        showToast('Failed to save settings', 'error');
    }
}

function updateApiKeyStatus() {
    const statusDiv = document.getElementById('apiKeyStatus');
    if (!statusDiv) return;

    if (AppState.apiKey) {
        statusDiv.textContent = '✓ API key saved';
        statusDiv.className = 'api-key-status success';
    } else {
        statusDiv.textContent = '⚠ No API key configured';
        statusDiv.className = 'api-key-status error';
    }
}

// ============================================================================
// MODELS POPULATION
// ============================================================================
function populateModels() {
    const modelSelect = document.getElementById('modelSelect');
    if (!modelSelect) return;

    // Clear existing options except first
    modelSelect.innerHTML = '<option value="">Select a model...</option>';

    // Add models by category
    for (const [category, models] of Object.entries(MODELS_BY_CATEGORY)) {
        const optgroup = document.createElement('optgroup');
        optgroup.label = category;

        models.forEach(model => {
            const option = document.createElement('option');
            option.value = model.id;
            const pricing = formatPricing(model.pricing);
            option.textContent = `${model.name} (${pricing})`;
            optgroup.appendChild(option);
        });

        modelSelect.appendChild(optgroup);
    }

    // Restore selected model
    if (AppState.selectedModel) {
        modelSelect.value = AppState.selectedModel;
    }
}

// ============================================================================
// SYNTAX CATEGORIES RENDERING
// ============================================================================
function renderSyntaxCategories() {
    const container = document.getElementById('syntaxCategories');
    if (!container) return;

    container.innerHTML = '';

    for (const [categoryName, categoryData] of Object.entries(UCL_SYNTAX)) {
        const category = createCategoryElement(categoryName, categoryData);
        container.appendChild(category);
    }
}

function createCategoryElement(name, data) {
    const category = document.createElement('div');
    category.className = 'category';

    const header = document.createElement('div');
    header.className = 'category-header';
    header.innerHTML = `
        <span class="category-name">${name}</span>
        <span class="category-toggle">▶</span>
    `;
    header.onclick = () => toggleCategory(category);

    const itemsContainer = document.createElement('div');
    itemsContainer.className = 'category-items';

    for (const [itemName, itemData] of Object.entries(data.items)) {
        const item = createSyntaxItem(itemName, itemData);
        itemsContainer.appendChild(item);
    }

    category.appendChild(header);
    category.appendChild(itemsContainer);

    return category;
}

function createSyntaxItem(name, data) {
    const item = document.createElement('div');
    item.className = 'syntax-item';

    let content = `<div class="syntax-name">${name}`;

    if (data.priority) {
        content += `<span class="priority-badge priority-${data.priority}">${data.priority}</span>`;
    }

    content += `</div>`;

    if (data.syntax && data.insertable) {
        content += `<div class="syntax-code">${escapeHtml(data.syntax)}</div>`;
    }

    content += `<div class="syntax-description">${data.description}</div>`;

    if (data.example) {
        content += `<div class="syntax-description" style="margin-top: 0.25rem; font-style: italic;">Ex: ${escapeHtml(data.example)}</div>`;
    }

    item.innerHTML = content;

    if (data.insertable && data.syntax) {
        item.style.cursor = 'pointer';
        item.onclick = () => insertSyntax(data.syntax);
    }

    return item;
}

function toggleCategory(categoryElement) {
    categoryElement.classList.toggle('expanded');
}

// ============================================================================
// SYNTAX INSERTION
// ============================================================================
function insertSyntax(syntax) {
    const textarea = document.getElementById('inputPrompt');
    if (!textarea) return;

    const start = textarea.selectionStart;
    const end = textarea.selectionEnd;
    const text = textarea.value;

    // Insert syntax at cursor position
    const before = text.substring(0, start);
    const after = text.substring(end);

    // Add newlines if needed
    const needsPrefix = before.length > 0 && !before.endsWith('\n');
    const needsSuffix = after.length > 0 && !after.startsWith('\n');

    const insertion = (needsPrefix ? '\n' : '') + syntax + (needsSuffix ? '\n' : '');

    textarea.value = before + insertion + after;

    // Move cursor to end of insertion
    const newPos = start + insertion.length;
    textarea.selectionStart = newPos;
    textarea.selectionEnd = newPos;
    textarea.focus();

    updateCharCounts();
    showToast('Syntax inserted', 'info');
}

// ============================================================================
// SEARCH FUNCTIONALITY
// ============================================================================
document.getElementById('syntaxSearch')?.addEventListener('input', (e) => {
    const query = e.target.value.toLowerCase();
    const categories = document.querySelectorAll('.category');

    categories.forEach(category => {
        const items = category.querySelectorAll('.syntax-item');
        let hasVisibleItems = false;

        items.forEach(item => {
            const text = item.textContent.toLowerCase();
            if (text.includes(query)) {
                item.style.display = '';
                hasVisibleItems = true;
            } else {
                item.style.display = 'none';
            }
        });

        // Show category if it has visible items
        if (query && hasVisibleItems) {
            category.classList.add('expanded');
            category.style.display = '';
        } else if (query && !hasVisibleItems) {
            category.style.display = 'none';
        } else {
            category.style.display = '';
        }
    });
});

// ============================================================================
// CHARACTER/WORD COUNTING
// ============================================================================
function updateCharCounts() {
    const inputPrompt = document.getElementById('inputPrompt');
    const outputPrompt = document.getElementById('outputPrompt');

    if (inputPrompt) {
        const text = inputPrompt.value;
        document.getElementById('inputCharCount').textContent = text.length;
        document.getElementById('inputWordCount').textContent = countWords(text);
    }

    if (outputPrompt) {
        const text = outputPrompt.value;
        document.getElementById('outputCharCount').textContent = text.length;
        document.getElementById('outputWordCount').textContent = countWords(text);
    }
}

function countWords(text) {
    return text.trim().split(/\s+/).filter(word => word.length > 0).length;
}

// Attach to textareas
document.getElementById('inputPrompt')?.addEventListener('input', updateCharCounts);
document.getElementById('outputPrompt')?.addEventListener('input', updateCharCounts);

// ============================================================================
// UCL OPTIMIZATION (OpenRouter API)
// ============================================================================
async function optimizeWithUCL() {
    const inputPrompt = document.getElementById('inputPrompt').value.trim();

    if (!inputPrompt) {
        showToast('Please enter a prompt to optimize', 'error');
        return;
    }

    // Get API key and model
    const apiKey = AppState.customModel || AppState.apiKey;
    const model = AppState.customModel || AppState.selectedModel;

    if (!AppState.apiKey) {
        showToast('Please configure your OpenRouter API key in settings', 'error');
        openModal('settingsModal');
        return;
    }

    if (!model) {
        showToast('Please select a model', 'error');
        return;
    }

    showLoading(true);
    setStatus('Optimizing...', 'warning');

    try {
        const response = await fetch('https://openrouter.ai/api/v1/chat/completions', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${AppState.apiKey}`,
                'HTTP-Referer': window.location.href,
                'X-Title': 'UCL Prompt Builder'
            },
            body: JSON.stringify({
                model: model,
                messages: [
                    {
                        role: 'system',
                        content: getUCLSystemPrompt()
                    },
                    {
                        role: 'user',
                        content: inputPrompt
                    }
                ],
                temperature: 0.7,
                max_tokens: 4000
            })
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.error?.message || `HTTP ${response.status}: ${response.statusText}`);
        }

        const data = await response.json();
        const optimizedPrompt = data.choices[0].message.content;

        document.getElementById('outputPrompt').value = optimizedPrompt;
        updateCharCounts();

        if (AppState.settings.autoCopy) {
            copyToClipboard(optimizedPrompt);
        }

        showToast('Prompt optimized successfully!', 'success');
        setStatus('Ready', 'success');

    } catch (error) {
        console.error('Optimization error:', error);
        showToast(`Optimization failed: ${error.message}`, 'error');
        setStatus('Error', 'error');
    } finally {
        showLoading(false);
    }
}

function getUCLSystemPrompt() {
    return `You are an expert in Universal Conditional Logic (UCL) prompt optimization.

Your task is to convert user prompts into UCL-optimized format following these rules:

1. ALWAYS start with [[CRITICAL:]] directive (+9.3% quality bonus)
2. Use keyword-based ^^CONDITION^^ instead of SWITCH (lower O_s)
3. Apply proper UCL syntax:
   - Concepts: {{concept:name:attribute}}
   - Directives: [[TYPE: content]]
   - Transforms: [[TRANSFORM: source TO target]]
   - Conditions: ^^CONDITION: keywords^^ ... ^^/CONDITION^^
   - Sections: <name> ... </name>
4. Avoid anti-patterns:
   - No role-questioning conditionals
   - Minimize SWITCH statements
   - Keep O_s (structural overhead) low
5. Optimize for S* ≈ 0.509 (optimal specification threshold)
6. Wrap domain-specific logic in keyword CONDITIONS
7. Use appropriate directive types (REQUIRE, ENFORCE, VALIDATE, etc.)

Return ONLY the optimized UCL prompt, no explanations or commentary.`;
}

// ============================================================================
// COPY TO CLIPBOARD
// ============================================================================
async function copyToClipboard(text) {
    try {
        await navigator.clipboard.writeText(text);
        showToast('Copied to clipboard!', 'success');
    } catch (error) {
        console.error('Copy failed:', error);
        showToast('Failed to copy', 'error');
    }
}

function copyOptimizedPrompt() {
    const text = document.getElementById('outputPrompt').value;
    if (!text) {
        showToast('No optimized prompt to copy', 'error');
        return;
    }
    copyToClipboard(text);
}

// ============================================================================
// SWAP PROMPTS
// ============================================================================
function swapPrompts() {
    const input = document.getElementById('inputPrompt');
    const output = document.getElementById('outputPrompt');

    const temp = input.value;
    input.value = output.value;
    output.value = temp;

    updateCharCounts();
    showToast('Prompts swapped', 'info');
}

// ============================================================================
// CLEAR ALL
// ============================================================================
function clearAll() {
    if (!confirm('Clear both prompts?')) return;

    document.getElementById('inputPrompt').value = '';
    document.getElementById('outputPrompt').value = '';
    updateCharCounts();
    showToast('Prompts cleared', 'info');
}

// ============================================================================
// DOWNLOAD
// ============================================================================
function downloadPrompt() {
    const text = document.getElementById('outputPrompt').value;
    if (!text) {
        showToast('No optimized prompt to download', 'error');
        return;
    }

    const blob = new Blob([text], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `ucl-optimized-prompt-${Date.now()}.txt`;
    a.click();
    URL.revokeObjectURL(url);

    showToast('Prompt downloaded', 'success');
}

// ============================================================================
// UI HELPERS
// ============================================================================
function showLoading(show) {
    const overlay = document.getElementById('loadingOverlay');
    if (overlay) {
        overlay.classList.toggle('active', show);
    }
}

function setStatus(text, type = 'info') {
    const statusText = document.querySelector('.status-text');
    const statusDot = document.querySelector('.status-dot');

    if (statusText) statusText.textContent = text;
    if (statusDot) {
        statusDot.style.background = type === 'success' ? 'var(--accent-success)' :
            type === 'error' ? 'var(--accent-error)' :
                type === 'warning' ? 'var(--accent-warning)' :
                    'var(--accent-primary)';
    }
}

function showToast(message, type = 'info') {
    const container = document.getElementById('toastContainer');
    if (!container) return;

    const toast = document.createElement('div');
    toast.className = `toast ${type}`;

    const icon = type === 'success' ? '✓' :
        type === 'error' ? '✗' :
            type === 'warning' ? '⚠' :
                'ℹ';

    toast.innerHTML = `
        <div class="toast-icon">${icon}</div>
        <div class="toast-content">
            <div class="toast-message">${escapeHtml(message)}</div>
        </div>
    `;

    container.appendChild(toast);

    setTimeout(() => {
        toast.style.opacity = '0';
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

function openModal(modalId) {
    const modal = document.getElementById(modalId);
    if (modal) modal.classList.add('active');
}

function closeModal(modalId) {
    const modal = document.getElementById(modalId);
    if (modal) modal.classList.remove('active');
}

// ============================================================================
// EVENT LISTENERS
// ============================================================================
function attachEventListeners() {
    // Settings
    document.getElementById('settingsBtn')?.addEventListener('click', () => openModal('settingsModal'));
    document.getElementById('closeSettingsBtn')?.addEventListener('click', () => closeModal('settingsModal'));
    document.getElementById('cancelSettingsBtn')?.addEventListener('click', () => closeModal('settingsModal'));
    document.getElementById('saveSettingsBtn')?.addEventListener('click', saveSettings);

    // Help
    document.getElementById('helpBtn')?.addEventListener('click', () => openModal('helpModal'));
    document.getElementById('closeHelpBtn')?.addEventListener('click', () => closeModal('helpModal'));

    // API Key toggle
    document.getElementById('toggleApiKey')?.addEventListener('click', function () {
        const input = document.getElementById('apiKeyInput');
        if (input.type === 'password') {
            input.type = 'text';
            this.textContent = '🙈';
        } else {
            input.type = 'password';
            this.textContent = '👁️';
        }
    });

    // Action buttons
    document.getElementById('optimizeBtn')?.addEventListener('click', optimizeWithUCL);
    document.getElementById('copyBtn')?.addEventListener('click', copyOptimizedPrompt);
    document.getElementById('swapBtn')?.addEventListener('click', swapPrompts);
    document.getElementById('clearBtn')?.addEventListener('click', clearAll);
    document.getElementById('downloadBtn')?.addEventListener('click', downloadPrompt);

    // Sidebar controls
    document.getElementById('collapseAllBtn')?.addEventListener('click', () => {
        document.querySelectorAll('.category').forEach(cat => cat.classList.remove('expanded'));
    });

    document.getElementById('expandAllBtn')?.addEventListener('click', () => {
        document.querySelectorAll('.category').forEach(cat => cat.classList.add('expanded'));
    });

    // Close modals on outside click
    document.querySelectorAll('.modal').forEach(modal => {
        modal.addEventListener('click', (e) => {
            if (e.target === modal) {
                modal.classList.remove('active');
            }
        });
    });

    // Keyboard shortcuts
    document.addEventListener('keydown', (e) => {
        // Ctrl/Cmd + Enter: Optimize
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
            e.preventDefault();
            optimizeWithUCL();
        }

        // Ctrl/Cmd + K: Clear
        if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
            e.preventDefault();
            clearAll();
        }
    });
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// ============================================================================
// CONSOLE BRANDING
// ============================================================================
console.log('%c⚡ UCL Prompt Builder', 'font-size: 24px; font-weight: bold; background: linear-gradient(135deg, #00d4ff, #a855f7); -webkit-background-clip: text; -webkit-text-fill-color: transparent;');
console.log('%cUniversal Conditional Logic Framework v2.0', 'font-size: 14px; color: #a0a0b0;');
console.log('%cBuilt with ❤️ for optimal prompt engineering', 'font-size: 12px; color: #6b7280;');
