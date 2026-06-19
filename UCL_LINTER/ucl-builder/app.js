// ============================================================================
// UCL PROMPT BUILDER - MAIN APPLICATION
// Beautiful dark mode web app for UCL prompt optimization with OpenRouter
// ============================================================================

// ============================================================================
// STATE MANAGEMENT
// ============================================================================
const AppState = {
    provider: 'openrouter', // 'openrouter' or 'lemonade'
    apiKey: '',
    selectedModel: '',
    customModel: '',
    lemonadeEndpoint: 'http://localhost:13305',
    lemonadeApiKey: '',
    settings: {
        autoCopy: false,
        syntaxHighlight: true
    },
    initialized: false
};

// Initialize Lemonade client
let lemonadeClient = new LemonadeClient(AppState.lemonadeEndpoint);

// ============================================================================
// INITIALIZATION
// ============================================================================
document.addEventListener('DOMContentLoaded', async () => {
    console.log('🚀  UCL Prompt Builder Initializing...');

    loadSettings();
    await populateModels();
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
        const provider = localStorage.getItem('ucl_provider') || 'openrouter';
        const apiKey = localStorage.getItem('ucl_api_key') || '';
        const model = localStorage.getItem('ucl_model') || '';
        const customModel = localStorage.getItem('ucl_custom_model') || '';
        const lemonadeEndpoint = localStorage.getItem('ucl_lemonade_endpoint') || 'http://localhost:13305';
        const lemonadeApiKey = localStorage.getItem('ucl_lemonade_api_key') || '';
        const autoCopy = localStorage.getItem('ucl_auto_copy') === 'true';
        const syntaxHighlight = localStorage.getItem('ucl_syntax_highlight') !== 'false';

        AppState.provider = provider;
        AppState.apiKey = apiKey;
        AppState.selectedModel = model;
        AppState.customModel = customModel;
        AppState.lemonadeEndpoint = lemonadeEndpoint;
        AppState.lemonadeApiKey = lemonadeApiKey;
        AppState.settings.autoCopy = autoCopy;
        AppState.settings.syntaxHighlight = syntaxHighlight;

        // Update UI
        const providerSelect = document.getElementById('providerSelect');
        const apiKeyInput = document.getElementById('apiKeyInput');
        const modelSelect = document.getElementById('modelSelect');
        const customModelInput = document.getElementById('customModelInput');
        const lemonadeEndpointInput = document.getElementById('lemonadeEndpointInput');
        const lemonadeApiKeyInput = document.getElementById('lemonadeApiKeyInput');
        const autoCopyCheck = document.getElementById('autoCopyCheck');
        const syntaxHighlightCheck = document.getElementById('syntaxHighlightCheck');

        if (providerSelect) providerSelect.value = provider;
        if (apiKeyInput) apiKeyInput.value = apiKey;
        if (modelSelect) modelSelect.value = model;
        if (customModelInput) customModelInput.value = customModel;
        if (lemonadeEndpointInput) lemonadeEndpointInput.value = lemonadeEndpoint;
        if (lemonadeApiKeyInput) lemonadeApiKeyInput.value = lemonadeApiKey;
        if (autoCopyCheck) autoCopyCheck.checked = autoCopy;
        if (syntaxHighlightCheck) syntaxHighlightCheck.checked = syntaxHighlight;

        updateApiKeyStatus();
        updateProviderUI();
    } catch (error) {
        console.error('Error loading settings:', error);
    }
}

async function saveSettings() {
    try {
        const providerSelect = document.getElementById('providerSelect');
        const apiKeyInput = document.getElementById('apiKeyInput');
        const modelSelect = document.getElementById('modelSelect');
        const customModelInput = document.getElementById('customModelInput');
        const lemonadeEndpointInput = document.getElementById('lemonadeEndpointInput');
        const lemonadeApiKeyInput = document.getElementById('lemonadeApiKeyInput');
        const autoCopyCheck = document.getElementById('autoCopyCheck');
        const syntaxHighlightCheck = document.getElementById('syntaxHighlightCheck');

        AppState.provider = providerSelect ? providerSelect.value : 'openrouter';
        AppState.apiKey = apiKeyInput ? apiKeyInput.value.trim() : '';
        AppState.selectedModel = modelSelect ? modelSelect.value : '';
        AppState.customModel = customModelInput ? customModelInput.value.trim() : '';
        AppState.lemonadeEndpoint = lemonadeEndpointInput ? lemonadeEndpointInput.value.trim() : 'http://localhost:13305';
        AppState.lemonadeApiKey = lemonadeApiKeyInput ? lemonadeApiKeyInput.value.trim() : '';
        AppState.settings.autoCopy = autoCopyCheck ? autoCopyCheck.checked : false;
        AppState.settings.syntaxHighlight = syntaxHighlightCheck ? syntaxHighlightCheck.checked : true;

        // Update Lemonade client with new endpoint
        lemonadeClient = new LemonadeClient(AppState.lemonadeEndpoint);

        localStorage.setItem('ucl_provider', AppState.provider);
        localStorage.setItem('ucl_api_key', AppState.apiKey);
        localStorage.setItem('ucl_model', AppState.selectedModel);
        localStorage.setItem('ucl_custom_model', AppState.customModel);
        localStorage.setItem('ucl_lemonade_endpoint', AppState.lemonadeEndpoint);
        localStorage.setItem('ucl_lemonade_api_key', AppState.lemonadeApiKey);
        localStorage.setItem('ucl_auto_copy', AppState.settings.autoCopy);
        localStorage.setItem('ucl_syntax_highlight', AppState.settings.syntaxHighlight);

        updateApiKeyStatus();
        updateProviderUI();
        await populateModels();
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

    if (AppState.provider === 'openrouter') {
        if (AppState.apiKey) {
            statusDiv.textContent = '✓ OpenRouter API key saved';
            statusDiv.className = 'api-key-status success';
        } else {
            statusDiv.textContent = '⚠ No OpenRouter API key configured';
            statusDiv.className = 'api-key-status error';
        }
    } else if (AppState.provider === 'lemonade') {
        statusDiv.textContent = '✓ Local Lemonade server: ' + AppState.lemonadeEndpoint;
        statusDiv.className = 'api-key-status success';
    } else if (AppState.provider === 'openai_compat') {
        statusDiv.textContent = '✓ Local server: ' + AppState.lemonadeEndpoint;
        statusDiv.className = 'api-key-status success';
    }
}

// Show/hide provider-specific settings sections
function updateProviderUI() {
    const openrouterSection = document.getElementById('openrouterSettings');
    const lemonadeSection = document.getElementById('lemonadeSettings');
    const providerSelect = document.getElementById('providerSelect');

    if (providerSelect) {
        const provider = providerSelect.value;
        if (openrouterSection) {
            openrouterSection.style.display = (provider === 'openrouter') ? '' : 'none';
        }
        if (lemonadeSection) {
            lemonadeSection.style.display = (provider === 'lemonade' || provider === 'openai_compat') ? '' : 'none';
        }
    }
}

// ============================================================================
// MODELS POPULATION
// ============================================================================
async function populateModels() {
    const modelSelect = document.getElementById('modelSelect');
    if (!modelSelect) return;

    // Clear existing options except first
    modelSelect.innerHTML = '<option value="">Select a model...</option>';

    // Get models based on provider
    if (AppState.provider === 'lemonade' || AppState.provider === 'openai_compat') {
        // Fetch actual models from Lemonade server
        await populateLemonadeModels(modelSelect);
    } else {
        // Use static OpenRouter models
        populateOpenRouterModels(modelSelect);
    }

    // Restore selected model
    if (AppState.selectedModel) {
        modelSelect.value = AppState.selectedModel;
    }
}

function populateOpenRouterModels(modelSelect) {
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
}

async function populateLemonadeModels(modelSelect) {
    try {
        // Show loading state
        const loadingOption = document.createElement('option');
        loadingOption.disabled = true;
        loadingOption.selected = true;
        loadingOption.textContent = 'Fetching models from Lemonade...';
        modelSelect.appendChild(loadingOption);

        // Fetch models from Lemonade server
        const models = await lemonadeClient.getAvailableModels(false);

        // Clear loading message
        modelSelect.innerHTML = '<option value="">Select a model...</option>';

        if (models.length === 0) {
            const noModelsOption = document.createElement('option');
            noModelsOption.disabled = true;
            noModelsOption.textContent = 'No models downloaded. Use Lemonade CLI to install models.';
            modelSelect.appendChild(noModelsOption);
            return;
        }

        // Group models by recipe/type
        const modelsByRecipe = {};
        models.forEach(model => {
            const recipe = model.recipe || 'other';
            if (!modelsByRecipe[recipe]) {
                modelsByRecipe[recipe] = [];
            }
            modelsByRecipe[recipe].push(model);
        });

        // Add models grouped by recipe
        for (const [recipe, recipeModels] of Object.entries(modelsByRecipe)) {
            const optgroup = document.createElement('optgroup');
            optgroup.label = `${recipe.toUpperCase()} Models`;

            recipeModels.forEach(model => {
                const option = document.createElement('option');
                option.value = model.id;

                // Build display text with metadata
                let displayText = model.id;
                const details = [];

                if (model.size) {
                    details.push(`${model.size}GB`);
                }
                if (model.max_context_window) {
                    details.push(`${(model.max_context_window / 1024).toFixed(1)}K ctx`);
                }
                if (model.labels && model.labels.length > 0) {
                    details.push(model.labels.join(', '));
                }

                if (details.length > 0) {
                    displayText += ` (${details.join(' · ')})`;
                }

                option.textContent = displayText;
                optgroup.appendChild(option);
            });

            modelSelect.appendChild(optgroup);
        }

        console.log(`✓ Loaded ${models.length} models from Lemonade server`);
    } catch (error) {
        console.error('Failed to fetch Lemonade models:', error);

        // Show error state
        modelSelect.innerHTML = '<option value="">Select a model...</option>';
        const errorOption = document.createElement('option');
        errorOption.disabled = true;
        errorOption.textContent = `Error: Cannot connect to Lemonade server (${AppState.lemonadeEndpoint})`;
        modelSelect.appendChild(errorOption);

        showToast('Cannot connect to Lemonade server. Is it running?', 'error');
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
// LEMONADE MODEL MANAGEMENT
// ============================================================================

// Ensure model is loaded before making inference requests
async function loadModelBeforeOptimize(modelName) {
    if (!lemonadeClient) return true;

    try {
        // Check server health first
        const isRunning = await lemonadeClient.isServerRunning();
        if (!isRunning) {
            showToast('Lemonade server not running. Start with: lemonade-server', 'error');
            setStatus('Server Offline', 'error');
            return false;
        }

        // Check if model is already loaded
        const isLoaded = await lemonadeClient.isModelLoaded(modelName);
        if (isLoaded) {
            console.log(`Model ${modelName} already loaded`);
            return true;
        }

        // Model not loaded - try to load it
        console.log(`Loading model ${modelName}...`);
        setStatus('Loading model...', 'warning');
        showLoading(true);

        await lemonadeClient.loadModel(modelName);

        showToast(`Model ${modelName} loaded successfully!`, 'success');
        setStatus('Model Ready', 'success');
        return true;

    } catch (error) {
        console.error('Model load error:', error);

        // If load fails, model might not be installed - offer to install
        if (error.message.includes('not found') || error.message.includes('not registered')) {
            const install = confirm(
                `Model "${modelName}" is not installed.\n\n` +
                `Would you like to download and install it now?`
            );
            if (install) {
                return await installModelWithProgress(modelName);
            }
            return false;
        }

        showToast(`Failed to load model: ${error.message}`, 'error');
        setStatus('Error', 'error');
        return false;
    }
}

// Install model with progress tracking via SSE
async function installModelWithProgress(modelName) {
    try {
        setStatus('Downloading model...', 'warning');

        // Show progress in the status area
        const progressEl = document.getElementById('downloadProgress');
        if (progressEl) {
            progressEl.style.display = 'block';
        }

        await lemonadeClient.pullModel(modelName, (event) => {
            if (event.type === 'progress') {
                const percent = event.percent || 0;
                const downloaded = lemonadeClient.formatBytes(event.bytes_downloaded || 0);
                const total = lemonadeClient.formatBytes(event.bytes_total || 0);

                // Update progress bar
                const progressBar = document.getElementById('downloadProgressBar');
                const progressText = document.getElementById('downloadProgressText');

                if (progressBar) {
                    progressBar.style.width = `${percent}%`;
                }
                if (progressText) {
                    progressText.textContent = `${percent}% (${downloaded} / ${total})`;
                }

                console.log(`Download progress: ${percent}%`);
            } else if (event.type === 'complete') {
                console.log('Download complete!');
            }
        });

        // Hide progress (reuse progressEl from line 441)
        if (progressEl) {
            progressEl.style.display = 'none';
        }

        showToast(`Model ${modelName} installed! Loading...`, 'success');

        // Auto-load after install
        await lemonadeClient.loadModel(modelName);
        showToast(`Model ${modelName} ready!`, 'success');
        setStatus('Model Ready', 'success');
        return true;

    } catch (error) {
        console.error('Install error:', error);
        showToast(`Install failed: ${error.message}`, 'error');
        setStatus('Error', 'error');

        // Hide progress (reuse progressEl from line 441)
        if (progressEl) {
            progressEl.style.display = 'none';
        }
        return false;
    }
}

// Fetch and display performance stats after optimization
async function showPerformanceStats() {
    if (!lemonadeClient || AppState.provider !== 'lemonade') return;

    try {
        const stats = await lemonadeClient.getStats();
        if (stats) {
            const ttft = stats.time_to_first_token?.toFixed(2) || '?';
            const tps = stats.tokens_per_second?.toFixed(1) || '?';
            console.log(`Performance: ${ttft}s to first token, ${tps} tokens/sec`);
            showToast(`${tps} tokens/sec, ${ttft}s to first token`, 'info');
        }
    } catch (error) {
        // Stats are optional, don't show error
        console.log('Could not fetch stats:', error.message);
    }
}

// ============================================================================
// UCL OPTIMIZATION (OpenRouter API / Lemonade / OpenAI-compatible)
// ============================================================================
async function optimizeWithUCL() {
    const inputPrompt = document.getElementById('inputPrompt').value.trim();

    if (!inputPrompt) {
        showToast('Please enter a prompt to optimize', 'error');
        return;
    }

    const model = AppState.customModel || AppState.selectedModel;

    // Validate based on provider
    if (AppState.provider === 'openrouter') {
        if (!AppState.apiKey) {
            showToast('Please configure your OpenRouter API key in settings', 'error');
            openModal('settingsModal');
            return;
        }
    } else if (AppState.provider === 'lemonade' || AppState.provider === 'openai_compat') {
        if (!AppState.lemonadeEndpoint) {
            showToast('Please configure your local server endpoint in settings', 'error');
            openModal('settingsModal');
            return;
        }

        // Initialize Lemonade client if needed
        if (!lemonadeClient || lemonadeClient.endpoint !== AppState.lemonadeEndpoint) {
            initLemonadeClient(AppState.lemonadeEndpoint);
        }

        // For Lemonade: ensure model is loaded before inference
        if (AppState.provider === 'lemonade') {
            const isLoaded = await loadModelBeforeOptimize(model);
            if (!isLoaded) return;
        }
    }

    if (!model) {
        showToast('Please select a model', 'error');
        return;
    }

    showLoading(true);
    setStatus('Optimizing...', 'warning');

    try {
        let response;

        if (AppState.provider === 'openrouter') {
            // OpenRouter API
            response = await fetch('https://openrouter.ai/api/v1/chat/completions', {
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
                        { role: 'system', content: getUCLSystemPrompt() },
                        { role: 'user', content: inputPrompt }
                    ],
                    temperature: 0.7,
                    max_tokens: 4000
                })
            });
        } else {
            // Lemonade or any OpenAI-compatible local server
            let endpoint = AppState.lemonadeEndpoint;
            // Ensure endpoint ends with /v1/chat/completions
            if (!endpoint.endsWith('/v1/chat/completions')) {
                endpoint = endpoint.replace(/\/$/, '') + '/v1/chat/completions';
            }

            const headers = {
                'Content-Type': 'application/json'
            };

            // Add API key if provided (optional for local servers)
            if (AppState.lemonadeApiKey) {
                headers['Authorization'] = `Bearer ${AppState.lemonadeApiKey}`;
            }

            response = await fetch(endpoint, {
                method: 'POST',
                headers: headers,
                body: JSON.stringify({
                    model: model,
                    messages: [
                        { role: 'system', content: getUCLSystemPrompt() },
                        { role: 'user', content: inputPrompt }
                    ],
                    temperature: 0.7,
                    max_tokens: 4000
                })
            });
        }

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

        // Show performance stats for Lemonade
        showPerformanceStats();

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
