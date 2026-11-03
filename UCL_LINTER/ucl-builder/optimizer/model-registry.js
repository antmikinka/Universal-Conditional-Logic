/**
 * ModelRegistry - Model Configuration Manager
 *
 * Manages model-specific configurations with a fallback chain:
 *   1. Static MODEL_DEFAULTS (built-in configs for known model families)
 *   2. Database overrides (user-customized configs)
 *   3. Generic fallback (auto-generated for unknown models)
 *
 * Supports 6 model families: Qwen, Claude, GPT, Gemini, DeepSeek, Llama.
 *
 * @module model-registry
 */

/**
 * @typedef {Object} ModelConfig
 * @property {string} modelId - Unique model identifier
 * @property {string} family - Detected model family ('qwen', 'claude', 'gpt', 'gemini', 'deepseek', 'llama', 'unknown')
 * @property {number} temperature - Sampling temperature (0.0-2.0)
 * @property {number} maxTokens - Maximum output tokens
 * @property {number} thinkingBudget - Token budget for thinking/reasoning (0 to disable)
 * @property {string} endpoint - API endpoint URL
 * @property {string} [thinkingStart] - Marker for thinking block start
 * @property {string} [thinkingEnd] - Marker for thinking block end
 * @property {Object} [extraParams] - Additional model-specific parameters
 * @property {number} updatedAt - Epoch ms timestamp of last modification
 * @property {string} [source] - 'default' | 'database' | 'generic'
 */

/**
 * Static default configurations for known model families.
 * @type {Map<string, ModelConfig>}
 */
const MODEL_DEFAULTS = new Map([
    // ── Qwen family ──────────────────────────────────────────────────────
    ['qwen', {
        modelId: 'qwen',
        family: 'qwen',
        temperature: 0.7,
        maxTokens: 8192,
        thinkingBudget: 4096,
        endpoint: 'http://localhost:8080/v1',
        thinkingStart: '<think>',
        thinkingEnd: '</think>',
        extraParams: { top_k: 20, top_p: 0.9 },
        updatedAt: 0,
        source: 'default',
    }],

    // ── Claude family ────────────────────────────────────────────────────
    ['claude', {
        modelId: 'claude',
        family: 'claude',
        temperature: 0.7,
        maxTokens: 8192,
        thinkingBudget: 10240,
        endpoint: 'https://api.anthropic.com/v1',
        thinkingStart: '<thinking>',
        thinkingEnd: '</thinking>',
        extraParams: { top_k: 40, top_p: 0.95 },
        updatedAt: 0,
        source: 'default',
    }],

    // ── GPT family ──────────────────────────────────────────────────────
    ['gpt', {
        modelId: 'gpt',
        family: 'gpt',
        temperature: 0.7,
        maxTokens: 4096,
        thinkingBudget: 0,
        endpoint: 'https://api.openai.com/v1',
        thinkingStart: '',
        thinkingEnd: '',
        extraParams: { frequency_penalty: 0, presence_penalty: 0 },
        updatedAt: 0,
        source: 'default',
    }],

    // ── Gemini family ────────────────────────────────────────────────────
    ['gemini', {
        modelId: 'gemini',
        family: 'gemini',
        temperature: 0.7,
        maxTokens: 8192,
        thinkingBudget: 0,
        endpoint: 'https://generativelanguage.googleapis.com/v1beta',
        thinkingStart: '',
        thinkingEnd: '',
        extraParams: { top_k: 40, top_p: 0.95 },
        updatedAt: 0,
        source: 'default',
    }],

    // ── DeepSeek family ──────────────────────────────────────────────────
    ['deepseek', {
        modelId: 'deepseek',
        family: 'deepseek',
        temperature: 0.7,
        maxTokens: 8192,
        thinkingBudget: 8192,
        endpoint: 'https://api.deepseek.com/v1',
        thinkingStart: '<think>',
        thinkingEnd: '</think>',
        extraParams: { top_k: 20, top_p: 0.95 },
        updatedAt: 0,
        source: 'default',
    }],

    // ── Llama family ─────────────────────────────────────────────────────
    ['llama', {
        modelId: 'llama',
        family: 'llama',
        temperature: 0.7,
        maxTokens: 4096,
        thinkingBudget: 0,
        endpoint: 'http://localhost:8080/v1',
        thinkingStart: '',
        thinkingEnd: '',
        extraParams: { top_k: 40, top_p: 0.9 },
        updatedAt: 0,
        source: 'default',
    }],
]);

/**
 * Family detection patterns (regex).
 * Order matters: more specific patterns first.
 * @type {Array<{family: string, pattern: RegExp}>}
 */
const FAMILY_PATTERNS = [
    { family: 'qwen',     pattern: /qwen/i },
    { family: 'claude',   pattern: /claude|anthropic/i },
    { family: 'gpt',      pattern: /gpt[-_\s]?|o1[-_\s]?|o3[-_\s]?|openai/i },
    { family: 'gemini',   pattern: /gemini|google/i },
    { family: 'deepseek', pattern: /deepseek/i },
    { family: 'llama',    pattern: /llama|meta[-_\s]?llama|llm/i },
];

class ModelRegistry {
    /**
     * @param {import('./storage-manager.js').StorageManager} storageManager
     */
    constructor(storageManager) {
        /** @type {import('./storage-manager.js').StorageManager} */
        this._storage = storageManager;
    }

    // ── Static Methods ────────────────────────────────────────────────────

    /**
     * Detect the model family from a modelId string.
     * Pure function — no side effects, no I/O.
     *
     * @param {string} modelId - The model identifier (e.g., 'qwen-2.5-72b', 'claude-3-opus')
     * @returns {string} Family name: 'qwen'|'claude'|'gpt'|'gemini'|'deepseek'|'llama'|'unknown'
     */
    static detectFamily(modelId) {
        if (!modelId || typeof modelId !== 'string') return 'unknown';

        for (const { family, pattern } of FAMILY_PATTERNS) {
            if (pattern.test(modelId)) {
                return family;
            }
        }
        return 'unknown';
    }

    /**
     * Get the default configuration for a known model family.
     * Returns a deep copy to prevent mutation of defaults.
     *
     * @param {string} modelId - The model identifier
     * @returns {ModelConfig|null} Default config for the family, or null if unknown
     */
    static getDefaultConfig(modelId) {
        const family = ModelRegistry.detectFamily(modelId);
        if (family === 'unknown') return null;

        const defaultConfig = MODEL_DEFAULTS.get(family);
        if (!defaultConfig) return null;

        // Return deep copy with modelId overridden
        return {
            ...structuredClone(defaultConfig),
            modelId,
        };
    }

    /**
     * Create a generic configuration for an unknown model.
     * Uses conservative defaults suitable for any model.
     *
     * @param {string} modelId - The model identifier
     * @param {string} [endpoint='http://localhost:8080/v1'] - API endpoint
     * @returns {ModelConfig} Generic config with safe defaults
     */
    static createGenericConfig(modelId, endpoint = 'http://localhost:8080/v1') {
        return {
            modelId,
            family: 'unknown',
            temperature: 0.7,
            maxTokens: 4096,
            thinkingBudget: 0,
            endpoint,
            thinkingStart: '',
            thinkingEnd: '',
            extraParams: {},
            updatedAt: Date.now(),
            source: 'generic',
        };
    }

    // ── Instance Methods ──────────────────────────────────────────────────

    /**
     * Get the configuration for a model.
     * Fallback chain: static defaults → database → generic.
     *
     * @param {string} modelId
     * @returns {Promise<ModelConfig>}
     */
    async getConfig(modelId) {
        // 1. Try database override first (user has customized this model)
        const dbConfig = await this._storage.getModelConfig(modelId);
        if (dbConfig) {
            return { ...dbConfig, source: dbConfig.source || 'database' };
        }

        // 2. Try static defaults for known families
        const defaultConfig = ModelRegistry.getDefaultConfig(modelId);
        if (defaultConfig) {
            return defaultConfig;
        }

        // 3. Fallback: create generic config
        return ModelRegistry.createGenericConfig(modelId);
    }

    /**
     * Save a model configuration to the database.
     * This creates a user override that takes precedence over defaults.
     *
     * @param {ModelConfig} config
     * @returns {Promise<void>}
     */
    async saveConfig(config) {
        config.updatedAt = Date.now();
        config.source = 'database';
        await this._storage.saveModelConfig(config);
    }

    /**
     * List all saved model configurations from the database.
     * Does NOT include default configs that haven't been overridden.
     *
     * @returns {Promise<ModelConfig[]>}
     */
    async listConfigs() {
        const configs = await this._storage.listModelConfigs();
        // Filter out the special _test_cases record
        return configs.filter(c => c.modelId !== '_test_cases');
    }

    /**
     * Reset a model configuration to its default.
     * Removes the database override so the default chain takes effect.
     *
     * @param {string} modelId
     * @returns {Promise<void>}
     */
    async resetToDefault(modelId) {
        await this._storage.deleteModelConfig(modelId);
    }
}

export { ModelRegistry, MODEL_DEFAULTS, FAMILY_PATTERNS };
