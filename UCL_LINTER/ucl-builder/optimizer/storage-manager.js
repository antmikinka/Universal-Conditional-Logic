/**
 * StorageManager - IndexedDB Persistence Layer
 *
 * CRITICAL QUALITY FIX #3: Fixed IndexedDB schema:
 * - REMOVED `sessions` store (session info derived from conversations)
 * - Uses auto-increment keys for versions and thinkingLogs (not composite [modelId, versionNum])
 * - Fixed `thinkingLogRef` to reference auto-increment ID
 * - Added proper indexes for efficient querying
 *
 * Schema:
 *   conversations:  keyPath = 'modelId' (one conversation per model)
 *   versions:       auto-increment 'id', indexed by modelId
 *   thinkingLogs:   auto-increment 'id', indexed by modelId
 *   templates:      keyPath = 'templateId' (base prompt templates)
 *
 * Quality fixes addressed:
 * - CRITICAL #3: IndexedDB Schema fixed
 * - CRITICAL #2: Template management added (templates store)
 * - HIGH: Partial failure recovery - atomic operations where possible
 *
 * @module storage-manager
 */

const DB_NAME = 'UCLPromptOptimizer';
const DB_VERSION = 2;

/**
 * Object store names.
 * NOTE: 'sessions' store was removed per quality review.
 * Session info is derived by scanning the 'conversations' store.
 */
const STORES = {
    // v1 stores
    CONVERSATIONS: 'conversations',
    VERSIONS: 'versions',
    THINKING_LOGS: 'thinkingLogs',
    TEMPLATES: 'templates',
    // v2 stores
    MODEL_CONFIGS: 'modelConfigs',
    TEST_RESULTS: 'testResults',
    OPTIMIZATION_RUNS: 'optimizationRuns',
};

/**
 * @typedef {Object} ConversationRecord
 * @property {string} modelId - Unique model identifier (keyPath)
 * @property {Array<{role: string, content: string}>} messages - Accumulated conversation
 * @property {string[]} contextFiles - Names of loaded context files
 * @property {number} updatedAt - Epoch ms timestamp
 * @property {number} tokenEstimate - Estimated token count
 */

/**
 * @typedef {Object} VersionRecord
 * @property {number} id - Auto-increment primary key
 * @property {string} modelId - Model this version belongs to
 * @property {number} versionNum - Sequential version number within model
 * @property {string} promptText - The UCL prompt text
 * @property {number|null} parentVersion - Parent versionNum (null for v1)
 * @property {string} basePrompt - Original raw input prompt
 * @property {number} timestamp - Epoch ms
 * @property {number} tokenCount - Token count of promptText
 * @property {number|null} thinkingLogId - FK to thinkingLogs.id (null until extraction)
 * @property {Object} metadata - Extra info (strategy, feedback, etc.)
 */

/**
 * @typedef {Object} ThinkingLogRecord
 * @property {number} id - Auto-increment primary key
 * @property {string} modelId - Model this log belongs to
 * @property {number} versionNum - Which version this log is for
 * @property {string} content - The extracted thinking/reasoning text
 * @property {string} strategy - Extraction strategy used ('qwen', 'deepseek', etc.)
 * @property {number} timestamp - Epoch ms
 */

/**
 * @typedef {Object} TemplateRecord
 * @property {string} templateId - 'base_convert' | 'base_update' | '{modelId}:convert' | '{modelId}:update'
 * @property {string} name - Human-readable name
 * @property {string} content - Template text
 * @property {string} scope - 'global' | 'model'
 * @property {string|null} modelId - null for global, model ID for per-model override
 * @property {number} updatedAt - Epoch ms
 */

class StorageManager {
    /**
     * @param {string} [dbName=DB_NAME]
     * @param {number} [version=DB_VERSION]
     */
    constructor(dbName = DB_NAME, version = DB_VERSION) {
        this._dbName = dbName;
        this._dbVersion = version;
        /** @type {IDBDatabase|null} */
        this._db = null;
    }

    // ── Lifecycle ─────────────────────────────────────────────────────────

    /**
     * Open/create the IndexedDB database.
     * Idempotent: calling init() multiple times is safe.
     * @returns {Promise<void>}
     */
    async init() {
        if (this._db) return;

        return new Promise((resolve, reject) => {
            const request = indexedDB.open(this._dbName, this._dbVersion);

            request.onupgradeneeded = (event) => {
                const db = event.target.result;
                const oldVersion = event.oldVersion;
                this._createObjectStores(db, oldVersion);
            };

            request.onsuccess = (event) => {
                this._db = event.target.result;
                resolve();
            };

            request.onerror = (event) => {
                reject(new Error(`Failed to open IndexedDB: ${event.target.error}`));
            };
        });
    }

    /**
     * Close the database connection.
     * Safe to call even if not initialized.
     */
    close() {
        if (this._db) {
            this._db.close();
            this._db = null;
        }
    }

    /**
     * Delete the entire database (for testing/reset).
     * @returns {Promise<void>}
     */
    async destroy() {
        this.close();
        return new Promise((resolve, reject) => {
            const request = indexedDB.deleteDatabase(this._dbName);
            request.onsuccess = () => resolve();
            request.onerror = () => reject(new Error('Failed to delete database'));
        });
    }

    // ── Conversations ─────────────────────────────────────────────────────

    /**
     * Save/overwrite the conversation for a model.
     * @param {string} modelId
     * @param {Array<{role: string, content: string}>} messages
     * @param {Object} [extra] - Additional fields (contextFiles, tokenEstimate)
     * @returns {Promise<void>}
     */
    async saveConversation(modelId, messages, extra = {}) {
        this._ensureInit();
        const record = {
            modelId,
            messages,
            contextFiles: extra.contextFiles || [],
            updatedAt: Date.now(),
            tokenEstimate: extra.tokenEstimate || 0,
        };
        await this._put(STORES.CONVERSATIONS, record);
    }

    /**
     * Load the conversation for a model.
     * @param {string} modelId
     * @returns {Promise<ConversationRecord|null>}
     */
    async loadConversation(modelId) {
        this._ensureInit();
        return this._get(STORES.CONVERSATIONS, modelId);
    }

    /**
     * Check if a conversation exists for a model.
     * @param {string} modelId
     * @returns {Promise<boolean>}
     */
    async hasConversation(modelId) {
        const convo = await this.loadConversation(modelId);
        return convo !== null && convo !== undefined;
    }

    /**
     * Delete the conversation for a model.
     * @param {string} modelId
     * @returns {Promise<void>}
     */
    async deleteConversation(modelId) {
        this._ensureInit();
        await this._delete(STORES.CONVERSATIONS, modelId);
    }

    // ── Versions ──────────────────────────────────────────────────────────

    /**
     * Save a new version record. Auto-increments the 'id' field.
     * @param {VersionRecord} versionObj
     * @returns {Promise<number>} - The auto-incremented id
     */
    async saveVersion(versionObj) {
        this._ensureInit();
        // Don't set 'id' — let auto-increment handle it
        const record = { ...versionObj };
        delete record.id;
        return this._add(STORES.VERSIONS, record);
    }

    /**
     * Update an existing version record in place (requires record.id).
     * Use this for thinkingLogId / metadata updates — never saveVersion for updates
     * (saveVersion always inserts and would create duplicate versionNum rows).
     *
     * @param {VersionRecord} versionObj - Must include IndexedDB auto-increment `id`
     * @returns {Promise<void>}
     */
    async updateVersion(versionObj) {
        this._ensureInit();
        if (versionObj == null || versionObj.id == null) {
            throw new Error('updateVersion requires a version record with id');
        }
        await this._put(STORES.VERSIONS, versionObj);
    }

    /**
     * Get all versions for a model, ordered by versionNum ascending.
     * @param {string} modelId
     * @returns {Promise<VersionRecord[]>}
     */
    async getVersions(modelId) {
        this._ensureInit();
        return this._getAllByIndex(STORES.VERSIONS, 'modelId', modelId);
    }

    /**
     * Get a specific version by modelId + versionNum.
     * @param {string} modelId
     * @param {number} versionNum
     * @returns {Promise<VersionRecord|null>}
     */
    async getVersion(modelId, versionNum) {
        const versions = await this.getVersions(modelId);
        return versions.find(v => v.versionNum === versionNum) || null;
    }

    /**
     * Get the latest version for a model (highest versionNum).
     * @param {string} modelId
     * @returns {Promise<VersionRecord|null>}
     */
    async getLatestVersion(modelId) {
        const versions = await this.getVersions(modelId);
        if (versions.length === 0) return null;
        return versions.reduce((latest, v) => v.versionNum > latest.versionNum ? v : latest, versions[0]);
    }

    /**
     * Delete all versions for a model.
     * @param {string} modelId
     * @returns {Promise<number>} - Number of records deleted
     */
    async deleteVersions(modelId) {
        this._ensureInit();
        return this._deleteByIndex(STORES.VERSIONS, 'modelId', modelId);
    }

    // ── Thinking Logs ─────────────────────────────────────────────────────

    /**
     * Save a thinking log. Returns the auto-incremented id.
     * @param {string} modelId
     * @param {number} versionNum
     * @param {string} content - Extracted reasoning text
     * @param {string} [strategy='generic'] - Extraction strategy name
     * @returns {Promise<number>} - The thinking log id (for referencing from version)
     */
    async saveThinkingLog(modelId, versionNum, content, strategy = 'generic') {
        this._ensureInit();
        const record = {
            modelId,
            versionNum,
            content,
            strategy,
            timestamp: Date.now(),
        };
        return this._add(STORES.THINKING_LOGS, record);
    }

    /**
     * Get a thinking log by its id.
     * @param {number} logId - Auto-incremented id from saveThinkingLog()
     * @returns {Promise<ThinkingLogRecord|null>}
     */
    async getThinkingLogById(logId) {
        this._ensureInit();
        return this._get(STORES.THINKING_LOGS, logId);
    }

    /**
     * Get thinking log(s) for a specific model + versionNum.
     * @param {string} modelId
     * @param {number} versionNum
     * @returns {Promise<ThinkingLogRecord[]>}
     */
    async getThinkingLogsForVersion(modelId, versionNum) {
        this._ensureInit();
        const allLogs = await this._getAllByIndex(STORES.THINKING_LOGS, 'modelId', modelId);
        return allLogs.filter(log => log.versionNum === versionNum);
    }

    /**
     * Delete all thinking logs for a model.
     * @param {string} modelId
     * @returns {Promise<number>}
     */
    async deleteThinkingLogs(modelId) {
        this._ensureInit();
        return this._deleteByIndex(STORES.THINKING_LOGS, 'modelId', modelId);
    }

    // ── Templates (CRITICAL FIX #2) ───────────────────────────────────────

    /**
     * Save a prompt template.
     * @param {TemplateRecord} template
     * @returns {Promise<void>}
     */
    async saveTemplate(template) {
        this._ensureInit();
        template.updatedAt = Date.now();
        await this._put(STORES.TEMPLATES, template);
    }

    /**
     * Get a template by ID.
     * Falls back to global template if model-specific override doesn't exist.
     *
     * @param {string} templateId - 'base_convert' or 'base_update'
     * @param {string} [modelId] - Optional model ID for per-model override
     * @returns {Promise<TemplateRecord|null>}
     */
    async getTemplate(templateId, modelId) {
        this._ensureInit();

        // Try model-specific override first
        if (modelId) {
            const modelKey = `${modelId}:${templateId.replace(/^base_/, '')}`;
            const modelTemplate = await this._get(STORES.TEMPLATES, modelKey);
            if (modelTemplate) return modelTemplate;
        }

        // Fall back to global
        return this._get(STORES.TEMPLATES, templateId);
    }

    /**
     * Get all templates, optionally filtered by scope.
     * @param {'global'|'model'} [scope]
     * @returns {Promise<TemplateRecord[]>}
     */
    async getAllTemplates(scope) {
        this._ensureInit();
        const all = await this._getAll(STORES.TEMPLATES);
        if (scope) {
            return all.filter(t => t.scope === scope);
        }
        return all;
    }

    /**
     * Initialize default templates if none exist.
     * Called during init to ensure templates are always available.
     * @returns {Promise<void>}
     */
    async initDefaultTemplates() {
        const existing = await this.getAllTemplates('global');
        if (existing.length > 0) return; // Already initialized

        const defaults = this._getDefaultTemplates();
        for (const template of defaults) {
            await this.saveTemplate(template);
        }
    }

    // ── Sessions (derived from conversations) ─────────────────────────────

    /**
     * List all model sessions (derived from conversations store).
     * CRITICAL FIX: No separate 'sessions' store needed.
     *
     * @returns {Promise<Array<{modelId: string, updatedAt: number, versionCount: number, contextFileCount: number, messageCount: number}>>}
     */
    async listSessions() {
        this._ensureInit();
        const conversations = await this._getAll(STORES.CONVERSATIONS);

        const sessions = [];
        for (const convo of conversations) {
            const versions = await this.getVersions(convo.modelId);
            sessions.push({
                modelId: convo.modelId,
                displayName: convo.modelId, // Could be enhanced with model registry
                updatedAt: convo.updatedAt,
                versionCount: versions.length,
                contextFileCount: (convo.contextFiles || []).length,
                messageCount: (convo.messages || []).length,
                tokenEstimate: convo.tokenEstimate || 0,
            });
        }

        // Sort by most recently updated
        sessions.sort((a, b) => b.updatedAt - a.updatedAt);
        return sessions;
    }

    /**
     * Delete all data for a model session (conversation, versions, thinking logs).
     * @param {string} modelId
     * @returns {Promise<void>}
     */
    async deleteSession(modelId) {
        this._ensureInit();
        await Promise.all([
            this.deleteConversation(modelId),
            this.deleteVersions(modelId),
            this.deleteThinkingLogs(modelId),
        ]);
    }

    /**
     * Export all data for a model session as JSON.
     * @param {string} modelId
     * @returns {Promise<Object>}
     */
    async exportSession(modelId) {
        this._ensureInit();
        const [conversation, versions, allLogs] = await Promise.all([
            this.loadConversation(modelId),
            this.getVersions(modelId),
            this._getAllByIndex(STORES.THINKING_LOGS, 'modelId', modelId),
        ]);

        return {
            exportVersion: 1,
            modelId,
            exportedAt: Date.now(),
            conversation,
            versions,
            thinkingLogs: allLogs,
        };
    }

    /**
     * Import session data from a JSON export.
     * @param {string} modelId
     * @param {Object} data - From exportSession()
     * @returns {Promise<void>}
     */
    async importSession(modelId, data) {
        this._ensureInit();

        // Clear existing data for this model
        await this.deleteSession(modelId);

        // Import conversation
        if (data.conversation) {
            await this.saveConversation(modelId, data.conversation.messages, {
                contextFiles: data.conversation.contextFiles || [],
                tokenEstimate: data.conversation.tokenEstimate || 0,
            });
        }

        // Import versions
        if (data.versions) {
            for (const version of data.versions) {
                await this.saveVersion({ ...version, modelId });
            }
        }

        // Import thinking logs
        if (data.thinkingLogs) {
            for (const log of data.thinkingLogs) {
                await this.saveThinkingLog(modelId, log.versionNum, log.content, log.strategy);
            }
        }
    }

    // ── Model Configs (v2) ────────────────────────────────────────────────

    /**
     * Save a model-specific configuration.
     * @param {ModelConfig} config - Model configuration object
     * @returns {Promise<void>}
     */
    async saveModelConfig(config) {
        this._ensureInit();
        await this._put(STORES.MODEL_CONFIGS, config);
    }

    /**
     * Get a model configuration by modelId.
     * @param {string} modelId
     * @returns {Promise<ModelConfig|null>}
     */
    async getModelConfig(modelId) {
        this._ensureInit();
        return this._get(STORES.MODEL_CONFIGS, modelId);
    }

    /**
     * List all saved model configurations.
     * @returns {Promise<ModelConfig[]>}
     */
    async listModelConfigs() {
        this._ensureInit();
        return this._getAll(STORES.MODEL_CONFIGS);
    }

    /**
     * Delete a model configuration by modelId.
     * @param {string} modelId
     * @returns {Promise<void>}
     */
    async deleteModelConfig(modelId) {
        this._ensureInit();
        await this._delete(STORES.MODEL_CONFIGS, modelId);
    }

    // ── Test Cases & Results (v2) ─────────────────────────────────────────

    /**
     * Save a set of test case definitions as a batch.
     * Stored as a single document under key '_test_cases' in modelConfigs.
     * @param {Object[]} testCases - Array of test case definitions
     * @returns {Promise<void>}
     */
    async saveTestCases(testCases) {
        this._ensureInit();
        await this._put(STORES.MODEL_CONFIGS, {
            modelId: '_test_cases',
            testCases,
            updatedAt: Date.now(),
        });
    }

    /**
     * Load previously saved test case definitions.
     * @returns {Promise<Object[]>} - Array of test cases, or empty array if none saved
     */
    async loadTestCases() {
        this._ensureInit();
        const record = await this._get(STORES.MODEL_CONFIGS, '_test_cases');
        return record ? (record.testCases || []) : [];
    }

    /**
     * Save a single test result (auto-incremented key).
     * @param {TestResult} result - Test result record
     * @returns {Promise<number>} - The auto-incremented id
     */
    async saveTestResult(result) {
        this._ensureInit();
        const record = { ...result };
        delete record.id;
        return this._add(STORES.TEST_RESULTS, record);
    }

    /**
     * Get all test results for a specific optimization run.
     * @param {string} runId
     * @returns {Promise<TestResult[]>}
     */
    async getTestResultsForRun(runId) {
        this._ensureInit();
        return this._getAllByIndex(STORES.TEST_RESULTS, 'runId', runId);
    }

    /**
     * Get all test results for a specific version number, filtered by model.
     * @param {string} modelId - Model ID to filter by
     * @param {number} versionNum - Version number to filter by
     * @returns {Promise<TestResult[]>}
     */
    async getTestResultsForVersion(modelId, versionNum) {
        this._ensureInit();
        const results = await this._getAllByIndex(STORES.TEST_RESULTS, 'versionNum', versionNum);
        return results.filter(r => r.modelId === modelId);
    }

    // ── Optimization Runs (v2) ────────────────────────────────────────────

    /**
     * Save an optimization run record.
     * @param {OptimizationRun} run - Optimization run record
     * @returns {Promise<void>}
     */
    async saveOptimizationRun(run) {
        this._ensureInit();
        await this._put(STORES.OPTIMIZATION_RUNS, run);
    }

    /**
     * Get an optimization run by runId.
     * @param {string} runId
     * @returns {Promise<OptimizationRun|null>}
     */
    async getOptimizationRun(runId) {
        this._ensureInit();
        return this._get(STORES.OPTIMIZATION_RUNS, runId);
    }

    /**
     * List all optimization runs, ordered by startedAt descending.
     * @returns {Promise<OptimizationRun[]>}
     */
    async listOptimizationRuns() {
        this._ensureInit();
        const all = await this._getAll(STORES.OPTIMIZATION_RUNS);
        all.sort((a, b) => b.startedAt - a.startedAt);
        return all;
    }

    /**
     * Update the quality score on a version record.
     * Finds the version by modelId + versionNum and updates its metadata.qualityScore.
     * @param {string} modelId
     * @param {number} versionNum
     * @param {number} qualityScore - Quality score 0.0-1.0
     * @returns {Promise<void>}
     */
    async updateVersionQuality(modelId, versionNum, qualityScore) {
        this._ensureInit();
        const versions = await this.getVersions(modelId);
        const version = versions.find(v => v.versionNum === versionNum);
        if (!version) {
            throw new Error(`Version ${versionNum} not found for model ${modelId}`);
        }
        version.metadata = version.metadata || {};
        version.metadata.qualityScore = qualityScore;
        await this._put(STORES.VERSIONS, version);
    }

    // ── Private: Schema ───────────────────────────────────────────────────

    /**
     * Create all object stores and indexes.
     * Version-guarded: v1 stores created when oldVersion < 1,
     * v2 stores created when oldVersion < 2.
     * @private
     * @param {IDBDatabase} db
     * @param {number} oldVersion - Previous database version from upgrade event
     */
    _createObjectStores(db, oldVersion = 0) {
        // ── v1 stores (original schema) ──────────────────────────────────
        if (oldVersion < 1) {
            // Conversations: keyPath = modelId (one per model)
            if (!db.objectStoreNames.contains(STORES.CONVERSATIONS)) {
                const convoStore = db.createObjectStore(STORES.CONVERSATIONS, { keyPath: 'modelId' });
                convoStore.createIndex('updatedAt', 'updatedAt', { unique: false });
            }

            // Versions: auto-increment 'id' (FIXED: was composite [modelId, versionNum])
            if (!db.objectStoreNames.contains(STORES.VERSIONS)) {
                const versionStore = db.createObjectStore(STORES.VERSIONS, { autoIncrement: true });
                versionStore.createIndex('modelId', 'modelId', { unique: false });
                versionStore.createIndex('modelId_versionNum', ['modelId', 'versionNum'], { unique: true });
            }

            // Thinking Logs: auto-increment 'id' (FIXED: was composite [modelId, versionNum])
            if (!db.objectStoreNames.contains(STORES.THINKING_LOGS)) {
                const logStore = db.createObjectStore(STORES.THINKING_LOGS, { autoIncrement: true });
                logStore.createIndex('modelId', 'modelId', { unique: false });
                logStore.createIndex('modelId_versionNum', ['modelId', 'versionNum'], { unique: false });
            }

            // Templates: keyPath = templateId
            if (!db.objectStoreNames.contains(STORES.TEMPLATES)) {
                db.createObjectStore(STORES.TEMPLATES, { keyPath: 'templateId' });
            }
        }

        // ── v2 stores (optimizer extension) ──────────────────────────────
        if (oldVersion < 2) {
            // Model Configs: keyPath = modelId (one config per model)
            if (!db.objectStoreNames.contains(STORES.MODEL_CONFIGS)) {
                db.createObjectStore(STORES.MODEL_CONFIGS, { keyPath: 'modelId' });
            }

            // Test Results: auto-increment, indexed for flexible querying
            if (!db.objectStoreNames.contains(STORES.TEST_RESULTS)) {
                const testStore = db.createObjectStore(STORES.TEST_RESULTS, { autoIncrement: true });
                testStore.createIndex('runId', 'runId', { unique: false });
                testStore.createIndex('testCaseId', 'testCaseId', { unique: false });
                testStore.createIndex('versionNum', 'versionNum', { unique: false });
                testStore.createIndex('runId_testCaseId', ['runId', 'testCaseId'], { unique: false });
            }

            // Optimization Runs: keyPath = runId
            if (!db.objectStoreNames.contains(STORES.OPTIMIZATION_RUNS)) {
                const runStore = db.createObjectStore(STORES.OPTIMIZATION_RUNS, { keyPath: 'runId' });
                runStore.createIndex('modelId', 'modelId', { unique: false });
                runStore.createIndex('status', 'status', { unique: false });
                runStore.createIndex('startedAt', 'startedAt', { unique: false });
            }
        }
    }

    /**
     * Default prompt templates.
     * CRITICAL FIX #2: Template management.
     * @private
     * @returns {TemplateRecord[]}
     */
    _getDefaultTemplates() {
        return [
            {
                templateId: 'base_convert',
                name: 'Base Conversion Prompt',
                scope: 'global',
                modelId: null,
                content: `You are an expert at converting natural language prompts into Universal Conditional Logic (UCL) format.

RULES:
1. ALWAYS start with [[CRITICAL:]] directive defining the primary role/identity
2. Use ^^CONDITION: keywords^^ blocks for conditional logic activation
3. Close all conditions with ^^/CONDITION^^
4. Group related tools under shared conditions
5. Use [[PARAM: name:type:required]] for parameters
6. Use [[REQUIRE:]], [[ENFORCE:]], [[VALIDATE:]], [[APPLY:]] for quality directives
7. Prefer keyword CONDITIONS over SWITCH statements
8. Keep conditions focused (3-5 keywords max)

Convert the following prompt to UCL format:`,
                updatedAt: 0,
            },
            {
                templateId: 'base_update',
                name: 'Base Update/Refinement Prompt',
                scope: 'global',
                modelId: null,
                content: `You are refining an existing UCL prompt based on testing feedback and reasoning analysis.

INPUT:
- The original context and requirements
- The current UCL prompt version
- The model's reasoning/thinking log from testing
- Specific feedback for improvement

INSTRUCTIONS:
1. Analyze the thinking log to understand where the model struggled or succeeded
2. Address the specific feedback provided
3. Maintain valid UCL syntax
4. Improve clarity, specificity, and effectiveness
5. Keep the same overall structure unless feedback warrants changes
6. Output ONLY the improved UCL prompt (no explanations)

Current UCL prompt:
{currentPrompt}

Model's reasoning from testing:
{thinkingLog}

Feedback for improvement:
{feedback}

Produce the improved UCL prompt:`,
                updatedAt: 0,
            },
        ];
    }

    // ── Private: IndexedDB helpers ────────────────────────────────────────

    /** @private */
    _ensureInit() {
        if (!this._db) {
            throw new Error('StorageManager not initialized. Call init() first.');
        }
    }

    /** @private */
    _get(storeName, key) {
        return new Promise((resolve, reject) => {
            const tx = this._db.transaction(storeName, 'readonly');
            const store = tx.objectStore(storeName);
            const request = store.get(key);
            request.onsuccess = () => resolve(request.result || null);
            request.onerror = () => reject(new Error(`GET failed: ${request.error}`));
        });
    }

    /** @private */
    _put(storeName, value) {
        return new Promise((resolve, reject) => {
            const tx = this._db.transaction(storeName, 'readwrite');
            const store = tx.objectStore(storeName);
            const request = store.put(value);
            request.onsuccess = () => resolve(request.result);
            request.onerror = () => reject(new Error(`PUT failed: ${request.error}`));
        });
    }

    /** @private - Returns auto-increment key */
    _add(storeName, value) {
        return new Promise((resolve, reject) => {
            const tx = this._db.transaction(storeName, 'readwrite');
            const store = tx.objectStore(storeName);
            const request = store.add(value);
            request.onsuccess = () => resolve(request.result);
            request.onerror = () => reject(new Error(`ADD failed: ${request.error}`));
        });
    }

    /** @private */
    _delete(storeName, key) {
        return new Promise((resolve, reject) => {
            const tx = this._db.transaction(storeName, 'readwrite');
            const store = tx.objectStore(storeName);
            const request = store.delete(key);
            request.onsuccess = () => resolve();
            request.onerror = () => reject(new Error(`DELETE failed: ${request.error}`));
        });
    }

    /** @private */
    _getAll(storeName) {
        return new Promise((resolve, reject) => {
            const tx = this._db.transaction(storeName, 'readonly');
            const store = tx.objectStore(storeName);
            const request = store.getAll();
            request.onsuccess = () => resolve(request.result || []);
            request.onerror = () => reject(new Error(`GET_ALL failed: ${request.error}`));
        });
    }

    /** @private */
    _getAllByIndex(storeName, indexName, value) {
        return new Promise((resolve, reject) => {
            const tx = this._db.transaction(storeName, 'readonly');
            const store = tx.objectStore(storeName);
            const index = store.index(indexName);
            const request = index.getAll(value);
            request.onsuccess = () => resolve(request.result || []);
            request.onerror = () => reject(new Error(`GET_ALL_BY_INDEX failed: ${request.error}`));
        });
    }

    /** @private */
    _deleteByIndex(storeName, indexName, value) {
        return new Promise((resolve, reject) => {
            const tx = this._db.transaction(storeName, 'readwrite');
            const store = tx.objectStore(storeName);
            const index = store.index(indexName);
            const request = index.openCursor(IDBKeyRange.only(value));
            let count = 0;
            request.onsuccess = (event) => {
                const cursor = event.target.result;
                if (cursor) {
                    cursor.delete();
                    count++;
                    cursor.continue();
                } else {
                    resolve(count);
                }
            };
            request.onerror = () => reject(new Error(`DELETE_BY_INDEX failed: ${request.error}`));
        });
    }
}

// ── Exports ────────────────────────────────────────────────────────────────
export { StorageManager, STORES, DB_NAME, DB_VERSION };
