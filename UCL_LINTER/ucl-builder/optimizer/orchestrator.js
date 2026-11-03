/**
 * ModelOptimizerOrchestrator - State Machine
 *
 * Coordinates the full optimization loop following the PipelineOrchestrator
 * pattern from pipeline/pipeline-orchestrator.js.
 *
 * State Machine with explicit loop semantics:
 *   IDLE → LOADING_CONTEXT → CONVERTING → TESTING → EXTRACTING → UPDATING → COMPLETE
 *     ↑                                                              │
 *     └──────────── LOOP_BACK (for next iteration) ─────────────────┘
 *
 * QUALITY FIXES:
 * - State machine loop semantics are EXPLICIT (LOOP_BACK event from COMPLETE → CONVERTING)
 * - User feedback mechanism for update phase (required feedback string)
 * - Partial failure recovery in context loading
 * - Resource cleanup via dispose()
 * - Uses shared LLMClient (no API duplication)
 * - Template management via StorageManager
 *
 * Usage:
 *   const orchestrator = new ModelOptimizerOrchestrator({
 *     modelId: 'qwen3-4b-FLM',
 *     endpoint: 'http://localhost:13305',
 *     onStateChange: (state, prev, event) => ...,
 *     onProgress: (phase, step, total, detail) => ...,
 *     onError: (phase, error) => ...,
 *   });
 *   await orchestrator.execute({ contextFiles, userPrompt });
 *
 * @module orchestrator
 */

import { LLMClient } from './llm-client.js';
import { StorageManager } from './storage-manager.js';
import { ConversationManager } from './conversation-manager.js';
import { ThinkingExtractor } from './thinking-extractor.js';
import { VersionManager } from './version-manager.js';
import { FileLoader } from './file-loader.js';
import { ModelRegistry } from './model-registry.js';
import { TestEvaluator } from './test-evaluator.js';
import { OptimizationHistory } from './optimization-history.js';
import { QualityScorer } from './quality-scorer.js';
import { PromptAdapter } from './prompt-adapter.js';
import { IterationController } from './iteration-controller.js';

// ── States & Events ───────────────────────────────────────────────────────

/**
 * Orchestrator states.
 * Loop semantics:
 *   - COMPLETE + LOOP_BACK → CONVERTING (start next iteration from existing context)
 *   - COMPLETE + RESET → IDLE (full reset)
 */
const STATES = {
    IDLE: 'IDLE',
    LOADING_CONTEXT: 'LOADING_CONTEXT',
    CONVERTING: 'CONVERTING',
    TESTING: 'TESTING',
    EXTRACTING: 'EXTRACTING',
    UPDATING: 'UPDATING',
    COMPLETE: 'COMPLETE',
    ERROR: 'ERROR',
};

/**
 * Orchestrator events.
 * LOOP_BACK: Explicit event for iteration loops (COMPLETE → CONVERTING).
 * This makes loop semantics unambiguous (quality fix).
 */
const EVENTS = {
    START: 'START',
    CONTEXT_LOADED: 'CONTEXT_LOADED',
    CONVERSION_COMPLETE: 'CONVERSION_COMPLETE',
    TEST_COMPLETE: 'TEST_COMPLETE',
    EXTRACTION_COMPLETE: 'EXTRACTION_COMPLETE',
    UPDATE_COMPLETE: 'UPDATE_COMPLETE',
    LOOP_BACK: 'LOOP_BACK',     // Explicit loop event
    ERROR: 'ERROR',
    RESET: 'RESET',
};

// ── Orchestrator ──────────────────────────────────────────────────────────

class ModelOptimizerOrchestrator {
    /**
     * @param {Object} config
     * @param {string} config.modelId - Model identifier (shared identity)
     * @param {string} [config.endpoint='http://localhost:13305'] - API endpoint
     * @param {string} [config.apiKey=''] - API key
     * @param {string} [config.model] - Model name for API (defaults to modelId)
     * @param {Function} [config.onStateChange] - (newState, prevState, event) => void
     * @param {Function} [config.onProgress] - (phase, step, total, detail) => void
     * @param {Function} [config.onError] - (phase, error) => void
     * @param {Function} [config.onVersionCreated] - (version) => void
     * @param {Function} [config.onIteration] - (iterationResult) => void - callback per iteration
     * @param {Function} [config.onOptimizationComplete] - (optimizationResult) => void
     * @param {Object} [config.modelRegistry] - Pre-configured ModelRegistry (optional, for DI/testing)
     * @param {Object} [config.testEvaluator] - Pre-configured TestEvaluator (optional, for DI/testing)
     * @param {Object} [config.optimizationHistory] - Pre-configured OptimizationHistory (optional, for DI/testing)
     * @param {Object} [config.optimization] - Optimization loop configuration
     */
    constructor(config = {}) {
        if (!config.modelId) throw new Error('ModelOptimizerOrchestrator requires modelId');

        this.modelId = config.modelId;

        // Callbacks
        this._onStateChange = config.onStateChange || (() => {});
        this._onProgress = config.onProgress || (() => {});
        this._onError = config.onError || (() => {});
        this._onVersionCreated = config.onVersionCreated || (() => {});
        this._onIteration = config.onIteration || null;
        this._onOptimizationComplete = config.onOptimizationComplete || null;

        // State
        this.currentState = STATES.IDLE;
        this.error = null;
        this._disposable = []; // Track resources for cleanup

        // Phase 4: Iterative optimization state
        this.cancelled = false;
        this.currentRunId = null;
        this.uclPrompt = null;
        this.currentPrompt = null;
        this.lastThinkingLog = null;
        this.lastMetrics = null;
        this.versionCounter = 0;
        this.testInput = null;
        this.iterationConfig = config.optimization || {
            maxIterations: 5,
            qualityThreshold: 0.85,
            convergenceDelta: 0.02,
            convergencePatience: 2,
            tokenBudget: 0,
        };

        // Optional injected dependencies (for DI/testing)
        this._modelRegistry = config.modelRegistry || null;
        this._testEvaluator = config.testEvaluator || null;
        this._optimizationHistory = config.optimizationHistory || null;

        // Shared LLMClient (CRITICAL FIX #1)
        this._llmClient = new LLMClient({
            endpoint: config.endpoint || 'http://localhost:13305',
            model: config.model || config.modelId,
            apiKey: config.apiKey || '',
            onRetry: (attempt, err, waitMs) => {
                console.warn(`[Optimizer] API retry attempt ${attempt}, waiting ${waitMs}ms`);
            },
        });

        // Storage (initialized lazily)
        this._storage = null;

        // Managers (initialized when storage is ready)
        this._conversationManager = null;
        this._versionManager = null;

        // Execution results
        this._result = null;
        this._currentVersion = null;
        this._lastTestOutput = null;
        this._lastThinking = null;
    }

    // ── Initialization ────────────────────────────────────────────────────

    /**
     * Initialize storage and managers. Idempotent.
     * @returns {Promise<void>}
     */
    async _ensureInitialized() {
        // Storage may be pre-injected by tests; still ensure managers exist
        if (!this._storage) {
            this._storage = new StorageManager();
            await this._storage.init();
            await this._storage.initDefaultTemplates();
        } else if (typeof this._storage.init === 'function' && !this._storage._db) {
            // Injected StorageManager instance that still needs init
            try {
                await this._storage.init();
                if (typeof this._storage.initDefaultTemplates === 'function') {
                    await this._storage.initDefaultTemplates();
                }
            } catch (_) {
                // Mock storage stubs may not implement full init — ignore
            }
        }

        if (!this._conversationManager) {
            this._conversationManager = new ConversationManager({
                modelId: this.modelId,
                llmClient: this._llmClient, // SHARED client — CRITICAL FIX #1
                storageManager: this._storage,
                onProgress: (loaded, total, detail) => {
                    this._onProgress('LOADING_CONTEXT', loaded, total, detail);
                },
                onError: (filename, error, recovered) => {
                    if (!recovered) {
                        this._onError('LOADING_CONTEXT', error);
                    }
                },
            });
        }

        if (!this._versionManager) {
            this._versionManager = new VersionManager(this.modelId, this._storage);
        }

        // Phase 4: Initialize optimization dependencies
        await this._initOptimizationDeps();
    }

    /**
     * Initialize optimization dependencies (Phase 4).
     * Uses injected dependencies if provided (for testing), otherwise creates from storage.
     * Also loads saved test cases into the evaluator.
     * @private
     * @returns {Promise<void>}
     */
    async _initOptimizationDeps() {
        if (!this._modelRegistry) {
            this._modelRegistry = new ModelRegistry(this._storage);
        }
        if (!this._testEvaluator) {
            this._testEvaluator = new TestEvaluator();
            // Load saved test cases from storage
            const savedCases = await this._storage.loadTestCases();
            if (savedCases.length > 0) {
                this._testEvaluator.addTestCases(savedCases);
            }
        }
        if (!this._optimizationHistory) {
            this._optimizationHistory = new OptimizationHistory(this._storage);
        }
    }

    // ── Full Pipeline ─────────────────────────────────────────────────────

    /**
     * Execute the full optimization pipeline.
     *
     * @param {Object} options
     * @param {Array<{name: string, content: string}>} [options.contextFiles] - Context files to load
     * @param {string} [options.userPrompt] - Raw prompt to convert
     * @param {boolean} [options.skipContextLoading=false] - Skip if context already loaded
     * @param {boolean} [options.runFullLoop=false] - Run convert→test→extract→update loop
     * @param {boolean} [options.autoIterate=false] - Run iterative optimization loop (Phase 4)
     * @param {string} [options.testInput] - Test input for optimization evaluation
     * @returns {Promise<Object>} - Full result
     */
    async execute(options = {}) {
        await this._ensureInitialized();

        console.log('[Optimizer] Starting execution...');
        this._result = { phases: {} };

        try {
            // Phase 1: Load Context (unless skipped)
            if (!options.skipContextLoading && options.contextFiles) {
                await this.transition(EVENTS.START);
                const loadResult = await this.loadModel(options.contextFiles);
                this._result.phases.loading = loadResult;
            } else if (options.skipContextLoading) {
                // Try to restore from storage
                await this.transition(EVENTS.START);
                const restored = await this._conversationManager.restore();
                if (!restored) {
                    throw new Error('No saved conversation found. Load context files first.');
                }
                await this.transition(EVENTS.CONTEXT_LOADED);
            }

            // Phase 2: Convert
            const convertResult = await this.convertPrompt(options.userPrompt || '');
            this._result.phases.conversion = convertResult;

            if (options.runFullLoop) {
                // Phase 3: Test
                const testResult = await this.testPrompt(convertResult.version.versionNum);
                this._result.phases.testing = testResult;

                // Phase 4: Extract
                const extractResult = await this.extractReasoning(
                    testResult.output, convertResult.version.versionNum
                );
                this._result.phases.extraction = extractResult;

                // Phase 5: Update (requires feedback or auto-feedback from thinking)
                const feedback = options.feedback || this._generateAutoFeedback(extractResult);
                const updateResult = await this.updatePrompt(feedback, convertResult.version.versionNum);
                this._result.phases.update = updateResult;
            }

            await this.transition(EVENTS.CONVERSION_COMPLETE);
            console.log('[Optimizer] Execution complete');

            // Phase 4: Iterative optimization loop
            if (options.autoIterate) {
                const optimizationResult = await this.runOptimizationLoop({
                    contextFiles: options.contextFiles,
                    userPrompt: options.userPrompt,
                    testInput: options.testInput,
                });
                this._result.optimization = optimizationResult;
            }

            return this._result;

        } catch (error) {
            await this.transition(EVENTS.ERROR, error);
            throw error;
        }
    }

    // ── Individual Phases ─────────────────────────────────────────────────

    /**
     * Phase 1: Load model context from files (load_model).
     * Default: sequential — each file is sent, response appended to loadedConvo, saved.
     *
     * @param {Array<{name: string, content: string}>} files
     * @param {Object} [options]
     * @param {'sequential'|'batch'} [options.mode='sequential']
     * @param {Function} [options.onFileResponse]
     * @param {Function} [options.shouldContinue]
     * @returns {Promise<Object>}
     */
    async loadModel(files, options = {}) {
        await this._ensureInitialized();
        await this.transition(EVENTS.START);

        const result = await this._conversationManager.loadContextFiles(files, {
            mode: options.mode || 'sequential',
            onFileResponse: options.onFileResponse,
            shouldContinue: options.shouldContinue,
        });
        await this.transition(EVENTS.CONTEXT_LOADED);

        return result;
    }

    /**
     * Phase 2: Convert a user prompt to UCL using loaded context.
     * sendRequest = loadedConvo + base_prompt + promptToConvert
     * Saves converted_prompt as version and appends exchange to history.
     *
     * @param {string} userPrompt - Raw prompt text
     * @returns {Promise<{version: Object, convertedText: string}>}
     */
    async convertPrompt(userPrompt) {
        if (!userPrompt?.trim()) {
            throw new Error('User prompt cannot be empty');
        }

        await this._ensureInitialized();
        this._onProgress('CONVERTING', 1, 1, 'Converting prompt to UCL...');

        // Ensure loadedConvo is available (restore from IndexedDB if needed)
        if (!this._conversationManager.getState().isLoaded) {
            await this._conversationManager.restore();
        }

        // Get the conversion template
        const template = await this._storage.getTemplate('base_convert', this.modelId);
        const templateContent = template?.content || 'Convert the following to UCL format:';

        const convertUserMsg = `${templateContent}\n\n---\nPROMPT TO CONVERT:\n${userPrompt}`;

        // Build messages: loadedConvo + base_prompt + promptToConvert
        const messages = [
            ...this._conversationManager.getLoadedConvo(),
            { role: 'user', content: convertUserMsg },
        ];

        const response = await this._llmClient.chat(messages, {
            temperature: 0.5,
            maxTokens: 4000,
        });

        const convertedText = LLMClient.extractContent(response);

        // Save as version (converted_prompt_v{n+1})
        const version = await this._versionManager.createVersion(convertedText, null, {
            basePrompt: userPrompt,
            strategy: ThinkingExtractor.detectStrategy(this.modelId),
        });

        // Persist exchange into loadedConvo for future modifications
        await this._conversationManager.appendExchange(
            convertUserMsg,
            convertedText,
            { phase: 'convert', versionNum: version.versionNum }
        );

        this._currentVersion = version;
        this.uclPrompt = convertedText;
        this.currentPrompt = convertedText;
        this._onVersionCreated(version);

        return { version, convertedText };
    }

    /**
     * Phase 3: Test a prompt version by sending it to the model.
     * Saves raw output for extractReasoning / thinking logs.
     *
     * @param {number} versionNum
     * @param {string} [testInput] - Optional test query
     * @returns {Promise<{output: string, usage: Object, versionNum: number}>}
     */
    async testPrompt(versionNum, testInput = null) {
        await this._ensureInitialized();
        this._onProgress('TESTING', 1, 1, `Testing version ${versionNum}...`);

        const version = await this._versionManager.getVersion(versionNum);
        if (!version) throw new Error(`Version ${versionNum} not found`);

        if (!this._conversationManager.getState().isLoaded) {
            await this._conversationManager.restore();
        }

        const testQuery = testInput ||
            'Demonstrate your understanding of these instructions with a short example.';

        const testUserMsg =
            `Apply the following UCL prompt (converted_prompt_v${versionNum}) and respond to the test query:\n\n` +
            `${version.promptText}\n\n---\nTest: ${testQuery}`;

        // Send: loadedConvo + converted prompt test
        const messages = [
            ...this._conversationManager.getLoadedConvo(),
            { role: 'user', content: testUserMsg },
        ];

        const response = await this._llmClient.chat(messages, {
            temperature: 0.7,
            maxTokens: 4000,
        });

        const output = LLMClient.extractContent(response);
        const usage = LLMClient.extractUsage(response);

        this._lastTestOutput = output;
        this.testInput = testQuery;

        // Keep test exchange in history for future updates
        await this._conversationManager.appendExchange(
            testUserMsg,
            output,
            { phase: 'test', versionNum }
        );

        return { output, usage, versionNum };
    }

    /**
     * Phase 4: Extract reasoning from a test output.
     * Saves thinking log named/linked to converted_prompt version.
     *
     * @param {string} rawOutput - Raw model response
     * @param {number} versionNum - Which version produced this output
     * @returns {Promise<{thinking: string, content: string, strategy: string, confidence: number}>}
     */
    async extractReasoning(rawOutput, versionNum) {
        await this._ensureInitialized();
        this._onProgress('EXTRACTING', 1, 1, `Extracting reasoning from v${versionNum}...`);

        const result = ThinkingExtractor.extract(rawOutput, this.modelId);

        // Save thinking log to storage (thinking_converted_prompt_v{n})
        const logId = await this._storage.saveThinkingLog(
            this.modelId, versionNum, result.thinking, result.strategy
        );

        // Link thinking log to version
        await this._versionManager.setThinkingLogRef(versionNum, logId);

        this._lastThinking = result;
        this.lastThinkingLog = result.thinking;

        return {
            ...result,
            thinkingLogId: logId,
            versionNum,
        };
    }

    /**
     * Phase 5: Update/refine a prompt based on feedback.
     *
     * historicalConvo = loadedConvo + base_prompt_update + converted_prompt_v{n}
     *                   + thinking_converted_prompt_v{n} + feedback
     * Version chain + conversation history are saved for future modifications.
     *
     * @param {string} feedback - User feedback or auto-generated feedback
     * @param {number} versionNum - Version to improve upon
     * @returns {Promise<Object>} - New version record
     */
    async updatePrompt(feedback, versionNum) {
        if (!feedback?.trim()) {
            throw new Error('Feedback is required for prompt update (user feedback mechanism)');
        }

        await this._ensureInitialized();
        this._onProgress('UPDATING', 1, 1, `Updating from v${versionNum}...`);

        const currentVersion = await this._versionManager.getVersion(versionNum);
        if (!currentVersion) throw new Error(`Version ${versionNum} not found`);

        if (!this._conversationManager.getState().isLoaded) {
            await this._conversationManager.restore();
        }

        // Get thinking log for context
        const thinkingLog = await this._versionManager.getThinkingLogForVersion(versionNum);

        // Build prior version history so future mods keep earlier changes in mind
        const allVersions = await this._versionManager.getAllVersions();
        const priorVersions = (allVersions || [])
            .filter(v => v.versionNum <= versionNum)
            .sort((a, b) => a.versionNum - b.versionNum);

        let versionHistoryBlock = '';
        if (priorVersions.length > 0) {
            versionHistoryBlock = '\n\n## Prior Version History (for continuity)\n';
            for (const v of priorVersions) {
                const snippet = (v.promptText || '').slice(0, 800);
                const fb = v.feedbackApplied || v.metadata?.feedbackApplied || '';
                versionHistoryBlock += `\n### converted_prompt_v${v.versionNum}` +
                    (fb ? ` [feedback: ${String(fb).slice(0, 200)}]` : '') +
                    `\n\`\`\`\n${snippet}${(v.promptText || '').length > 800 ? '\n... (truncated)' : ''}\n\`\`\`\n`;
            }
        }

        // Get update template
        const template = await this._storage.getTemplate('base_update', this.modelId);
        let templateContent = template?.content || 'Improve this UCL prompt based on feedback.';

        // Replace template placeholders
        templateContent = templateContent
            .replace('{currentPrompt}', currentVersion.promptText)
            .replace('{thinkingLog}', thinkingLog || this.lastThinkingLog || '[No thinking log available]')
            .replace('{feedback}', feedback);

        // Always include explicit converted prompt + thinking if placeholders missing
        if (!templateContent.includes(currentVersion.promptText)) {
            templateContent += `\n\n## Current converted_prompt_v${versionNum}\n\`\`\`\n${currentVersion.promptText}\n\`\`\``;
        }
        if (thinkingLog && !templateContent.includes(thinkingLog.slice(0, 80))) {
            templateContent += `\n\n## thinking_converted_prompt_v${versionNum}\n${thinkingLog}`;
        }
        if (!templateContent.includes(feedback)) {
            templateContent += `\n\n## Feedback\n${feedback}`;
        }
        templateContent += versionHistoryBlock;
        templateContent += '\n\nProduce the improved UCL prompt only (valid UCL syntax).';

        // historicalConvo = loadedConvo + base_prompt_update + converted + thinking
        const messages = [
            ...this._conversationManager.getLoadedConvo(),
            { role: 'user', content: templateContent },
        ];

        const response = await this._llmClient.chat(messages, {
            temperature: 0.5,
            maxTokens: 4000,
        });

        const updatedText = LLMClient.extractContent(response);

        // Create new version linked to parent (preserves modification chain)
        const newVersion = await this._versionManager.createVersion(updatedText, versionNum, {
            basePrompt: currentVersion.basePrompt,
            strategy: ThinkingExtractor.detectStrategy(this.modelId),
            feedbackApplied: feedback,
        });

        // Save update exchange so NEXT modification has full history
        await this._conversationManager.appendExchange(
            templateContent,
            updatedText,
            { phase: 'update', versionNum: newVersion.versionNum, parentVersion: versionNum }
        );

        this._currentVersion = newVersion;
        this.uclPrompt = updatedText;
        this.currentPrompt = updatedText;
        this._onVersionCreated(newVersion);

        await this.transition(EVENTS.UPDATE_COMPLETE);

        return newVersion;
    }

    /**
     * Resume iteration from the latest version.
     * Explicit loop semantics: COMPLETE → CONVERTING via LOOP_BACK event.
     *
     * @param {string} feedback - Feedback for the update
     * @returns {Promise<Object>} - New version
     */
    async resumeIteration(feedback) {
        const latest = await this._versionManager?.getLatestVersion();
        if (!latest) {
            throw new Error('No versions exist. Run convertPrompt() first.');
        }

        // Explicit loop-back transition
        if (this.currentState === STATES.COMPLETE) {
            await this.transition(EVENTS.LOOP_BACK);
        }

        return this.updatePrompt(feedback, latest.versionNum);
    }

    // ── State Machine ─────────────────────────────────────────────────────

    /**
     * Execute a state transition.
     * Follows PipelineOrchestrator pattern exactly.
     *
     * @param {string} event - Event from EVENTS
     * @param {*} [payload] - Optional payload (e.g., error object)
     * @returns {Promise<void>}
     */
    async transition(event, payload = null) {
        const previousState = this.currentState;
        let nextState = previousState;

        switch (this.currentState) {
            case STATES.IDLE:
                if (event === EVENTS.START) nextState = STATES.LOADING_CONTEXT;
                break;

            case STATES.LOADING_CONTEXT:
                if (event === EVENTS.CONTEXT_LOADED) nextState = STATES.CONVERTING;
                else if (event === EVENTS.ERROR) { nextState = STATES.ERROR; this.error = payload; }
                break;

            case STATES.CONVERTING:
                if (event === EVENTS.CONVERSION_COMPLETE) nextState = STATES.COMPLETE;
                else if (event === EVENTS.ERROR) { nextState = STATES.ERROR; this.error = payload; }
                break;

            case STATES.TESTING:
                if (event === EVENTS.TEST_COMPLETE) nextState = STATES.EXTRACTING;
                else if (event === EVENTS.ERROR) { nextState = STATES.ERROR; this.error = payload; }
                break;

            case STATES.EXTRACTING:
                if (event === EVENTS.EXTRACTION_COMPLETE) nextState = STATES.UPDATING;
                else if (event === EVENTS.ERROR) { nextState = STATES.ERROR; this.error = payload; }
                break;

            case STATES.UPDATING:
                if (event === EVENTS.UPDATE_COMPLETE) nextState = STATES.COMPLETE;
                else if (event === EVENTS.ERROR) { nextState = STATES.ERROR; this.error = payload; }
                break;

            case STATES.COMPLETE:
                // EXPLICIT LOOP SEMANTICS: LOOP_BACK goes to CONVERTING
                if (event === EVENTS.LOOP_BACK) nextState = STATES.CONVERTING;
                else if (event === EVENTS.RESET) nextState = STATES.IDLE;
                break;

            case STATES.ERROR:
                if (event === EVENTS.RESET) { nextState = STATES.IDLE; this.error = null; }
                break;
        }

        if (nextState !== previousState) {
            this.currentState = nextState;
            this._onStateChange(nextState, previousState, event);
            console.log(`[Optimizer] State: ${previousState} → ${nextState} (${event})`);
        }
    }

    // ── State Management ──────────────────────────────────────────────────

    /**
     * Reset orchestrator to IDLE state and clear all data.
     */
    async reset() {
        await this.transition(EVENTS.RESET);
        this._result = null;
        this._currentVersion = null;
        this._lastTestOutput = null;
        this._lastThinking = null;
        // Phase 4: Reset optimization state
        this.cancelled = false;
        this.currentRunId = null;
        this.uclPrompt = null;
        this.currentPrompt = null;
        this.lastThinkingLog = null;
        this.lastMetrics = null;
        this.versionCounter = 0;
        this.testInput = null;
        if (this._conversationManager) {
            this._conversationManager.reset();
        }
        if (this._versionManager) {
            this._versionManager.invalidateCache();
        }
    }

    /**
     * Get current state summary.
     * @returns {Object}
     */
    getState() {
        return {
            currentState: this.currentState,
            modelId: this.modelId,
            error: this.error,
            conversation: this._conversationManager?.getState() || null,
            versions: this._currentVersion?.versionNum || 0,
            hasResult: this._result !== null,
            // Phase 4 fields
            cancelled: this.cancelled,
            currentRunId: this.currentRunId,
            hasOptimizationDeps: this._modelRegistry !== null,
        };
    }

    /**
     * Get the result object.
     * @returns {Object|null}
     */
    getResult() {
        if (this.currentState !== STATES.COMPLETE) return null;
        return this._result;
    }

    /**
     * Access the version manager (for UI to list versions, etc.)
     * @returns {VersionManager|null}
     */
    getVersionManager() {
        return this._versionManager;
    }

    /**
     * Access the storage manager (for UI session management).
     * @returns {StorageManager|null}
     */
    getStorageManager() {
        return this._storage;
    }

    /**
     * Access the conversation manager (loadedConvo).
     * @returns {ConversationManager|null}
     */
    getConversationManager() {
        return this._conversationManager;
    }

    /**
     * Cleanup all resources.
     * QUALITY FIX: OptimizerUI holds long-lived resources — this ensures cleanup.
     */
    dispose() {
        // Phase 4: Cancel any ongoing optimization
        if (this.currentRunId && !this.cancelled) {
            this.cancelOptimization();
        }
        this._llmClient.abort();
        if (this._storage) {
            this._storage.close();
        }
        this._conversationManager = null;
        this._versionManager = null;
        this._modelRegistry = null;
        this._testEvaluator = null;
        this._optimizationHistory = null;
        this._storage = null;
        this._llmClient = null;
    }

    // ── Phase 4: Iterative Optimization ───────────────────────────────────

    /**
     * Run a single optimization iteration.
     *
     * Steps:
     *   1. Get ModelConfig from ModelRegistry
     *   2. Adapt prompt (first: from UCL, subsequent: from feedback)
     *   3. Test the prompt against the model
     *   4. Extract reasoning from response
     *   5. Evaluate test results
     *   6. Compute composite quality score
     *   7. Record iteration in optimization history
     *   8. Update version quality in storage
     *   9. Store state for next iteration
     *
     * @param {number} iterationNum - Iteration number (1-based)
     * @param {string|null} feedback - Feedback from previous iteration (null for first)
     * @returns {Promise<Object>} Iteration result
     */
    async runIteration(iterationNum, feedback = null) {
        await this._ensureInitialized();

        const startTime = Date.now();

        // 1. Get ModelConfig from ModelRegistry
        const modelConfig = await this._modelRegistry.getConfig(this.modelId);

        // 2. Convert or update prompt
        let adaptedPrompt;
        if (iterationNum === 1) {
            // First iteration: adapt UCL prompt for the target model
            adaptedPrompt = PromptAdapter.adaptForModel(this.uclPrompt || '', modelConfig);
        } else {
            // Subsequent iterations: use feedback to improve prompt
            const history = await this._optimizationHistory.getRun(this.currentRunId);
            const metrics = this.lastMetrics
                ? { compositeScore: this.lastMetrics.qualityScore?.composite || 0, iteration: iterationNum - 1 }
                : {};
            const messages = PromptAdapter.buildUpdateMessages(
                this.currentPrompt || '',
                this.lastThinkingLog || '',
                metrics,
                feedback || 'Improve the prompt based on test results.',
                history?.iterations || [],
                modelConfig
            );
            const response = await this._llmClient.chat(messages, {
                temperature: 0.5,
                maxTokens: modelConfig.maxTokens || 4000,
            });
            adaptedPrompt = LLMClient.extractContent(response);
        }

        // 3. Test the prompt against the model
        const testInputText = this.testInput || 'Generate a UCL prompt for file management';
        const testMessages = PromptAdapter.buildTestMessages(adaptedPrompt, testInputText, modelConfig);
        const testResponse = await this._llmClient.chat(testMessages, {
            temperature: modelConfig.temperature || 0.7,
            maxTokens: modelConfig.maxTokens || 4000,
        });
        const testOutput = LLMClient.extractContent(testResponse);
        const usage = LLMClient.extractUsage(testResponse);

        // 4. Extract reasoning from response
        const { thinking, content: cleanResponse } = ThinkingExtractor.extract(
            testOutput,
            this.modelId
        );

        // 5. Persist this iteration as a real version (parent = previous latest)
        // CRITICAL: must createVersion — updateVersionQuality on phantom nums broke multi-iter
        const latestBefore = await this._versionManager.getLatestVersion();
        const parentNum = latestBefore?.versionNum ?? null;
        const versionRecord = await this._versionManager.createVersion(adaptedPrompt, parentNum, {
            basePrompt: this.uclPrompt || '',
            strategy: ThinkingExtractor.detectStrategy(this.modelId),
            feedbackApplied: feedback || (iterationNum === 1 ? 'initial_loop_iteration' : ''),
            loopIteration: iterationNum,
            runId: this.currentRunId,
        });
        const versionNum = versionRecord.versionNum;
        this.versionCounter = versionNum;
        this._currentVersion = versionRecord;
        this._onVersionCreated(versionRecord);

        // 6. Evaluate test results
        const testResults = this._testEvaluator.evaluateAll(cleanResponse, {
            thinkingExtracted: thinking,
            versionNum,
            runId: this.currentRunId,
            iteration: iterationNum,
            usage,
            duration: Date.now() - startTime,
        });

        // 7. Compute composite quality score
        const previousScore = this.lastMetrics?.qualityScore?.composite || null;
        const qualityScore = QualityScorer.computeScore({
            promptText: adaptedPrompt,
            testResults,
            thinkingText: thinking,
            previousScore,
        });

        // 8. Save thinking log linked to this version (update in place, no duplicates)
        if (thinking) {
            const logId = await this._storage.saveThinkingLog(
                this.modelId, versionNum, thinking, ThinkingExtractor.detectStrategy(this.modelId)
            );
            await this._versionManager.setThinkingLogRef(versionNum, logId);
        }

        // 9. Record iteration in optimization history
        await this._optimizationHistory.recordIteration(this.currentRunId, {
            iterationNum,
            versionNum,
            promptText: adaptedPrompt,
            qualityScore: qualityScore.composite,
            testResults,
            thinkingExtracted: thinking,
            usage,
            duration: Date.now() - startTime,
        });

        // 10. Update version quality in storage (real version only)
        await this._storage.updateVersionQuality(
            this.modelId,
            versionNum,
            qualityScore.composite
        );

        // 11. Append to loadedConvo so future mods keep loop history
        try {
            await this._conversationManager.appendExchange(
                `[optimize iter ${iterationNum}] Adapted/refined prompt scored ${qualityScore.composite.toFixed(3)}`,
                adaptedPrompt.slice(0, 4000),
                { phase: 'optimize', versionNum }
            );
        } catch (histErr) {
            console.warn('[Optimizer] Could not append loop exchange to loadedConvo:', histErr.message);
        }

        // 12. Store current state for next iteration
        this.currentPrompt = adaptedPrompt;
        this.uclPrompt = adaptedPrompt;
        this.lastThinkingLog = thinking;
        this.lastMetrics = { qualityScore, testResults };

        return {
            iterationNum,
            versionNum,
            adaptedPrompt,
            qualityScore,
            testResults,
            thinkingExtracted: thinking,
            usage,
            duration: Date.now() - startTime,
        };
    }

    /**
     * Run the full optimization loop with automatic iteration control.
     *
     * Uses IterationController for 6 stopping criteria:
     *   1. Manual cancel
     *   2. Max iterations reached
     *   3. Quality threshold met
     *   4. Convergence detected
     *   5. Token budget exhausted
     *   6. Score degradation detected
     *
     * Includes error recovery: aborts after 3 consecutive iteration failures.
     *
     * @param {Object} [options={}]
     * @param {Array} [options.contextFiles] - Context files (already loaded by execute())
     * @param {string} [options.userPrompt] - Original user prompt
     * @param {string} [options.testInput] - Test input for evaluation
     * @returns {Promise<Object>} Optimization run result
     */
    async runOptimizationLoop(options = {}) {
        await this._ensureInitialized();

        const { testInput = null } = options;

        // Initialize loop state
        this.testInput = testInput;
        this.versionCounter = 1;
        this.cancelled = false;

        // Use current version's prompt as the UCL baseline
        if (!this.uclPrompt && this._currentVersion) {
            this.uclPrompt = this._currentVersion.promptText;
        }
        if (!this.uclPrompt) {
            throw new Error('No UCL prompt available. Run convertPrompt() first.');
        }

        // Get model config for optimization settings
        const modelConfig = await this._modelRegistry.getConfig(this.modelId);

        // Create iteration controller with merged config
        const controller = new IterationController({
            ...this.iterationConfig,
            ...(modelConfig.extraParams?.optimization || {}),
        });

        // Start optimization run in history
        const run = await this._optimizationHistory.startRun({
            modelId: this.modelId,
            basePrompt: this.uclPrompt,
            maxIterations: controller.getState().config.maxIterations,
            qualityThreshold: controller.getState().config.qualityThreshold,
        });
        this.currentRunId = run.runId;

        const iterationHistory = [];
        let feedback = null;
        let consecutiveFailures = 0;
        const MAX_FAILURES = 3;
        let stopReason = 'max_iterations';

        // Main optimization loop
        while (true) {
            // Check cancellation
            if (this.cancelled) {
                stopReason = 'cancelled';
                break;
            }

            // Check if should continue (IterationController's 6 criteria)
            const decision = controller.shouldContinue();
            if (!decision.continue) {
                stopReason = decision.reason;
                break;
            }

            try {
                // Run iteration
                const iterationResult = await this.runIteration(
                    controller.getState().iterationCount + 1,
                    feedback
                );

                // Reset failure counter on success
                consecutiveFailures = 0;

                // Record in controller
                controller.recordIteration({
                    iterationNum: iterationResult.iterationNum,
                    compositeScore: iterationResult.qualityScore.composite,
                    tokensUsed: iterationResult.usage?.total_tokens || iterationResult.usage?.totalTokens || 0,
                    duration: iterationResult.duration || 0,
                });

                iterationHistory.push(iterationResult);

                // Invoke per-iteration callback
                if (this._onIteration) {
                    this._onIteration(iterationResult);
                }

                // Check if quality threshold met (early exit)
                const threshold = controller.getState().config.qualityThreshold;
                if (iterationResult.qualityScore.composite >= threshold) {
                    stopReason = 'quality_threshold';
                    break;
                }

                // Generate feedback for next iteration
                feedback = TestEvaluator.generateFeedback(iterationResult.testResults);

            } catch (error) {
                consecutiveFailures++;
                this._onError('OPTIMIZATION', error);
                console.warn(`[Optimizer] Iteration error (${consecutiveFailures}/${MAX_FAILURES}): ${error.message}`);

                if (consecutiveFailures >= MAX_FAILURES) {
                    stopReason = 'error';
                    await this._optimizationHistory.failRun(
                        this.currentRunId,
                        `Optimization failed after ${MAX_FAILURES} consecutive errors: ${error.message}`
                    );
                    throw new Error(`Optimization failed after ${MAX_FAILURES} consecutive errors: ${error.message}`);
                }
            }
        }

        // Complete the run (unless already failed)
        if (stopReason !== 'error') {
            await this._optimizationHistory.completeRun(this.currentRunId, stopReason);
        }

        // Find best version across all completed runs
        const bestVersion = await this._optimizationHistory.findBestVersion(this.modelId);

        const result = {
            runId: this.currentRunId,
            totalIterations: iterationHistory.length,
            bestVersionNum: bestVersion?.versionNum || null,
            bestQualityScore: bestVersion?.qualityScore || 0,
            stopReason,
            iterationHistory,
            bestPrompt: bestVersion?.promptText || null,
        };

        // Invoke completion callback
        if (this._onOptimizationComplete) {
            this._onOptimizationComplete(result);
        }

        return result;
    }

    /**
     * Cancel an ongoing optimization loop.
     * Sets the cancelled flag and marks the current run as cancelled in history.
     */
    cancelOptimization() {
        this.cancelled = true;
        if (this.currentRunId && this._optimizationHistory) {
            this._optimizationHistory.cancelRun(this.currentRunId).catch(err => {
                console.warn(`[Optimizer] Failed to cancel run ${this.currentRunId}: ${err.message}`);
            });
        }
    }

    // ── Private ───────────────────────────────────────────────────────────

    /**
     * Generate auto-feedback from extraction result when user doesn't provide feedback.
     * @private
     */
    _generateAutoFeedback(extractResult) {
        if (!extractResult.thinking) {
            return 'Improve the prompt to be more specific and structured.';
        }
        return `Based on the reasoning log, the model's thinking process was:\n${extractResult.thinking.substring(0, 500)}...\nImprove the prompt to address any gaps or issues identified in the reasoning.`;
    }
}

// ── Exports ────────────────────────────────────────────────────────────────
export { ModelOptimizerOrchestrator, STATES, EVENTS };
