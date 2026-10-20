/**
 * ConversationManager - Accumulated Conversation (loadedConvo) Management
 *
 * CRITICAL QUALITY FIX #1: Uses shared LLMClient instead of duplicating API logic.
 * HIGH QUALITY FIX: Context loading uses batch mode (all files in one request)
 *   instead of O(n) individual API calls.
 * HIGH QUALITY FIX: Token management extracted to pure utility functions.
 *
 * Manages the accumulated conversation history per model:
 * - Loads context files → sends to model → accumulates responses
 * - Provides conversation as messages[] for subsequent API calls
 * - Handles token estimation and truncation
 * - Persists to IndexedDB via StorageManager
 *
 * Usage:
 *   const cm = new ConversationManager({
 *     modelId: 'qwen3-4b-FLM',
 *     llmClient: sharedLLMClient,
 *     storageManager: storageManager,
 *     onProgress: (loaded, total, filename) => console.log(`Loaded ${loaded}/${total}: ${filename}`),
 *   });
 *   await cm.loadContextFiles(files);
 *   const messages = cm.getLoadedConvo();
 *
 * @module conversation-manager
 */

import { LLMClient } from './llm-client.js';

// ── Token Utilities (extracted for testability / low coupling) ─────────────

/**
 * Estimate token count from text. Rough: chars / 4.
 * @param {string} text
 * @returns {number}
 */
function estimateTokens(text) {
    if (!text) return 0;
    return Math.ceil(text.length / 4);
}

/**
 * Estimate total tokens for a message array.
 * @param {Array<{role: string, content: string}>} messages
 * @returns {number}
 */
function estimateMessagesTokens(messages) {
    return messages.reduce((sum, m) => sum + estimateTokens(m.content) + 4, 0); // +4 for role overhead
}

/**
 * Truncate messages to fit within a token limit.
 * Strategy: keep first N messages (system/identity) and most recent exchanges.
 * Drop oldest file exchanges first.
 *
 * @param {Array<{role: string, content: string}>} messages
 * @param {number} maxTokens
 * @param {number} [keepFirst=2] - Number of initial messages to always keep
 * @returns {{ truncated: Array, dropped: number, tokenEstimate: number }}
 */
function truncateToLimit(messages, maxTokens, keepFirst = 2) {
    const totalTokens = estimateMessagesTokens(messages);
    if (totalTokens <= maxTokens) {
        return { truncated: [...messages], dropped: 0, tokenEstimate: totalTokens };
    }

    // Keep first N messages (usually identity/instructions)
    const head = messages.slice(0, keepFirst);
    const body = messages.slice(keepFirst);

    // Add messages from the end until we hit the limit
    const headTokens = estimateMessagesTokens(head);
    const remaining = maxTokens - headTokens;
    const kept = [];
    let usedTokens = 0;

    for (let i = body.length - 1; i >= 0; i--) {
        const msgTokens = estimateTokens(body[i].content) + 4;
        if (usedTokens + msgTokens > remaining) break;
        kept.unshift(body[i]);
        usedTokens += msgTokens;
    }

    const result = [...head, ...kept];
    return {
        truncated: result,
        dropped: body.length - kept.length,
        tokenEstimate: headTokens + usedTokens,
    };
}

// ── ConversationManager ───────────────────────────────────────────────────

class ConversationManager {
    /**
     * @param {Object} config
     * @param {string} config.modelId - Model identifier
     * @param {LLMClient} config.llmClient - Shared LLM client (CRITICAL FIX #1)
     * @param {import('./storage-manager.js').StorageManager} config.storageManager
     * @param {number} [config.maxContextTokens=28000] - Max tokens before truncation
     * @param {number} [config.batchSize=5] - Files per batch in context loading
     * @param {Function} [config.onProgress] - (loaded, total, filename) => void
     * @param {Function} [config.onError] - (filename, error, recovered) => void
     */
    constructor(config = {}) {
        if (!config.modelId) throw new Error('ConversationManager requires modelId');
        if (!config.llmClient) throw new Error('ConversationManager requires llmClient (shared LLMClient instance)');
        if (!config.storageManager) throw new Error('ConversationManager requires storageManager');

        this.modelId = config.modelId;
        this._llm = config.llmClient;
        this._storage = config.storageManager;
        this.maxContextTokens = config.maxContextTokens ?? 28000;
        this.batchSize = config.batchSize ?? 5;

        // Callbacks
        this._onProgress = config.onProgress || (() => {});
        this._onError = config.onError || (() => {});

        // State
        /** @type {Array<{role: string, content: string}>} */
        this._messages = [];
        /** @type {string[]} */
        this._loadedFiles = [];
        this._isLoaded = false;
    }

    // ── Context Loading (HIGH FIX: batch mode) ────────────────────────────

    /**
     * Load context files into the conversation.
     *
     * Modes:
     * - `sequential` (default, matches load_model): for each file, sendMsg, append
     *   response to loadedConvo, save after each response.
     * - `batch`: groups files into batches (legacy / faster path).
     *
     * @param {Array<{name: string, content: string}>} files
     * @param {Object} [options]
     * @param {'sequential'|'batch'} [options.mode='sequential']
     * @param {Function} [options.onFileResponse] - (file, responseText, index, total) => void
     * @param {Function} [options.shouldContinue] - () => boolean — return false to abort
     * @returns {Promise<{messages: Array, tokenEstimate: number, filesLoaded: number, batchesUsed: number, mode: string, partialFailure: boolean, fileResponses: Array}>}
     */
    async loadContextFiles(files, options = {}) {
        const mode = options.mode || 'sequential';
        if (mode === 'batch') {
            return this._loadContextFilesBatch(files, options);
        }
        return this.loadContextFilesSequential(files, options);
    }

    /**
     * Sequential load_model: for every file in folder, sendMsg, append to loadedConvo, save.
     * @param {Array<{name: string, content: string}>} files
     * @param {Object} [options]
     * @param {Function} [options.onFileResponse]
     * @param {Function} [options.shouldContinue]
     * @returns {Promise<Object>}
     */
    async loadContextFilesSequential(files, options = {}) {
        if (!files || files.length === 0) {
            throw new Error('No context files provided');
        }

        const onFileResponse = options.onFileResponse || (() => {});
        const shouldContinue = options.shouldContinue || (() => true);

        this._messages = [];
        this._loadedFiles = [];
        this._isLoaded = false;
        const fileResponses = [];
        let partialFailure = false;

        // Introduce all files once (order is intentional)
        this._messages.push({
            role: 'user',
            content: this._buildSystemContextMessage(files),
        });

        for (let i = 0; i < files.length; i++) {
            if (!shouldContinue()) {
                console.warn('[ConversationManager] Sequential load aborted by caller');
                break;
            }

            const file = files[i];
            const userContent =
                `--- File ${i + 1}/${files.length}: ${file.name} ---\n` +
                `${file.content}\n` +
                `--- End: ${file.name} ---\n\n` +
                `Acknowledge that you have processed this file. Summarize key points relevant to prompt optimization.`;

            this._messages.push({ role: 'user', content: userContent });

            try {
                const response = await this._llm.chat(this._messages, {
                    temperature: 0.3,
                    maxTokens: 1500,
                });

                const assistantContent = LLMClient.extractContent(response);
                this._messages.push({ role: 'assistant', content: assistantContent });
                this._loadedFiles.push(file.name);

                const entry = {
                    fileName: file.name,
                    response: assistantContent,
                    timestamp: new Date().toISOString(),
                    tokens: LLMClient.extractUsage(response)?.total || 0,
                    index: i,
                };
                fileResponses.push(entry);
                onFileResponse(file, assistantContent, i, files.length, entry);

                // Persist after EACH response so history survives interruption
                await this.save();

                this._onProgress(i + 1, files.length, file.name);
            } catch (error) {
                partialFailure = true;
                this._onError(file.name, error, true);

                const failMsg = `[Note: Could not process ${file.name}: ${error.message}. Continuing.]`;
                this._messages.push({ role: 'assistant', content: failMsg });

                const entry = {
                    fileName: file.name,
                    response: `Error: ${error.message}`,
                    timestamp: new Date().toISOString(),
                    tokens: 0,
                    index: i,
                    error: true,
                };
                fileResponses.push(entry);
                onFileResponse(file, failMsg, i, files.length, entry);
                await this.save();
            }
        }

        this._truncateIfNeeded();
        this._isLoaded = this._loadedFiles.length > 0;
        await this.save();

        return {
            messages: this._messages,
            tokenEstimate: estimateMessagesTokens(this._messages),
            filesLoaded: this._loadedFiles.length,
            batchesUsed: this._loadedFiles.length, // 1 file = 1 "batch" in sequential mode
            mode: 'sequential',
            partialFailure,
            fileResponses,
        };
    }

    /**
     * Batch load (legacy): groups files into batches for fewer API calls.
     * @private
     */
    async _loadContextFilesBatch(files, options = {}) {
        if (!files || files.length === 0) {
            throw new Error('No context files provided');
        }

        this._messages = [];
        this._loadedFiles = [];
        this._isLoaded = false;

        // Add system identity message
        this._messages.push({
            role: 'user',
            content: this._buildSystemContextMessage(files),
        });

        // Batch files for API calls
        const batches = this._createBatches(files, this.batchSize);
        let batchesCompleted = 0;
        let partialFailure = false;

        for (const batch of batches) {
            const batchContent = this._buildBatchMessage(batch);

            this._messages.push({
                role: 'user',
                content: batchContent,
            });

            try {
                const response = await this._llm.chat(this._messages, {
                    temperature: 0.3, // Low temperature for context acknowledgment
                    maxTokens: 1000,  // Short acknowledgment responses
                });

                const assistantContent = LLMClient.extractContent(response);
                this._messages.push({
                    role: 'assistant',
                    content: assistantContent,
                });

                // Track loaded files
                for (const file of batch) {
                    this._loadedFiles.push(file.name);
                }

                batchesCompleted++;
                this._onProgress(batchesCompleted, batches.length,
                    batch.map(f => f.name).join(', '));

            } catch (error) {
                // Partial failure recovery: log and continue
                partialFailure = true;
                this._onError(
                    batch.map(f => f.name).join(', '),
                    error,
                    true // recovered = true, we continue
                );

                // Add a system note about the failure
                this._messages.push({
                    role: 'assistant',
                    content: `[Note: Some context files could not be processed: ${error.message}. Continuing with available context.]`,
                });
            }
        }

        this._truncateIfNeeded();
        this._isLoaded = true;

        // Persist to storage
        await this.save();

        const tokenEstimate = estimateMessagesTokens(this._messages);
        return {
            messages: this._messages,
            tokenEstimate,
            filesLoaded: this._loadedFiles.length,
            batchesUsed: batchesCompleted,
            mode: 'batch',
            partialFailure,
            fileResponses: [],
        };
    }

    /**
     * Append a user/assistant exchange to loadedConvo and persist.
     * Used after convert/test/update so future modifications retain history.
     *
     * @param {string} userContent
     * @param {string} assistantContent
     * @param {Object} [meta]
     * @returns {Promise<void>}
     */
    async appendExchange(userContent, assistantContent, meta = {}) {
        // Only role/content — extra fields can break some OpenAI-compatible APIs
        if (userContent) {
            const prefix = meta.phase ? `[${meta.phase}${meta.versionNum != null ? ' v' + meta.versionNum : ''}]\n` : '';
            this._messages.push({
                role: 'user',
                content: prefix + userContent,
            });
        }
        if (assistantContent) {
            this._messages.push({
                role: 'assistant',
                content: assistantContent,
            });
        }
        this._truncateIfNeeded();
        this._isLoaded = true;
        await this.save();
    }

    /**
     * Truncate messages if over maxContextTokens.
     * @private
     */
    _truncateIfNeeded() {
        if (estimateMessagesTokens(this._messages) > this.maxContextTokens) {
            const { truncated, dropped } = truncateToLimit(
                this._messages, this.maxContextTokens, 1
            );
            this._messages = truncated;
            if (dropped > 0) {
                console.warn(`[ConversationManager] Truncated ${dropped} oldest messages to fit context window`);
            }
        }
    }

    // ── Conversation State ────────────────────────────────────────────────

    /**
     * Get the current accumulated conversation messages.
     * @returns {Array<{role: string, content: string}>}
     */
    getLoadedConvo() {
        return [...this._messages];
    }

    /**
     * Get list of loaded file names.
     * @returns {string[]}
     */
    getLoadedFiles() {
        return [...this._loadedFiles];
    }

    /**
     * Restore conversation from IndexedDB.
     * @returns {Promise<boolean>} - True if restored successfully
     */
    async restore() {
        const record = await this._storage.loadConversation(this.modelId);
        if (!record || !record.messages || record.messages.length === 0) {
            return false;
        }

        this._messages = record.messages;
        this._loadedFiles = record.contextFiles || [];
        this._isLoaded = this._messages.length > 0;
        return true;
    }

    /**
     * Persist current conversation to IndexedDB.
     * @returns {Promise<void>}
     */
    async save() {
        await this._storage.saveConversation(this.modelId, this._messages, {
            contextFiles: this._loadedFiles,
            tokenEstimate: estimateMessagesTokens(this._messages),
        });
    }

    /**
     * Reset conversation state (in-memory only, call save() to persist).
     */
    reset() {
        this._messages = [];
        this._loadedFiles = [];
        this._isLoaded = false;
    }

    /**
     * Full reset: clear in-memory and persist empty state.
     * @returns {Promise<void>}
     */
    async resetAndSave() {
        this.reset();
        await this.save();
    }

    // ── API Interaction (uses LLMClient — CRITICAL FIX #1) ───────────────

    /**
     * Send messages to the model and get a response.
     * Uses the shared LLMClient — no API duplication.
     *
     * @param {Array<{role: string, content: string}>} messages - Override messages (or use loaded convo)
     * @param {Object} [options] - LLM options (temperature, maxTokens, etc.)
     * @returns {Promise<{content: string, usage: Object}>}
     */
    async sendToModel(messages = null, options = {}) {
        const msgs = messages || this._messages;
        const response = await this._llm.chat(msgs, options);
        const content = LLMClient.extractContent(response);
        const usage = LLMClient.extractUsage(response);
        return { content, usage, rawResponse: response };
    }

    // ── State ─────────────────────────────────────────────────────────────

    /**
     * Get current state summary.
     * @returns {Object}
     */
    getState() {
        return {
            modelId: this.modelId,
            messageCount: this._messages.length,
            tokenEstimate: estimateMessagesTokens(this._messages),
            filesLoaded: this._loadedFiles.length,
            loadedFileNames: this._loadedFiles,
            isLoaded: this._isLoaded,
            maxContextTokens: this.maxContextTokens,
        };
    }

    // ── Private ───────────────────────────────────────────────────────────

    /**
     * Build the initial system context message that introduces all files.
     * @private
     */
    _buildSystemContextMessage(files) {
        const fileList = files.map(f => `- ${f.name} (${estimateTokens(f.content)} tokens)`).join('\n');
        return `You are about to receive context files for model optimization. ` +
            `There are ${files.length} files to process. ` +
            `Acknowledge each batch of files as you receive them. ` +
            `Files to process:\n${fileList}`;
    }

    /**
     * Build a batch message containing multiple files.
     * @private
     */
    _buildBatchMessage(files) {
        const parts = files.map(f =>
            `--- File: ${f.name} ---\n${f.content}\n--- End: ${f.name} ---`
        );
        return `Here are the next context files:\n\n${parts.join('\n\n')}\n\nAcknowledge that you have processed these files.`;
    }

    /**
     * Split files into batches.
     * @private
     */
    _createBatches(files, batchSize) {
        const batches = [];
        for (let i = 0; i < files.length; i += batchSize) {
            batches.push(files.slice(i, i + batchSize));
        }
        return batches;
    }
}

// ── Exports ────────────────────────────────────────────────────────────────
export {
    ConversationManager,
    estimateTokens,
    estimateMessagesTokens,
    truncateToLimit,
};
