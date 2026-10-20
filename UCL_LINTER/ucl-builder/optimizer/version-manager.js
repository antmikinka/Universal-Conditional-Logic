/**
 * VersionManager - Versioned Prompt Chain Management
 *
 * Manages the versioned prompt chain with lineage tracking.
 * Each version stores the prompt text, parent version, thinking log,
 * and supports diffing between versions.
 *
 * Features:
 * - Sequential version numbering per model
 * - Parent-child lineage chain
 * - Thinking log correlation via storage
 * - Simple diff between versions
 * - Metadata tracking per version
 *
 * Dependencies: StorageManager
 *
 * Usage:
 *   const vm = new VersionManager('qwen3-4b-FLM', storageManager);
 *   const v1 = await vm.createVersion(promptText, null, { basePrompt: 'raw input' });
 *   const v2 = await vm.createVersion(updatedText, 1, { feedbackApplied: '...' });
 *   const chain = await vm.getVersionChain(2); // [v1, v2]
 *
 * @module version-manager
 */

import { ThinkingExtractor } from './thinking-extractor.js';

class VersionManager {
    /**
     * @param {string} modelId - Model identifier
     * @param {import('./storage-manager.js').StorageManager} storageManager
     */
    constructor(modelId, storageManager) {
        if (!modelId) throw new Error('VersionManager requires a modelId');
        if (!storageManager) throw new Error('VersionManager requires a storageManager');

        this.modelId = modelId;
        this._storage = storageManager;

        // In-memory cache of versions (refreshed from storage)
        this._cache = null;
    }

    /**
     * Create a new version in the chain.
     *
     * @param {string} promptText - The UCL prompt text
     * @param {number|null} parentVersion - Parent version number (null for v1)
     * @param {Object} [metadata] - Extra info (basePrompt, tokenCount, strategy, feedbackApplied)
     * @returns {Promise<Object>} - The created version record
     */
    async createVersion(promptText, parentVersion = null, metadata = {}) {
        const versions = await this._loadVersions();
        const nextNum = versions.length > 0
            ? Math.max(...versions.map(v => v.versionNum)) + 1
            : 1;

        const record = {
            modelId: this.modelId,
            versionNum: nextNum,
            promptText,
            parentVersion,
            basePrompt: metadata.basePrompt || '',
            timestamp: Date.now(),
            tokenCount: metadata.tokenCount || this._estimateTokens(promptText),
            thinkingLogId: null, // Set later when thinking is extracted
            metadata: {
                strategy: metadata.strategy || 'unknown',
                contextWindowSize: metadata.contextWindowSize || 32000,
                feedbackApplied: metadata.feedbackApplied || '',
                ...metadata,
            },
        };

        const id = await this._storage.saveVersion(record);
        record.id = id;

        // Invalidate cache
        this._cache = null;

        return record;
    }

    /**
     * Update a version's thinking log reference.
     *
     * @param {number} versionNum
     * @param {number} thinkingLogId - ID from StorageManager.saveThinkingLog()
     * @returns {Promise<void>}
     */
    async setThinkingLogRef(versionNum, thinkingLogId) {
        const version = await this.getVersion(versionNum);
        if (!version) {
            throw new Error(`Version ${versionNum} not found for model ${this.modelId}`);
        }
        if (version.id == null) {
            // Ensure we have the storage id for in-place update
            const fromStore = await this._storage.getVersion(this.modelId, versionNum);
            if (!fromStore?.id) {
                throw new Error(`Version ${versionNum} missing storage id — cannot update thinking log ref`);
            }
            version.id = fromStore.id;
        }
        version.thinkingLogId = thinkingLogId;
        // CRITICAL: update in place — saveVersion would insert a duplicate row
        await this._storage.updateVersion(version);
        this._cache = null;
    }

    /**
     * Get a specific version by number.
     * Prefer the most recently saved row if duplicates exist (legacy bug).
     * @param {number} versionNum
     * @returns {Promise<Object|null>}
     */
    async getVersion(versionNum) {
        const versions = await this._loadVersions();
        const matches = versions.filter(v => v.versionNum === versionNum);
        if (matches.length === 0) return null;
        // Prefer record with thinkingLogId, else highest id (most recent)
        const withLog = matches.filter(v => v.thinkingLogId != null);
        if (withLog.length > 0) {
            return withLog.reduce((a, b) => ((a.id || 0) >= (b.id || 0) ? a : b));
        }
        return matches.reduce((a, b) => ((a.id || 0) >= (b.id || 0) ? a : b));
    }

    /**
     * Get all versions ordered by versionNum ascending.
     * @returns {Promise<Object[]>}
     */
    async getAllVersions() {
        return this._loadVersions();
    }

    /**
     * Get the latest (highest versionNum) version.
     * @returns {Promise<Object|null>}
     */
    async getLatestVersion() {
        const versions = await this._loadVersions();
        if (versions.length === 0) return null;
        return versions[versions.length - 1];
    }

    /**
     * Walk the parent chain from a given version back to v1.
     * Returns versions in order [v1, v2, ..., versionNum].
     *
     * @param {number} versionNum
     * @returns {Promise<Object[]>} - Ordered chain
     */
    async getVersionChain(versionNum) {
        const versions = await this._loadVersions();
        const versionMap = new Map(versions.map(v => [v.versionNum, v]));

        const chain = [];
        let current = versionMap.get(versionNum);

        while (current) {
            chain.unshift(current);
            current = current.parentVersion != null
                ? versionMap.get(current.parentVersion)
                : null;
        }

        return chain;
    }

    /**
     * Get the thinking log text for a specific version.
     *
     * @param {number} versionNum
     * @returns {Promise<string>} - Thinking log text, or empty string
     */
    async getThinkingLogForVersion(versionNum) {
        const version = await this.getVersion(versionNum);
        if (!version || version.thinkingLogId == null) {
            return '';
        }

        const log = await this._storage.getThinkingLogById(version.thinkingLogId);
        return log ? log.content : '';
    }

    /**
     * Get the next version number without creating a version.
     * @returns {Promise<number>}
     */
    async getNextVersionNumber() {
        const versions = await this._loadVersions();
        if (versions.length === 0) return 1;
        return Math.max(...versions.map(v => v.versionNum)) + 1;
    }

    /**
     * Get total version count.
     * @returns {Promise<number>}
     */
    async getVersionCount() {
        const versions = await this._loadVersions();
        return versions.length;
    }

    /**
     * Simple line-based diff between two versions.
     *
     * @param {number} v1Num - Earlier version number
     * @param {number} v2Num - Later version number
     * @returns {Promise<{added: string[], removed: string[], changed: number, v1Text: string, v2Text: string}>}
     */
    async diffVersions(v1Num, v2Num) {
        const [v1, v2] = await Promise.all([
            this.getVersion(v1Num),
            this.getVersion(v2Num),
        ]);

        if (!v1) throw new Error(`Version ${v1Num} not found`);
        if (!v2) throw new Error(`Version ${v2Num} not found`);

        const lines1 = v1.promptText.split('\n');
        const lines2 = v2.promptText.split('\n');

        const set1 = new Set(lines1);
        const set2 = new Set(lines2);

        const added = lines2.filter(l => !set1.has(l));
        const removed = lines1.filter(l => !set2.has(l));

        return {
            added,
            removed,
            changed: added.length + removed.length,
            v1Text: v1.promptText,
            v2Text: v2.promptText,
        };
    }

    /**
     * Invalidate the in-memory cache, forcing reload from storage.
     */
    invalidateCache() {
        this._cache = null;
    }

    // ── Private ───────────────────────────────────────────────────────────

    /**
     * Load and cache versions from storage.
     * @private
     * @returns {Promise<Object[]>}
     */
    async _loadVersions() {
        if (this._cache) return this._cache;
        const versions = await this._storage.getVersions(this.modelId);
        // Sort by versionNum ascending
        versions.sort((a, b) => a.versionNum - b.versionNum);
        this._cache = versions;
        return versions;
    }

    /**
     * Rough token estimate: chars / 4.
     * @private
     */
    _estimateTokens(text) {
        return Math.ceil((text || '').length / 4);
    }
}

// ── Exports ────────────────────────────────────────────────────────────────
export { VersionManager };
