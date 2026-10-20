/**
 * OptimizationHistory - Optimization Run Tracker
 *
 * Tracks full optimization runs with per-iteration metrics.
 * Manages the lifecycle of optimization runs (start → iterate → complete/fail)
 * and provides cross-run analysis capabilities.
 *
 * @module optimization-history
 */

/**
 * @typedef {Object} OptimizationRun
 * @property {string} runId - Unique run identifier
 * @property {string} modelId - Model being optimized
 * @property {number} startedAt - Epoch ms when run started
 * @property {number|null} completedAt - Epoch ms when run finished (null if running)
 * @property {'running'|'completed'|'failed'|'cancelled'} status - Run status
 * @property {string} basePrompt - Original base prompt being optimized
 * @property {Object} config - Run configuration
 * @property {number} config.maxIterations - Maximum number of iterations
 * @property {number} config.qualityThreshold - Target quality score (0.0-1.0)
 * @property {Object[]} iterations - Per-iteration data
 * @property {number|null} finalScore - Final quality score (null if not completed)
 * @property {number|null} bestVersionNum - Best version number found
 * @property {string|null} stopReason - Why the run stopped
 * @property {string|null} errorMessage - Error message if failed
 */

/**
 * @typedef {Object} IterationRecord
 * @property {number} iterationNum - Iteration number (1-based)
 * @property {number} versionNum - Version number produced
 * @property {string} promptText - The prompt text for this iteration
 * @property {number} qualityScore - Quality score for this iteration
 * @property {Object[]} testResults - Test results for this iteration
 * @property {string} thinkingExtracted - Extracted thinking from model
 * @property {Object} usage - Token usage stats
 * @property {number} duration - Iteration duration in ms
 */

class OptimizationHistory {
    /**
     * @param {import('./storage-manager.js').StorageManager} storageManager
     */
    constructor(storageManager) {
        /** @type {import('./storage-manager.js').StorageManager} */
        this._storage = storageManager;
    }

    // ── Lifecycle Management ──────────────────────────────────────────────

    /**
     * Start a new optimization run.
     * Creates and persists the initial run record.
     *
     * @param {Object} params
     * @param {string} params.modelId - Model being optimized
     * @param {string} params.basePrompt - Original base prompt
     * @param {number} [params.maxIterations=10] - Maximum iterations
     * @param {number} [params.qualityThreshold=0.8] - Target quality score
     * @param {Object} [params.config={}] - Additional config
     * @returns {Promise<OptimizationRun>} The created run record
     */
    async startRun({ modelId, basePrompt, maxIterations = 10, qualityThreshold = 0.8, config = {} }) {
        const run = {
            runId: this._generateRunId(),
            modelId,
            startedAt: Date.now(),
            completedAt: null,
            status: 'running',
            basePrompt,
            config: {
                maxIterations,
                qualityThreshold,
                ...config,
            },
            iterations: [],
            finalScore: null,
            bestVersionNum: null,
            stopReason: null,
            errorMessage: null,
        };

        await this._storage.saveOptimizationRun(run);
        return run;
    }

    /**
     * Record an iteration within a running optimization.
     * Updates the run with new iteration data and tracks the best version.
     *
     * @param {string} runId - The run to update
     * @param {IterationRecord} iteration - Iteration data
     * @returns {Promise<OptimizationRun>} Updated run record
     */
    async recordIteration(runId, iteration) {
        const run = await this._storage.getOptimizationRun(runId);
        if (!run) {
            throw new Error(`Optimization run ${runId} not found`);
        }
        if (run.status !== 'running') {
            throw new Error(`Cannot record iteration: run ${runId} is ${run.status}`);
        }

        run.iterations.push(iteration);

        // Track best version: compare against previous best (before this iteration)
        const prevBestScore = run.iterations.length > 1
            ? Math.max(...run.iterations.slice(0, -1).map(i => i.qualityScore || 0))
            : -1;

        if (iteration.qualityScore > prevBestScore) {
            run.bestVersionNum = iteration.versionNum;
        }

        await this._storage.saveOptimizationRun(run);
        return run;
    }

    /**
     * Mark a run as successfully completed.
     *
     * @param {string} runId
     * @param {string} [stopReason='completed'] - Reason for stopping
     * @returns {Promise<OptimizationRun>} Updated run record
     */
    async completeRun(runId, stopReason = 'completed') {
        const run = await this._storage.getOptimizationRun(runId);
        if (!run) {
            throw new Error(`Optimization run ${runId} not found`);
        }

        run.status = 'completed';
        run.completedAt = Date.now();
        run.stopReason = stopReason;

        // Calculate final score from best iteration
        run.finalScore = this._getBestScore(run);

        await this._storage.saveOptimizationRun(run);
        return run;
    }

    /**
     * Mark a run as failed.
     *
     * @param {string} runId
     * @param {string} errorMessage - Error description
     * @returns {Promise<OptimizationRun>} Updated run record
     */
    async failRun(runId, errorMessage) {
        const run = await this._storage.getOptimizationRun(runId);
        if (!run) {
            throw new Error(`Optimization run ${runId} not found`);
        }

        run.status = 'failed';
        run.completedAt = Date.now();
        run.errorMessage = errorMessage;
        run.stopReason = 'error';

        // Calculate final score from whatever iterations we have
        run.finalScore = run.iterations.length > 0 ? this._getBestScore(run) : null;

        await this._storage.saveOptimizationRun(run);
        return run;
    }

    /**
     * Cancel a running optimization.
     *
     * @param {string} runId
     * @returns {Promise<OptimizationRun>} Updated run record
     */
    async cancelRun(runId) {
        const run = await this._storage.getOptimizationRun(runId);
        if (!run) {
            throw new Error(`Optimization run ${runId} not found`);
        }

        run.status = 'cancelled';
        run.completedAt = Date.now();
        run.stopReason = 'cancelled';
        run.finalScore = run.iterations.length > 0 ? this._getBestScore(run) : null;

        await this._storage.saveOptimizationRun(run);
        return run;
    }

    // ── Retrieval ─────────────────────────────────────────────────────────

    /**
     * Get a specific optimization run by ID.
     *
     * @param {string} runId
     * @returns {Promise<OptimizationRun|null>}
     */
    async getRun(runId) {
        return this._storage.getOptimizationRun(runId);
    }

    /**
     * List all optimization runs, ordered by startedAt descending.
     * Optionally filter by model or status.
     *
     * @param {Object} [filters]
     * @param {string} [filters.modelId] - Filter by model
     * @param {string} [filters.status] - Filter by status
     * @returns {Promise<OptimizationRun[]>}
     */
    async listRuns(filters = {}) {
        let runs = await this._storage.listOptimizationRuns();

        if (filters.modelId) {
            runs = runs.filter(r => r.modelId === filters.modelId);
        }
        if (filters.status) {
            runs = runs.filter(r => r.status === filters.status);
        }

        return runs;
    }

    // ── Analysis ──────────────────────────────────────────────────────────

    /**
     * Get cross-run trends for a specific model.
     * Analyzes how optimization quality has improved over time.
     *
     * @param {string} modelId
     * @returns {Promise<Object>} Trend analysis
     */
    async getTrends(modelId) {
        const runs = await this.listRuns({ modelId, status: 'completed' });

        if (runs.length === 0) {
            return {
                modelId,
                totalRuns: 0,
                trend: 'none',
                avgScore: 0,
                scoreHistory: [],
                avgIterations: 0,
                improvementRate: 0,
            };
        }

        // Sort by startedAt ascending for trend analysis
        const sorted = [...runs].sort((a, b) => a.startedAt - b.startedAt);
        const scores = sorted.map(r => r.finalScore || 0);

        const avgScore = scores.reduce((s, v) => s + v, 0) / scores.length;
        const avgIterations = sorted.reduce((s, r) => s + r.iterations.length, 0) / sorted.length;

        // Determine trend direction
        let trend = 'stable';
        if (scores.length >= 2) {
            const firstHalf = scores.slice(0, Math.floor(scores.length / 2));
            const secondHalf = scores.slice(Math.floor(scores.length / 2));
            const firstAvg = firstHalf.reduce((s, v) => s + v, 0) / firstHalf.length;
            const secondAvg = secondHalf.reduce((s, v) => s + v, 0) / secondHalf.length;

            if (secondAvg - firstAvg > 0.05) {
                trend = 'improving';
            } else if (firstAvg - secondAvg > 0.05) {
                trend = 'declining';
            }
        }

        // Improvement rate: how much score improved from first to last run
        const improvementRate = scores.length >= 2
            ? scores[scores.length - 1] - scores[0]
            : 0;

        return {
            modelId,
            totalRuns: runs.length,
            trend,
            avgScore,
            scoreHistory: scores,
            avgIterations,
            improvementRate,
        };
    }

    /**
     * Find the best version across all completed runs for a model.
     *
     * @param {string} modelId
     * @returns {Promise<Object|null>} Best version info or null if no runs
     */
    async findBestVersion(modelId) {
        const runs = await this.listRuns({ modelId, status: 'completed' });

        if (runs.length === 0) return null;

        let bestRun = null;
        let bestIteration = null;
        let bestScore = -1;

        for (const run of runs) {
            for (const iter of run.iterations) {
                if (iter.qualityScore > bestScore) {
                    bestScore = iter.qualityScore;
                    bestRun = run;
                    bestIteration = iter;
                }
            }
        }

        if (!bestIteration || !bestRun) return null;

        return {
            runId: bestRun.runId,
            modelId,
            versionNum: bestIteration.versionNum,
            iterationNum: bestIteration.iterationNum,
            qualityScore: bestIteration.qualityScore,
            promptText: bestIteration.promptText,
            testResults: bestIteration.testResults,
            thinkingExtracted: bestIteration.thinkingExtracted,
            timestamp: bestRun.startedAt,
        };
    }

    // ── Private Helpers ───────────────────────────────────────────────────

    /**
     * Generate a unique run ID.
     * @private
     * @returns {string}
     */
    _generateRunId() {
        const timestamp = Date.now().toString(36);
        const random = Math.random().toString(36).slice(2, 8);
        return `run_${timestamp}_${random}`;
    }

    /**
     * Get the best quality score from a run's iterations.
     * @private
     * @param {OptimizationRun} run
     * @returns {number}
     */
    _getBestScore(run) {
        if (run.iterations.length === 0) return 0;
        return run.iterations.reduce((max, i) => Math.max(max, i.qualityScore || 0), -Infinity);
    }
}

export { OptimizationHistory };
