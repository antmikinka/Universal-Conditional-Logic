/**
 * IterationController - Optimization Loop Control with 5 Stopping Criteria
 *
 * Controls the optimization iteration loop, tracking scores, tokens, and
 * convergence. Evaluates 6 stopping criteria in priority order:
 *
 *   1. Manual cancel → stop immediately ('cancelled')
 *   2. iteration >= maxIterations → 'max_iterations'
 *   3. compositeScore >= qualityThreshold → 'quality_threshold'
 *   4. isConverged() → 'converged'
 *   5. totalTokens >= tokenBudget (when budget > 0) → 'token_budget'
 *   6. Score drops > 0.15 from best → 'degradation'
 *
 * @module iteration-controller
 */

/**
 * @typedef {Object} OptimizationConfig
 * @property {number} [maxIterations=10] - Maximum number of iterations
 * @property {number} [qualityThreshold=0.8] - Target quality score (0.0-1.0)
 * @property {number} [convergenceDelta=0.01] - Minimum improvement to count as progress
 * @property {number} [convergencePatience=3] - Number of iterations to check for convergence
 * @property {number} [tokenBudget=0] - Maximum total tokens (0 = unlimited)
 * @property {number} [degradationThreshold=0.15] - Score drop from best that triggers stop
 */

/**
 * @typedef {Object} IterationData
 * @property {number} iterationNum - Iteration number (1-based)
 * @property {number} compositeScore - Quality score for this iteration
 * @property {number} tokensUsed - Tokens consumed in this iteration
 * @property {number} duration - Iteration duration in ms
 */

/**
 * @typedef {Object} IterationDecision
 * @property {boolean} continue - Whether to continue iterating
 * @property {string|null} reason - Reason for stopping (null if continuing)
 * @property {Object} details - Current state snapshot
 * @property {number} details.currentScore - Most recent score
 * @property {number} details.iteration - Current iteration number
 * @property {number} details.maxIterations - Max iterations configured
 * @property {number} details.bestScore - Best score achieved so far
 * @property {boolean} details.converged - Whether convergence detected
 */

/**
 * Default optimization configuration.
 */
const DEFAULT_CONFIG = {
    maxIterations: 10,
    qualityThreshold: 0.8,
    convergenceDelta: 0.01,
    convergencePatience: 3,
    tokenBudget: 0,
    degradationThreshold: 0.15,
};

class IterationController {
    /**
     * @param {OptimizationConfig} [config={}] - Optimization configuration
     */
    constructor(config = {}) {
        /** @type {OptimizationConfig} */
        this._config = { ...DEFAULT_CONFIG, ...config };

        /** @type {boolean} */
        this._cancelled = false;

        /** @type {Object[]} */
        this._history = [];

        /** @type {number} */
        this._totalTokensUsed = 0;
    }

    // ── Public API ──────────────────────────────────────────────────────────

    /**
     * Record an iteration's results.
     * Updates history, token tracking, and best score.
     *
     * @param {IterationData} data - Iteration results
     */
    recordIteration(data) {
        const record = {
            iterationNum: data.iterationNum || this._history.length + 1,
            compositeScore: data.compositeScore || 0,
            tokensUsed: data.tokensUsed || 0,
            duration: data.duration || 0,
            timestamp: Date.now(),
        };

        this._history.push(record);
        this._totalTokensUsed += record.tokensUsed;
    }

    /**
     * Determine whether the optimization loop should continue.
     * Checks stopping criteria in priority order.
     *
     * @returns {IterationDecision} Decision with continue flag and reason
     */
    shouldContinue() {
        const currentIteration = this._history.length;
        const currentScore = currentIteration > 0
            ? this._history[currentIteration - 1].compositeScore
            : 0;
        const bestScore = this._getBestScore();
        const converged = this.isConverged();

        const details = {
            currentScore,
            iteration: currentIteration,
            maxIterations: this._config.maxIterations,
            bestScore,
            converged,
        };

        // ── Priority 1: Manual cancel ──────────────────────────────────
        if (this._cancelled) {
            return { continue: false, reason: 'cancelled', details };
        }

        // ── Priority 2: Max iterations reached ─────────────────────────
        if (currentIteration >= this._config.maxIterations) {
            return { continue: false, reason: 'max_iterations', details };
        }

        // ── Priority 3: Quality threshold met ──────────────────────────
        if (currentIteration > 0 && currentScore >= this._config.qualityThreshold) {
            return { continue: false, reason: 'quality_threshold', details };
        }

        // ── Priority 4: Converged ──────────────────────────────────────
        if (converged) {
            return { continue: false, reason: 'converged', details };
        }

        // ── Priority 5: Token budget exhausted ─────────────────────────
        if (this._config.tokenBudget > 0 && this._totalTokensUsed >= this._config.tokenBudget) {
            return { continue: false, reason: 'token_budget', details };
        }

        // ── Priority 6: Score degradation ──────────────────────────────
        if (currentIteration > 0 && bestScore > 0) {
            const drop = bestScore - currentScore;
            if (drop > this._config.degradationThreshold) {
                return { continue: false, reason: 'degradation', details };
            }
        }

        // ── Continue ───────────────────────────────────────────────────
        return { continue: true, reason: null, details };
    }

    /**
     * Cancel the optimization loop.
     * Sets the cancelled flag — shouldContinue() will return false.
     */
    cancel() {
        this._cancelled = true;
    }

    /**
     * Reset all state. Clears history, tokens, and cancelled flag.
     * Does NOT change the configuration.
     */
    reset() {
        this._cancelled = false;
        this._history = [];
        this._totalTokensUsed = 0;
    }

    /**
     * Get the current state of the controller.
     *
     * @returns {Object} State snapshot
     */
    getState() {
        return {
            cancelled: this._cancelled,
            iterationCount: this._history.length,
            totalTokensUsed: this._totalTokensUsed,
            bestScore: this._getBestScore(),
            converged: this.isConverged(),
            config: { ...this._config },
        };
    }

    /**
     * Get the full iteration history.
     *
     * @returns {Object[]} Array of iteration records
     */
    getHistory() {
        return [...this._history];
    }

    /**
     * Get the iteration with the highest composite score.
     *
     * @returns {Object|null} Best iteration record, or null if no history
     */
    getBestIteration() {
        if (this._history.length === 0) return null;

        let best = this._history[0];
        for (let i = 1; i < this._history.length; i++) {
            if (this._history[i].compositeScore > best.compositeScore) {
                best = this._history[i];
            }
        }
        return { ...best };
    }

    /**
     * Detect whether the optimization has converged.
     * Convergence = all recent improvements are below convergenceDelta.
     *
     * Requires at least (patience + 1) iterations to evaluate.
     *
     * @returns {boolean} True if converged
     */
    isConverged() {
        const { convergencePatience: patience, convergenceDelta: delta } = this._config;

        if (this._history.length < patience + 1) return false;

        const recent = this._history.slice(-(patience + 1));
        const improvements = [];

        for (let i = 1; i < recent.length; i++) {
            improvements.push(recent[i].compositeScore - recent[i - 1].compositeScore);
        }

        return improvements.every(imp => imp < delta);
    }

    /**
     * Get the total tokens consumed across all iterations.
     *
     * @returns {number} Total tokens used
     */
    getTotalTokensUsed() {
        return this._totalTokensUsed;
    }

    // ── Private Helpers ─────────────────────────────────────────────────────

    /**
     * Get the best composite score from history.
     * @private
     * @returns {number} Best score, or 0 if no history
     */
    _getBestScore() {
        if (this._history.length === 0) return 0;
        return this._history.reduce((max, h) => Math.max(max, h.compositeScore), -Infinity);
    }
}

export { IterationController, DEFAULT_CONFIG };
