/**
 * QualityScorer - Composite Quality Scoring Engine
 *
 * Computes a composite quality score from 4 dimensions:
 *   1. Syntax (25%) - UCL structural correctness
 *   2. Test Pass Rate (40%) - Weighted test case results
 *   3. Reasoning Depth (20%) - Thinking/reasoning quality
 *   4. Conciseness (15%) - Prompt bloat penalty
 *
 * All scores normalized to 0.0-1.0 range.
 *
 * @module quality-scorer
 */

/**
 * @typedef {Object} ScoreInputs
 * @property {string} promptText - The UCL prompt text to evaluate
 * @property {import('./test-evaluator.js').TestResult[]} testResults - Test results from TestEvaluator
 * @property {import('./test-evaluator.js').TestCase[]} [testCases] - Test cases for weight lookup
 * @property {string} [thinkingText=''] - Extracted thinking/reasoning text
 * @property {number} [previousScore=null] - Previous composite score for delta calculation
 */

/**
 * @typedef {Object} QualityScore
 * @property {number} composite - Weighted composite score (0.0-1.0)
 * @property {Object} breakdown - Individual dimension scores
 * @property {number} breakdown.syntaxScore - Syntax quality (0.0-1.0)
 * @property {number} breakdown.testPassRate - Test pass rate (0.0-1.0)
 * @property {number} breakdown.reasoningDepth - Reasoning depth (0.0-1.0)
 * @property {number} breakdown.conciseness - Conciseness score (0.0-1.0)
 * @property {number|null} delta - Change from previous score (null on first iteration)
 * @property {'excellent'|'good'|'needs_work'|'poor'} verdict - Quality verdict
 */

/**
 * Scoring dimension weights (must sum to 1.0).
 */
const WEIGHTS = {
    syntax: 0.25,
    testPassRate: 0.40,
    reasoningDepth: 0.20,
    conciseness: 0.15,
};

class QualityScorer {
    // ── Public API ──────────────────────────────────────────────────────────

    /**
     * Compute the composite quality score from all 4 dimensions.
     *
     * @param {ScoreInputs} inputs - Scoring inputs
     * @returns {QualityScore} Composite quality score with breakdown
     */
    static computeScore(inputs) {
        const {
            promptText = '',
            testResults = [],
            testCases = [],
            thinkingText = '',
            previousScore = null,
        } = inputs;

        const syntaxScore = QualityScorer.scoreSyntax(promptText);
        const testPassRate = QualityScorer.scoreTestPassRate(testResults, testCases);
        const reasoningDepth = QualityScorer.scoreReasoningDepth(thinkingText);
        const conciseness = QualityScorer.scoreConciseness(promptText);

        const composite =
            syntaxScore * WEIGHTS.syntax +
            testPassRate * WEIGHTS.testPassRate +
            reasoningDepth * WEIGHTS.reasoningDepth +
            conciseness * WEIGHTS.conciseness;

        const clampedComposite = Math.max(0, Math.min(1, composite));
        const delta = previousScore !== null && previousScore !== undefined
            ? QualityScorer.computeDelta(clampedComposite, previousScore)
            : null;

        return {
            composite: clampedComposite,
            breakdown: {
                syntaxScore,
                testPassRate,
                reasoningDepth,
                conciseness,
            },
            delta,
            verdict: QualityScorer._computeVerdict(clampedComposite),
        };
    }

    /**
     * Score UCL syntax quality.
     *
     * Required checks (each contributes 0.25 to syntax score):
     *   - hasCritical: Contains [[CRITICAL: directive
     *   - hasCondition: Contains ^^CONDITION block
     *   - hasCloseCondition: Contains ^^/CONDITION^^ closer
     *   - noOrphanTags: Every opened condition is properly closed
     *
     * Bonus checks (each contributes 0.125, capped at 1.0):
     *   - hasParam: Contains [[PARAM: definitions
     *   - hasDirective: Contains other directive patterns like [[TOOL: or [[DIRECTIVE:
     *
     * @param {string} promptText - UCL prompt text
     * @returns {number} Syntax score (0.0-1.0)
     */
    static scoreSyntax(promptText) {
        if (!promptText || typeof promptText !== 'string') return 0;

        const text = promptText.trim();
        if (text.length === 0) return 0;

        let score = 0;

        // ── Required checks (4 × 0.25 = 1.0) ────────────────────────────
        const hasCritical = /\[\[CRITICAL:/.test(text);
        const hasCondition = /\^\^CONDITION/.test(text);
        const hasCloseCondition = /\^\^\/CONDITION\^\^/.test(text);

        // Count opens and closes for orphan detection
        const openCount = (text.match(/\^\^CONDITION/g) || []).length;
        const closeCount = (text.match(/\^\^\/CONDITION\^\^/g) || []).length;
        const noOrphanTags = openCount > 0 && openCount === closeCount;

        if (hasCritical) score += 0.25;
        if (hasCondition) score += 0.25;
        if (hasCloseCondition) score += 0.25;
        if (noOrphanTags) score += 0.25;

        // ── Bonus checks (2 × 0.125, capped at 1.0) ─────────────────────
        const hasParam = /\[\[PARAM:/.test(text);
        const hasDirective = /\[\[(TOOL|DIRECTIVE):/.test(text);

        if (hasParam) score += 0.125;
        if (hasDirective) score += 0.125;

        return Math.min(1.0, score);
    }

    /**
     * Score test pass rate as a weighted average.
     * Each result's score is weighted by the test case's weight.
     * If no test cases provided, uses unweighted average.
     *
     * @param {import('./test-evaluator.js').TestResult[]} results - Test results
     * @param {import('./test-evaluator.js').TestCase[]} [testCases=[]] - Test cases for weight lookup
     * @returns {number} Weighted test pass rate (0.0-1.0)
     */
    static scoreTestPassRate(results, testCases = []) {
        if (!results || results.length === 0) return 0;

        // Build weight lookup from test cases
        const weightMap = new Map();
        for (const tc of testCases) {
            weightMap.set(tc.id, tc.weight || 1.0);
        }

        let totalWeight = 0;
        let weightedSum = 0;

        for (const result of results) {
            const weight = weightMap.get(result.testCaseId) || 1.0;
            weightedSum += (result.score || 0) * weight;
            totalWeight += weight;
        }

        return totalWeight > 0 ? weightedSum / totalWeight : 0;
    }

    /**
     * Score reasoning depth based on thinking text length and structure.
     *
     * Length-based tiers:
     *   - <50 chars → 0.1 (minimal reasoning)
     *   - 50-200 chars → 0.4 (basic reasoning)
     *   - 200-500 chars → 0.7 (substantial reasoning)
     *   - >500 chars → 1.0 (deep reasoning)
     *
     * Structure markers boost (+0.15, capped at 1.0):
     *   - Numbered lists (1., 2., etc.)
     *   - Bullet points (- or *)
     *   - Step markers ("Step", "Phase")
     *   - Logical connectors ("Therefore", "Because", "However")
     *
     * @param {string} thinkingText - Extracted thinking/reasoning text
     * @returns {number} Reasoning depth score (0.0-1.0)
     */
    static scoreReasoningDepth(thinkingText) {
        if (!thinkingText || typeof thinkingText !== 'string') return 0;

        const text = thinkingText.trim();
        if (text.length === 0) return 0;

        // ── Length-based tier ────────────────────────────────────────────
        let score;
        if (text.length < 50) {
            score = 0.1;
        } else if (text.length < 200) {
            score = 0.4;
        } else if (text.length < 500) {
            score = 0.7;
        } else {
            score = 1.0;
        }

        // ── Structure marker boost ───────────────────────────────────────
        const structureMarkers = [
            /\d+\.\s/g,                          // Numbered lists
            /[-*]\s/g,                           // Bullet points
            /\b(step|phase)\b/gi,                // Step markers
            /\b(therefore|because|however|thus|hence|consequently)\b/gi,  // Logical connectors
        ];

        let markerCount = 0;
        for (const pattern of structureMarkers) {
            const matches = text.match(pattern);
            if (matches && matches.length > 0) {
                markerCount++;
            }
        }

        // Boost: +0.05 per marker type found (max 4 types × 0.05 = 0.20)
        const boost = Math.min(0.20, markerCount * 0.05);
        score = Math.min(1.0, score + boost);

        return score;
    }

    /**
     * Score conciseness with penalty for bloat.
     *
     * Length-based tiers:
     *   - <500 chars → 1.0 (concise)
     *   - 500-1000 chars → 0.9 (moderate)
     *   - 1000-2000 chars → 0.7 (verbose)
     *   - >2000 chars → 0.5 (bloated)
     *
     * @param {string} promptText - UCL prompt text
     * @returns {number} Conciseness score (0.0-1.0)
     */
    static scoreConciseness(promptText) {
        if (promptText === null || promptText === undefined || typeof promptText !== 'string') return 0;

        const length = promptText.trim().length;

        if (length === 0) return 1.0; // Empty prompt is trivially concise
        if (length < 500) return 1.0;
        if (length < 1000) return 0.9;
        if (length < 2000) return 0.7;
        return 0.5;
    }

    /**
     * Compute the delta (change) between current and previous scores.
     *
     * @param {number} current - Current composite score
     * @param {number} previous - Previous composite score
     * @returns {number} Delta (positive = improvement, negative = regression)
     */
    static computeDelta(current, previous) {
        return Math.round((current - previous) * 1000) / 1000;
    }

    /**
     * Generate a human-readable quality report.
     *
     * @param {QualityScore} score - The quality score to report on
     * @returns {string} Formatted quality report text
     */
    static generateReport(score) {
        if (!score || typeof score !== 'object') {
            return 'Invalid score data.';
        }

        const lines = [];
        const pct = (v) => `${(v * 100).toFixed(1)}%`;

        lines.push('=== Quality Score Report ===');
        lines.push('');
        lines.push(`Composite Score: ${pct(score.composite)} (${score.verdict.replace('_', ' ')})`);

        if (score.delta !== null && score.delta !== undefined) {
            const sign = score.delta >= 0 ? '+' : '';
            lines.push(`Delta: ${sign}${(score.delta * 100).toFixed(1)}%`);
        }

        lines.push('');
        lines.push('Dimension Breakdown:');
        lines.push(`  Syntax:          ${pct(score.breakdown.syntaxScore)} (weight: ${pct(WEIGHTS.syntax)})`);
        lines.push(`  Test Pass Rate:  ${pct(score.breakdown.testPassRate)} (weight: ${pct(WEIGHTS.testPassRate)})`);
        lines.push(`  Reasoning Depth: ${pct(score.breakdown.reasoningDepth)} (weight: ${pct(WEIGHTS.reasoningDepth)})`);
        lines.push(`  Conciseness:     ${pct(score.breakdown.conciseness)} (weight: ${pct(WEIGHTS.conciseness)})`);

        lines.push('');
        if (score.verdict === 'excellent') {
            lines.push('Verdict: Excellent quality. Prompt is well-structured and effective.');
        } else if (score.verdict === 'good') {
            lines.push('Verdict: Good quality. Minor improvements may push this to excellent.');
        } else if (score.verdict === 'needs_work') {
            lines.push('Verdict: Needs work. Focus on the weakest dimensions for improvement.');
        } else {
            lines.push('Verdict: Poor quality. Significant restructuring recommended.');
        }

        return lines.join('\n');
    }

    // ── Private Helpers ─────────────────────────────────────────────────────

    /**
     * Compute verdict from composite score.
     * @private
     * @param {number} composite - Composite score (0.0-1.0)
     * @returns {'excellent'|'good'|'needs_work'|'poor'}
     */
    static _computeVerdict(composite) {
        if (composite >= 0.9) return 'excellent';
        if (composite >= 0.75) return 'good';
        if (composite >= 0.5) return 'needs_work';
        return 'poor';
    }
}

export { QualityScorer, WEIGHTS };
