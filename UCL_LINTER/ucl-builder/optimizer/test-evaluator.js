/**
 * TestEvaluator - Structured Test Case Evaluation Engine
 *
 * Evaluates model responses against structured test cases with multiple
 * expectation types: contains, notContains, minlength, thinkingCheck.
 *
 * Supports weighted scoring, batch evaluation, summary aggregation,
 * and human-readable feedback generation.
 *
 * @module test-evaluator
 */

/**
 * @typedef {Object} TestCase
 * @property {string} id - Unique test case identifier
 * @property {string} name - Human-readable test name
 * @property {'syntax'|'behavior'|'custom'} category - Test category
 * @property {string} input - Input prompt to send to the model
 * @property {Object} expectations - What to check in the response
 * @property {string[]} [expectations.contains] - Strings that must appear in response
 * @property {string[]} [expectations.notContains] - Strings that must NOT appear
 * @property {number} [expectations.minlength] - Minimum response length
 * @property {Object} [expectations.thinkingCheck] - Thinking/reasoning validation
 * @property {boolean} expectations.thinkingCheck.enabled - Whether thinking check is active
 * @property {string[]} [expectations.thinkingCheck.mustContain] - Strings required in thinking block
 * @property {number} weight - Importance weight (0.0-1.0)
 * @property {boolean} enabled - Whether this test is active
 */

/**
 * @typedef {Object} TestResult
 * @property {string} testCaseId - ID of the test case
 * @property {number} versionNum - Version number being tested
 * @property {string} runId - Optimization run ID
 * @property {number} iteration - Iteration number within the run
 * @property {boolean} passed - Whether the test passed overall
 * @property {number} score - Score from 0.0 to 1.0
 * @property {Object} details - Detailed check results
 * @property {Object} details.checks - Individual check results
 * @property {string[]} details.failures - List of failure descriptions
 * @property {string} modelResponse - The model's response text
 * @property {string} thinkingExtracted - Extracted thinking/reasoning text
 * @property {Object} usage - Token usage stats
 * @property {number} usage.promptTokens
 * @property {number} usage.completionTokens
 * @property {number} usage.totalTokens
 * @property {number} duration - Response time in ms
 * @property {number} timestamp - Epoch ms
 */

class TestEvaluator {
    constructor() {
        /** @type {Map<string, TestCase>} */
        this._testCases = new Map();
    }

    // ── Test Case Management ──────────────────────────────────────────────

    /**
     * Add one or more test cases.
     * @param {TestCase|TestCase[]} testCases - Test case(s) to add
     * @returns {number} Total number of test cases after adding
     */
    addTestCases(testCases) {
        const cases = Array.isArray(testCases) ? testCases : [testCases];
        for (const tc of cases) {
            if (!tc.id) {
                tc.id = `tc_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
            }
            if (tc.enabled === undefined) tc.enabled = true;
            if (tc.weight === undefined) tc.weight = 1.0;
            this._testCases.set(tc.id, tc);
        }
        return this._testCases.size;
    }

    /**
     * Remove a test case by ID.
     * @param {string} testCaseId
     * @returns {boolean} True if removed, false if not found
     */
    removeTestCase(testCaseId) {
        return this._testCases.delete(testCaseId);
    }

    /**
     * Enable or disable a test case.
     * @param {string} testCaseId
     * @param {boolean} enabled
     * @returns {boolean} True if updated, false if not found
     */
    setTestCaseEnabled(testCaseId, enabled) {
        const tc = this._testCases.get(testCaseId);
        if (!tc) return false;
        tc.enabled = enabled;
        return true;
    }

    /**
     * Get all test cases (optionally filtered by enabled status).
     * @param {boolean} [enabledOnly=false]
     * @returns {TestCase[]}
     */
    getTestCases(enabledOnly = false) {
        const all = [...this._testCases.values()];
        return enabledOnly ? all.filter(tc => tc.enabled) : all;
    }

    /**
     * Get a single test case by ID.
     * @param {string} testCaseId
     * @returns {TestCase|null}
     */
    getTestCase(testCaseId) {
        return this._testCases.get(testCaseId) || null;
    }

    /**
     * Clear all test cases.
     */
    clearTestCases() {
        this._testCases.clear();
    }

    // ── Evaluation ────────────────────────────────────────────────────────

    /**
     * Evaluate a single model response against a single test case.
     *
     * @param {string} modelResponse - The model's response text
     * @param {TestCase} testCase - The test case to evaluate against
     * @param {Object} [options] - Additional context
     * @param {string} [options.thinkingExtracted=''] - Extracted thinking text
     * @param {number} [options.versionNum=0] - Version number
     * @param {string} [options.runId=''] - Run ID
     * @param {number} [options.iteration=0] - Iteration number
     * @param {Object} [options.usage={}] - Token usage stats
     * @param {number} [options.duration=0] - Response duration in ms
     * @returns {TestResult}
     */
    evaluateOne(modelResponse, testCase, options = {}) {
        const {
            thinkingExtracted = '',
            versionNum = 0,
            runId = '',
            iteration = 0,
            usage = { promptTokens: 0, completionTokens: 0, totalTokens: 0 },
            duration = 0,
        } = options;

        const expectations = testCase.expectations || {};
        const checks = {};
        const failures = [];
        let checkCount = 0;
        let checkPasses = 0;

        // ── contains check ────────────────────────────────────────────
        if (expectations.contains && Array.isArray(expectations.contains)) {
            const containsResults = {};
            for (const str of expectations.contains) {
                checkCount++;
                const found = modelResponse.includes(str);
                containsResults[str] = found;
                if (found) {
                    checkPasses++;
                } else {
                    failures.push(`Missing expected string: "${str}"`);
                }
            }
            checks.contains = containsResults;
        }

        // ── notContains check ─────────────────────────────────────────
        if (expectations.notContains && Array.isArray(expectations.notContains)) {
            const notContainsResults = {};
            for (const str of expectations.notContains) {
                checkCount++;
                const absent = !modelResponse.includes(str);
                notContainsResults[str] = absent;
                if (absent) {
                    checkPasses++;
                } else {
                    failures.push(`Found forbidden string: "${str}"`);
                }
            }
            checks.notContains = notContainsResults;
        }

        // ── minlength check ───────────────────────────────────────────
        if (expectations.minlength !== undefined && expectations.minlength !== null) {
            checkCount++;
            const meetsLength = modelResponse.length >= expectations.minlength;
            checks.minlength = { required: expectations.minlength, actual: modelResponse.length, passed: meetsLength };
            if (meetsLength) {
                checkPasses++;
            } else {
                failures.push(`Response too short: ${modelResponse.length} < ${expectations.minlength}`);
            }
        }

        // ── thinkingCheck ─────────────────────────────────────────────
        if (expectations.thinkingCheck && expectations.thinkingCheck.enabled) {
            const tc = expectations.thinkingCheck;
            checkCount++;

            const hasThinking = thinkingExtracted.length > 0;
            checks.thinkingCheck = { hasThinking };

            if (!hasThinking) {
                failures.push('No thinking/reasoning block extracted');
            } else {
                checkPasses++;

                // Check mustContain within thinking block
                if (tc.mustContain && Array.isArray(tc.mustContain)) {
                    for (const str of tc.mustContain) {
                        checkCount++;
                        const found = thinkingExtracted.includes(str);
                        checks.thinkingCheck[`mustContain_${str}`] = found;
                        if (found) {
                            checkPasses++;
                        } else {
                            failures.push(`Thinking block missing: "${str}"`);
                        }
                    }
                }
            }
        }

        // ── Calculate score ───────────────────────────────────────────
        const score = checkCount > 0 ? checkPasses / checkCount : 1.0;
        const passed = failures.length === 0;

        return {
            testCaseId: testCase.id,
            versionNum,
            runId,
            iteration,
            passed,
            score,
            details: { checks, failures },
            modelResponse,
            thinkingExtracted,
            usage,
            duration,
            timestamp: Date.now(),
        };
    }

    /**
     * Evaluate a model response against all enabled test cases.
     *
     * @param {string} modelResponse - The model's response text
     * @param {Object} [options] - Context options (passed to evaluateOne)
     * @returns {TestResult[]} Array of test results
     */
    evaluateAll(modelResponse, options = {}) {
        const enabledTests = this.getTestCases(true);
        return enabledTests.map(tc => this.evaluateOne(modelResponse, tc, options));
    }

    // ── Static Analysis Methods ───────────────────────────────────────────

    /**
     * Calculate a weighted summary from an array of test results.
     * Each result's score is weighted by its corresponding test case weight.
     *
     * @param {TestResult[]} results - Array of test results
     * @param {TestCase[]} [testCases] - Test cases for weight lookup (if not embedded)
     * @returns {{ overallScore: number, passedCount: number, failedCount: number, totalCount: number, weightedScore: number, categoryScores: Object }}
     */
    static getSummary(results, testCases = []) {
        if (!results || results.length === 0) {
            return {
                overallScore: 0,
                passedCount: 0,
                failedCount: 0,
                totalCount: 0,
                weightedScore: 0,
                categoryScores: {},
            };
        }

        // Build weight lookup from test cases
        const weightMap = new Map();
        const categoryMap = new Map();
        for (const tc of testCases) {
            weightMap.set(tc.id, tc.weight || 1.0);
            categoryMap.set(tc.id, tc.category || 'custom');
        }

        let totalWeight = 0;
        let weightedSum = 0;
        let passedCount = 0;
        let failedCount = 0;
        const categoryScores = {};
        const categoryWeights = {};

        for (const result of results) {
            const weight = weightMap.get(result.testCaseId) || 1.0;
            const category = categoryMap.get(result.testCaseId) || 'custom';

            weightedSum += result.score * weight;
            totalWeight += weight;

            if (result.passed) passedCount++;
            else failedCount++;

            // Accumulate category scores
            if (!categoryScores[category]) {
                categoryScores[category] = { totalScore: 0, totalWeight: 0, count: 0 };
            }
            categoryScores[category].totalScore += result.score * weight;
            categoryScores[category].totalWeight += weight;
            categoryScores[category].count++;
        }

        // Calculate category averages
        for (const [cat, data] of Object.entries(categoryScores)) {
            categoryScores[cat] = data.totalWeight > 0
                ? data.totalScore / data.totalWeight
                : 0;
        }

        const weightedScore = totalWeight > 0 ? weightedSum / totalWeight : 0;
        const overallScore = results.length > 0
            ? results.reduce((sum, r) => sum + r.score, 0) / results.length
            : 0;

        return {
            overallScore,
            passedCount,
            failedCount,
            totalCount: results.length,
            weightedScore,
            categoryScores,
        };
    }

    /**
     * Generate human-readable feedback from test results.
     *
     * @param {TestResult[]} results - Array of test results
     * @param {TestCase[]} [testCases] - Test cases for context
     * @returns {string} Human-readable feedback text
     */
    static generateFeedback(results, testCases = []) {
        if (!results || results.length === 0) {
            return 'No test results to evaluate.';
        }

        const summary = TestEvaluator.getSummary(results, testCases);
        const lines = [];

        lines.push(`Test Results: ${summary.passedCount}/${summary.totalCount} passed`);
        lines.push(`Overall Score: ${(summary.overallScore * 100).toFixed(1)}%`);

        if (summary.weightedScore !== summary.overallScore) {
            lines.push(`Weighted Score: ${(summary.weightedScore * 100).toFixed(1)}%`);
        }

        // Category breakdown
        const cats = Object.entries(summary.categoryScores);
        if (cats.length > 1) {
            lines.push('');
            lines.push('Category Breakdown:');
            for (const [cat, score] of cats) {
                lines.push(`  ${cat}: ${(score * 100).toFixed(1)}%`);
            }
        }

        // Failures
        const failures = results
            .filter(r => !r.passed)
            .flatMap(r => r.details?.failures || []);

        if (failures.length > 0) {
            lines.push('');
            lines.push(`Failures (${failures.length}):`);
            for (const f of failures) {
                lines.push(`  - ${f}`);
            }
        }

        // Suggestions
        if (summary.overallScore < 0.5) {
            lines.push('');
            lines.push('Suggestion: Significant issues detected. Consider reviewing the prompt structure and core logic.');
        } else if (summary.overallScore < 0.8) {
            lines.push('');
            lines.push('Suggestion: Some tests failing. Review failures above for targeted improvements.');
        } else if (summary.overallScore < 1.0) {
            lines.push('');
            lines.push('Suggestion: Close to passing. Minor refinements may resolve remaining failures.');
        }

        return lines.join('\n');
    }

    /**
     * Get the default UCL syntax test cases.
     * These test basic UCL formatting rules and are a good starting point.
     *
     * @returns {TestCase[]} Array of default syntax test cases
     */
    static getDefaultSyntaxTests() {
        return [
            {
                id: 'ucl_has_critical_directive',
                name: 'Has [[CRITICAL:]] Directive',
                category: 'syntax',
                input: 'Convert: "Make a chatbot that helps with math"',
                expectations: {
                    contains: ['[[CRITICAL:'],
                },
                weight: 1.0,
                enabled: true,
            },
            {
                id: 'ucl_has_condition_blocks',
                name: 'Has ^^CONDITION^^ Blocks',
                category: 'syntax',
                input: 'Convert: "If the user asks about pricing, show the pricing table"',
                expectations: {
                    contains: ['^^CONDITION'],
                },
                weight: 1.0,
                enabled: true,
            },
            {
                id: 'ucl_no_raw_json',
                name: 'No Raw JSON Output',
                category: 'syntax',
                input: 'Convert: "Build a tool that converts text to uppercase"',
                expectations: {
                    notContains: ['{', '}'],
                },
                weight: 0.5,
                enabled: true,
            },
            {
                id: 'ucl_min_length',
                name: 'Response Not Trivial',
                category: 'syntax',
                input: 'Convert: "Make a summarization tool"',
                expectations: {
                    minlength: 50,
                },
                weight: 0.5,
                enabled: true,
            },
            {
                id: 'ucl_has_close_condition',
                name: 'Conditions Are Closed',
                category: 'syntax',
                input: 'Convert: "When asked about weather, check the weather API"',
                expectations: {
                    contains: ['^^/CONDITION^^'],
                },
                weight: 1.0,
                enabled: true,
            },
        ];
    }
}

export { TestEvaluator };
