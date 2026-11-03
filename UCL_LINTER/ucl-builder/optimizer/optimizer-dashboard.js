/**
 * OptimizerDashboard - Dashboard UI for the Optimizer Tab
 *
 * Renders a comprehensive dashboard in the editor area when the optimizer
 * tab is active. Contains 8 panels:
 *   1. Quality Score Gauges (4 dimensions)
 *   2. Composite Score + Verdict
 *   3. Score Chart (SVG line chart)
 *   4. Test Results List
 *   5. Version History with Scores
 *   6. Realtime Progress Grid
 *   7. Thinking Log Viewer (accordion)
 *   8. Optimization History (cross-run trends)
 *
 * @module optimizer-dashboard
 */

// ── Dashboard State ──────────────────────────────────────────────────────────

/**
 * Escape HTML special characters to prevent XSS when injecting text
 * into innerHTML template literals.
 * @param {*} str - Value to escape (converted to string)
 * @returns {string} Safe HTML string
 */
function escapeHtml(str) {
    if (!str) return '';
    return String(str)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
}

const DashboardState = {
    initialized: false,
    iterations: [],
    currentRun: null,
    chartPoints: [],
    bestScore: 0,
    bestVersionNum: null,
    runStartTime: null,
    tokensUsed: 0,
    thinkingLogs: [],
    qualityThreshold: 0.85,  // Synced from OptimizerState.iterationConfig.qualityThreshold
};

// ── Dashboard HTML ──────────────────────────────────────────────────────────

function buildDashboardHTML() {
    return `
        <div class="dashboard-header">
            <h2 class="dashboard-title">Optimization Dashboard</h2>
            <div class="dashboard-status">
                <div class="dashboard-status-indicator" id="dashStatusIndicator" aria-live="polite">
                    <span id="dashStatusIcon">&#x26AA;</span>
                    <span id="dashStatusText">Idle</span>
                </div>
                <button class="optimizer-btn optimizer-btn-start" id="dashStartBtn" aria-label="Start optimization loop">
                    &#x25B6; Start Optimization
                </button>
                <button class="optimizer-btn optimizer-btn-stop" id="dashStopBtn" style="display:none;" aria-label="Stop optimization loop">
                    &#x23F9; Stop
                </button>
            </div>
        </div>

        <div class="dashboard-grid">
            <!-- Panel 1: Quality Score Gauges -->
            <div class="dashboard-panel" id="dashQualityPanel">
                <div class="dashboard-panel-title">
                    &#x1F4CA; Quality Scores
                </div>
                <div class="quality-gauges" id="dashQualityGauges" role="group" aria-label="Quality score gauges">
                    <div class="quality-gauge gauge-syntax">
                        <div class="gauge-container">
                            <svg class="gauge-svg" viewBox="0 0 100 100">
                                <circle class="gauge-bg" cx="50" cy="50" r="42"/>
                                <circle class="gauge-fill" cx="50" cy="50" r="42"
                                    stroke-dasharray="263.89" stroke-dashoffset="263.89"/>
                            </svg>
                            <div class="gauge-value" id="gaugeSyntaxVal">--</div>
                        </div>
                        <div class="gauge-label">Syntax</div>
                    </div>
                    <div class="quality-gauge gauge-testPassRate">
                        <div class="gauge-container">
                            <svg class="gauge-svg" viewBox="0 0 100 100">
                                <circle class="gauge-bg" cx="50" cy="50" r="42"/>
                                <circle class="gauge-fill" cx="50" cy="50" r="42"
                                    stroke-dasharray="263.89" stroke-dashoffset="263.89"/>
                            </svg>
                            <div class="gauge-value" id="gaugeTestVal">--</div>
                        </div>
                        <div class="gauge-label">Test Pass Rate</div>
                    </div>
                    <div class="quality-gauge gauge-reasoningDepth">
                        <div class="gauge-container">
                            <svg class="gauge-svg" viewBox="0 0 100 100">
                                <circle class="gauge-bg" cx="50" cy="50" r="42"/>
                                <circle class="gauge-fill" cx="50" cy="50" r="42"
                                    stroke-dasharray="263.89" stroke-dashoffset="263.89"/>
                            </svg>
                            <div class="gauge-value" id="gaugeReasonVal">--</div>
                        </div>
                        <div class="gauge-label">Reasoning Depth</div>
                    </div>
                    <div class="quality-gauge gauge-conciseness">
                        <div class="gauge-container">
                            <svg class="gauge-svg" viewBox="0 0 100 100">
                                <circle class="gauge-bg" cx="50" cy="50" r="42"/>
                                <circle class="gauge-fill" cx="50" cy="50" r="42"
                                    stroke-dasharray="263.89" stroke-dashoffset="263.89"/>
                            </svg>
                            <div class="gauge-value" id="gaugeConciseVal">--</div>
                        </div>
                        <div class="gauge-label">Conciseness</div>
                    </div>
                </div>
                <div class="composite-score">
                    <div class="composite-score-value" id="dashCompositeScore" aria-live="polite">--</div>
                    <div class="composite-score-label">Composite Score</div>
                    <div class="composite-score-bar">
                        <div class="composite-score-fill" id="dashCompositeFill" style="width: 0%"></div>
                    </div>
                    <div style="text-align: center;">
                        <span class="verdict-badge" id="dashVerdict" style="display:none;"></span>
                    </div>
                </div>
            </div>

            <!-- Panel 2: Score Chart -->
            <div class="dashboard-panel" id="dashChartPanel">
                <div class="dashboard-panel-title">
                    &#x1F4C8; Score Progression
                </div>
                <div class="score-chart" id="dashScoreChart" role="img" aria-label="Quality score over iterations chart">
                    <svg viewBox="0 0 400 200" preserveAspectRatio="xMidYMid meet" aria-hidden="true">
                        <defs>
                            <linearGradient id="scoreGradient" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="0%" stop-color="var(--accent-primary)" stop-opacity="0.3"/>
                                <stop offset="100%" stop-color="var(--accent-primary)" stop-opacity="0"/>
                            </linearGradient>
                        </defs>
                        <!-- Grid lines -->
                        <line class="chart-grid-line" x1="40" y1="20" x2="390" y2="20"/>
                        <line class="chart-grid-line" x1="40" y1="65" x2="390" y2="65"/>
                        <line class="chart-grid-line" x1="40" y1="110" x2="390" y2="110"/>
                        <line class="chart-grid-line" x1="40" y1="155" x2="390" y2="155"/>
                        <!-- Y-axis labels -->
                        <text class="chart-label" x="30" y="24" text-anchor="end">1.0</text>
                        <text class="chart-label" x="30" y="69" text-anchor="end">0.75</text>
                        <text class="chart-label" x="30" y="114" text-anchor="end">0.5</text>
                        <text class="chart-label" x="30" y="159" text-anchor="end">0.25</text>
                        <!-- Threshold line -->
                        <line class="chart-threshold-line" id="dashThresholdLine"
                            x1="40" y1="32" x2="390" y2="32"/>
                        <!-- Area fill -->
                        <path class="chart-area" id="dashChartArea" d=""/>
                        <!-- Line -->
                        <polyline class="chart-line" id="dashChartLine" points=""/>
                        <!-- Dots rendered dynamically -->
                        <g id="dashChartDots"></g>
                    </svg>
                </div>
                <div class="convergence-indicator" id="dashConvergence" style="display:none;">
                    <span class="convergence-icon" id="dashConvergenceIcon">&#x2705;</span>
                    <span class="convergence-text" id="dashConvergenceText">Converged</span>
                </div>
            </div>

            <!-- Panel 3: Test Results -->
            <div class="dashboard-panel" id="dashTestPanel">
                <div class="dashboard-panel-title">
                    &#x1F9EA; Test Results
                </div>
                <div class="test-results-list" id="dashTestResults">
                    <div class="optimizer-empty">No test results yet</div>
                </div>
            </div>

            <!-- Panel 4: Realtime Progress -->
            <div class="dashboard-panel" id="dashProgressPanel">
                <div class="dashboard-panel-title">
                    &#x26A1; Progress
                </div>
                <div class="progress-grid" id="dashProgressGrid" aria-live="polite">
                    <div class="progress-item">
                        <div class="progress-item-label">Iteration</div>
                        <div class="progress-item-value" id="dashIteration">0 / 0</div>
                    </div>
                    <div class="progress-item">
                        <div class="progress-item-label">Status</div>
                        <div class="progress-item-value" id="dashRunStatus">Idle</div>
                    </div>
                    <div class="progress-item">
                        <div class="progress-item-label">Tokens Used</div>
                        <div class="progress-item-value" id="dashTokensUsed">0</div>
                    </div>
                    <div class="progress-item">
                        <div class="progress-item-label">Best Score</div>
                        <div class="progress-item-value highlight" id="dashBestScore">--</div>
                    </div>
                    <div class="progress-item">
                        <div class="progress-item-label">Duration</div>
                        <div class="progress-item-value" id="dashDuration">0s</div>
                    </div>
                    <div class="progress-item">
                        <div class="progress-item-label">Stop Reason</div>
                        <div class="progress-item-value" id="dashStopReason">--</div>
                    </div>
                </div>
            </div>

            <!-- Panel 5: Version History -->
            <div class="dashboard-panel" id="dashVersionPanel">
                <div class="dashboard-panel-title">
                    &#x1F4DD; Version History
                </div>
                <div class="version-list" id="dashVersionList">
                    <div class="optimizer-empty">No versions yet</div>
                </div>
            </div>

            <!-- Panel 6: Thinking Log -->
            <div class="dashboard-panel" id="dashThinkingPanel">
                <div class="dashboard-panel-title">
                    &#x1F4AD; Thinking Log
                </div>
                <div class="thinking-accordion" id="dashThinkingLog">
                    <div class="optimizer-empty">No thinking logs yet</div>
                </div>
            </div>

            <!-- Panel 7: Optimization History -->
            <div class="dashboard-panel" id="dashHistoryPanel">
                <div class="dashboard-panel-title">
                    &#x1F4C0; Run History
                </div>
                <div id="dashOptimHistory">
                    <div class="optimizer-empty">No optimization runs yet</div>
                </div>
            </div>

            <!-- Panel 8: Test Case Management -->
            <div class="dashboard-panel" id="dashTestCasePanel">
                <div class="dashboard-panel-title">
                    &#x1F4CB; Test Cases
                </div>
                <div class="test-case-list" id="dashTestCases">
                    <div class="optimizer-empty">No test cases configured</div>
                </div>
                <div style="margin-top: var(--gap-md); display: flex; gap: var(--gap-sm);">
                    <button class="optimizer-btn optimizer-btn-sm" id="dashAddTestCaseBtn">+ Add Test</button>
                    <button class="optimizer-btn optimizer-btn-sm" id="dashExportRunBtn">Export Run</button>
                    <button class="optimizer-btn optimizer-btn-sm" id="dashImportRunBtn">Import Run</button>
                </div>
            </div>
        </div>

        <!-- Scoring Transparency Panel -->
        <div class="scoring-info-panel" id="dashScoringPanel">
            <button class="scoring-info-toggle" id="dashScoringToggle" aria-expanded="false">
                <span>❓ How Scoring Works</span>
                <span class="scoring-toggle-arrow" id="dashScoringArrow">▼</span>
            </button>
            <div class="scoring-info-content" id="dashScoringContent" style="display:none;">
                <div class="scoring-info-intro">
                    Quality is scored across <strong>4 dimensions</strong> with a weighted composite:
                </div>
                <div class="scoring-dimensions">
                    <div class="scoring-dimension">
                        <div class="scoring-dim-header">
                            <span class="scoring-dim-name">🧪 Test Pass Rate</span>
                            <span class="scoring-dim-weight">40%</span>
                        </div>
                        <div class="scoring-dim-desc">
                            Weighted average of test case results. Each test has an expectation type (contains, notContains, minlength, thinkingCheck) and a weight. Score = Σ(pass × weight) / Σ(weight).
                        </div>
                    </div>
                    <div class="scoring-dimension">
                        <div class="scoring-dim-header">
                            <span class="scoring-dim-name">⚡ Syntax</span>
                            <span class="scoring-dim-weight">25%</span>
                        </div>
                        <div class="scoring-dim-desc">
                            Checks UCL structural correctness:<br>
                            • <code>[[CRITICAL:]]</code> present (+high bonus)<br>
                            • <code>^^CONDITION^^</code> / <code>^^/CONDITION^^</code> balanced<br>
                            • No orphan tags or broken nesting<br>
                            • Bonus for <code>[[PARAM:]]</code>, <code>[[TOOL:]]</code>, other directives
                        </div>
                    </div>
                    <div class="scoring-dimension">
                        <div class="scoring-dim-header">
                            <span class="scoring-dim-name">💭 Reasoning Depth</span>
                            <span class="scoring-dim-weight">20%</span>
                        </div>
                        <div class="scoring-dim-desc">
                            Measures depth of chain-of-thought reasoning: length-based scoring with a structure bonus for numbered steps, bullet points, or logical connectors (therefore, because, if-then).
                        </div>
                    </div>
                    <div class="scoring-dimension">
                        <div class="scoring-dim-header">
                            <span class="scoring-dim-name">✂️ Conciseness</span>
                            <span class="scoring-dim-weight">15%</span>
                        </div>
                        <div class="scoring-dim-desc">
                            Penalizes bloat. Optimal prompt length is model-dependent. Shorter prompts that achieve the same test pass rate score higher. Bloat ratio = actual_length / ideal_length.
                        </div>
                    </div>
                </div>
                <div class="scoring-verdicts">
                    <strong>Verdicts:</strong>
                    <span class="verdict-excellent">★ Excellent (≥0.90)</span> ·
                    <span class="verdict-good">● Good (≥0.75)</span> ·
                    <span class="verdict-needswork">▲ Needs Work (≥0.50)</span> ·
                    <span class="verdict-poor">▼ Poor (&lt;0.50)</span>
                </div>
            </div>
        </div>
    `;
}

// ── Initialization ──────────────────────────────────────────────────────────

function initOptimizerDashboard(container) {
    if (DashboardState.initialized && container.children.length > 0) return;

    container.innerHTML = buildDashboardHTML();
    attachDashboardListeners();
    DashboardState.initialized = true;
    console.log('[Dashboard] Initialized');
}

function destroyOptimizerDashboard() {
    DashboardState.initialized = false;
    DashboardState.iterations = [];
    DashboardState.currentRun = null;
    DashboardState.chartPoints = [];
    DashboardState.bestScore = 0;
    DashboardState.bestVersionNum = null;
    DashboardState.runStartTime = null;
    DashboardState.tokensUsed = 0;
    DashboardState.thinkingLogs = [];
}

function attachDashboardListeners() {
    const startBtn = document.getElementById('dashStartBtn');
    const stopBtn = document.getElementById('dashStopBtn');

    if (startBtn) {
        startBtn.addEventListener('click', () => {
            if (typeof onStartOptimization === 'function') {
                onStartOptimization();
            }
        });
    }

    if (stopBtn) {
        stopBtn.addEventListener('click', () => {
            if (typeof onStopOptimization === 'function') {
                onStopOptimization();
            }
        });
    }

    const exportBtn = document.getElementById('dashExportRunBtn');
    if (exportBtn) {
        exportBtn.addEventListener('click', () => onExportRun());
    }

    const importBtn = document.getElementById('dashImportRunBtn');
    if (importBtn) {
        importBtn.addEventListener('click', () => onImportRun());
    }

    const addTestBtn = document.getElementById('dashAddTestCaseBtn');
    if (addTestBtn) {
        addTestBtn.addEventListener('click', () => onAddTestCase());
    }

    // Scoring info toggle
    const scoringToggle = document.getElementById('dashScoringToggle');
    const scoringContent = document.getElementById('dashScoringContent');
    const scoringArrow = document.getElementById('dashScoringArrow');
    if (scoringToggle && scoringContent) {
        scoringToggle.addEventListener('click', () => {
            const expanded = scoringContent.style.display !== 'none';
            scoringContent.style.display = expanded ? 'none' : '';
            scoringToggle.setAttribute('aria-expanded', !expanded);
            if (scoringArrow) scoringArrow.style.transform = expanded ? '' : 'rotate(180deg)';
        });
    }
}

// ── Update Functions (called from optimizer-ui.js callbacks) ────────────────

/**
 * Update dashboard after each iteration completes.
 * @param {Object} result - Iteration result from orchestrator
 */
function updateDashboardIteration(result) {
    if (!result) return;

    DashboardState.iterations.push(result);

    // Update gauges
    if (result.qualityScore) {
        const { breakdown, composite, verdict } = result.qualityScore;
        updateGauge('gaugeSyntaxVal', breakdown.syntaxScore);
        updateGauge('gaugeTestVal', breakdown.testPassRate);
        updateGauge('gaugeReasonVal', breakdown.reasoningDepth);
        updateGauge('gaugeConciseVal', breakdown.conciseness);

        // Composite
        const compositeEl = document.getElementById('dashCompositeScore');
        if (compositeEl) compositeEl.textContent = (composite * 100).toFixed(1) + '%';

        const fillEl = document.getElementById('dashCompositeFill');
        if (fillEl) fillEl.style.width = (composite * 100) + '%';

        // Verdict
        const verdictEl = document.getElementById('dashVerdict');
        if (verdictEl) {
            verdictEl.textContent = verdict.replace('_', ' ');
            verdictEl.className = 'verdict-badge verdict-' + verdict;
            verdictEl.style.display = '';
        }

        // Track best
        if (composite > DashboardState.bestScore) {
            DashboardState.bestScore = composite;
            DashboardState.bestVersionNum = result.versionNum;
        }
    }

    // Update chart
    DashboardState.chartPoints.push({
        iteration: result.iterationNum,
        score: result.qualityScore?.composite || 0,
    });
    renderScoreChart();

    // Update test results
    if (result.testResults) {
        renderTestResults(result.testResults);
    }

    // Update progress
    updateProgressGrid(result);

    // Update thinking log
    if (result.thinkingExtracted) {
        DashboardState.thinkingLogs.push({
            iteration: result.iterationNum,
            thinking: result.thinkingExtracted,
        });
        renderThinkingLog();
    }

    // Update version history
    renderVersionHistory();

    // Update status
    updateDashboardStatus('running', `Iteration ${result.iterationNum} complete`);
}

/**
 * Update dashboard when optimization loop completes.
 * @param {Object} result - Optimization result from orchestrator
 */
function updateDashboardComplete(result) {
    if (!result) return;

    DashboardState.currentRun = result;
    updateDashboardStatus('complete', `Complete: ${result.stopReason}`);

    // Update stop reason
    const stopReasonEl = document.getElementById('dashStopReason');
    if (stopReasonEl) stopReasonEl.textContent = result.stopReason || '--';

    // Show convergence indicator
    if (result.stopReason === 'converged' || result.stopReason === 'quality_threshold') {
        const convEl = document.getElementById('dashConvergence');
        if (convEl) {
            convEl.style.display = '';
            convEl.classList.add('converged');
        }
        const convText = document.getElementById('dashConvergenceText');
        if (convText) convText.textContent = `Converged: ${result.stopReason}`;
    }

    // Show/hide buttons
    const startBtn = document.getElementById('dashStartBtn');
    const stopBtn = document.getElementById('dashStopBtn');
    if (startBtn) startBtn.style.display = '';
    if (stopBtn) stopBtn.style.display = 'none';

    // Update optimization history panel
    renderOptimizationHistory(result);
}

/**
 * Update dashboard status indicator.
 * @param {string} status - 'idle' | 'running' | 'complete' | 'error'
 * @param {string} text - Status text
 */
function updateDashboardStatus(status, text) {
    const iconMap = {
        idle: '\u26AA',
        running: '\uD83D\uDFE2',
        complete: '\u2705',
        error: '\u274C',
    };

    const iconEl = document.getElementById('dashStatusIcon');
    const textEl = document.getElementById('dashStatusText');

    if (iconEl) iconEl.textContent = iconMap[status] || '\u26AA';
    if (textEl) textEl.textContent = text || status;

    // Update run status in progress grid
    const runStatusEl = document.getElementById('dashRunStatus');
    if (runStatusEl) runStatusEl.textContent = text || status;
}

// ── Render Helpers ──────────────────────────────────────────────────────────

function updateGauge(elementId, value) {
    const el = document.getElementById(elementId);
    if (!el) return;

    const pct = Math.round(value * 100);
    el.textContent = pct + '%';

    // Update ARIA label on parent gauge element for screen readers
    const gaugeEl = el.closest('.quality-gauge');
    if (gaugeEl) {
        const label = gaugeEl.querySelector('.gauge-label');
        const labelText = label ? label.textContent : elementId;
        gaugeEl.setAttribute('aria-label', `${labelText}: ${pct}%`);
        gaugeEl.setAttribute('role', 'meter');
        gaugeEl.setAttribute('aria-valuenow', pct);
        gaugeEl.setAttribute('aria-valuemin', 0);
        gaugeEl.setAttribute('aria-valuemax', 100);
    }

    // Update the SVG gauge fill
    const gaugeContainer = el.closest('.gauge-container');
    if (gaugeContainer) {
        const fill = gaugeContainer.querySelector('.gauge-fill');
        if (fill) {
            const circumference = 2 * Math.PI * 42; // r=42
            const offset = circumference * (1 - value);
            fill.setAttribute('stroke-dashoffset', offset);
        }
    }
}

function renderScoreChart() {
    const points = DashboardState.chartPoints;
    if (points.length === 0) return;

    const lineEl = document.getElementById('dashChartLine');
    const areaEl = document.getElementById('dashChartArea');
    const dotsEl = document.getElementById('dashChartDots');
    const thresholdEl = document.getElementById('dashThresholdLine');
    if (!lineEl || !areaEl || !dotsEl) return;

    // Chart dimensions
    const xStart = 50, xEnd = 380;
    const yStart = 20, yEnd = 180;
    const xRange = xEnd - xStart;
    const yRange = yEnd - yStart;

    // Update threshold line position (read from DashboardState, synced from OptimizerState)
    if (thresholdEl) {
        const threshold = DashboardState.qualityThreshold || 0.85;
        const threshY = yEnd - (threshold * yRange);
        thresholdEl.setAttribute('y1', threshY);
        thresholdEl.setAttribute('y2', threshY);
    }

    // Compute points
    const maxIter = Math.max(points.length, 5);
    const svgPoints = points.map((p, i) => {
        const x = xStart + (i / Math.max(maxIter - 1, 1)) * xRange;
        const y = yEnd - (p.score * yRange);
        return { x, y, score: p.score, iteration: p.iteration };
    });

    // Polyline
    const pointsStr = svgPoints.map(p => `${p.x},${p.y}`).join(' ');
    lineEl.setAttribute('points', pointsStr);

    // Area fill
    if (svgPoints.length > 1) {
        const areaStr = `M ${svgPoints[0].x},${yEnd} ` +
            svgPoints.map(p => `L ${p.x},${p.y}`).join(' ') +
            ` L ${svgPoints[svgPoints.length - 1].x},${yEnd} Z`;
        areaEl.setAttribute('d', areaStr);
    }

    // Dots
    dotsEl.innerHTML = svgPoints.map(p =>
        `<circle class="chart-dot" cx="${p.x}" cy="${p.y}" r="4">
            <title>Iteration ${p.iteration}: ${(p.score * 100).toFixed(1)}%</title>
        </circle>`
    ).join('');
}

function renderTestResults(testResults) {
    const container = document.getElementById('dashTestResults');
    if (!container) return;

    if (!testResults || testResults.length === 0) {
        container.innerHTML = '<div class="optimizer-empty">No test results</div>';
        return;
    }

    container.innerHTML = testResults.map(r => {
        const pass = r.score >= 0.5;
        const icon = pass ? '\u2705' : '\u274C';
        const iconClass = pass ? 'pass' : 'fail';
        const name = r.testCaseId || r.testName || 'Test';
        const score = ((r.score || 0) * 100).toFixed(1) + '%';
        const detail = r.detail || r.message || '';

        return `
            <div class="test-result-item">
                <span class="test-result-icon ${iconClass}">${icon}</span>
                <div class="test-result-content">
                    <div class="test-result-name">${escapeHtml(name)}</div>
                    ${detail ? `<div class="test-result-detail">${escapeHtml(detail)}</div>` : ''}
                </div>
                <span class="test-result-score">${escapeHtml(score)}</span>
            </div>
        `;
    }).join('');
}

function renderVersionHistory() {
    const container = document.getElementById('dashVersionList');
    if (!container) return;

    const iterations = DashboardState.iterations;
    if (iterations.length === 0) {
        container.innerHTML = '<div class="optimizer-empty">No versions yet</div>';
        return;
    }

    container.innerHTML = [...iterations].reverse().map(iter => {
        const score = iter.qualityScore?.composite || 0;
        const isBest = iter.versionNum === DashboardState.bestVersionNum;
        return `
            <div class="version-item ${isBest ? 'best' : ''}"
                 role="button" tabindex="0"
                 data-version-num="${iter.versionNum}"
                 aria-label="Version ${iter.versionNum}, score ${(score * 100).toFixed(1)}%${isBest ? ', best version' : ''}">
                <span class="version-item-num">v${iter.versionNum}</span>
                <span class="version-item-score">${(score * 100).toFixed(1)}%</span>
                ${isBest ? '<span class="version-item-badge">BEST</span>' : ''}
            </div>
        `;
    }).join('');

    // Attach keyboard handlers for Enter/Space on version items
    container.querySelectorAll('.version-item[role="button"]').forEach(item => {
        item.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                item.click();
            }
        });
    });
}

function renderThinkingLog() {
    const container = document.getElementById('dashThinkingLog');
    if (!container) return;

    const logs = DashboardState.thinkingLogs;
    if (logs.length === 0) {
        container.innerHTML = '<div class="optimizer-empty">No thinking logs yet</div>';
        return;
    }

    container.innerHTML = logs.map((log, i) => `
        <div class="thinking-accordion-item">
            <div class="thinking-accordion-header" data-thinking-idx="${i}" role="button" tabindex="0" aria-expanded="false">
                <span class="thinking-accordion-toggle">&#x25B6;</span>
                <span class="thinking-accordion-title">Iteration ${escapeHtml(log.iteration)}</span>
            </div>
            <div class="thinking-accordion-content" id="thinkingContent${i}" style="display:none;">
                ${escapeHtml(log.thinking) || '<em>No thinking text extracted</em>'}
            </div>
        </div>
    `).join('');

    // Attach toggle listeners (click + keyboard)
    container.querySelectorAll('.thinking-accordion-header').forEach(header => {
        const toggleAccordion = () => {
            const idx = header.dataset.thinkingIdx;
            const content = document.getElementById('thinkingContent' + idx);
            if (content) {
                const isHidden = content.style.display === 'none';
                content.style.display = isHidden ? '' : 'none';
                header.classList.toggle('expanded', isHidden);
                header.setAttribute('aria-expanded', String(isHidden));
            }
        };
        header.addEventListener('click', toggleAccordion);
        header.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                toggleAccordion();
            }
        });
    });
}

function updateProgressGrid(latestResult) {
    const iterEl = document.getElementById('dashIteration');
    const tokensEl = document.getElementById('dashTokensUsed');
    const bestEl = document.getElementById('dashBestScore');
    const durationEl = document.getElementById('dashDuration');

    if (iterEl) {
        const total = latestResult.maxIterations || DashboardState.iterations.length;
        iterEl.textContent = `${DashboardState.iterations.length} / ${total}`;
    }

    if (latestResult.usage) {
        DashboardState.tokensUsed += latestResult.usage.total_tokens || latestResult.usage.totalTokens || 0;
    }
    if (tokensEl) tokensEl.textContent = DashboardState.tokensUsed.toLocaleString();

    if (bestEl) {
        bestEl.textContent = DashboardState.bestScore > 0
            ? (DashboardState.bestScore * 100).toFixed(1) + '%'
            : '--';
    }

    if (durationEl && DashboardState.runStartTime) {
        const elapsed = Math.floor((Date.now() - DashboardState.runStartTime) / 1000);
        if (elapsed < 60) {
            durationEl.textContent = elapsed + 's';
        } else {
            durationEl.textContent = Math.floor(elapsed / 60) + 'm ' + (elapsed % 60) + 's';
        }
    }
}

function renderOptimizationHistory(runResult) {
    const container = document.getElementById('dashOptimHistory');
    if (!container) return;

    if (!runResult) {
        container.innerHTML = '<div class="optimizer-empty">No optimization runs yet</div>';
        return;
    }

    const totalIter = runResult.totalIterations || 0;
    const bestScore = runResult.bestQualityScore || 0;
    const stopReason = runResult.stopReason || '--';

    container.innerHTML = `
        <div class="progress-grid">
            <div class="progress-item">
                <div class="progress-item-label">Total Iterations</div>
                <div class="progress-item-value">${totalIter}</div>
            </div>
            <div class="progress-item">
                <div class="progress-item-label">Best Score</div>
                <div class="progress-item-value highlight">${(bestScore * 100).toFixed(1)}%</div>
            </div>
            <div class="progress-item">
                <div class="progress-item-label">Stop Reason</div>
                <div class="progress-item-value">${stopReason}</div>
            </div>
            <div class="progress-item">
                <div class="progress-item-label">Best Version</div>
                <div class="progress-item-value">v${runResult.bestVersionNum || '--'}</div>
            </div>
        </div>
    `;
}

// ── Start/Stop Handlers ─────────────────────────────────────────────────────

function showRunningState(isRunning) {
    const startBtn = document.getElementById('dashStartBtn');
    const stopBtn = document.getElementById('dashStopBtn');

    if (startBtn) startBtn.style.display = isRunning ? 'none' : '';
    if (stopBtn) stopBtn.style.display = isRunning ? '' : 'none';

    if (isRunning) {
        DashboardState.runStartTime = Date.now();
        DashboardState.tokensUsed = 0;
    }
}

// ── Test Case Management ────────────────────────────────────────────────────

function onAddTestCase() {
    const name = prompt('Test case name:');
    if (!name?.trim()) return;

    const description = prompt('Test description / expected behavior:') || '';

    const container = document.getElementById('dashTestCases');
    if (!container) return;

    // Remove empty placeholder
    const empty = container.querySelector('.optimizer-empty');
    if (empty) empty.remove();

    const id = 'tc_' + Date.now();
    const item = document.createElement('div');
    item.className = 'test-case-item';
    item.dataset.testCaseId = id;
    item.innerHTML = `
        <input type="checkbox" checked data-test-id="${escapeHtml(id)}">
        <div class="test-case-item-content">
            <div class="test-case-item-name">${escapeHtml(name)}</div>
            <div class="test-case-item-desc">${escapeHtml(description)}</div>
        </div>
        <div class="test-case-item-actions">
            <button class="optimizer-btn optimizer-btn-sm optimizer-btn-danger" data-remove-test="${escapeHtml(id)}">&times;</button>
        </div>
    `;
    container.appendChild(item);

    // Remove handler
    item.querySelector('[data-remove-test]').addEventListener('click', () => {
        item.remove();
        if (container.children.length === 0) {
            container.innerHTML = '<div class="optimizer-empty">No test cases configured</div>';
        }
    });
}

// ── Export/Import ────────────────────────────────────────────────────────────

function onExportRun() {
    const data = {
        run: DashboardState.currentRun,
        iterations: DashboardState.iterations,
        chartPoints: DashboardState.chartPoints,
        bestScore: DashboardState.bestScore,
        bestVersionNum: DashboardState.bestVersionNum,
        exportedAt: new Date().toISOString(),
    };

    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `optimization-run-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);

    if (typeof showToast === 'function') showToast('Run exported', 'success');
}

function onImportRun() {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = '.json';
    input.addEventListener('change', async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        try {
            const text = await file.text();
            const data = JSON.parse(text);

            if (data.iterations) {
                DashboardState.iterations = data.iterations;
                DashboardState.chartPoints = data.chartPoints || [];
                DashboardState.bestScore = data.bestScore || 0;
                DashboardState.bestVersionNum = data.bestVersionNum || null;
                DashboardState.currentRun = data.run || null;

                // Re-render everything WITHOUT calling updateDashboardIteration()
                // which would push() the last item again and duplicate it.
                if (DashboardState.iterations.length > 0) {
                    const last = DashboardState.iterations[DashboardState.iterations.length - 1];

                    // Update gauges directly (inline, no state mutation)
                    if (last.qualityScore) {
                        const { breakdown, composite, verdict } = last.qualityScore;
                        updateGauge('gaugeSyntaxVal', breakdown.syntaxScore);
                        updateGauge('gaugeTestVal', breakdown.testPassRate);
                        updateGauge('gaugeReasonVal', breakdown.reasoningDepth);
                        updateGauge('gaugeConciseVal', breakdown.conciseness);

                        const compositeEl = document.getElementById('dashCompositeScore');
                        if (compositeEl) compositeEl.textContent = (composite * 100).toFixed(1) + '%';

                        const fillEl = document.getElementById('dashCompositeFill');
                        if (fillEl) fillEl.style.width = (composite * 100) + '%';

                        const verdictEl = document.getElementById('dashVerdict');
                        if (verdictEl) {
                            verdictEl.textContent = verdict.replace('_', ' ');
                            verdictEl.className = 'verdict-badge verdict-' + verdict;
                            verdictEl.style.display = '';
                        }
                    }

                    // Render chart (reads from DashboardState.chartPoints — already set)
                    renderScoreChart();

                    // Render test results
                    if (last.testResults) {
                        renderTestResults(last.testResults);
                    }

                    // Update progress grid
                    updateProgressGrid(last);

                    // Render thinking logs (rebuild from full state)
                    renderThinkingLog();

                    // Render version history (reads from DashboardState.iterations — already set)
                    renderVersionHistory();

                    // Update status
                    updateDashboardStatus('complete', 'Imported run loaded');
                }
                if (DashboardState.currentRun) {
                    updateDashboardComplete(DashboardState.currentRun);
                }

                if (typeof showToast === 'function') showToast('Run imported', 'success');
            }
        } catch (err) {
            if (typeof showToast === 'function') showToast('Import failed: ' + err.message, 'error');
        }
    });
    input.click();
}

// ── Exports ──────────────────────────────────────────────────────────────────
export {
    DashboardState,
    initOptimizerDashboard,
    destroyOptimizerDashboard,
    updateDashboardIteration,
    updateDashboardComplete,
    updateDashboardStatus,
    showRunningState,
    renderScoreChart,
    renderTestResults,
    renderVersionHistory,
    renderThinkingLog,
    updateProgressGrid,
};
