/**
 * UCL Pipeline Orchestrator - State Machine
 *
 * Manages the 3-phase UCL Preparation Pipeline:
 * 1. Source Discovery: Parse source files to discover tools
 * 2. Context Preloading: Execute 8-message warm-up sequence
 * 3. UCL Generation: Generate UCL from test prompt with warm-up context
 *
 * Architecture:
 * - State machine with 5 states: IDLE, DISCOVERING, PRELOADING, GENERATING, COMPLETE
 * - Event-driven transitions
 * - Error recovery and retry logic
 * - Progress tracking with callbacks
 * - Integration with SourceParser and WarmUpEngine
 *
 * Usage:
 *   const orchestrator = new PipelineOrchestrator({
 *     sourceDir: './src/tools',
 *     endpoint: 'http://localhost:13305',
 *     model: 'qwen3-4b-FLM',
 *     onStateChange: (state) => console.log(`State: ${state}`)
 *   });
 *   const result = await orchestrator.execute('Build a file management assistant');
 */

import { SourceParser } from './source-parser.js';
import { WarmUpEngine } from './warm-up-engine.js';

// Pipeline states
const STATES = {
    IDLE: 'IDLE',
    DISCOVERING: 'DISCOVERING',
    PRELOADING: 'PRELOADING',
    GENERATING: 'GENERATING',
    COMPLETE: 'COMPLETE',
    ERROR: 'ERROR',
};

// Pipeline events
const EVENTS = {
    START: 'START',
    DISCOVERY_COMPLETE: 'DISCOVERY_COMPLETE',
    PRELOADING_COMPLETE: 'PRELOADING_COMPLETE',
    GENERATION_COMPLETE: 'GENERATION_COMPLETE',
    ERROR: 'ERROR',
    RESET: 'RESET',
};

class PipelineOrchestrator {
    constructor(config = {}) {
        this.sourceDir = config.sourceDir || './tools';
        this.endpoint = config.endpoint || 'http://localhost:13305';
        this.model = config.model || 'qwen3-4b-FLM';
        this.apiKey = config.apiKey || '';

        // Callbacks
        this.onStateChange = config.onStateChange || (() => {});
        this.onProgress = config.onProgress || (() => {});
        this.onError = config.onError || (() => {});

        // State
        this.currentState = STATES.IDLE;
        this.discoveredTools = [];
        this.warmUpResult = null;
        this.generationResult = null;
        this.error = null;

        // Components
        this.sourceParser = new SourceParser();
        this.warmUpEngine = new WarmUpEngine({
            endpoint: this.endpoint,
            model: this.model,
            apiKey: this.apiKey,
            onProgress: (step, total, title) => {
                this.onProgress('PRELOADING', step, total, title);
            },
        });
    }

    /**
     * Execute the full pipeline
     * @param {string} testPrompt - Natural language prompt to convert to UCL
     * @returns {Promise<Object>} - Pipeline result with UCL output
     */
    async execute(testPrompt) {
        console.log('🚀 Starting UCL Preparation Pipeline...');

        try {
            // Phase 1: Source Discovery
            await this.transition(EVENTS.START);
            await this.executeSourceDiscovery();

            // Phase 2: Context Preloading
            await this.transition(EVENTS.DISCOVERY_COMPLETE);
            await this.executeContextPreloading();

            // Phase 3: UCL Generation
            await this.transition(EVENTS.PRELOADING_COMPLETE);
            await this.executeUCLGeneration(testPrompt);

            // Complete
            await this.transition(EVENTS.GENERATION_COMPLETE);

            console.log('✅ Pipeline execution complete!');

            return {
                success: true,
                tools: this.discoveredTools,
                warmUp: this.warmUpResult,
                generation: this.generationResult,
            };

        } catch (error) {
            await this.transition(EVENTS.ERROR, error);
            throw error;
        }
    }

    /**
     * Execute Phase 1: Source Discovery
     */
    async executeSourceDiscovery() {
        console.log('🔍 Phase 1: Source Discovery...');

        try {
            this.discoveredTools = await this.sourceParser.parseDirectory(this.sourceDir);
            console.log(`✓ Discovered ${this.discoveredTools.length} tools`);
        } catch (error) {
            console.error('✗ Source discovery failed:', error.message);
            throw error;
        }
    }

    /**
     * Execute Phase 2: Context Preloading
     */
    async executeContextPreloading() {
        console.log('🔥 Phase 2: Context Preloading (8-message warm-up)...');

        try {
            this.warmUpResult = await this.warmUpEngine.warmUp();
            console.log(`✓ Warm-up complete: ${this.warmUpResult.totalTokens} tokens`);
        } catch (error) {
            console.error('✗ Context preloading failed:', error.message);
            throw error;
        }
    }

    /**
     * Execute Phase 3: UCL Generation
     */
    async executeUCLGeneration(testPrompt) {
        console.log('🧪 Phase 3: UCL Generation...');

        try {
            this.generationResult = await this.warmUpEngine.generateUCL(testPrompt);
            console.log(`✓ UCL generation complete: ${this.generationResult.outputLength} chars`);
        } catch (error) {
            console.error('✗ UCL generation failed:', error.message);
            throw error;
        }
    }

    /**
     * State transition handler
     */
    async transition(event, payload = null) {
        const previousState = this.currentState;
        let nextState = previousState;

        switch (this.currentState) {
            case STATES.IDLE:
                if (event === EVENTS.START) {
                    nextState = STATES.DISCOVERING;
                }
                break;

            case STATES.DISCOVERING:
                if (event === EVENTS.DISCOVERY_COMPLETE) {
                    nextState = STATES.PRELOADING;
                } else if (event === EVENTS.ERROR) {
                    nextState = STATES.ERROR;
                    this.error = payload;
                }
                break;

            case STATES.PRELOADING:
                if (event === EVENTS.PRELOADING_COMPLETE) {
                    nextState = STATES.GENERATING;
                } else if (event === EVENTS.ERROR) {
                    nextState = STATES.ERROR;
                    this.error = payload;
                }
                break;

            case STATES.GENERATING:
                if (event === EVENTS.GENERATION_COMPLETE) {
                    nextState = STATES.COMPLETE;
                } else if (event === EVENTS.ERROR) {
                    nextState = STATES.ERROR;
                    this.error = payload;
                }
                break;

            case STATES.ERROR:
                if (event === EVENTS.RESET) {
                    nextState = STATES.IDLE;
                    this.error = null;
                }
                break;

            case STATES.COMPLETE:
                if (event === EVENTS.RESET) {
                    nextState = STATES.IDLE;
                }
                break;
        }

        if (nextState !== previousState) {
            this.currentState = nextState;
            this.onStateChange(nextState, previousState, event);
            console.log(`🔄 State: ${previousState} → ${nextState} (${event})`);
        }
    }

    /**
     * Reset pipeline to IDLE state
     */
    reset() {
        this.transition(EVENTS.RESET);
        this.discoveredTools = [];
        this.warmUpResult = null;
        this.generationResult = null;
        this.error = null;
    }

    /**
     * Get current pipeline state
     */
    getState() {
        return {
            currentState: this.currentState,
            discoveredTools: this.discoveredTools.length,
            warmUpComplete: this.warmUpResult !== null,
            generationComplete: this.generationResult !== null,
            error: this.error,
        };
    }

    /**
     * Get pipeline result
     */
    getResult() {
        if (this.currentState !== STATES.COMPLETE) {
            return null;
        }

        return {
            tools: this.discoveredTools,
            warmUp: this.warmUpResult,
            generation: this.generationResult,
        };
    }
}

// Export for use in other modules
export { PipelineOrchestrator, STATES, EVENTS };
