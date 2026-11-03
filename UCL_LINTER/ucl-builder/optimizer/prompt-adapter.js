/**
 * PromptAdapter - Model Family Prompt Adaptation Layer
 *
 * Adapts UCL prompts for different model families by applying family-specific
 * formatting, role structures, and instruction conventions.
 *
 * Supported families: qwen, claude, gpt, gemini, deepseek, llama, unknown.
 *
 * Key concerns:
 *   - System vs user role message structure (supportsSystem flag)
 *   - Family-specific thinking/reasoning instructions
 *   - UCL directive preservation across all adaptations
 *
 * @module prompt-adapter
 */

/**
 * @typedef {Object} Message
 * @property {'system'|'user'|'assistant'} role - Message role
 * @property {string} content - Message content
 */

/**
 * @typedef {Object} ModelConfig
 * @property {string} modelId - Model identifier
 * @property {string} family - Model family ('qwen'|'claude'|'gpt'|'gemini'|'deepseek'|'llama'|'unknown')
 * @property {number} temperature - Sampling temperature
 * @property {number} maxTokens - Maximum output tokens
 * @property {number} thinkingBudget - Token budget for thinking
 * @property {string} endpoint - API endpoint
 * @property {string} [thinkingStart] - Thinking block start marker
 * @property {string} [thinkingEnd] - Thinking block end marker
 * @property {Object} [extraParams] - Additional parameters
 */

/**
 * Map of model families to their supportsSystem capability.
 * Models that support system messages can separate UCL instructions
 * from user queries via system/user roles.
 * @type {Map<string, boolean>}
 */
const SYSTEM_SUPPORT = new Map([
    ['qwen', true],
    ['claude', true],
    ['gpt', true],
    ['gemini', true],
    ['deepseek', true],
    ['llama', false],
    ['unknown', false],
]);

class PromptAdapter {
    // ── Public API ──────────────────────────────────────────────────────────

    /**
     * Adapt a UCL prompt for a specific model family.
     * Routes to the appropriate family-specific adapter.
     *
     * @param {string} uclPrompt - The UCL prompt to adapt
     * @param {ModelConfig} modelConfig - Model configuration
     * @returns {string} Adapted prompt text
     */
    static adaptForModel(uclPrompt, modelConfig) {
        if (!uclPrompt || typeof uclPrompt !== 'string') return '';
        if (!modelConfig || typeof modelConfig !== 'object') return uclPrompt;

        const family = (modelConfig.family || 'unknown').toLowerCase();

        switch (family) {
            case 'qwen':     return PromptAdapter.adaptForQwen(uclPrompt, modelConfig);
            case 'claude':   return PromptAdapter.adaptForClaude(uclPrompt, modelConfig);
            case 'gpt':      return PromptAdapter.adaptForGPT(uclPrompt, modelConfig);
            case 'gemini':   return PromptAdapter.adaptForGemini(uclPrompt, modelConfig);
            case 'deepseek': return PromptAdapter.adaptForDeepSeek(uclPrompt, modelConfig);
            default:         return PromptAdapter.adaptGeneric(uclPrompt, modelConfig);
        }
    }

    /**
     * Adapt for Qwen family.
     * Adds step-by-step thinking instruction. Preserves UCL directives.
     *
     * @param {string} uclPrompt - UCL prompt
     * @param {ModelConfig} modelConfig - Model config
     * @returns {string} Adapted prompt
     */
    static adaptForQwen(uclPrompt, modelConfig) {
        const header = 'You are a precise instruction-following assistant. Think step by step using <think> tags before responding.\n\n';
        return header + uclPrompt;
    }

    /**
     * Adapt for Claude family.
     * System role handles UCL, user role for queries.
     * Adds precision instruction.
     *
     * @param {string} uclPrompt - UCL prompt
     * @param {ModelConfig} modelConfig - Model config
     * @returns {string} Adapted prompt
     */
    static adaptForClaude(uclPrompt, modelConfig) {
        const footer = '\n\nFollow the system instructions precisely and apply all conditional logic as specified.';
        return uclPrompt + footer;
    }

    /**
     * Adapt for GPT family.
     * Similar to Claude with GPT-specific conventions.
     *
     * @param {string} uclPrompt - UCL prompt
     * @param {ModelConfig} modelConfig - Model config
     * @returns {string} Adapted prompt
     */
    static adaptForGPT(uclPrompt, modelConfig) {
        const header = 'You are a helpful assistant that follows instructions exactly.\n\n';
        const footer = '\n\nApply all conditional rules and directives precisely as written above.';
        return header + uclPrompt + footer;
    }

    /**
     * Adapt for Gemini family.
     * Gemini-specific adaptations for instruction following.
     *
     * @param {string} uclPrompt - UCL prompt
     * @param {ModelConfig} modelConfig - Model config
     * @returns {string} Adapted prompt
     */
    static adaptForGemini(uclPrompt, modelConfig) {
        const header = 'Follow these instructions carefully and apply all conditional logic.\n\n';
        return header + uclPrompt;
    }

    /**
     * Adapt for DeepSeek family.
     * DeepSeek-specific thinking format with explicit reasoning guidance.
     *
     * @param {string} uclPrompt - UCL prompt
     * @param {ModelConfig} modelConfig - Model config
     * @returns {string} Adapted prompt
     */
    static adaptForDeepSeek(uclPrompt, modelConfig) {
        const header = 'You are a precise assistant. Before responding, reason through the problem step by step in <think> tags.\n\n';
        return header + uclPrompt;
    }

    /**
     * Generic adaptation for unknown model families.
     * Minimal changes — preserves UCL directives as-is.
     *
     * @param {string} uclPrompt - UCL prompt
     * @param {ModelConfig} modelConfig - Model config
     * @returns {string} Adapted prompt
     */
    static adaptGeneric(uclPrompt, modelConfig) {
        // No adaptation — preserve UCL as-is
        return uclPrompt;
    }

    // ── Message Building ────────────────────────────────────────────────────

    /**
     * Check if a model family supports system messages.
     *
     * @param {ModelConfig} modelConfig - Model configuration
     * @returns {boolean} True if the model supports system role messages
     */
    static supportsSystem(modelConfig) {
        if (!modelConfig || typeof modelConfig !== 'object') return false;

        // Check explicit extraParams override first
        if (modelConfig.extraParams && typeof modelConfig.extraParams.supportsSystem === 'boolean') {
            return modelConfig.extraParams.supportsSystem;
        }

        const family = (modelConfig.family || 'unknown').toLowerCase();
        return SYSTEM_SUPPORT.get(family) || false;
    }

    /**
     * Build messages for a model call with UCL prompt and user query.
     *
     * If supportsSystem: system=UCL prompt, user=query
     * If not supportsSystem: user=adapted prompt (instruction + UCL + query combined)
     *
     * @param {string} uclPrompt - The UCL prompt text
     * @param {ModelConfig} modelConfig - Model configuration
     * @returns {Message[]} Array of messages
     */
    static buildMessages(uclPrompt, modelConfig) {
        const adapted = PromptAdapter.adaptForModel(uclPrompt, modelConfig);

        if (PromptAdapter.supportsSystem(modelConfig)) {
            return [
                { role: 'system', content: adapted },
                { role: 'user', content: 'Follow the instructions above.' },
            ];
        }

        // No system message support — combine everything into user message
        return [
            { role: 'user', content: adapted },
        ];
    }

    /**
     * Build messages for testing: UCL prompt + test case input.
     * Used by TestEvaluator during optimization to get model responses.
     *
     * @param {string} uclPrompt - The UCL prompt text
     * @param {string} testInput - The test case input
     * @param {ModelConfig} modelConfig - Model configuration
     * @returns {Message[]} Array of messages for test evaluation
     */
    static buildTestMessages(uclPrompt, testInput, modelConfig) {
        const adapted = PromptAdapter.adaptForModel(uclPrompt, modelConfig);

        if (PromptAdapter.supportsSystem(modelConfig)) {
            return [
                { role: 'system', content: adapted },
                { role: 'user', content: testInput || 'Respond according to the instructions.' },
            ];
        }

        // Combine into single user message
        const combined = testInput
            ? `${adapted}\n\n---\n\nUser Query: ${testInput}`
            : adapted;

        return [
            { role: 'user', content: combined },
        ];
    }

    /**
     * Build messages for the optimization update phase.
     * Includes current prompt, thinking log, metrics, feedback, and history.
     *
     * @param {string} currentPrompt - Current UCL prompt being optimized
     * @param {string} thinkingLog - Extracted thinking/reasoning from model
     * @param {Object} metrics - Performance metrics (scores, test results)
     * @param {string} feedback - Human or automated feedback text
     * @param {Object[]} history - Array of previous iteration records
     * @param {ModelConfig} modelConfig - Model configuration
     * @returns {Message[]} Messages for the optimization LLM call
     */
    static buildUpdateMessages(currentPrompt, thinkingLog, metrics, feedback, history, modelConfig) {
        const adapted = PromptAdapter.adaptForModel(currentPrompt, modelConfig);

        // Build the optimization instruction
        let instruction = 'You are a prompt optimization expert. Your task is to improve the following UCL prompt based on the provided feedback and metrics.\n\n';

        // Current prompt
        instruction += '## Current Prompt\n';
        instruction += '```\n' + adapted + '\n```\n\n';

        // Thinking log
        if (thinkingLog && thinkingLog.trim().length > 0) {
            instruction += '## Model Reasoning\n';
            instruction += thinkingLog.trim() + '\n\n';
        }

        // Metrics
        if (metrics && typeof metrics === 'object') {
            instruction += '## Performance Metrics\n';
            instruction += '- Quality Score: ' + ((metrics.compositeScore || 0) * 100).toFixed(1) + '%\n';
            if (metrics.testPassRate !== undefined) {
                instruction += '- Test Pass Rate: ' + ((metrics.testPassRate || 0) * 100).toFixed(1) + '%\n';
            }
            if (metrics.iteration !== undefined) {
                instruction += '- Current Iteration: ' + metrics.iteration + '\n';
            }
            instruction += '\n';
        }

        // History
        if (history && Array.isArray(history) && history.length > 0) {
            instruction += '## Optimization History\n';
            for (const entry of history) {
                const score = ((entry.score || entry.compositeScore || 0) * 100).toFixed(1);
                instruction += `- Iteration ${entry.iteration || entry.iterationNum || '?'}: Score ${score}%\n`;
            }
            instruction += '\n';
        }

        // Feedback
        if (feedback && feedback.trim().length > 0) {
            instruction += '## Feedback\n';
            instruction += feedback.trim() + '\n\n';
        }

        instruction += '## Task\n';
        instruction += 'Produce an improved version of the UCL prompt that addresses the feedback and improves the quality score. Maintain all UCL syntax ([[CRITICAL:]], ^^CONDITION^^, [[PARAM:]], etc.).';

        if (PromptAdapter.supportsSystem(modelConfig)) {
            return [
                { role: 'system', content: instruction },
                { role: 'user', content: 'Optimize the prompt based on the analysis above.' },
            ];
        }

        return [
            { role: 'user', content: instruction },
        ];
    }
}

export { PromptAdapter, SYSTEM_SUPPORT };
