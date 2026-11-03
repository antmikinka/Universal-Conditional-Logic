/**
 * ThinkingExtractor - Multi-Strategy Reasoning Extraction
 *
 * Pure functions for parsing model responses to separate thinking/reasoning
 * from final output content. Supports multiple model families.
 *
 * Features:
 * - Strategy auto-detection from model ID
 * - Qwen/DeepSeek <think>...</think> extraction
 * - Claude thinking block extraction
 * - Gemini thought parts extraction
 * - Generic fallback (no extraction, full response as content)
 * - Handles malformed/missing tags gracefully
 *
 * Quality fixes addressed:
 * - HIGH: Thinking extraction fails silently — always returns content even if thinking is empty
 * - HIGH: Logs extraction confidence
 *
 * Usage:
 *   const { thinking, content, strategy, confidence } = ThinkingExtractor.extract(rawResponse, 'qwen3-4b-FLM');
 *
 * @module thinking-extractor
 */

/**
 * Extraction result.
 * @typedef {Object} ExtractionResult
 * @property {string} thinking - Extracted reasoning text (empty string if none)
 * @property {string} content - Usable output (always non-null, may be full response)
 * @property {string} strategy - Strategy name used ('qwen', 'deepseek', 'claude', 'gemini', 'generic')
 * @property {number} confidence - 0.0 to 1.0 confidence in extraction quality
 * @property {string} rawResponse - Original input for reference
 */

/**
 * Strategy definition.
 * @typedef {Object} StrategyDef
 * @property {string} name - Strategy identifier
 * @property {RegExp} pattern - Regex to match model ID
 * @property {Function} extract - (rawResponse: string) => ExtractionResult
 */

/**
 * Multi-strategy reasoning extraction engine.
 * All methods are static — no instance state needed.
 */
class ThinkingExtractor {
    /**
     * Main entry point. Auto-detects strategy from model ID and extracts.
     *
     * @param {string} rawResponse - Raw model response text
     * @param {string} modelId - Model identifier (e.g. 'qwen3-4b-FLM')
     * @returns {ExtractionResult}
     */
    static extract(rawResponse, modelId = '') {
        if (!rawResponse || typeof rawResponse !== 'string') {
            return {
                thinking: '',
                content: '',
                strategy: 'generic',
                confidence: 1.0,
                rawResponse: rawResponse || '',
            };
        }

        const strategy = ThinkingExtractor.detectStrategy(modelId);
        const extractFn = ThinkingExtractor._getExtractFn(strategy);

        try {
            const result = extractFn(rawResponse);
            return {
                ...result,
                strategy: result.strategy || strategy,
                rawResponse,
            };
        } catch (error) {
            // NEVER fail silently — return full response as content
            console.warn(`[ThinkingExtractor] Extraction failed for strategy '${strategy}':`, error.message);
            return {
                thinking: '',
                content: rawResponse,
                strategy: 'generic',
                confidence: 0.5,
                rawResponse,
            };
        }
    }

    /**
     * Detect which extraction strategy to use based on model ID.
     *
     * @param {string} modelId - Model identifier
     * @returns {string} Strategy name
     */
    static detectStrategy(modelId) {
        if (!modelId || typeof modelId !== 'string') {
            return 'generic';
        }

        const lower = modelId.toLowerCase();

        for (const [name, pattern] of ThinkingExtractor._strategyPatterns) {
            if (pattern.test(lower)) {
                return name;
            }
        }

        return 'generic';
    }

    /**
     * Extract reasoning from Qwen models.
     * Pattern: <think>...</think>
     *
     * @param {string} response
     * @returns {Partial<ExtractionResult>}
     */
    static extractQwen(response) {
        return ThinkingExtractor._extractThinkTags(response, 'qwen');
    }

    /**
     * Extract reasoning from DeepSeek models.
     * Pattern: <think>...</think> (same as Qwen)
     *
     * @param {string} response
     * @returns {Partial<ExtractionResult>}
     */
    static extractDeepSeek(response) {
        return ThinkingExtractor._extractThinkTags(response, 'deepseek');
    }

    /**
     * Extract reasoning from Claude models.
     * Claude uses API-level thinking blocks, not in-text tags.
     * For text-only responses, looks for structured reasoning patterns.
     *
     * @param {string} response
     * @returns {Partial<ExtractionResult>}
     */
    static extractClaude(response) {
        // Claude's extended thinking is API-level, not in response text.
        // Try to detect reasoning sections by common patterns.
        const patterns = [
            // "Let me think through this..." followed by content
            /(?:let me think|reasoning|analysis|thought process)[:\s]*\n([\s\S]*?)(?:\n---|\n#|\n\*\*|$)/i,
        ];

        for (const pattern of patterns) {
            const match = response.match(pattern);
            if (match && match[1]?.trim()) {
                const thinking = match[1].trim();
                const content = response.replace(match[0], '').trim();
                return {
                    thinking,
                    content: content || response, // Don't lose data
                    confidence: 0.6, // Lower confidence since this is heuristic
                };
            }
        }

        // Fallback: no thinking extracted
        return {
            thinking: '',
            content: response,
            confidence: 0.9, // High confidence that no thinking is present
        };
    }

    /**
     * Extract reasoning from Gemini models.
     * Gemini uses "thought" parts in candidates, but in text form
     * we look for reasoning markers.
     *
     * @param {string} response
     * @returns {Partial<ExtractionResult>}
     */
    static extractGemini(response) {
        // Gemini thinking is typically in API response structure, not text.
        // For text responses, look for thinking markers.
        const patterns = [
            /(?:thinking|reasoning|let me analyze)[:\s]*\n([\s\S]*?)(?:\n\n---|\n\n##|\n\n\*\*Response|$)/i,
        ];

        for (const pattern of patterns) {
            const match = response.match(pattern);
            if (match && match[1]?.trim()) {
                const thinking = match[1].trim();
                const content = response.replace(match[0], '').trim();
                return {
                    thinking,
                    content: content || response,
                    confidence: 0.6,
                };
            }
        }

        return {
            thinking: '',
            content: response,
            confidence: 0.9,
        };
    }

    /**
     * Generic fallback. No extraction — full response is content.
     *
     * @param {string} response
     * @returns {Partial<ExtractionResult>}
     */
    static extractGeneric(response) {
        return {
            thinking: '',
            content: response,
            confidence: 1.0, // Full confidence that content is the full response
        };
    }

    /**
     * Remove specific HTML/XML-style tags from text.
     *
     * @param {string} text
     * @param {string} tagName - Tag name without brackets
     * @returns {string} - Text with tags removed
     */
    static stripTags(text, tagName) {
        if (!text || !tagName) return text || '';
        const openPattern = new RegExp(`<${tagName}[^>]*>`, 'gi');
        const closePattern = new RegExp(`</${tagName}>`, 'gi');
        return text.replace(openPattern, '').replace(closePattern, '');
    }

    // ── Private helpers ───────────────────────────────────────────────────

    /**
     * Strategy name → regex pattern for model ID matching.
     * Order matters: first match wins.
     * @private
     */
    static _strategyPatterns = new Map([
        ['qwen',     /qwen/i],
        ['deepseek', /deepseek/i],
        ['claude',   /claude|anthropic/i],
        ['gemini',   /gemini|google/i],
    ]);

    /**
     * Get the extraction function for a strategy name.
     * @private
     */
    static _getExtractFn(strategy) {
        const fns = {
            qwen: ThinkingExtractor.extractQwen,
            deepseek: ThinkingExtractor.extractDeepSeek,
            claude: ThinkingExtractor.extractClaude,
            gemini: ThinkingExtractor.extractGemini,
            generic: ThinkingExtractor.extractGeneric,
        };
        return fns[strategy] || fns.generic;
    }

    /**
     * Shared extraction for <think>...</think> tags (Qwen, DeepSeek).
     * @private
     * @param {string} response
     * @param {string} strategyName
     * @returns {Partial<ExtractionResult>}
     */
    static _extractThinkTags(response, strategyName) {
        // Match <think>...</think> — non-greedy, handles multiline
        const thinkPattern = /<think>([\s\S]*?)<\/think>/i;
        const match = response.match(thinkPattern);

        if (match) {
            const thinking = match[1].trim();
            // Content is everything outside the <think> tags
            const content = response.replace(thinkPattern, '').trim();

            return {
                thinking,
                content: content || '', // Could be empty if response was only thinking
                confidence: thinking ? 0.95 : 0.8,
            };
        }

        // Check for unclosed <think> tag (malformed response)
        const unclosedMatch = /<think>([\s\S]*)/i;
        const unclosed = response.match(unclosedMatch);

        if (unclosed) {
            const thinking = unclosed[1].trim();
            return {
                thinking,
                content: '', // No content after unclosed think
                confidence: 0.4, // Low confidence — malformed
            };
        }

        // No <think> tags found at all
        return {
            thinking: '',
            content: response,
            confidence: 0.85, // High confidence the model just didn't use thinking
        };
    }
}

// ── Exports ────────────────────────────────────────────────────────────────
export { ThinkingExtractor };
