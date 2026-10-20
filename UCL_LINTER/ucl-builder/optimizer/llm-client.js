/**
 * LLMClient - Shared API Client for LLM interactions
 *
 * CRITICAL QUALITY FIX #1: Extracts API logic from WarmUpEngine.callAPI()
 * into a reusable, shared client. Both WarmUpEngine and ConversationManager
 * should use this instead of duplicating fetch/retry logic.
 *
 * Features:
 * - OpenAI-compatible chat completions API
 * - Retry logic with exponential backoff
 * - Token-bucket rate limiting
 * - Token usage tracking
 * - Abort support for cancellation
 * - Timeout support
 *
 * Quality fixes addressed:
 * - CRITICAL #1: API Logic Duplication - this IS the shared client
 * - HIGH #9: API Rate Limiting - token bucket implementation
 * - HIGH: Partial failure recovery - retries with backoff
 *
 * Usage:
 *   const client = new LLMClient({
 *     endpoint: 'http://localhost:13305',
 *     model: 'qwen3-4b-FLM',
 *     apiKey: '',
 *   });
 *   const response = await client.chat(messages, { temperature: 0.7 });
 *
 * @module llm-client
 */

/**
 * Token bucket rate limiter.
 * Allows burst up to `capacity`, refills at `refillRate` tokens per ms.
 */
class TokenBucket {
    /**
     * @param {number} capacity - Max tokens in bucket
     * @param {number} refillRate - Tokens added per millisecond
     */
    constructor(capacity, refillRate) {
        this.capacity = capacity;
        this.tokens = capacity;
        this.refillRate = refillRate;
        this.lastRefill = Date.now();
    }

    /**
     * Try to consume tokens. Returns true if successful.
     * @param {number} count - Tokens to consume
     * @returns {boolean}
     */
    tryConsume(count = 1) {
        this._refill();
        if (this.tokens >= count) {
            this.tokens -= count;
            return true;
        }
        return false;
    }

    /**
     * Wait until tokens are available, then consume.
     * @param {number} count - Tokens to consume
     * @returns {Promise<void>}
     */
    async waitAndConsume(count = 1) {
        this._refill();
        if (this.tokens >= count) {
            this.tokens -= count;
            return;
        }
        const deficit = count - this.tokens;
        const waitMs = Math.ceil(deficit / this.refillRate);
        await new Promise(resolve => setTimeout(resolve, waitMs));
        this._refill();
        this.tokens -= Math.min(count, this.tokens);
    }

    /** @private */
    _refill() {
        const now = Date.now();
        const elapsed = now - this.lastRefill;
        this.tokens = Math.min(this.capacity, this.tokens + elapsed * this.refillRate);
        this.lastRefill = now;
    }
}

/**
 * Shared LLM API client.
 * Follows the callAPI/callAPIWithRetry pattern from WarmUpEngine
 * but is reusable across all optimizer modules.
 */
class LLMClient {
    /**
     * @param {Object} config
     * @param {string} config.endpoint - API base URL (e.g. 'http://localhost:13305')
     * @param {string} config.model - Model identifier (e.g. 'qwen3-4b-FLM')
     * @param {string} [config.apiKey=''] - Bearer token for authentication
     * @param {number} [config.temperature=0.7] - Default sampling temperature
     * @param {number} [config.maxTokens=4000] - Default max output tokens
     * @param {number} [config.maxRetries=3] - Max retry attempts on failure
     * @param {number} [config.timeout=30000] - Request timeout in ms
     * @param {number} [config.rateLimitRPM=60] - Requests per minute (0 = disabled)
     * @param {Function} [config.onRetry] - Callback on retry: (attempt, error, waitMs) => void
     */
    constructor(config = {}) {
        this.endpoint = config.endpoint || 'http://localhost:13305';
        this.model = config.model || 'qwen3-4b-FLM';
        this.apiKey = config.apiKey || '';
        this.temperature = config.temperature ?? 0.7;
        this.maxTokens = config.maxTokens ?? 4000;
        this.maxRetries = config.maxRetries ?? 3;
        this.timeout = config.timeout ?? 30000;
        this.onRetry = config.onRetry || (() => {});

        // Rate limiting: token bucket (requests per minute -> tokens per ms)
        const rpm = config.rateLimitRPM ?? 60;
        if (rpm > 0) {
            const tokensPerMs = rpm / 60000;
            this._rateLimiter = new TokenBucket(rpm / 60, tokensPerMs);
        } else {
            this._rateLimiter = null;
        }

        // Cumulative token tracking
        this.totalPromptTokens = 0;
        this.totalCompletionTokens = 0;
        this.totalRequests = 0;

        // Active abort controller for cancellation
        this._activeController = null;
    }

    /**
     * Send a chat completion request with retry logic.
     *
     * @param {Array<{role: string, content: string}>} messages - Conversation messages
     * @param {Object} [options] - Override defaults for this request
     * @param {number} [options.temperature] - Sampling temperature
     * @param {number} [options.maxTokens] - Max output tokens
     * @param {string} [options.model] - Model override
     * @param {AbortSignal} [options.signal] - External abort signal
     * @returns {Promise<Object>} - Full API response with choices, usage
     */
    async chat(messages, options = {}) {
        const requestConfig = {
            model: options.model || this.model,
            temperature: options.temperature ?? this.temperature,
            max_tokens: options.maxTokens ?? this.maxTokens,
        };

        return this._callWithRetry(messages, requestConfig, options.signal);
    }

    /**
     * Extract content string from API response.
     * Convenience method: response.choices[0].message.content
     *
     * @param {Object} response - API response from chat()
     * @returns {string}
     * @throws {Error} If response format is unexpected
     */
    static extractContent(response) {
        if (!response?.choices?.[0]?.message?.content) {
            throw new Error('Unexpected API response format: missing choices[0].message.content');
        }
        return response.choices[0].message.content;
    }

    /**
     * Extract token usage from API response.
     *
     * @param {Object} response - API response from chat()
     * @returns {{ prompt: number, completion: number, total: number }}
     */
    static extractUsage(response) {
        const usage = response?.usage || {};
        return {
            prompt: usage.prompt_tokens || 0,
            completion: usage.completion_tokens || 0,
            total: usage.total_tokens || 0,
        };
    }

    /**
     * Get cumulative usage statistics.
     * @returns {{ totalRequests: number, totalPromptTokens: number, totalCompletionTokens: number, totalTokens: number }}
     */
    getUsageStats() {
        return {
            totalRequests: this.totalRequests,
            totalPromptTokens: this.totalPromptTokens,
            totalCompletionTokens: this.totalCompletionTokens,
            totalTokens: this.totalPromptTokens + this.totalCompletionTokens,
        };
    }

    /**
     * Reset cumulative usage counters.
     */
    resetUsageStats() {
        this.totalPromptTokens = 0;
        this.totalCompletionTokens = 0;
        this.totalRequests = 0;
    }

    /**
     * Abort any in-flight request.
     */
    abort() {
        if (this._activeController) {
            this._activeController.abort();
            this._activeController = null;
        }
    }

    /**
     * Update the model identifier.
     * @param {string} modelId
     */
    setModel(modelId) {
        this.model = modelId;
    }

    /**
     * Update the endpoint URL.
     * @param {string} endpoint
     */
    setEndpoint(endpoint) {
        this.endpoint = endpoint;
    }

    // ── Private implementation ──────────────────────────────────────────

    /**
     * Call API with retry logic and exponential backoff.
     * Extracted from WarmUpEngine.callAPIWithRetry().
     *
     * @private
     * @param {Array} messages
     * @param {Object} requestConfig
     * @param {AbortSignal} [externalSignal]
     * @returns {Promise<Object>}
     */
    async _callWithRetry(messages, requestConfig, externalSignal) {
        let lastError;

        for (let attempt = 1; attempt <= this.maxRetries; attempt++) {
            try {
                // Rate limit
                if (this._rateLimiter) {
                    await this._rateLimiter.waitAndConsume(1);
                }
                return await this._callAPI(messages, requestConfig, externalSignal);
            } catch (error) {
                lastError = error;

                // Don't retry on abort
                if (error.name === 'AbortError') {
                    throw error;
                }

                if (attempt < this.maxRetries) {
                    const waitMs = 1000 * Math.pow(2, attempt - 1); // 1s, 2s, 4s...
                    this.onRetry(attempt, error, waitMs);
                    console.warn(
                        `[LLMClient] Request failed (attempt ${attempt}/${this.maxRetries}), ` +
                        `retrying in ${waitMs}ms: ${error.message}`
                    );
                    await this._sleep(waitMs);
                }
            }
        }

        throw lastError;
    }

    /**
     * Execute a single API call.
     * Extracted from WarmUpEngine.callAPI().
     *
     * @private
     * @param {Array} messages
     * @param {Object} requestConfig
     * @param {AbortSignal} [externalSignal]
     * @returns {Promise<Object>}
     */
    async _callAPI(messages, requestConfig, externalSignal) {
        // Set up abort: link external signal + internal controller for timeout
        const controller = new AbortController();
        this._activeController = controller;

        if (externalSignal) {
            if (externalSignal.aborted) {
                controller.abort();
            } else {
                externalSignal.addEventListener('abort', () => controller.abort(), { once: true });
            }
        }

        // Timeout
        const timeoutId = setTimeout(() => controller.abort(), this.timeout);

        try {
            const url = `${this.endpoint}/v1/chat/completions`;
            const headers = {
                'Content-Type': 'application/json',
            };

            if (this.apiKey) {
                headers['Authorization'] = `Bearer ${this.apiKey}`;
            }

            const body = JSON.stringify({
                model: requestConfig.model,
                messages,
                temperature: requestConfig.temperature,
                max_tokens: requestConfig.max_tokens,
            });

            const response = await fetch(url, {
                method: 'POST',
                headers,
                body,
                signal: controller.signal,
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                const msg = errorData.error?.message || `HTTP ${response.status}: ${response.statusText}`;
                throw new Error(msg);
            }

            const result = await response.json();

            // Track cumulative tokens
            this.totalRequests++;
            if (result.usage) {
                this.totalPromptTokens += result.usage.prompt_tokens || 0;
                this.totalCompletionTokens += result.usage.completion_tokens || 0;
            }

            return result;
        } finally {
            clearTimeout(timeoutId);
            this._activeController = null;
        }
    }

    /**
     * @private
     */
    _sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}

// ── Exports ────────────────────────────────────────────────────────────────
export { LLMClient, TokenBucket };
