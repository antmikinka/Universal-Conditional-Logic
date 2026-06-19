// ============================================================================
// OPENROUTER MODELS CONFIGURATION
// Curated list of recommended models with metadata
// ============================================================================

const OPENROUTER_MODELS = [
    // === Reasoning Models ===
    {
        id: 'anthropic/claude-3.7-sonnet',
        name: 'Claude 3.7 Sonnet',
        provider: 'Anthropic',
        category: 'Reasoning',
        recommended: true,
        description: 'Extended thinking, best for complex reasoning',
        contextWindow: 200000,
        pricing: { prompt: 3, completion: 15 }
    },
    {
        id: 'openai/o1',
        name: 'OpenAI o1',
        provider: 'OpenAI',
        category: 'Reasoning',
        recommended: true,
        description: 'Advanced reasoning model',
        contextWindow: 200000,
        pricing: { prompt: 15, completion: 60 }
    },
    {
        id: 'openai/o1-mini',
        name: 'OpenAI o1-mini',
        provider: 'OpenAI',
        category: 'Reasoning',
        recommended: true,
        description: 'Faster reasoning model, cost-effective',
        contextWindow: 128000,
        pricing: { prompt: 3, completion: 12 }
    },
    {
        id: 'google/gemini-2.0-flash-thinking-exp:free',
        name: 'Gemini 2.0 Flash Thinking (Free)',
        provider: 'Google',
        category: 'Reasoning',
        recommended: true,
        description: 'Free reasoning model with thinking capability',
        contextWindow: 32000,
        pricing: { prompt: 0, completion: 0 }
    },
    {
        id: 'deepseek/deepseek-chat',
        name: 'DeepSeek Chat',
        provider: 'DeepSeek',
        category: 'Reasoning',
        recommended: true,
        description: 'Strong reasoning at low cost',
        contextWindow: 64000,
        pricing: { prompt: 0.14, completion: 0.28 }
    },

    // === Standard Models ===
    {
        id: 'openai/gpt-4o',
        name: 'GPT-4o',
        provider: 'OpenAI',
        category: 'Standard',
        recommended: true,
        description: 'Latest GPT-4 model, multimodal',
        contextWindow: 128000,
        pricing: { prompt: 2.5, completion: 10 }
    },
    {
        id: 'openai/gpt-4o-mini',
        name: 'GPT-4o Mini',
        provider: 'OpenAI',
        category: 'Standard',
        recommended: true,
        description: 'Fast and affordable GPT-4 variant',
        contextWindow: 128000,
        pricing: { prompt: 0.15, completion: 0.6 }
    },
    {
        id: 'anthropic/claude-3.5-sonnet',
        name: 'Claude 3.5 Sonnet',
        provider: 'Anthropic',
        category: 'Standard',
        recommended: true,
        description: 'Balanced performance and speed',
        contextWindow: 200000,
        pricing: { prompt: 3, completion: 15 }
    },
    {
        id: 'anthropic/claude-3.5-haiku',
        name: 'Claude 3.5 Haiku',
        provider: 'Anthropic',
        category: 'Standard',
        recommended: true,
        description: 'Fast and cost-effective',
        contextWindow: 200000,
        pricing: { prompt: 0.8, completion: 4 }
    },
    {
        id: 'google/gemini-2.0-flash-exp:free',
        name: 'Gemini 2.0 Flash (Free)',
        provider: 'Google',
        category: 'Standard',
        recommended: true,
        description: 'Free high-performance model',
        contextWindow: 1000000,
        pricing: { prompt: 0, completion: 0 }
    },
    {
        id: 'google/gemini-pro-1.5',
        name: 'Gemini Pro 1.5',
        provider: 'Google',
        category: 'Standard',
        recommended: true,
        description: 'Large context window, multimodal',
        contextWindow: 2000000,
        pricing: { prompt: 1.25, completion: 5 }
    },
    {
        id: 'qwen/qwen-2.5-72b-instruct',
        name: 'Qwen 2.5 72B',
        provider: 'Qwen',
        category: 'Standard',
        recommended: true,
        description: 'Strong open model, good for UCL',
        contextWindow: 32000,
        pricing: { prompt: 0.35, completion: 0.4 }
    },

    // === Vision Models ===
    {
        id: 'openai/gpt-4o',
        name: 'GPT-4o (Vision)',
        provider: 'OpenAI',
        category: 'Vision',
        recommended: false,
        description: 'Vision + text capabilities',
        contextWindow: 128000,
        pricing: { prompt: 2.5, completion: 10 }
    },
    {
        id: 'anthropic/claude-3.5-sonnet',
        name: 'Claude 3.5 Sonnet (Vision)',
        provider: 'Anthropic',
        category: 'Vision',
        recommended: false,
        description: 'Vision + text capabilities',
        contextWindow: 200000,
        pricing: { prompt: 3, completion: 15 }
    },

    // === Budget/Free Options ===
    {
        id: 'meta-llama/llama-3.1-70b-instruct:free',
        name: 'Llama 3.1 70B (Free)',
        provider: 'Meta',
        category: 'Free',
        recommended: true,
        description: 'Free open model',
        contextWindow: 131072,
        pricing: { prompt: 0, completion: 0 }
    },
    {
        id: 'meta-llama/llama-3.3-70b-instruct:free',
        name: 'Llama 3.3 70B (Free)',
        provider: 'Meta',
        category: 'Free',
        recommended: true,
        description: 'Latest free Llama model',
        contextWindow: 131072,
        pricing: { prompt: 0, completion: 0 }
    },
    {
        id: 'microsoft/phi-4:free',
        name: 'Phi-4 (Free)',
        provider: 'Microsoft',
        category: 'Free',
        recommended: true,
        description: 'Compact high-performance model',
        contextWindow: 16384,
        pricing: { prompt: 0, completion: 0 }
    },
    {
        id: 'nousresearch/hermes-3-llama-3.1-405b:free',
        name: 'Hermes 3 405B (Free)',
        provider: 'Nous Research',
        category: 'Free',
        recommended: true,
        description: 'Massive free model',
        contextWindow: 128000,
        pricing: { prompt: 0, completion: 0 }
    }
];

// Group models by category for UI display
const MODELS_BY_CATEGORY = {
    'Reasoning (Best for UCL)': OPENROUTER_MODELS.filter(m => m.category === 'Reasoning'),
    'Standard': OPENROUTER_MODELS.filter(m => m.category === 'Standard'),
    'Free Models': OPENROUTER_MODELS.filter(m => m.category === 'Free'),
    'Vision': OPENROUTER_MODELS.filter(m => m.category === 'Vision')
};

// Default recommended models for quick access
const RECOMMENDED_MODELS = OPENROUTER_MODELS.filter(m => m.recommended);

// Helper function to format pricing
function formatPricing(pricing) {
    if (pricing.prompt === 0 && pricing.completion === 0) {
        return 'FREE';
    }
    return `$${pricing.prompt}/M in · $${pricing.completion}/M out`;
}

// Helper function to get model by ID
function getModelById(modelId) {
    return OPENROUTER_MODELS.find(m => m.id === modelId);
}

// Export for use in main app
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        OPENROUTER_MODELS,
        MODELS_BY_CATEGORY,
        RECOMMENDED_MODELS,
        formatPricing,
        getModelById
    };
}
