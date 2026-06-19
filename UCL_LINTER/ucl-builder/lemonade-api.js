// ============================================================================
// LEMONADE API CLIENT
// Deep integration with Lemonade Server for local LLM management
// ============================================================================

class LemonadeClient {
    constructor(endpoint = 'http://localhost:13305') {
        this.endpoint = endpoint.replace(/\/$/, ''); // Remove trailing slash
        this.healthCache = null;
        this.healthCacheTime = 0;
        this.cacheTTL = 5000; // 5 seconds
    }

    // ========================================================================
    // SERVER STATUS
    // ========================================================================

    async getHealth(forceRefresh = false) {
        try {
            // Use cache if fresh
            if (!forceRefresh && this.healthCache && (Date.now() - this.healthCacheTime) < this.cacheTTL) {
                return this.healthCache;
            }

            const response = await fetch(`${this.endpoint}/v1/health`);
            if (!response.ok) {
                throw new Error(`Health check failed: ${response.status}`);
            }

            const data = await response.json();
            this.healthCache = data;
            this.healthCacheTime = Date.now();
            return data;
        } catch (error) {
            console.error('Health check error:', error);
            throw error;
        }
    }

    async getSystemInfo() {
        try {
            const response = await fetch(`${this.endpoint}/v1/system-info`);
            if (!response.ok) {
                throw new Error(`System info failed: ${response.status}`);
            }
            return await response.json();
        } catch (error) {
            console.error('System info error:', error);
            throw error;
        }
    }

    async getStats() {
        try {
            const response = await fetch(`${this.endpoint}/v1/stats`);
            if (!response.ok) {
                throw new Error(`Stats failed: ${response.status}`);
            }
            return await response.json();
        } catch (error) {
            console.error('Stats error:', error);
            throw error;
        }
    }

    async isServerRunning() {
        try {
            await this.getHealth();
            return true;
        } catch {
            return false;
        }
    }

    // ========================================================================
    // MODEL MANAGEMENT
    // ========================================================================

    async isModelLoaded(modelName) {
        try {
            const health = await this.getHealth();
            return health.all_models_loaded.some(m => m.model_name === modelName);
        } catch {
            return false;
        }
    }

    async getLoadedModels() {
        try {
            const health = await this.getHealth();
            return health.all_models_loaded || [];
        } catch {
            return [];
        }
    }

    async getAvailableModels(showAll = false) {
        try {
            const url = showAll
                ? `${this.endpoint}/v1/models?show_all=true`
                : `${this.endpoint}/v1/models`;

            const response = await fetch(url);
            if (!response.ok) {
                throw new Error(`Get models failed: ${response.status}`);
            }

            const data = await response.json();
            // Filter for downloaded models by default
            const models = data.data || [];
            return models.filter(m => showAll || m.downloaded === true);
        } catch (error) {
            console.error('Get available models error:', error);
            throw error;
        }
    }

    async loadModel(modelName, options = {}) {
        try {
            const body = {
                model_name: modelName,
                ...options
            };

            const response = await fetch(`${this.endpoint}/v1/load`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body)
            });

            if (!response.ok) {
                const error = await response.json().catch(() => ({}));
                throw new Error(error.message || `Load failed: ${response.status}`);
            }

            // Refresh health cache
            await this.getHealth(true);

            return await response.json();
        } catch (error) {
            console.error('Load model error:', error);
            throw error;
        }
    }

    async unloadModel(modelName) {
        try {
            const body = modelName ? { model_name: modelName } : {};

            const response = await fetch(`${this.endpoint}/v1/unload`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body)
            });

            if (!response.ok) {
                const error = await response.json().catch(() => ({}));
                throw new Error(error.message || `Unload failed: ${response.status}`);
            }

            // Refresh health cache
            await this.getHealth(true);

            return await response.json();
        } catch (error) {
            console.error('Unload model error:', error);
            throw error;
        }
    }

    async deleteModel(modelName) {
        try {
            const response = await fetch(`${this.endpoint}/v1/delete`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ model_name: modelName })
            });

            if (!response.ok) {
                const error = await response.json().catch(() => ({}));
                throw new Error(error.message || `Delete failed: ${response.status}`);
            }

            // Refresh health cache
            await this.getHealth(true);

            return await response.json();
        } catch (error) {
            console.error('Delete model error:', error);
            throw error;
        }
    }

    // ========================================================================
    // MODEL INSTALLATION (with SSE progress)
    // ========================================================================

    async pullModel(modelName, onProgress = null) {
        try {
            const response = await fetch(`${this.endpoint}/v1/pull`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    model_name: modelName,
                    stream: true,
                    subscribe: true
                })
            });

            if (!response.ok) {
                const error = await response.json().catch(() => ({}));
                throw new Error(error.message || `Pull failed: ${response.status}`);
            }

            // Parse Server-Sent Events (SSE)
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });

                // Parse SSE events
                const lines = buffer.split('\n');
                buffer = lines.pop() || ''; // Keep incomplete line in buffer

                let eventType = '';
                for (const line of lines) {
                    if (line.startsWith('event:')) {
                        eventType = line.substring(6).trim();
                    } else if (line.startsWith('data:')) {
                        const data = JSON.parse(line.substring(5).trim());

                        if (onProgress) {
                            onProgress({
                                type: eventType,
                                ...data
                            });
                        }

                        if (eventType === 'complete') {
                            // Refresh health cache
                            await this.getHealth(true);
                            return data;
                        } else if (eventType === 'error') {
                            throw new Error(data.error || 'Download failed');
                        }
                    }
                }
            }

            throw new Error('Download stream ended unexpectedly');
        } catch (error) {
            console.error('Pull model error:', error);
            throw error;
        }
    }

    // ========================================================================
    // UTILITY METHODS
    // ========================================================================

    getModelStatus(modelName, healthData) {
        if (!healthData) return 'unknown';

        const loadedModels = healthData.all_models_loaded || [];
        const isLoaded = loadedModels.some(m => m.model_name === modelName);

        if (isLoaded) return 'loaded';

        // Check if installed but not loaded
        // Note: Lemonade doesn't have a direct "list installed" endpoint
        // We'll assume if it's not loaded, it might need to be pulled
        return 'not_loaded';
    }

    formatBytes(bytes) {
        if (bytes === 0) return '0 B';
        const k = 1024;
        const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
    }

    getDeviceLabel(device) {
        if (!device) return 'Unknown';
        if (device.includes('npu')) return 'NPU';
        if (device.includes('gpu')) return 'GPU';
        if (device.includes('cpu')) return 'CPU';
        return device;
    }
}

// Global instance (initialized when Lemonade provider is selected)
let lemonadeClient = null;

function initLemonadeClient(endpoint) {
    lemonadeClient = new LemonadeClient(endpoint);
    return lemonadeClient;
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        LemonadeClient,
        initLemonadeClient
    };
}
