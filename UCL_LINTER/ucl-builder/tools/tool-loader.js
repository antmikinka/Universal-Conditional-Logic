// ============================================================================
// UCL TOOL LOADER
// Dynamic tool loading from tools/ folder JSON files
// ============================================================================

const ToolLoader = {
    // Cache for loaded domains
    _cache: new Map(),

    // Loading state
    _loading: false,
    _loaded: false,
    _error: null,

    // Metadata from index.json
    _metadata: null,

    /**
     * Load the tool index to discover available domains
     */
    async loadIndex() {
        if (this._metadata) return this._metadata;

        try {
            const response = await fetch('tools/index.json');
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            this._metadata = await response.json();
            return this._metadata;
        } catch (error) {
            console.error('Failed to load tool index:', error);
            throw error;
        }
    },

    /**
     * Load a single domain's tools from JSON file
     */
    async loadDomain(domainName) {
        // Check cache first
        if (this._cache.has(domainName)) {
            return this._cache.get(domainName);
        }

        try {
            const response = await fetch(`tools/${domainName}.json`);
            if (!response.ok) throw new Error(`HTTP ${response.status}`);

            const domain = await response.json();

            // Cache the domain
            this._cache.set(domainName, domain);

            return domain;
        } catch (error) {
            console.error(`Failed to load domain '${domainName}':`, error);
            throw error;
        }
    },

    /**
     * Load all domains in parallel
     */
    async loadAll() {
        if (this._loaded) {
            return this.getAllDomains();
        }

        if (this._loading) {
            // Wait for ongoing load
            while (this._loading) {
                await new Promise(resolve => setTimeout(resolve, 50));
            }
            return this.getAllDomains();
        }

        this._loading = true;
        this._error = null;

        try {
            // Load index first
            const index = await this.loadIndex();

            // Load all domains in parallel
            const domainPromises = index.domains.map(domain => this.loadDomain(domain));
            await Promise.all(domainPromises);

            this._loaded = true;
            this._loading = false;

            return this.getAllDomains();
        } catch (error) {
            this._loading = false;
            this._error = error;
            throw error;
        }
    },

    /**
     * Get all loaded domains as a single object (mirrors UCL_TOOL_DOMAINS structure)
     */
    getAllDomains() {
        const domains = {};
        for (const [name, domain] of this._cache) {
            domains[name] = domain;
        }
        return domains;
    },

    /**
     * Get metadata about available tools
     */
    getMetadata() {
        return this._metadata || {
            totalDomains: 0,
            totalTools: 0,
            categories: {}
        };
    },

    /**
     * Check if a domain is loaded
     */
    hasDomain(domainName) {
        return this._cache.has(domainName);
    },

    /**
     * Get a specific domain (must be loaded first)
     */
    getDomain(domainName) {
        return this._cache.get(domainName);
    },

    /**
     * Get list of loaded domain names
     */
    getDomainNames() {
        return Array.from(this._cache.keys());
    },

    /**
     * Add or replace a domain produced by POST /api/convert-tools.
     * If the JSON catalog failed to load, seed the cache from the
     * hardcoded fallback first so the built-in domains stay visible.
     */
    registerDomain(domain) {
        if (!domain || !domain.name || !domain.tools) {
            throw new Error('Imported domain is missing a name or tools');
        }
        if (!this._loaded && typeof UCL_TOOL_DOMAINS !== 'undefined') {
            Object.entries(UCL_TOOL_DOMAINS).forEach(([key, value]) => {
                if (!this._cache.has(key)) this._cache.set(key, value);
            });
        }
        this._cache.set(domain.name, domain);
        this._loaded = true;
        if (this._metadata && Array.isArray(this._metadata.domains)) {
            if (!this._metadata.domains.includes(domain.name)) {
                this._metadata.domains.push(domain.name);
            }
            if (this._metadata.metadata) {
                this._metadata.metadata.totalDomains = this._cache.size;
            }
        }
    },

    /**
     * Clear the cache and reset state
     */
    clearCache() {
        this._cache.clear();
        this._metadata = null;
        this._loaded = false;
        this._loading = false;
        this._error = null;
    },

    /**
     * Get loading state
     */
    isLoading() {
        return this._loading;
    },

    /**
     * Check if tools are loaded
     */
    isLoaded() {
        return this._loaded;
    },

    /**
     * Get any error that occurred during loading
     */
    getError() {
        return this._error;
    },

    /**
     * Calculate total tool count across all loaded domains
     */
    getTotalToolCount() {
        let count = 0;
        for (const domain of this._cache.values()) {
            count += Object.keys(domain.tools).length;
        }
        return count;
    },

    /**
     * Get tool count for a specific domain
     */
    getDomainToolCount(domainName) {
        const domain = this._cache.get(domainName);
        return domain ? Object.keys(domain.tools).length : 0;
    }
};

// Export for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ToolLoader;
}
