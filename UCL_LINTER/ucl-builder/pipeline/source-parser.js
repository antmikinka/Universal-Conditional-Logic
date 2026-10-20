/**
 * UCL Source Parser - Tool Discovery Engine
 *
 * Parses .ts/.js source files to extract tool definitions, parameters, types, and descriptions.
 * Outputs JSON tool definitions for the UCL Preparation Pipeline warm-up engine.
 *
 * Architecture:
 * - Regex-based pattern matching for tool definitions
 * - Support for multiple tool definition formats (TypeScript interfaces, JSDoc, etc.)
 * - Parameter extraction with type inference
 * - Description extraction from comments
 *
 * Usage:
 *   const parser = new SourceParser();
 *   const tools = await parser.parseDirectory('./src/tools');
 *   console.log(tools); // JSON tool definitions
 */

import { promises as fsPromises } from 'fs';
import { join, extname } from 'path';

class SourceParser {
    constructor() {
        // Tool definition patterns
        this.patterns = {
            // TypeScript interface: interface ToolName { ... }
            typescriptInterface: /interface\s+(\w+Tool)\s*\{([^}]+)\}/g,

            // JSDoc tool definition: @tool {name} - description
            jsdocTool: /@tool\s+\{(\w+)\}\s*-\s*(.+)/g,

            // Function-based tool: function toolName(params) { ... }
            functionTool: /function\s+(\w+Tool)\s*\(([^)]*)\)/g,

            // Class-based tool: class ToolName { ... }
            classTool: /class\s+(\w+Tool)\s*\{([^}]+)\}/g,

            // Parameter patterns
            param: /(\w+)\s*:\s*(\w+)(?:\s*=\s*([^,}]+))?/g,

            // JSDoc parameter: @param {type} name - description
            jsdocParam: /@param\s+\{(\w+)\}\s+(\w+)\s*-\s*(.+)/g,

            // Description patterns
            description: /\/\/\s*(.+)|\/\*\*\s*(.+)\s*\*\//g,

            // Tool metadata
            toolName: /name:\s*['"](\w+)['"]/,
            toolDescription: /description:\s*['"](.+)['"]/,
        };

        // Tool metadata storage
        this.tools = new Map();
    }

    /**
     * Parse a directory recursively for tool definitions
     * @param {string} dirPath - Directory path to scan
     * @returns {Promise<Array>} - Array of tool definitions
     */
    async parseDirectory(dirPath) {
        const files = await this.getAllFiles(dirPath);

        for (const file of files) {
            if (file.endsWith('.ts') || file.endsWith('.js')) {
                const content = await fsPromises.readFile(file, 'utf-8');
                this.parseFile(content, file);
            }
        }

        return Array.from(this.tools.values());
    }

    /**
     * Parse a single file for tool definitions
     * @param {string} content - File content
     * @param {string} filePath - File path (for metadata)
     */
    parseFile(content, filePath) {
        // Extract TypeScript interfaces
        this.extractTypeScriptInterfaces(content, filePath);

        // Extract JSDoc tool definitions
        this.extractJsdocTools(content, filePath);

        // Extract function-based tools
        this.extractFunctionTools(content, filePath);

        // Extract class-based tools
        this.extractClassTools(content, filePath);
    }

    /**
     * Extract TypeScript interface-based tool definitions
     */
    extractTypeScriptInterfaces(content, filePath) {
        let match;
        while ((match = this.patterns.typescriptInterface.exec(content)) !== null) {
            const toolName = match[1];
            const interfaceBody = match[2];

            const tool = {
                name: toolName,
                type: 'interface',
                source: filePath,
                parameters: this.extractParameters(interfaceBody),
                description: this.extractDescription(content, match.index),
            };

            this.tools.set(toolName, tool);
        }
    }

    /**
     * Extract JSDoc-based tool definitions
     */
    extractJsdocTools(content, filePath) {
        let match;
        while ((match = this.patterns.jsdocTool.exec(content)) !== null) {
            const toolName = match[1];
            const description = match[2];

            const tool = {
                name: toolName,
                type: 'jsdoc',
                source: filePath,
                description: description,
                parameters: this.extractJsdocParameters(content, match.index),
            };

            this.tools.set(toolName, tool);
        }
    }

    /**
     * Extract function-based tool definitions
     */
    extractFunctionTools(content, filePath) {
        let match;
        while ((match = this.patterns.functionTool.exec(content)) !== null) {
            const toolName = match[1];
            const params = match[2];

            const tool = {
                name: toolName,
                type: 'function',
                source: filePath,
                parameters: this.extractFunctionParameters(params),
                description: this.extractDescription(content, match.index),
            };

            this.tools.set(toolName, tool);
        }
    }

    /**
     * Extract class-based tool definitions
     */
    extractClassTools(content, filePath) {
        let match;
        while ((match = this.patterns.classTool.exec(content)) !== null) {
            const toolName = match[1];
            const classBody = match[2];

            const tool = {
                name: toolName,
                type: 'class',
                source: filePath,
                parameters: this.extractParameters(classBody),
                description: this.extractDescription(content, match.index),
            };

            this.tools.set(toolName, tool);
        }
    }

    /**
     * Extract parameters from interface/class body
     */
    extractParameters(body) {
        const params = [];
        let match;

        while ((match = this.patterns.param.exec(body)) !== null) {
            params.push({
                name: match[1],
                type: match[2],
                required: !match[3], // If no default value, it's required
                default: match[3] || null,
            });
        }

        return params;
    }

    /**
     * Extract JSDoc parameters
     */
    extractJsdocParameters(content, startIndex) {
        const params = [];
        const endIndex = content.indexOf('*/', startIndex);
        const jsdocBlock = content.substring(startIndex, endIndex);

        let match;
        while ((match = this.patterns.jsdocParam.exec(jsdocBlock)) !== null) {
            params.push({
                name: match[2],
                type: match[1],
                description: match[3],
                required: true, // JSDoc params are required by default
            });
        }

        return params;
    }

    /**
     * Extract function parameters
     */
    extractFunctionParameters(paramsString) {
        const params = [];
        const paramList = paramsString.split(',').map(p => p.trim());

        for (const param of paramList) {
            if (!param) continue;

            // Parse: name: type = default
            const match = param.match(/(\w+)\s*:\s*(\w+)(?:\s*=\s*(.+))?/);
            if (match) {
                params.push({
                    name: match[1],
                    type: match[2],
                    required: !match[3],
                    default: match[3] || null,
                });
            }
        }

        return params;
    }

    /**
     * Extract description from comments
     */
    extractDescription(content, index) {
        // Look backwards from match index for comments
        const beforeMatch = content.substring(0, index);
        const lines = beforeMatch.split('\n');

        // Check last 5 lines for comments
        for (let i = lines.length - 1; i >= Math.max(0, lines.length - 5); i--) {
            const line = lines[i].trim();

            // Single-line comment: // description
            if (line.startsWith('//')) {
                return line.replace(/^\/\/\s*/, '');
            }

            // Multi-line comment: /** description */
            if (line.startsWith('/**') || line.startsWith('*')) {
                return line.replace(/^\/\*\*\s*|\*\s*|\*\/$/g, '').trim();
            }
        }

        return null;
    }

    /**
     * Get all files in directory recursively
     */
    async getAllFiles(dirPath) {
        const files = [];

        const entries = await fsPromises.readdir(dirPath, { withFileTypes: true });

        for (const entry of entries) {
            const fullPath = join(dirPath, entry.name);

            if (entry.isDirectory()) {
                const subFiles = await this.getAllFiles(fullPath);
                files.push(...subFiles);
            } else {
                files.push(fullPath);
            }
        }

        return files;
    }

    /**
     * Generate UCL-formatted tool definitions
     * Converts parsed tools to UCL syntax for warm-up messages
     */
    generateUCLDefinitions(tools) {
        const uclBlocks = [];

        for (const tool of tools) {
            const keywords = this.extractKeywords(tool.name, tool.description);
            const params = tool.parameters.map(p =>
                `  [[PARAM: ${p.name}:${p.type}${p.required ? ':required' : ''}]]`
            ).join('\n');

            const block = `^^CONDITION: ${keywords.join(', ')}^^
  [[TOOL: ${tool.name}]]
${params}
  // ${tool.description || 'No description'}
^^/CONDITION^^`;

            uclBlocks.push(block);
        }

        return uclBlocks.join('\n\n');
    }

    /**
     * Extract keywords from tool name and description
     */
    extractKeywords(name, description) {
        const keywords = new Set();

        // Extract from name (camelCase to words)
        const nameWords = name.replace(/Tool$/, '').replace(/([A-Z])/g, ' $1').trim().split(/\s+/);
        nameWords.forEach(w => keywords.add(w.toLowerCase()));

        // Extract from description (first 5 significant words)
        if (description) {
            const descWords = description.split(/\s+/)
                .filter(w => w.length > 3)
                .slice(0, 5);
            descWords.forEach(w => keywords.add(w.toLowerCase()));
        }

        return Array.from(keywords).slice(0, 5); // Max 5 keywords per UCL rules
    }

    /**
     * Export tools to JSON format
     */
    exportToJSON(tools, outputPath) {
        const json = JSON.stringify(tools, null, 2);
        return fsPromises.writeFile(outputPath, json, 'utf-8');
    }
}

// Export for use in other modules (ES modules)
export { SourceParser };
