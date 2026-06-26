/**
 * UCL-Tool Integration Types
 *
 * TypeScript interfaces for integrating UCL-Tool framework
 * into the pi.dev system prompt builder.
 */

/**
 * UCL-Tool parameter definition
 */
export interface UCLToolParameter {
  name: string;
  type: string;
  required: boolean;
  description?: string;
  default?: any;
  constraints?: Record<string, any>;
}

/**
 * UCL-Tool example (few-shot)
 */
export interface UCLToolExample {
  input: string;
  output: string;
}

/**
 * UCL-Tool definition
 */
export interface UCLTool {
  name: string;
  description: string;
  parameters: UCLToolParameter[];
  keywords?: string[];
  when?: string[];
  examples?: UCLToolExample[];
  returns?: string;
}

/**
 * UCL-Tool domain configuration
 */
export interface UCLToolDomain {
  name: string;
  description: string;
  category: string;
  keywords: string[];
  tools: UCLTool[];
}

/**
 * UCL-Tool configuration for system prompt injection
 */
export interface UCLToolConfig {
  /** Pre-built domain names to include (e.g., ["filesystem", "web", "playwright"]) */
  domains?: string[];

  /** Full domain data (optional - if provided, overrides getBuiltinDomains()) */
  domainData?: UCLToolDomain[];

  /** Custom tool definitions to add */
  customTools?: UCLTool[];

  /** Output format: "ucl" (directive-based) or "json" (schema-based) or "both" */
  format?: "ucl" | "json" | "both";

  /** Group tools under shared conditions (reduces O_s overhead) */
  groupTools?: boolean;

  /** Maximum number of tools to include (prevents prompt bloat) */
  maxTools?: number;

  /** Agent identity/role for the system prompt */
  identity?: string;
}

/**
 * JSON schema for UCL-Tool export from Python
 */
export interface UCLToolExport {
  version: string;
  generatedAt: string;
  domains: UCLToolDomain[];
  customTools: UCLTool[];
  metadata: {
    totalTools: number;
    estimatedTokens: number;
    overheadReduction: number;
  };
  /** Agent identity/role (optional) */
  identity?: string;
}
