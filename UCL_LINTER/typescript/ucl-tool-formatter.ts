/**
 * UCL-Tool Formatter for System Prompts
 *
 * Converts UCL-Tool definitions into formatted text blocks
 * suitable for injection into LLM system prompts.
 */

import type {
  UCLToolConfig,
  UCLTool,
  UCLToolParameter,
  UCLToolExample,
  UCLToolDomain,
} from "./ucl-tool-types.ts";

/**
 * Format a single parameter for UCL directive format
 */
function formatParameter(param: UCLToolParameter): string {
  const required = param.required ? "required" : "optional";
  const desc = param.description ? `\n    [[DESC: ${param.description}]]` : "";
  return `    [[PARAM: ${param.name}, ${param.type}, ${required}]]${desc}`;
}

/**
 * Format a single tool for UCL directive format
 */
function formatToolUCL(tool: UCLTool): string {
  const lines: string[] = [];

  lines.push(`    [[TOOL: ${tool.name}]]`);
  lines.push(`    [[DESC: ${tool.description}]]`);

  // Parameters
  for (const param of tool.parameters) {
    lines.push(formatParameter(param));
  }

  // WHEN condition (keyword-based activation)
  if (tool.keywords && tool.keywords.length > 0) {
    lines.push(`    [[WHEN: ${tool.keywords.join(", ")}]]`);
  }

  // Examples (few-shot)
  if (tool.examples && tool.examples.length > 0) {
    for (const example of tool.examples) {
      lines.push(`    [[EXAMPLE: ${example.input} -> ${example.output}]]`);
    }
  }

  return lines.join("\n");
}

/**
 * Format a tool domain as a UCL condition group
 */
function formatDomainUCL(domain: UCLToolDomain): string {
  const lines: string[] = [];

  // Create condition based on domain keywords
  if (domain.keywords.length > 0) {
    const condition = domain.keywords
      .map((kw) => `"${kw}"`)
      .join(" OR ");
    lines.push(
      `^^CONDITION: {{concept:user_input}} CONTAINS ${condition}^^`,
    );
  }

  lines.push(`<${domain.name}>`);
  lines.push(`[[DESC: ${domain.description}]]`);

  // Format each tool
  for (const tool of domain.tools) {
    lines.push(formatToolUCL(tool));
  }

  lines.push(`</${domain.name}>`);

  if (domain.keywords.length > 0) {
    lines.push("^^/CONDITION^^");
  }

  return lines.join("\n");
}

/**
 * Format a single tool for JSON schema format
 */
function formatToolJSON(tool: UCLTool): object {
  const parameters: Record<string, any> = {
    type: "object",
    properties: {},
    required: [],
  };

  for (const param of tool.parameters) {
    parameters.properties[param.name] = {
      type: param.type,
      description: param.description || "",
    };
    if (param.required) {
      parameters.required.push(param.name);
    }
  }

  return {
    type: "function",
    function: {
      name: tool.name,
      description: tool.description,
      parameters: parameters,
    },
  };
}

/**
 * Format UCL-Tool configuration for prompt injection
 *
 * @param config - UCL-Tool configuration
 * @returns Formatted string ready for system prompt injection
 *
 * @example
 * ```typescript
 * const config: UCLToolConfig = {
 *   domains: ["filesystem", "web"],
 *   format: "ucl",
 *   groupTools: true,
 *   identity: "CodeAssistant"
 * };
 * const formatted = formatUCLToolsForPrompt(config);
 * // Returns: <system_role>...</system_role>\n^^CONDITION:...^^\n<filesystem>...</filesystem>
 * ```
 */
export function formatUCLToolsForPrompt(config: UCLToolConfig): string {
  const sections: string[] = [];

  // System role (identity)
  if (config.identity) {
    sections.push(`<system_role>
    {{concept:ai_identity}} = "${config.identity}"
</system_role>`);
  }

  // Collect all tools (from domains + custom)
  const allTools: UCLTool[] = [...(config.customTools || [])];

  // Format domains
  if (config.domains && config.domains.length > 0) {
    // Note: In real implementation, domains would be loaded from a registry
    // For now, we'll use placeholder domain definitions
    const domainMap: Record<string, UCLToolDomain> = getBuiltinDomains();

    for (const domainName of config.domains) {
      const domain = domainMap[domainName];
      if (domain) {
        if (config.format === "json") {
          // JSON format: add tools to allTools
          allTools.push(...domain.tools);
        } else {
          // UCL format: format as condition group
          sections.push(formatDomainUCL(domain));
        }
      }
    }
  }

  // Apply maxTools limit
  const toolsToFormat = config.maxTools
    ? allTools.slice(0, config.maxTools)
    : allTools;

  // JSON format
  if (config.format === "json" || config.format === "both") {
    const jsonTools = toolsToFormat.map(formatToolJSON);
    const jsonBlock = JSON.stringify({ tools: jsonTools }, null, 2);
    sections.push(`<ucl_tools_json>\n${jsonBlock}\n</ucl_tools_json>`);
  }

  // Group tools under shared condition (if requested and not already grouped)
  if (
    config.groupTools &&
    config.format === "ucl" &&
    toolsToFormat.length > 0
  ) {
    const allKeywords = toolsToFormat.flatMap((t) => t.keywords || []);
    const uniqueKeywords = [...new Set(allKeywords)].slice(0, 10);

    if (uniqueKeywords.length > 0) {
      const condition = uniqueKeywords
        .map((kw) => `"${kw}"`)
        .join(" OR ");
      const groupedTools = toolsToFormat
        .map((t) => formatToolUCL(t))
        .join("\n");

      sections.push(
        `^^CONDITION: {{concept:user_input}} CONTAINS ${condition}^^\n<tools>\n${groupedTools}\n</tools>\n^^/CONDITION^^`,
      );
    }
  }

  return sections.join("\n\n");
}

/**
 * Replace {{UCL_TOOLS}} placeholder in a prompt template
 *
 * @param template - Prompt template with {{UCL_TOOLS}} placeholder
 * @param config - UCL-Tool configuration
 * @returns Prompt with placeholder replaced
 *
 * @example
 * ```typescript
 * const template = "You are an assistant.\n\n{{UCL_TOOLS}}\n\nGuidelines: Be helpful.";
 * const config: UCLToolConfig = { domains: ["filesystem"], identity: "Helper" };
 * const result = replaceUCLPlaceholder(template, config);
 * // Returns prompt with UCL-Tool block injected
 * ```
 */
export function replaceUCLPlaceholder(
  template: string,
  config: UCLToolConfig,
): string {
  if (!template.includes("{{UCL_TOOLS}}")) {
    return template;
  }

  const formatted = formatUCLToolsForPrompt(config);
  return template.replace("{{UCL_TOOLS}}", formatted);
}

/**
 * Get built-in domain definitions
 * (In production, this would load from a registry or JSON file)
 */
function getBuiltinDomains(): Record<string, UCLToolDomain> {
  return {
    filesystem: {
      name: "filesystem",
      description: "Local file system operations",
      category: "core",
      keywords: [
        "file",
        "path",
        "directory",
        "read",
        "write",
        "delete",
        "move",
        "copy",
      ],
      tools: [
        {
          name: "read_file",
          description: "Read file content from disk",
          parameters: [
            {
              name: "path",
              type: "string",
              required: true,
              description: "Path to the file",
            },
            {
              name: "encoding",
              type: "string",
              required: false,
              description: "File encoding",
              default: "utf-8",
            },
          ],
          keywords: ["read", "file", "open", "load"],
        },
        {
          name: "write_file",
          description: "Write content to a file",
          parameters: [
            {
              name: "path",
              type: "string",
              required: true,
              description: "Path to the file",
            },
            {
              name: "content",
              type: "string",
              required: true,
              description: "Content to write",
            },
          ],
          keywords: ["write", "save", "create"],
        },
        {
          name: "list_dir",
          description: "List directory contents",
          parameters: [
            {
              name: "path",
              type: "string",
              required: true,
              description: "Directory path",
            },
          ],
          keywords: ["list", "directory", "folder", "ls"],
        },
      ],
    },
    // Add more domains as needed...
  };
}
