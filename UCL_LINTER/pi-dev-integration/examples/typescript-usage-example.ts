/**
 * UCL-Tool x pi.dev Integration: TypeScript Usage Examples
 *
 * Demonstrates how to use the UCL-Tool TypeScript modules:
 * 1. Import exported JSON configuration
 * 2. Format tools for prompt injection
 * 3. Use with buildSystemPrompt()
 * 4. Custom tool definitions
 * 5. Placeholder-based injection
 *
 * Usage (conceptual — requires pi.dev harness to run fully):
 *   npx tsx pi-dev-integration/examples/typescript-usage-example.ts
 */

import type {
  UCLToolConfig,
  UCLTool,
  UCLToolExport,
} from "../src/ucl-tool-types.ts";
import {
  formatUCLToolsForPrompt,
  replaceUCLPlaceholder,
} from "../src/ucl-tool-formatter.ts";

// ============================================================================
// EXAMPLE 1: Basic formatting
// ============================================================================

function example1_BasicFormatting(): void {
  console.log("=".repeat(70));
  console.log("EXAMPLE 1: Basic Formatting");
  console.log("=".repeat(70));
  console.log();

  const config: UCLToolConfig = {
    domains: ["filesystem"],
    format: "ucl",
    identity: "CodeAssistant",
  };

  const result = formatUCLToolsForPrompt(config);
  console.log("Formatted UCL output:");
  console.log(result);
  console.log();
}

// ============================================================================
// EXAMPLE 2: Multiple domains with grouping
// ============================================================================

function example2_MultipleDomains(): void {
  console.log("=".repeat(70));
  console.log("EXAMPLE 2: Multiple Domains with Grouping");
  console.log("=".repeat(70));
  console.log();

  const config: UCLToolConfig = {
    domains: ["filesystem", "web"],
    format: "ucl",
    groupTools: true,  // Group all tools under one ^^CONDITION^^
    identity: "FullStackAssistant",
  };

  const result = formatUCLToolsForPrompt(config);
  console.log("Grouped output:");
  console.log(result);
  console.log();
}

// ============================================================================
// EXAMPLE 3: JSON format (for OpenAI-compatible APIs)
// ============================================================================

function example3_JsonFormat(): void {
  console.log("=".repeat(70));
  console.log("EXAMPLE 3: JSON Format (OpenAI-compatible)");
  console.log("=".repeat(70));
  console.log();

  const config: UCLToolConfig = {
    domains: ["filesystem"],
    format: "json",
    identity: "CodeAssistant",
  };

  const result = formatUCLToolsForPrompt(config);
  console.log("JSON Schema output:");
  console.log(result);
  console.log();
}

// ============================================================================
// EXAMPLE 4: Custom tool definitions
// ============================================================================

function example4_CustomTools(): void {
  console.log("=".repeat(70));
  console.log("EXAMPLE 4: Custom Tool Definitions");
  console.log("=".repeat(70));
  console.log();

  const calculateBMI: UCLTool = {
    name: "calculate_bmi",
    description: "Calculate Body Mass Index from weight and height",
    parameters: [
      {
        name: "weight_kg",
        type: "number",
        required: true,
        description: "Weight in kilograms",
      },
      {
        name: "height_m",
        type: "number",
        required: true,
        description: "Height in meters",
      },
    ],
    keywords: ["bmi", "body", "mass", "weight", "height", "calculate"],
    examples: [
      {
        input: "calculate_bmi(weight_kg=70, height_m=1.75)",
        output: "22.86 (normal weight)",
      },
    ],
  };

  const config: UCLToolConfig = {
    customTools: [calculateBMI],
    format: "ucl",
    identity: "HealthAssistant",
  };

  const result = formatUCLToolsForPrompt(config);
  console.log("Custom tool output:");
  console.log(result);
  console.log();
}

// ============================================================================
// EXAMPLE 5: Placeholder injection
// ============================================================================

function example5_PlaceholderInjection(): void {
  console.log("=".repeat(70));
  console.log("EXAMPLE 5: Placeholder Injection ({{UCL_TOOLS}})");
  console.log("=".repeat(70));
  console.log();

  const template = `You are a helpful coding assistant.

{{UCL_TOOLS}}

Guidelines:
- Be concise and helpful
- Show file paths clearly
- Use tools when appropriate
`;

  const config: UCLToolConfig = {
    domains: ["filesystem"],
    format: "ucl",
    identity: "CodeHelper",
  };

  const result = replaceUCLPlaceholder(template, config);
  console.log("Prompt with injected tools:");
  console.log(result);
  console.log();
}

// ============================================================================
// EXAMPLE 6: Dynamic tool selection based on task
// ============================================================================

function example6_DynamicToolSelection(): void {
  console.log("=".repeat(70));
  console.log("EXAMPLE 6: Dynamic Tool Selection");
  console.log("=".repeat(70));
  console.log();

  function getToolsForTask(task: string): UCLToolConfig {
    const config: UCLToolConfig = {
      domains: [],
      format: "ucl",
      groupTools: true,
      identity: "TaskAssistant",
    };

    if (task.includes("file") || task.includes("read") || task.includes("write")) {
      config.domains!.push("filesystem");
    }
    if (task.includes("web") || task.includes("api") || task.includes("http")) {
      config.domains!.push("web");
    }
    if (task.includes("browser") || task.includes("test") || task.includes("playwright")) {
      config.domains!.push("playwright");
    }

    // Fallback: if no domain matched, include filesystem as default
    if (config.domains!.length === 0) {
      config.domains!.push("filesystem");
    }

    return config;
  }

  const tasks = [
    "Read the config file and update it",
    "Search the web for API documentation",
    "Test the login page with Playwright",
    "Help me with a math problem",
  ];

  for (const task of tasks) {
    const config = getToolsForTask(task);
    console.log(`Task: "${task}"`);
    console.log(`  Domains: [${config.domains!.join(", ")}]`);

    const formatted = formatUCLToolsForPrompt(config);
    console.log(`  Output: ${formatted.length} chars`);
    console.log();
  }
}

// ============================================================================
// EXAMPLE 7: Importing from exported JSON file
// ============================================================================

function example7_ImportFromJSON(): void {
  console.log("=".repeat(70));
  console.log("EXAMPLE 7: Import from Exported JSON");
  console.log("=".repeat(70));
  console.log();

  // In a real project, you would import the JSON file directly:
  //   import uclConfig from "../ucl-tools-export.json";
  //
  // For this example, we simulate the structure:

  const simulatedConfig: UCLToolExport = {
    version: "2.1",
    generatedAt: new Date().toISOString(),
    domains: [
      {
        name: "filesystem",
        description: "Local file system operations",
        category: "filesystem",
        keywords: ["file", "path", "directory", "read", "write"],
        tools: [
          {
            name: "read_file",
            description: "Read file content from disk",
            parameters: [
              { name: "path", type: "string", required: true },
            ],
            keywords: ["read", "file", "open"],
          },
        ],
      },
    ],
    customTools: [],
    metadata: {
      totalTools: 1,
      estimatedTokens: 100,
      overheadReduction: 78,
    },
  };

  // Use the imported config
  const config: UCLToolConfig = {
    domains: simulatedConfig.domains.map((d) => d.name),
    format: "ucl",
    identity: "CodeAssistant",
  };

  const result = formatUCLToolsForPrompt(config);
  console.log("Formatted from imported JSON:");
  console.log(result);
  console.log();
  console.log("JSON metadata:");
  console.log(`  Total tools: ${simulatedConfig.metadata.totalTools}`);
  console.log(`  Overhead reduction: ${simulatedConfig.metadata.overheadReduction}%`);
  console.log();
}

// ============================================================================
// Main
// ============================================================================

function main(): void {
  console.log();
  console.log("#".repeat(70));
  console.log("# UCL-Tool x pi.dev Integration: TypeScript Examples");
  console.log("#".repeat(70));
  console.log();

  example1_BasicFormatting();
  example2_MultipleDomains();
  example3_JsonFormat();
  example4_CustomTools();
  example5_PlaceholderInjection();
  example6_DynamicToolSelection();
  example7_ImportFromJSON();

  console.log("#".repeat(70));
  console.log("# All examples complete");
  console.log("#".repeat(70));
  console.log();
  console.log("Next steps:");
  console.log("  1. Generate ucl-tools-export.json with Python export_for_typescript()");
  console.log("  2. Import it in your TypeScript project");
  console.log("  3. Use buildSystemPrompt() with uclTools option");
  console.log();
}

main();
