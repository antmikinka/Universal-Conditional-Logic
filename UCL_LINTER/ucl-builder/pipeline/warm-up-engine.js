/**
 * UCL Warm-Up Engine - Production Version
 *
 * Implements the 8-message progressive teaching sequence validated by spike test.
 * Includes error handling, retry logic, progress tracking, and conversation state management.
 *
 * Architecture:
 * - Progressive teaching sequence (8 steps)
 * - Conversation state management (accumulating context)
 * - Error handling with retry logic (3 attempts per step)
 * - Progress tracking with callbacks
 * - Integration with Lemonade API
 *
 * Usage:
 *   const engine = new WarmUpEngine({
 *     endpoint: 'http://localhost:13305',
 *     model: 'qwen3-4b-FLM',
 *     onProgress: (step, total) => console.log(`Step ${step}/${total}`)
 *   });
 *   const result = await engine.warmUp();
 */

import { SourceParser } from './source-parser.js';

class WarmUpEngine {
    constructor(config = {}) {
        this.endpoint = config.endpoint || 'http://localhost:13305';
        this.model = config.model || 'qwen3-4b-FLM';
        this.apiKey = config.apiKey || '';
        this.temperature = config.temperature || 0.7;
        this.maxTokens = config.maxTokens || 4000;
        this.maxRetries = config.maxRetries || 3;

        // Progress callback
        this.onProgress = config.onProgress || (() => {});
        this.onStepComplete = config.onStepComplete || (() => {});
        this.onError = config.onError || (() => {});

        // Conversation state
        this.conversation = [];
        this.currentStep = 0;
        this.totalSteps = 8;
        this.totalTokens = 0;

        // Warm-up message templates
        this.warmupMessages = this.initializeWarmupMessages();
    }

    /**
     * Initialize 8 warm-up message templates
     * Based on spike test validation (5.25x quality improvement)
     */
    initializeWarmupMessages() {
        return [
            {
                step: 1,
                phase: 'Syntax',
                title: 'UCL Syntax Basics',
                user: `You are learning Universal Conditional Logic (UCL) — a mathematical framework for prompt optimization. Here are the core directives you must know:

[[CRITICAL: content]]     — HIGHEST PRIORITY. +9.3% quality. Always place at top.
[[REQUIRE: requirement]]  — Specify required conditions
[[TRANSFORM: src TO dst]] — Define transformation rules
[[VALIDATE: rule]]        — Specify validation requirements
[[ENFORCE: rule]]         — Strictly enforce a constraint
[[APPLY: rule]]           — Apply specific logic
[[TOOL: tool_name]]       — Reference a tool
[[PARAM: name:type]]      — Define a tool parameter

Key principle: The Over-Specification Paradox — more detail can degrade performance. Optimal specification threshold S* ≈ 0.509.

Acknowledge that you understand these directives and their purposes.`,
            },
            {
                step: 2,
                phase: 'Syntax',
                title: 'Conditional Logic Patterns',
                user: `UCL uses conditional blocks to activate logic based on keywords. Here are the patterns:

PATTERN 1 - Keyword Condition (RECOMMENDED):
^^CONDITION: keyword1, keyword2^^
  [[APPLY: domain-specific rules]]
^^/CONDITION^^

Example:
^^CONDITION: matrix, eigenvalue, linear^^
  [[APPLY: Linear algebra methods]]
  [[REQUIRE: Matrix dimension validation]]
^^/CONDITION^^

PATTERN 2 - Tool Activation:
^^CONDITION: file, read, open^^
  [[TOOL: read_file]]
  [[PARAM: path:string:required]]
^^/CONDITION^^

PATTERN 3 - SWITCH (NOT RECOMMENDED - high overhead):
^^SWITCH: domain^^
  ^^CASE: math^^ [[APPLY: Math rules]] ^^/CASE^^
  ^^CASE: physics^^ [[APPLY: Physics]] ^^/CASE^^
^^/SWITCH^^

Key rule: Use keyword CONDITIONS instead of SWITCH to minimize structural overhead (O_s).

Confirm you understand these conditional patterns.`,
            },
            {
                step: 3,
                phase: 'Tools',
                title: 'Tool Definitions (Filesystem Domain)',
                user: `Here are the tools you will work with, formatted as UCL directives:

^^CONDITION: read, file, open, load^^
  [[TOOL: read_file]]
  [[PARAM: path:string:required]]
  [[PARAM: encoding:string]]
  // Read file content from disk
^^/CONDITION^^

^^CONDITION: write, save, create^^
  [[TOOL: write_file]]
  [[PARAM: path:string:required]]
  [[PARAM: content:string:required]]
  // Write content to a file
^^/CONDITION^^

^^CONDITION: delete, remove^^
  [[TOOL: delete_file]]
  [[PARAM: path:string:required]]
  // Delete a file from disk
^^/CONDITION^^

^^CONDITION: list, directory, folder^^
  [[TOOL: list_dir]]
  [[PARAM: path:string:required]]
  [[PARAM: recursive:boolean]]
  // List files in a directory
^^/CONDITION^^

^^CONDITION: execute, run, shell, command^^
  [[TOOL: exec_command]]
  [[PARAM: command:string:required]]
  [[PARAM: cwd:string]]
  // Execute shell command
^^/CONDITION^^

Acknowledge these tools and their activation patterns.`,
            },
            {
                step: 4,
                phase: 'Tools',
                title: 'Tool Parameters and Return Types',
                user: `UCL tool parameters follow this syntax:

[[PARAM: name:type]]              — Optional parameter
[[PARAM: name:type:required]]     — Required parameter

Types: string, integer, boolean, object, array

Return types can be specified:
[[TOOL: read_file]] → string
[[TOOL: list_dir]] → array
[[TOOL: exec_command]] → object

Example with mixed parameters:
^^CONDITION: search, find, query^^
  [[TOOL: search_files]]
  [[PARAM: pattern:string:required]]
  [[PARAM: path:string]]
  [[PARAM: max_results:integer]]
  [[PARAM: case_sensitive:boolean]]
  // Search for files matching pattern
^^/CONDITION^^

Confirm you understand parameter syntax and return types.`,
            },
            {
                step: 5,
                phase: 'Examples',
                title: 'Conversion Examples (Natural Language → UCL)',
                user: `Here are examples of converting natural language prompts into UCL format:

EXAMPLE 1:
Input: "Build me a chatbot that can search the web and answer questions"
Output:
[[CRITICAL: You are a research chatbot]]

^^CONDITION: search, web, query, find^^
  [[TOOL: web_search]]
  [[PARAM: query:string:required]]
  [[REQUIRE: Return top 5 results with URLs]]
^^/CONDITION^^

^^CONDITION: question, answer, explain^^
  [[APPLY: Synthesize answers from search results]]
  [[REQUIRE: Cite sources]]
^^/CONDITION^^

EXAMPLE 2:
Input: "Create a code reviewer that checks Python style"
Output:
[[CRITICAL: You are a Python code reviewer]]

^^CONDITION: review, check, lint, style^^
  [[ENFORCE: PEP 8 compliance]]
  [[VALIDATE: Type hint presence]]
  [[APPLY: Static analysis rules]]
^^/CONDITION^^

EXAMPLE 3:
Input: "Make an assistant that can manage files and run commands"
Output:
[[CRITICAL: You are a system administration assistant]]

^^CONDITION: file, read, write, delete^^
  [[TOOL: read_file]]
  [[TOOL: write_file]]
  [[TOOL: delete_file]]
  [[PARAM: path:string:required]]
^^/CONDITION^^

^^CONDITION: execute, run, command, shell^^
  [[TOOL: exec_command]]
  [[PARAM: command:string:required]]
  [[ENFORCE: Security validation]]
^^/CONDITION^^

Study these patterns. Acknowledge what you've learned.`,
            },
            {
                step: 6,
                phase: 'Examples',
                title: 'Edge Cases and Complex Examples',
                user: `Here are edge cases and complex conversion examples:

EDGE CASE 1 - Multiple tools in one condition:
Input: "Build a data analyst that can read CSVs, process data, and create visualizations"
Output:
[[CRITICAL: You are a data analysis assistant]]

^^CONDITION: read, csv, data, file^^
  [[TOOL: read_file]]
  [[PARAM: path:string:required]]
  [[APPLY: Parse CSV format]]
^^/CONDITION^^

^^CONDITION: process, analyze, transform, filter^^
  [[APPLY: Statistical analysis methods]]
  [[TRANSFORM: raw data TO insights]]
  [[VALIDATE: Data integrity]]
^^/CONDITION^^

^^CONDITION: visualize, chart, plot, graph^^
  [[APPLY: Create visualizations]]
  [[REQUIRE: Clear labels and titles]]
^^/CONDITION^^

EDGE CASE 2 - Nested requirements:
Input: "Create a secure API client that handles authentication"
Output:
[[CRITICAL: You are a secure API client]]

^^CONDITION: authenticate, login, token^^
  [[TOOL: http_request]]
  [[PARAM: url:string:required]]
  [[PARAM: credentials:object:required]]
  [[ENFORCE: HTTPS only]]
  [[VALIDATE: Token format]]
  [[REQUIRE: Secure credential storage]]
^^/CONDITION^^

Note edge case handling strategies. Acknowledge understanding.`,
            },
            {
                step: 7,
                phase: 'Rules',
                title: 'Conversion Rules and Heuristics',
                user: `Here are the conversion rules for mapping natural language to UCL:

RULE 1 - Extract core identity:
- Start with [[CRITICAL: You are X]] where X is the primary role
- Example: "build a chatbot" → [[CRITICAL: You are a chatbot]]

RULE 2 - Map capabilities to conditions:
- Each capability becomes a ^^CONDITION^^ block
- Extract keywords from the capability description
- Example: "can search the web" → ^^CONDITION: search, web^^

RULE 3 - Group related tools:
- Tools with similar activation keywords go in one condition
- Example: read_file, write_file, delete_file all under ^^CONDITION: file^^

RULE 4 - Add quality directives:
- Use [[REQUIRE:]] for output quality
- Use [[ENFORCE:]] for constraints
- Use [[VALIDATE:]] for verification
- Use [[APPLY:]] for domain logic

RULE 5 - Optimize for low overhead:
- Prefer keyword CONDITIONS over SWITCH
- Keep conditions focused (3-5 keywords max)
- Avoid nested conditions

RULE 6 - Parameter inference:
- Required params marked with :required
- Types inferred from context (path:string, count:integer)
- Optional params omit :required

RULE 7 - Anti-patterns to avoid:
- No role-questioning: ^^CONDITION: if answer provided^^
- No unconditional procedures
- No large SWITCH statements (>4 cases)

Summarize these conversion rules in your own words.`,
            },
            {
                step: 8,
                phase: 'Rules',
                title: 'Output Format Requirements',
                user: `Final requirements for UCL output format:

FORMAT REQUIREMENTS:
1. ALWAYS start with [[CRITICAL:]] directive at the very top
2. Use ^^CONDITION: keywords^^ blocks for conditional logic
3. Close all conditions with ^^/CONDITION^^
4. Use proper directive syntax: [[TYPE: content]]
5. Indent content inside conditions for readability
6. Group related tools under shared conditions
7. Add comments (// style) to explain complex logic

QUALITY CRITERIA:
- Valid UCL syntax (no malformed directives)
- Appropriate keyword selection (3-5 per condition)
- Proper parameter typing
- Low structural overhead (O_s)
- Clear, maintainable structure
- No anti-patterns (role-questioning, large SWITCH, etc.)

OUTPUT STRUCTURE:
[[CRITICAL: Primary identity/role]]

^^CONDITION: capability1_keywords^^
  [[TOOL: tool_name]] or [[APPLY: logic]]
  [[PARAM: name:type]]
  [[REQUIRE/ENFORCE/VALIDATE: quality]]
^^/CONDITION^^

^^CONDITION: capability2_keywords^^
  ...
^^/CONDITION^^

Confirm you are ready to convert prompts to UCL format following all rules and format requirements.`,
            },
        ];
    }

    /**
     * Execute warm-up sequence
     * @returns {Promise<Object>} - Conversation history and token count
     */
    async warmUp() {
        console.log('🔥 Starting warm-up sequence...');
        this.conversation = [];
        this.totalTokens = 0;
        this.currentStep = 0;

        for (let i = 0; i < this.warmupMessages.length; i++) {
            const msg = this.warmupMessages[i];
            this.currentStep = i + 1;

            this.onProgress(this.currentStep, this.totalSteps, msg.title);

            try {
                const response = await this.executeStep(msg);
                this.onStepComplete(this.currentStep, msg.title, response);
            } catch (error) {
                this.onError(this.currentStep, msg.title, error);
                throw error;
            }
        }

        console.log(`✅ Warm-up complete! Total tokens: ${this.totalTokens}`);

        return {
            conversation: this.conversation,
            totalTokens: this.totalTokens,
            stepsCompleted: this.currentStep,
        };
    }

    /**
     * Execute a single warm-up step
     */
    async executeStep(message) {
        // Add user message to conversation
        this.conversation.push({
            role: 'user',
            content: message.user,
        });

        // Call API with retry logic
        const response = await this.callAPIWithRetry(this.conversation);

        const assistantResponse = response.choices[0].message.content;

        // Track tokens
        if (response.usage) {
            this.totalTokens = response.usage.total_tokens;
        }

        // Add assistant response to conversation
        this.conversation.push({
            role: 'assistant',
            content: assistantResponse,
        });

        return assistantResponse;
    }

    /**
     * Call Lemonade API with retry logic
     */
    async callAPIWithRetry(messages, attempt = 1) {
        try {
            return await this.callAPI(messages);
        } catch (error) {
            if (attempt < this.maxRetries) {
                console.warn(`⚠️ API call failed (attempt ${attempt}/${this.maxRetries}), retrying...`);
                await this.sleep(1000 * attempt); // Exponential backoff
                return this.callAPIWithRetry(messages, attempt + 1);
            }
            throw error;
        }
    }

    /**
     * Call Lemonade API
     */
    async callAPI(messages) {
        const url = `${this.endpoint}/v1/chat/completions`;
        const headers = {
            'Content-Type': 'application/json',
        };

        if (this.apiKey) {
            headers['Authorization'] = `Bearer ${this.apiKey}`;
        }

        const response = await fetch(url, {
            method: 'POST',
            headers,
            body: JSON.stringify({
                model: this.model,
                messages,
                temperature: this.temperature,
                max_tokens: this.maxTokens,
            }),
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.error?.message || `HTTP ${response.status}: ${response.statusText}`);
        }

        return await response.json();
    }

    /**
     * Generate UCL from test prompt with warm-up context
     * @param {string} testPrompt - Natural language prompt to convert
     * @returns {Promise<string>} - UCL-formatted output
     */
    async generateUCL(testPrompt) {
        if (this.conversation.length === 0) {
            throw new Error('Must call warmUp() before generateUCL()');
        }

        console.log('🧪 Generating UCL from test prompt...');

        // Add test prompt to conversation
        this.conversation.push({
            role: 'user',
            content: testPrompt + '\n\nConvert this to UCL format following all the rules and patterns you learned.',
        });

        const response = await this.callAPIWithRetry(this.conversation);
        const uclOutput = response.choices[0].message.content;

        if (response.usage) {
            this.totalTokens = response.usage.total_tokens;
        }

        console.log(`✅ UCL generation complete! Output length: ${uclOutput.length} chars`);

        return {
            output: uclOutput,
            totalTokens: this.totalTokens,
            outputLength: uclOutput.length,
        };
    }

    /**
     * Sleep utility
     */
    sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    /**
     * Get conversation state
     */
    getState() {
        return {
            currentStep: this.currentStep,
            totalSteps: this.totalSteps,
            totalTokens: this.totalTokens,
            conversationLength: this.conversation.length,
        };
    }
}

// Export for use in other modules
export { WarmUpEngine };
