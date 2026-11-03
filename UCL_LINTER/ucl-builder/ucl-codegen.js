/**
 * Builds a host-side checker from an OpenAI tools list or an MCP inputSchema list.
 * The result is source text. This page does not run it, and the alias table was
 * not part of the 408-cell tool run.
 */
const UCLCodegen = (() => {
    const COMMON_ALIASES = {
        path: ["file_path", "filepath", "file", "filename", "dir", "directory", "target_path", "folder"],
        file_path: ["path", "filepath", "file", "filename"],
        query: ["q", "search_query", "pattern", "search", "keyword", "term"],
        pattern: ["query", "regex", "search_pattern", "glob", "match"],
        command: ["cmd", "shell_cmd", "script", "bash", "exec"],
        content: ["text", "body", "data", "contents", "value"],
        url: ["uri", "endpoint", "href", "link"],
        repo: ["repository", "repo_name", "project"],
        owner: ["user", "org", "organization", "username"],
        sql: ["query", "statement", "sql_query"]
    };

    function normalizeSchemas(parsed) {
        const rawList = Array.isArray(parsed)
            ? parsed
            : (parsed && (parsed.tools || parsed.functions)) || [parsed];

        return rawList.map((item) => {
            const fn = (item && (item.function || item)) || {};
            const name = String(fn.name || "unnamed_tool");
            const schema = fn.parameters || fn.inputSchema || fn.input_schema || {};
            const props = schema.properties || {};
            const required = Array.isArray(schema.required) ? schema.required.map(String) : [];
            const cleanProps = {};
            const aliases = {};

            for (const [propName, propSpec] of Object.entries(props)) {
                const spec = propSpec && typeof propSpec === "object" ? propSpec : {};
                const rawType = Array.isArray(spec.type) ? spec.type[0] : spec.type;
                const entry = { type: rawType || "string" };
                if (Array.isArray(spec.enum)) entry.enum = spec.enum;
                if (spec.default !== undefined) entry.default = spec.default;
                cleanProps[propName] = entry;

                const candidates = COMMON_ALIASES[propName] || [];
                for (const alias of candidates) {
                    if (!Object.prototype.hasOwnProperty.call(props, alias) && !Object.prototype.hasOwnProperty.call(aliases, alias)) {
                        aliases[alias] = propName;
                    }
                }
            }

            return { name, required, properties: cleanProps, aliases };
        }).filter((tool) => tool.name);
    }

    function toPyLiteral(val, indent) {
        const pad = " ".repeat(indent || 0);
        const nextPad = " ".repeat((indent || 0) + 4);
        if (val === null || val === undefined) return "None";
        if (typeof val === "boolean") return val ? "True" : "False";
        if (typeof val === "number") return String(val);
        if (typeof val === "string") return JSON.stringify(val);
        if (Array.isArray(val)) {
            if (val.length === 0) return "[]";
            return "[" + val.map((item) => toPyLiteral(item, indent)).join(", ") + "]";
        }
        if (typeof val === "object") {
            const entries = Object.entries(val);
            if (entries.length === 0) return "{}";
            const lines = entries.map(([key, value]) => nextPad + JSON.stringify(key) + ": " + toPyLiteral(value, (indent || 0) + 4));
            return "{\n" + lines.join(",\n") + "\n" + pad + "}";
        }
        return "None";
    }

    function rulebook(parsed) {
        const book = {};
        for (const spec of normalizeSchemas(parsed)) {
            book[spec.name] = {
                required: spec.required,
                properties: spec.properties,
                aliases: spec.aliases
            };
        }
        return book;
    }

    function generatePython(parsed) {
        const specs = normalizeSchemas(parsed);
        const header = [
            "# Host rulebook generated from " + specs.length + " tool schema(s).",
            "# Keep this on the host. The name list is what the model reads.",
            "# Alias keys are a fixed convenience list. They were not part of the 408-cell run.",
            "from typing import Any, Dict, Tuple",
            "",
            "TOOL_RULEBOOK: Dict[str, Dict[str, Any]] = " + toPyLiteral(rulebook(parsed), 0),
            "",
            "TYPE_MAP = {",
            '    "string": str,',
            '    "integer": int,',
            '    "number": (int, float),',
            '    "boolean": bool,',
            '    "array": list,',
            '    "object": dict,',
            "}",
            "",
            "",
            "def validate_and_coerce(call: Dict[str, Any]) -> Tuple[bool, Dict[str, Any], str]:",
            '    """Check one tool call against this rulebook.',
            "",
            "    Returns (ok, normalized_call, error). On a missing required field the",
            "    error includes only that tool's property map.",
            '    """',
            '    name = call.get("name")',
            '    raw_args = call.get("arguments") or {}',
            "    if not isinstance(raw_args, dict):",
            '        return False, call, f"{name} arguments must be an object"',
            "",
            "    if name not in TOOL_RULEBOOK:",
            '        return False, call, f"Unknown tool {name!r}. Allowed: {list(TOOL_RULEBOOK)}"',
            "",
            "    spec = TOOL_RULEBOOK[name]",
            '    props = spec["properties"]',
            '    aliases = spec["aliases"]',
            "    coerced: Dict[str, Any] = {}",
            "",
            "    for key, value in raw_args.items():",
            "        canonical = aliases.get(key, key)",
            "        if canonical not in coerced or key in props:",
            "            coerced[canonical] = value",
            "",
            "    for prop, prop_spec in props.items():",
            '        if prop not in coerced and "default" in prop_spec:',
            '            coerced[prop] = prop_spec["default"]',
            "",
            '    missing = [field for field in spec["required"] if field not in coerced]',
            "    if missing:",
            '        return False, call, f"{name} missing required arg(s): {missing}. Schema: {props}"',
            "",
            "    for key, value in list(coerced.items()):",
            "        if key not in props:",
            "            continue",
            "        prop_spec = props[key]",
            '        expected = prop_spec.get("type", "string")',
            "        py_type = TYPE_MAP.get(expected, str)",
            "",
            '        if expected == "integer" and isinstance(value, str) and value.lstrip("-").isdigit() and value.count("-") <= 1:',
            "            coerced[key] = int(value)",
            '        elif expected == "number" and isinstance(value, str):',
            "            try:",
            "                coerced[key] = float(value)",
            "            except ValueError:",
            "                pass",
            '        elif expected == "boolean" and isinstance(value, str) and value.lower() in ("true", "false"):',
            '            coerced[key] = value.lower() == "true"',
            "",
            "        current = coerced[key]",
            "        if isinstance(current, bool) and expected in (\"integer\", \"number\"):",
            '            return False, call, f"{name}.{key} must be {expected}, got bool"',
            "        if not isinstance(current, py_type):",
            '            return False, call, f"{name}.{key} must be {expected}, got {type(current).__name__}"',
            '        if "enum" in prop_spec and current not in prop_spec["enum"]:',
            '            return False, call, f"{name}.{key} must be one of {prop_spec[\'enum\']}"',
            "",
            '    return True, {"name": name, "arguments": coerced}, ""',
            ""
        ];
        return header.join("\n");
    }

    function generateTypeScript(parsed) {
        const specs = normalizeSchemas(parsed);
        const jsonRulebook = JSON.stringify(rulebook(parsed), null, 2);
        return [
            "// Host rulebook generated from " + specs.length + " tool schema(s).",
            "// Keep this on the host. The name list is what the model reads.",
            "// Alias keys are a fixed convenience list. They were not part of the 408-cell run.",
            "",
            "export interface ToolCall {",
            "  name: string;",
            "  arguments?: Record<string, unknown>;",
            "}",
            "",
            "export interface ValidationResult {",
            "  ok: boolean;",
            "  call: ToolCall;",
            "  error?: string;",
            "}",
            "",
            "const TOOL_RULEBOOK: Record<string, {",
            "  required: string[];",
            "  properties: Record<string, { type: string; enum?: unknown[]; default?: unknown }>;",
            "  aliases: Record<string, string>;",
            "}> = " + jsonRulebook + ";",
            "",
            "export function validateAndCoerce(call: ToolCall): ValidationResult {",
            "  const spec = TOOL_RULEBOOK[call.name];",
            "  if (!spec) {",
            "    return {",
            "      ok: false,",
            "      call,",
            "      error: \"Unknown tool '\" + call.name + \"'. Allowed: \" + Object.keys(TOOL_RULEBOOK).join(\", \"),",
            "    };",
            "  }",
            "",
            "  const rawArgs = call.arguments ?? {};",
            "  if (typeof rawArgs !== \"object\" || rawArgs === null || Array.isArray(rawArgs)) {",
            "    return { ok: false, call, error: call.name + \" arguments must be an object\" };",
            "  }",
            "  const coerced: Record<string, unknown> = {};",
            "",
            "  for (const [key, val] of Object.entries(rawArgs)) {",
            "    const canonical = spec.aliases[key] ?? key;",
            "    if (!(canonical in coerced) || key in spec.properties) {",
            "      coerced[canonical] = val;",
            "    }",
            "  }",
            "",
            "  for (const [prop, propSpec] of Object.entries(spec.properties)) {",
            "    if (!(prop in coerced) && propSpec.default !== undefined) {",
            "      coerced[prop] = propSpec.default;",
            "    }",
            "  }",
            "",
            "  const missing = spec.required.filter((req) => !(req in coerced));",
            "  if (missing.length > 0) {",
            "    return {",
            "      ok: false,",
            "      call,",
            "      error: call.name + \" missing required arg(s): [\" + missing.join(\", \") + \"]. Schema: \" + JSON.stringify(spec.properties),",
            "    };",
            "  }",
            "",
            "  for (const [key, val] of Object.entries(coerced)) {",
            "    const propSpec = spec.properties[key];",
            "    if (!propSpec) continue;",
            "",
            "    if (propSpec.type === \"integer\" && typeof val === \"string\" && /^-?\\d+$/.test(val)) {",
            "      coerced[key] = parseInt(val, 10);",
            "    } else if (propSpec.type === \"number\" && typeof val === \"string\" && /^-?\\d+(\\.\\d+)?$/.test(val)) {",
            "      coerced[key] = Number(val);",
            "    } else if (propSpec.type === \"boolean\" && typeof val === \"string\" && /^(true|false)$/i.test(val)) {",
            "      coerced[key] = val.toLowerCase() === \"true\";",
            "    }",
            "",
            "    const actual = coerced[key];",
            "    const typeOk =",
            "      (propSpec.type === \"string\" && typeof actual === \"string\") ||",
            "      (propSpec.type === \"integer\" && typeof actual === \"number\" && Number.isInteger(actual)) ||",
            "      (propSpec.type === \"number\" && typeof actual === \"number\" && !Number.isNaN(actual)) ||",
            "      (propSpec.type === \"boolean\" && typeof actual === \"boolean\") ||",
            "      (propSpec.type === \"array\" && Array.isArray(actual)) ||",
            "      (propSpec.type === \"object\" && typeof actual === \"object\" && actual !== null && !Array.isArray(actual));",
            "",
            "    if (!typeOk) {",
            "      return { ok: false, call, error: call.name + \".\" + key + \" expected \" + propSpec.type + \", got \" + typeof actual };",
            "    }",
            "    if (propSpec.enum && !propSpec.enum.includes(actual)) {",
            "      return { ok: false, call, error: call.name + \".\" + key + \" must be one of [\" + propSpec.enum.join(\", \") + \"]\" };",
            "    }",
            "  }",
            "",
            "  return { ok: true, call: { name: call.name, arguments: coerced } };",
            "}",
            ""
        ].join("\n");
    }

    return { generatePython, generateTypeScript, normalizeSchemas };
})();
