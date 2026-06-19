# UCL Prompt Builder 🚀

> **Beautiful dark mode web application for optimizing AI prompts using Universal Conditional Logic (UCL) with OpenRouter and local LLM support**

![Version](https://img.shields.io/badge/version-2.1-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## Features ✨

- **🎨 Stunning Dark Mode UI** - Glassmorphism effects, animated gradients, smooth transitions
- **⚡ UCL Optimization** - Mathematical prompt optimization based on empirical research
- **🤖 Multiple LLM Providers** - Support for OpenRouter (20+ cloud models) AND local LLMs
- **🏠 Local LLM Support** - Lemonade Server, Ollama, LM Studio, and any OpenAI-compatible API
- **📚 Comprehensive Syntax Library** - Complete UCL reference with click-to-insert
- **🔍 Smart Search** - Find UCL patterns instantly
- **💾 Auto-Save Settings** - API key and preferences stored securely in browser
- **⌨️ Keyboard Shortcuts** - Optimize with Ctrl+Enter, Clear with Ctrl+K
- **📱 Responsive Design** - Works on desktop, tablet, and mobile

## Quick Start

### 1. Open the Application

Simply open `index.html` in your browser:

```bash
# Navigate to the UCL builder directory
cd Universal-Conditional-Logic/ucl-builder

# Open in your default browser (Windows)
start index.html

# Or drag and drop index.html into your browser
```

No build process, no dependencies, no npm install required! ✅

### 2. Configure Your LLM Provider

#### Option A: OpenRouter (Cloud)

1. Click the **⚙️ Settings** icon in the top right
2. Select **OpenRouter** as provider
3. Enter your [OpenRouter API key](https://openrouter.ai/keys)
4. Select a model (or use custom model ID)
5. Save settings

#### Option B: Lemonade Server (Local - Free)

1. Install and start [Lemonade Server](https://lemonade-server.ai/):
   ```bash
   # Install Lemonade
   pip install lemonade-server
   
   # Start the server (default port 13305)
   lemonade-server
   ```

2. In UCL Builder:
   - Click **⚙️ Settings**
   - Select **Lemonade** as provider
   - Endpoint: `http://localhost:13305` (default)
   - No API key required!
   - Select a local model (Qwen3, Llama 3.2, Phi-4, etc.)
   - Save settings

#### Option C: Ollama (Local)

1. Install and start [Ollama](https://ollama.com/):
   ```bash
   ollama serve
   ```

2. In UCL Builder:
   - Select **Custom OpenAI-Compatible** as provider
   - Endpoint: `http://localhost:11434`
   - Enter model name (e.g., `llama3.2`, `qwen2.5`)
   - Save settings

#### Option D: LM Studio (Local)

1. Start [LM Studio](https://lmstudio.ai/) and load a model
2. Enable the local server (default port 1234)
3. In UCL Builder:
   - Select **Custom OpenAI-Compatible** as provider
   - Endpoint: `http://localhost:1234`
   - Enter model name
   - Save settings

### 3. Optimize Your Prompts

1. Enter your prompt in the left pane
2. Click **Optimize with UCL** (or press Ctrl+Enter)
3. Get your mathematically optimized prompt in the right pane
4. Copy and use in your favorite LLM!

## Supported Providers

### OpenRouter (Cloud)

20+ models including:
- **Reasoning**: Claude 3.7 Sonnet, OpenAI o1, Gemini 2.0 Flash Thinking
- **Standard**: GPT-4o, Claude 3.5, Gemini Pro 1.5, Qwen 2.5
- **Free**: Llama 3.3 70B, Gemini 2.0 Flash, Phi-4, Hermes 3 405B

### Local LLMs

**Lemonade Server** (10+ models):
- Qwen3 (0.6B, 1.7B, 4B, 8B)
- Llama 3.2 (1B, 3B)
- Phi-3.5 Mini, Phi-4
- DeepSeek R1 Distill (7B, 8B)

**Any OpenAI-Compatible Server**:
- Ollama
- LM Studio
- text-generation-webui
- vLLM
- Custom servers

## How It Works

The UCL Prompt Builder uses the **Universal Conditional Logic Framework** to transform your prompts according to proven optimization principles:

### The Over-Specification Paradox

Research shows (N=305, p < 2.89×10⁻¹³) that **more detailed prompts can actually degrade AI performance**. There's an optimal threshold:

```
Q(S) = a×S              for S ≤ S* (linear growth)
Q(S) = Q_max - b(S-S*)²  for S > S* (quadratic degradation)
```

Where **S* ≈ 0.509** is the sweet spot.

### Key Optimizations Applied

1. **[[CRITICAL:]] Directive** - Added at top for +9.3% quality bonus
2. **Keyword-based CONDITIONS** - Lower structural overhead than SWITCH
3. **Proper UCL Syntax** - Concepts, Directives, Transforms, Sections
4. **Anti-pattern Removal** - Eliminates role-questioning, excessive branching
5. **Domain Logic** - Wraps specialized knowledge in conditional blocks

## UCL Syntax Reference

### Core Elements

| Element | Syntax | Purpose |
|---------|--------|---------|
| **Concept** | `{{concept:name:attribute}}` | Variable reference |
| **Directive** | `[[TYPE: content]]` | Core instruction |
| **Condition** | `^^CONDITION: keywords^^ ... ^^/CONDITION^^` | Conditional logic |
| **Section** | `<name>...</name>` | Organized content block |
| **Transform** | `[[TRANSFORM: source TO target]]` | Conversion rule |

### Directives (16 Types)

- **[[CRITICAL:]]** ⭐ - Highest priority (+9.3% quality)
- **[[REQUIRE:]]** - Specify requirements
- **[[ENFORCE:]]** - Strict enforcement
- **[[VALIDATE:]]** - Validation rules
- **[[APPLY:]]** - Apply logic/rules
- **[[TRANSFORM:]]** - Transformation
- And 10 more...

See the sidebar in the app for complete reference!

## Examples

### Before UCL Optimization

```
Write a Python function to calculate the factorial of a number.
```

### After UCL Optimization

```
[[CRITICAL: Return ONLY Python code with no explanations]]

<output_format>
  [[REQUIRE: Language: Python]]
  [[ENFORCE: PEP 8 compliance]]
  [[VALIDATE: Function signature correctness]]
</output_format>

^^CONDITION: factorial, recursive^^
  [[APPLY: Recursive implementation with base case]]
^^/CONDITION^^

^^CONDITION: factorial, iterative^^
  [[APPLY: Iterative loop-based approach]]
^^/CONDITION^^

[[REQUIRE: Include docstring]]
[[REQUIRE: Type hints]]
```

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl + Enter` | Optimize with UCL |
| `Ctrl + K` | Clear all |
| Click syntax item | Insert into prompt |

## Technical Details

### Stack
- **Frontend**: Pure HTML5, CSS3, Vanilla JavaScript
- **No frameworks** - Zero dependencies
- **No build process** - Works directly in browser
- **API**: OpenRouter (OpenAI-compatible) or any local OpenAI-compatible server

### Files
```
ucl-builder/
├── index.html      # Main application structure
├── styles.css      # Dark mode design system
├── app.js          # Application logic
├── ucl-data.js     # UCL syntax reference
├── models.js       # Model configurations
└── README.md       # This file
```

### Browser Compatibility
- ✅ Chrome/Edge (recommended)
- ✅ Firefox
- ✅ Safari
- ✅ Modern mobile browsers

## Privacy & Security

- **Local-first** - Works entirely in your browser
- **API keys stored locally** - Never sent to any server except your chosen provider
- **Local LLMs** - Zero data leaves your machine when using Lemonade/Ollama/LM Studio
- **No tracking** - No analytics or telemetry
- **No backend** - Runs entirely in your browser
- **Open source** - Inspect all code

## Research Foundation

Based on the paper:
> "The Over-Specification Paradox: A Mathematical Framework for Prompt Optimization in Large Language Models"
> by Anthony Mikinka, 2025

Key findings:
- Sample size: N = 305
- Statistical significance: p < 2.89×10⁻¹³
- Effect size: Cohen's d = 1.15 (large)
- Models tested: 11 different LLMs

## Tips for Best Results

1. ✅ **Always let UCL add [[CRITICAL:]]** - Don't remove it!
2. ✅ **Be specific about output format** - JSON, code, structured data
3. ✅ **Use keyword conditions** - Better than complex logic
4. ✅ **Test with different models** - Reasoning models work best
5. ✅ **Try local LLMs** - Free, private, and surprisingly effective
6. ❌ **Don't over-specify** - Trust the optimization

## Troubleshooting

### OpenRouter Issues
- Make sure you have credits on OpenRouter
- Check key format: should start with `sk-or-v1-`
- Try refreshing the page

### Local LLM Issues

**Lemonade Server:**
- Ensure Lemonade is running: `lemonade-server`
- Check endpoint: `http://localhost:13305`
- Verify model is loaded in Lemonade

**Ollama:**
- Ensure Ollama is running: `ollama serve`
- Pull a model first: `ollama pull llama3.2`
- Check endpoint: `http://localhost:11434`

**LM Studio:**
- Enable local server in LM Studio settings
- Load a model
- Check endpoint: `http://localhost:1234`

**General:**
- Check CORS settings if using custom server
- Verify server supports `/v1/chat/completions` endpoint
- Look at browser console (F12) for errors

### Model Not Working
- Some cloud models require payment
- Free cloud models have rate limits
- Local models require sufficient RAM/VRAM
- Try a different model from the dropdown

### Optimization Failed
- Check your internet connection (for cloud)
- Verify server is running (for local)
- Verify API key is valid (if required)
- Look at browser console (F12) for errors

## Contributing

Found a bug or have a feature request? This is part of the UCL Framework project.

## License

MIT License - See main UCL Framework repository

## Author

**Anthony Mikinka**

Part of the [Universal Conditional Logic Framework](https://github.com/antmikinka/Universal-Conditional-Logic)

---

**Made with ❤️ and mathematics** ⚡
