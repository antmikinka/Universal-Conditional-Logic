# UCL Prompt Builder 🚀

> **Beautiful dark mode web application for optimizing AI prompts using Universal Conditional Logic (UCL) with OpenRouter integration**

![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## Features ✨

- **🎨 Stunning Dark Mode UI** - Glassmorphism effects, animated gradients, smooth transitions
- **⚡ UCL Optimization** - Mathematical prompt optimization based on empirical research
- **🤖 OpenRouter Integration** - Support for 20+ LLM models (Claude, GPT-4, Gemini, etc.)
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

### 2. Configure Your API Key

1. Click the **⚙️ Settings** icon in the top right
2. Enter your [OpenRouter API key](https://openrouter.ai/keys)
3. Select a default model (or use custom model ID)
4. Save settings

### 3. Optimize Your Prompts

1. Enter your prompt in the left pane
2. Click **Optimize with UCL** (or press Ctrl+Enter)
3. Get your mathematically optimized prompt in the right pane
4. Copy and use in your favorite LLM!

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

## Supported Models

The builder supports 20+ OpenRouter models across categories:

### Reasoning Models (Best for UCL)
- Claude 3.7 Sonnet
- OpenAI o1 / o1-mini
- DeepSeek Chat
- Gemini 2.0 Flash Thinking

### Standard Models
- GPT-4o / GPT-4o Mini
- Claude 3.5 Sonnet / Haiku
- Gemini Pro 1.5
- Qwen 2.5 72B

### Free Models
- Llama 3.3 70B
- Gemini 2.0 Flash
- Phi-4
- Hermes 3 405B

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
- **API**: OpenRouter (OpenAI-compatible)

### Files
```
ucl-builder/
├── index.html      # Main application structure
├── styles.css      # Dark mode design system
├── app.js          # Application logic
├── ucl-data.js     # UCL syntax reference
├── models.js       # OpenRouter model configs
└── README.md       # This file
```

### Browser Compatibility
- ✅ Chrome/Edge (recommended)
- ✅ Firefox
- ✅ Safari
- ✅ Modern mobile browsers

## Privacy & Security

- **API keys stored locally** - Never sent to any server except OpenRouter
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
5. ❌ **Don't over-specify** - Trust the optimization

## Troubleshooting

### API Key Issues
- Make sure you have credits on OpenRouter
- Check key format: should start with `sk-or-v1-`
- Try refreshing the page

### Model Not Working
- Some models require payment
- Free models have rate limits
- Try a different model from the dropdown

### Optimization Failed
- Check your internet connection
- Verify API key is valid
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
