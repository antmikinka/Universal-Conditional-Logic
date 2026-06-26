# UCL Linter Rules and Anti-Patterns

The UCL Linter identifies architectural flaws that increase structural overhead ($O_s$) or violate the $S^*$ optimal specification threshold. These rules are implemented in `prompt_analyzer.py` and `core.py`.

## 🚫 Critical Anti-Patterns

### 1. High-Overhead Branching (SWITCH Architecture)
UCL evidence shows that using SWITCH/CASE structures significantly degrades quality due to high procedural overhead.

- **Detection**: Identifies `^^SWITCH:^^` and `^^CASE:^^` blocks.
- **Penalty**: **-0.02** off $Q_{final}$ **per case**.
- **Structural Cost**: $1.0 \times \ln(\text{num\_cases})$.
- **Recommendation**: Convert to **Keyword-based CONDITIONS**.

### 2. Missing CRITICAL Header
The `[[CRITICAL:]]` directive is the single most effective tool for blocking perceived penalty ($P_{perceived}$).

- **Detection**: Checks for absence of `[[CRITICAL:` in a complex prompt.
- **Penalty**: Loss of **+9.3%** quality bonus.
- **Recommendation**: Add `[[CRITICAL:]]` as the first item in the system role.

### 3. Displaced CRITICAL Header
Placement of the CRITICAL directive matters. If it appears late in the prompt, its effectiveness in priming the LLM's attention mechanism is reduced.

- **Detection**: Checks if `[[CRITICAL:` appears after the first 10 lines.
- **Penalty**: Bonus reduced by **30%**.
- **Recommendation**: Move to the very top.

---

## 🟡 Medium-Severity Issues

### 4. Forced (Non-Keyword) Conditions
Conditions that do not target specific keywords are "forced," meaning the LLM must parse them on every turn, increasing $O_s$.

- **Detection**: Identifies `^^CONDITION:^^` blocks that use boolean expressions instead of `CONTAINS "keyword"`.
- **Penalty**: **-0.03** per forced condition.
- **Recommendation**: Use keyword-based conditions to allow the LLM to ignore irrelevant branches.

### 5. Role-Questioning / Meta-Commentary
Instructions that ask the LLM to "think about its role" or "verify its identity" create catastrophic overhead.

- **Detection**: Regex-based detection of phrases like "Are you sure?", "Keep in mind who you are", "Do you understand?".
- **Penalty**: **-0.15** ($Q_{final}$).
- **Recommendation**: Use `{{concept:ai_identity:name}}` assignment instead.

---

## 🟢 Low-Severity / Info

### 6. Low Specification ($S < 0.2$)
The prompt is too vague and does not leverage the linear growth region of the $Q(S)$ curve.

- **Detection**: Calculated when word/element density is minimal.
- **Impact**: Score capped by linear $a \cdot S$ function.
- **Recommendation**: Add more requirements or domain-specific logic.

### 7. Over-Specification ($S > 0.6$)
The prompt has crossed the $S^*$ threshold and has entered the quadratic degradation region.

- **Detection**: Calculated when $O_s$ or line counts exceed safe thresholds.
- **Impact**: Rapid drop in quality due to the Over-Specification Paradox.
- **Recommendation**: Remove redundant instructions; keep only CRITICAL elements.

---

## 🛠️ Detection Logic (Regex Primitives)

The Linter uses the following patterns for identification:

| Pattern | Regex |
|---------|-------|
| **Directive** | `\[\[(\w+):\s*([^\]]+)\]\]` |
| **Condition** | `\^\^CONDITION:\s*([^\^]+)\^\^` |
| **Switch** | `\^\^SWITCH:\s*([^\^]+)\^\^` |
| **Concept** | `\{\{concept:(\w+)(?::(\w+))?\}\}` |
| **Transform** | `\[\[TRANSFORM:\s*([^\]]+)\]\]` |
