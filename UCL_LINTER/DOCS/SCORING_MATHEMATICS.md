# The Mathematics of UCL Scoring

The Universal Conditional Logic (UCL) framework is governed by a mathematical model derived from empirical testing (N=305). This model predicts AI response quality ($Q$) based on architectural complexity and specification density.

## 🧠 The Over-Specification Paradox

The core finding of UCL research is that prompt quality does not increase linearly with detail. Instead, it follows an "Inverted-U" curve.

### The Quality Function: $Q(S)$

The quality $Q$ of a prompt at specification level $S$ is defined as:

$$
Q(S) = 
\begin{cases} 
a \cdot S & \text{for } S \le S^* \\
Q_{max} - b(S - S^*)^2 & \text{for } S > S^* 
\end{cases}
$$

Where:
- **$S^*$ (0.509)**: The Optimal Specification Threshold.
- **$Q_{max}$ (1.0)**: Maximum theoretical quality.
- **$a$ (1.96)**: Linear growth coefficient.
- **$b$ (4.0)**: Quadratic degradation rate (The Penalty of Over-Specification).

---

## 🏗️ Structural Overhead ($O_s$)

Structural Overhead measures the "clutter" of a prompt—the procedural instructions that take up context without providing new information.

### Calculation Formula:

$$O_s(A) = \sum_{i=1}^{n} w_i(T_i) + \gamma \ln(C)$$

Where:
- **$w_i(T_i)$**: The weight of directive type $T_i$.
- **$\gamma$ (1.0)**: The Branching Complexity Weight.
- **$C$**: The number of branches (cases) in a SWITCH statement.

### Element Weights ($w_i$):

| Element Type | $O_s$ Impact | Rationale |
|--------------|--------------|-----------|
| **Directive: CRITICAL** | **-0.05** | **Negative Overhead!** Reduces perceived complexity. |
| **Concept** | +0.10 | Minimal identity overhead. |
| **Directive: REQUIRE** | +0.50 | Hard constraints increase parsing load. |
| **Condition (Keyword)** | +0.50 | Efficient branching (only parses body when active). |
| **Condition (Forced)** | +2.00 | Inefficient; always evaluated. |
| **Transform** | +1.00 | High procedural overhead for mapping logic. |

---

## 📈 Specification Level ($S$)

The specification level $S$ is an estimate of how "full" the prompt's instruction set is. In the UCL Builder, it is estimated using:

$$S \approx (0.3 \cdot \text{ElementFactor} + 0.4 \cdot O_s \text{Factor} + 0.3 \cdot \text{LineFactor})$$

- **ElementFactor**: Density of UCL primitives.
- **$O_s$ Factor**: Normalized structural complexity.
- **LineFactor**: Literal length of the prompt.

---

## 🏆 Final Quality Calculation

The final quality $Q_{final}$ combines the base quality with discrete bonuses and penalties:

$$Q_{final} = \text{clamp}(0, 1, Q_{base} - \sum \text{Penalties} + \sum \text{Bonuses})$$

### Empirical Bonuses/Penalties:

| Feature | Impact | Rule |
|---------|--------|------|
| **CRITICAL Bonus** | **+0.093** | Added if `[[CRITICAL:]]` is at the very top. |
| **Keyword Bonus** | **+0.050** | Awarded for using keyword-based CONDITIONS. |
| **Role Penalty** | **-0.150** | Deducted for role-questioning instructions. |
| **Switch Penalty** | **-0.020** | Deducted **per case** in a SWITCH/CASE block. |

---

## 📊 Interpretation of Scores

| Grade | $Q_{final}$ | Interpretation |
|-------|-------------|----------------|
| **A** | 95% - 100% | Mathematically optimal; minimal overhead. |
| **B** | 85% - 94% | Highly effective; slight overhead or sub-optimal $S$. |
| **C** | 70% - 84% | Over-specified or inefficient architecture. |
| **D** | 50% - 69% | Critical architectural flaws (Role questioning, high $O_s$). |
| **F** | < 50% | Catastrophic failure (Structural Paradox exceeded). |
