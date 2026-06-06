# Requirements & Implementation Brief: Mini Attention Notebook

Welcome, learner! This document serves as the **Production-Ready Implementation Brief** for the **Mini Attention Notebook**. It is written from the perspective of a Principal Architect to guide you through building this project step-by-step.

---

## 1. System Scope & Boundaries

To make self-attention intuitive, we must constrain the computation sizes so they are easily visualizable on a standard desktop screen without overwhelming the user with massive, unreadable grids.

| Parameter | Constraint | Rationale / Architectural Boundary |
| :--- | :--- | :--- |
| **Input Sequence Length ($N$)** | $2 \leq N \leq 12$ tokens | Below 2 tokens makes attention trivial (always 1.0). Above 12 tokens causes horizontal scrolling on tables and ruins the spatial comparison of matrices. |
| **Model Dimension ($d_{model}$)** | $\{8, 16, 32, 64, 128\}$ | Standard transformer dimensions. Hardcapped to 128 to prevent memory blowing and slow API responses. Default to 16 for UI density. |
| **Attention Heads ($h$)** | $\{1, 2, 4, 8\}$ | Must satisfy the validation: $d_{model} \pmod h == 0$. If invalid, the UI must display a validation warning and prevent computation. |
| **Numerical Format** | Floating point decimal | Support formatting toggles (e.g., `.3f` or `.4f`) to control precision display on screen. |

---

## 2. API Contract

The system split is simple: the frontend handles state and rendering; the Python/FastAPI backend handles numpy-based mathematical calculations to ensure exact mathematical precision matching PyTorch frameworks.

### Endpoint: `POST /api/attention/compute`

#### Request Payload Schema (TypeScript Interface)

```typescript
interface AttentionComputeRequest {
  text: string;        // The raw input sentence (2 to 12 tokens after tokenization)
  dModel: number;      // Embedding dimension (e.g., 8, 16, 32, 64, 128)
  heads: number;       // Number of attention heads (e.g., 1, 2, 4, 8)
  applyMask: boolean;  // Whether to apply causal masking (upper triangle = -inf)
  seed: number;        // Random seed for reproducibility of projections & toy embeddings
}
```

#### Response Payload Schema (JSON Structure)

```json
{
  "tokens": ["The", "sky", "is", "blue"],
  "dimensions": {
    "seqLen": 4,
    "dModel": 16,
    "heads": 2,
    "dKey": 8
  },
  "matrices": {
    "X": [[0.21, -0.41, 0.33, "..."], "..."],      // Shape: (seqLen, dModel) - Input embeddings
    "WQ": [[[0.01, "..."], "..."], "..."],         // Shape: (heads, dModel, dKey) - Projection Weights
    "WK": [[[0.02, "..."], "..."], "..."],         // Shape: (heads, dModel, dKey)
    "WV": [[[0.03, "..."], "..."], "..."],         // Shape: (heads, dModel, dKey)
    "Q": [[[0.12, "..."], "..."], "..."],          // Shape: (heads, seqLen, dKey) - Projected Queries
    "K": [[[0.23, "..."], "..."], "..."],          // Shape: (heads, seqLen, dKey) - Projected Keys
    "V": [[[-0.31, "..."], "..."], "..."],         // Shape: (heads, seqLen, dKey) - Projected Values
    "rawScores": [[[0.86, "..."], "..."], "..."],   // Shape: (heads, seqLen, seqLen) - Q @ K.T
    "scaledScores": [[[0.30, "..."], "..."], "..."],// Shape: (heads, seqLen, seqLen) - Q @ K.T / sqrt(d_k)
    "maskedScores": [[[0.30, -9e9, "..."], "..."]], // Shape: (heads, seqLen, seqLen) - Causal mask applied
    "attentionWeights": [[[1.0, 0.0, "..."], "..."]],// Shape: (heads, seqLen, seqLen) - Softmax applied
    "contextOutputs": [[[0.21, "..."], "..."], "..."],// Shape: (heads, seqLen, dKey) - Attn @ V
    "mergedOutput": [[-0.31, 0.08, "..."], "..."]   // Shape: (seqLen, dModel) - Concatenated context outputs
  }
}
```

---

## 3. UI/UX Panel Layout Specs

The UI layout is designed as a single-page educational dashboard. It is split into three main areas (matching the provided layout architecture):

```
+--------------------------------------------------------------------------------------------------+
|  Mini Attention Notebook Title                                   [Docs] [Export] [Reset All]     |
+------------------------------------+-------------------------------------------------------------+
| 1. INPUT SEQUENCE                  |  TOKEN EMBEDDINGS (X)      QUERY (Q)   KEY (K)     VALUE (V)    |
|    [Text input box]                |  [Matrix table/grid]       [Matrix]    [Matrix]    [Matrix]     |
| 2. MODEL DIMENSIONS                +-------------------------------------------------------------+
|    d_model: [16]   Heads: [2]      | 5. ATTENTION COMPUTATION STEPS (Wizard Steps 1-5)           |
| 3. INITIALIZATION                  |  [Raw QK^T Heatmap] -> [Scaled Heatmap] -> [Causal Mask]    |
|    Seed: [42]                      |  -> [Softmax Heatmap] -> [Context Vectors V Matrix]         |
| 4. VIEW OPTIONS                    +-------------------------------------------------------------+
|    Causal Mask: [ON/OFF]           | 6. MULTI-HEAD ATTENTION    7. ATTENTION OVERLAY  8. INSPECT |
|    Precision: [.3f]                |  [Head Selector & Weights] | [Average Attn Map] | [Cell info]|
+------------------------------------+-------------------------------------------------------------+
```

### Column 1: Left Control Sidebar
1. **Input Sequence Box**: Input string with validation. Displays sequential token items tagged with index (e.g., `0: The`, `1: sky`).
2. **Model Dimensions**: Dropdowns for `d_model` and `Heads`. Dynamic formula readout showing $d_k = d_{model} / h$ and total weights parameters.
3. **Initialization**: Seed value input to allow generating reproducible random weight matrices.
4. **View Options**: Causal masking toggle and decimal display format configuration (e.g. `.2f`, `.3f`).

### Column 2: Center Workspace (Main Visualizer)
1. **Embeddings & QKV Panels**: Displays $X$, $Q$, $K$, and $V$ matrices in standard grids with color coding for high/low weights.
2. **Attention Wizard Steps**:
   - **Step 1 (Raw Scores)**: Heatmap representing $Q K^T$. Shows metrics: shape, min, max, mean.
   - **Step 2 (Scaled Scores)**: Heatmap representing $Q K^T / \sqrt{d_k}$. Illustrates the shrinkage of magnitude.
   - **Step 3 (Masked Scores)**: Heatmap showing future tokens blanked out (replaced by $-\infty$).
   - **Step 4 (Softmax Weights)**: Heatmap showing the probability distribution of attention.
   - **Step 5 (Output Context)**: Vector table demonstrating final values after multiplying attention weights with $V$.

### Column 3: Multi-Head & Cell Inspector
1. **Multi-Head View**: Visual representation of the projections split by heads. Head selector changes the active step maps in the wizard.
2. **Inspect Cell Panel**: Interactive component. When a cell in any attention map is hovered or clicked, this panel updates to show:
   - Source Token -> Target Token
   - Active Head index
   - Mathematical formula evaluated for that exact cell
   - Plain English translation: *"Token 'sky' attends to 'The' with 62.4% probability."*

---

## 4. Phase-by-Phase Build Order

We recommend implementing the system in these sequential phases to avoid integration bottlenecks:

* **Phase 1: Math Engine Development**
  - Implement standard matrix arithmetic in python.
  - Setup test suites with Pytest using tiny pre-computed PyTorch shapes as golden tests.
* **Phase 2: FastAPI Web Server Setup**
  - Build endpoints matching request/response schemas.
  - Test payload serialization and response time benchmark ($< 300\text{ms}$).
* **Phase 3: Frontend Foundation**
  - Next.js folder initialization.
  - Setup Zustand store to handle active seed, token sizes, active head, and step states.
* **Phase 4: Component Rendering (Tables & Matrices)**
  - Build custom React table grids that adapt to $N \times d_{model}$ layout.
  - Handle hover state communication between grid cells.
* **Phase 5: Heatmap Visualizations**
  - Integrate color-mapped visualizer cells representing weights.
  - Implement Causal Masking visualization step with clean $-\infty$ indicator labels.
* **Phase 6: Multi-Head Selector & Cell Inspector**
  - Wire up dynamic switching between heads.
  - Build interactive explanation engine in the Inspect Cell sidebar.

---

## 5. Acceptance Criteria (QA Checklist)

Before releasing the notebook, ensure you can check off every requirement below:

- [ ] **Boundary Check**: Entering a 1-character string or a sentence > 12 words triggers a clear error message.
- [ ] **Dimension Check**: Selecting $d_{model} = 16$ and $heads = 8$ is allowed; selecting $d_{model} = 16$ and $heads = 5$ fails validation.
- [ ] **Precision Sync**: Changing the seed creates different values, but using the same seed always results in identical floating-point output.
- [ ] **Mathematical Integrity**: Step 4 attention weights for any row must sum to exactly $1.0$ (or $1.0$ up to rounding error) when mask is OFF. When mask is ON, row tokens must only attend to index $\leq$ current index.
- [ ] **Visualization Clarity**: Cells with higher values must have stronger background highlight colors (diverging or sequential color maps).
- [ ] **Performance Threshold**: Fetching and computing token configurations must load in under $300\text{ms}$ on typical local runs.
