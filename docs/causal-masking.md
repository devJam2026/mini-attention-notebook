# Causal Masking: Why GPT Cannot See the Future

In GPT-style (decoder-only) language models, we use **Causal Masking**. This document explains why we mask, how it works mathematically, and how it is implemented.

---

## 1. The Core Problem: Autoregressive "Cheating"

Decoder-only models are trained to predict the **next token** in a sequence given the previous ones.
Suppose the training sentence is:
`"The sky is blue"`

When predicting the word at index $i$, the model should only look at tokens $0$ to $i$.
* To predict `"sky"`, the input is `["The"]`.
* To predict `"is"`, the input is `["The", "sky"]`.
* To predict `"blue"`, the input is `["The", "sky", "is"]`.

If we do not apply a mask, the attention step allows token $1$ (`"sky"`) to attend to token $2$ (`"is"`) and token $3$ (`"blue"`). During training, the model would simply "look ahead" and copy the future tokens rather than learning to predict them. This is known as **information leakage** or **cheating**.

---

## 2. The Lower Triangular Mask Matrix

To prevent look-ahead, we block attention from any token at row $i$ to any token at column $j$ where $j > i$.
This creates a **lower triangular structure**:

```
Row (i) attends to Column (j)

       The   sky   is    blue
The   [ OK    X     X     X  ]  <- Token 0 can only attend to Token 0 ("The")
sky   [ OK   OK     X     X  ]  <- Token 1 can attend to 0 and 1 ("The", "sky")
is    [ OK   OK    OK     X  ]  <- Token 2 can attend to 0, 1, and 2
blue  [ OK   OK    OK    OK  ]  <- Token 3 can attend to all tokens
```

* `OK` = Valid Attention
* `X` = Masked Out (Attention Blocked)

---

## 3. The $-\infty$ Masking Trick

How do we implement this "X" in a differentiable, matrix-multiplication-friendly way? We use negative infinity ($-\infty$).

### Step 1: Generate the Causal Mask Matrix ($M$)
We define a mask matrix $M$ of size $N \times N$:
$$M_{i,j} = \begin{cases} 
0 & \text{if } j \leq i \\
-\infty & \text{if } j > i 
\end{cases}$$

For a 4-token sequence:
$$M = \begin{pmatrix}
0 & -\infty & -\infty & -\infty \\
0 & 0 & -\infty & -\infty \\
0 & 0 & 0 & -\infty \\
0 & 0 & 0 & 0
\end{pmatrix}$$

*(Note: In practice, we use a large negative value like $-1e9$ or $-1e15$ because float systems cannot represent true $-\infty$ without math errors).*

### Step 2: Add Mask to Scaled Scores
We add the mask matrix directly to the scaled attention scores:
$$\text{Masked Scores} = \frac{QK^T}{\sqrt{d_k}} + M$$

If a scaled score was $0.85$, adding $-\infty$ makes it $-\infty$.

### Step 3: Run Softmax
We apply the Softmax function:
$$\text{Attention Weights} = \text{Softmax}\left(\text{Masked Scores}\right)$$

Since $e^{-\infty} = 0$:
* The probability of attending to future tokens becomes exactly **$0.0$**.
* The remaining attention weights are automatically re-scaled to sum to $1.0$ among the past and current tokens.

---

## 4. Comparing Mask Off vs. Mask On

Here is what the attention weights look like in the visualizer when you toggle the causal mask:

### Mask OFF (Bidirectional - BERT style)
Tokens can attend in all directions. Useful for understanding full sentences after they are written (e.g. classification or translation encoders).
```
       The   sky   is   blue
The   [0.40, 0.20, 0.10, 0.30]
sky   [0.15, 0.50, 0.15, 0.20]
is    [0.10, 0.20, 0.60, 0.10]
blue  [0.25, 0.15, 0.10, 0.50]
```

### Mask ON (Causal - GPT style)
Tokens can only attend to themselves and their predecessors.
```
       The   sky   is   blue
The   [1.00, 0.00, 0.00, 0.00]
sky   [0.42, 0.58, 0.00, 0.00]
is    [0.33, 0.35, 0.32, 0.00]
blue  [0.28, 0.21, 0.23, 0.28]
```
*(Observe that the upper-right corner elements are forced to exactly `0.00`)*.
