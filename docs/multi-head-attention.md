# Multi-Head Attention: Exploring Multiple Relationships

In standard Self-Attention, a single query matrix searches a single key matrix. In modern transformers, we use **Multi-Head Attention (MHA)**. This guide explains why MHA is superior and how the math divides dimensions across multiple heads.

---

## 1. Why Multiple Heads?

If a model only has **one head**, it can only attend to the sequence in a single way. For example:
- In the sentence `"The cat sat on the mat because it was tired"`, does the word `"it"` refer to `"cat"` or `"mat"`?
- A single head has to make a single compromise, distributing its attention weights.

With **Multi-Head Attention**, the model can split the work:
* **Head 1** can learn to focus on **syntax** (e.g., matching verbs to their subjects).
* **Head 2** can learn to focus on **coreference resolution** (e.g., mapping `"it"` to `"cat"`).
* **Head 3** can learn to focus on **locality** (e.g., attending to immediately adjacent words).

By having multiple heads, the model learns different relationships in parallel across different **representation subspaces**.

---

## 2. The Dimension Splitting Math

To keep the model computationally efficient, we do not simply replicate the single-head calculations $h$ times (which would increase computational cost by a factor of $h$).

Instead, we **split** the embedding dimension $d_{model}$ among the heads.

### The Dimension Constraint
The number of heads ($h$) must divide the model dimension ($d_{model}$) with no remainder:
$$d_{model} \pmod h == 0$$

The dimension of each individual head ($d_k$) is:
$$d_k = \frac{d_{model}}{h}$$

### Numerical Example
Let's look at the default settings from the visualizer dashboard:
* $d_{model} = 16$
* $h = 2$ heads
* Head dimension: $d_k = 16 / 2 = 8$

Instead of projecting the input $X$ ($4 \times 16$) to $Q$ of size $4 \times 16$, we project it using separate weights for each head to $Q_1$ and $Q_2$ of shape $4 \times 8$:

```
                             Input X (N x d_model)
                                  /         \
                                 /           \
                 Proj Weights Head 1       Proj Weights Head 2
                  (d_model x d_k)           (d_model x d_k)
                     (16 x 8)                  (16 x 8)
                       /                         \
                      v                           v
                 Q1 (N x d_k)                Q2 (N x d_k)
                    (4 x 8)                     (4 x 8)
```

We do the same for Key ($K$) and Value ($V$) projections.

---

## 3. Concatenation and the Output Projection ($W_O$)

After calculating the attention outputs for each head independently:

$$\text{head}_i = \text{Attention}(Q_i, K_i, V_i) \quad (\text{Shape: } N \times d_k)$$

We need to merge them back into a single matrix of the original model dimension $d_{model}$ to pass to the next block (like a feed-forward layer).

### Step 1: Concatenate the Heads
We glue the head matrices together horizontally:
$$\text{Concat} = [\text{head}_1, \text{head}_2, \dots, \text{head}_h] \quad (\text{Shape: } N \times (h \cdot d_k))$$

Since $h \cdot d_k = d_{model}$, the shape of $\text{Concat}$ is exactly $(N \times d_{model})$.

### Step 2: Apply the Output Projection ($W_O$)
To allow information to mix across different heads, we multiply the concatenated matrix by a final learnable projection weight matrix $W_O$ of shape $(d_{model} \times d_{model})$:

$$\text{MultiHeadOutput} = \text{Concat} \cdot W_O \quad (\text{Shape: } N \times d_{model})$$

---

## 4. Summary of Multi-Head Dimensions

For sequence length $N$, model dimension $d_{model}$, and number of heads $h$:

| Matrix | PyTorch / NumPy Shape | Role in the Network |
| :--- | :--- | :--- |
| **Input $X$** | $(N \times d_{model})$ | Embedded sequence representation |
| **Weights $W_Q, W_K, W_V$** | $(h \times d_{model} \times d_k)$ | Projections parameters per head |
| **Queries $Q$ / Keys $K$** | $(h \times N \times d_k)$ | Split projections for matching |
| **Values $V$** | $(h \times N \times d_k)$ | Split projections for context values |
| **Attention Weights $A$** | $(h \times N \times N)$ | Attention probability map per head |
| **Merged Output** | $(N \times d_{model})$ | Re-assembled multi-head output |
