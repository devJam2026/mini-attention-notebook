# Self-Attention Mechanics: Q, K, and V Explained

In this guide, we dive deep into the core equation of the modern Transformer:

$$\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

We will explore what these matrices represent and walk through the step-by-step attention pipeline.

---

## 1. The Analogy: The Database Retrieval System

To understand Query ($Q$), Key ($K$), and Value ($V$), think of a database search:

* **Query ($Q$)**: The search term you type into the database. (What you are looking for).
* **Key ($K$)**: The indexing system of the database (e.g., titles, tags, categories). What each record is registered under.
* **Value ($V$)**: The actual content of the database record. What gets returned to you when a key matches your query.

In self-attention, **every token in the sentence plays all three roles simultaneously**. Each token searches for other tokens ($Q$), registers its own features ($K$), and offers its contextual content ($V$).

---

## 2. From Input ($X$) to Projections ($Q, K, V$)

Let's start with our input token embeddings $X$ of shape $(N \times d_{model})$.
To extract Query, Key, and Value representations, we pass $X$ through three separate linear projection layers:

$$Q = X W_Q \quad (\text{Shape: } N \times d_k)$$
$$K = X W_K \quad (\text{Shape: } N \times d_k)$$
$$V = X W_V \quad (\text{Shape: } N \times d_k)$$

* $W_Q, W_K, W_V$ are weight matrices of shape $(d_{model} \times d_k)$.
* $d_k$ is the projection dimension (for single head, $d_k = d_{model}$; for multi-head, $d_k = d_{model} / h$).

```
        X (Input Matrix)
           (N x d_model)
             /   |   \
            /    |    \
           /     |     \
     x W_Q      x W_K     x W_V
     (d_model x d_k)   (d_model x d_k)   (d_model x d_k)
       /         |         \
      v          v          v
  Q (Query)   K (Key)    V (Value)
  (N x d_k)   (N x d_k)  (N x d_k)
```

---

## 3. Step-by-Step Attention Computation Pipeline

Once we have $Q, K, V$, we run the 5-step attention process shown in the wizard:

### Step 1: Compute Raw Scores ($Q K^T$)
We compute the raw dot product of all queries with all keys:
$$\text{Raw Scores} = Q K^T \quad (\text{Shape: } N \times N)$$

* Cell $(i, j)$ in this matrix represents how similar token $i$'s Query is to token $j$'s Key.

### Step 2: Scaling by $\sqrt{d_k}$
We divide all scores by the square root of the key dimension:
$$\text{Scaled Scores} = \frac{QK^T}{\sqrt{d_k}}$$

* **Why?** If $d_k$ is large, dot products grow large in magnitude. This pushes the Softmax function into regions with extremely small gradients. Dividing by $\sqrt{d_k}$ stabilizes variance.

### Step 3: Causal Masking (Optional)
If causal masking is enabled (e.g., in autoregressive models like GPT), we replace all values in the upper-right triangle of the matrix with $-\infty$:
$$\text{Masked Scores}_{i,j} = \begin{cases} 
\text{Scaled Scores}_{i,j} & \text{if } j \leq i \\
-\infty & \text{if } j > i 
\end{cases}$$

### Step 4: Apply Softmax to Get Attention Weights
We apply the Softmax function row-wise:
$$A = \text{Softmax}\left(\text{Scores}\right)$$
$$A_{i,j} = \frac{e^{S_{i,j}}}{\sum_{k=1}^N e^{S_{i,k}}}$$

* Since $e^{-\infty} = 0$, masked cells become exactly $0.0$ in the probability matrix.
* Every row in this attention weights matrix sums to exactly $1.0$.

### Step 5: Compute Final Context Output ($A V$)
We multiply the attention weights matrix $A$ by the Value matrix $V$:
$$\text{Output} = A V \quad (\text{Shape: } N \times d_k)$$

* Each row of the output is a **weighted sum of the Value vectors** of all tokens.
* If token $i$ attends strongly to token $j$ (high weight $A_{i,j}$), the representation of token $i$ will absorb a large fraction of token $j$'s Value vector $V_j$.
