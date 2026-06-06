# Interview Defense Guide: Mastering Transformer Self-Attention

If you are interviewing for an ML/NLP role, you *will* be asked about the self-attention formula. This guide provides deep, production-ready answers to the 5 most common interview questions regarding attention mechanics.

---

## Question 1: Why do we divide by $\sqrt{d_k}$?

### Simple Answer
Dividing by $\sqrt{d_k}$ prevents the dot products from growing too large in magnitude, which would push the Softmax function into flat regions with extremely small gradients (vanishing gradients).

### Mathematical Explanation
Assume that components of Query vector $\vec{q}$ and Key vector $\vec{k}$ are independent random variables with mean $0$ and variance $1$:
$$q_i, k_i \sim \mathcal{N}(0, 1)$$

The dot product is:
$$q \cdot k = \sum_{i=1}^{d_k} q_i k_i$$

Since they are independent:
* The mean of $q_i k_i$ is:
  $$\mathbb{E}[q_i k_i] = \mathbb{E}[q_i]\mathbb{E}[k_i] = 0$$
* The variance of $q_i k_i$ is:
  $$\text{Var}(q_i k_i) = \text{Var}(q_i)\text{Var}(k_i) = 1 \cdot 1 = 1$$

Since the sum contains $d_k$ independent terms, the variance of the sum is the sum of the variances:
$$\text{Var}(q \cdot k) = \sum_{i=1}^{d_k} \text{Var}(q_i k_i) = d_k$$

Therefore, the standard deviation of the raw dot product is **$\sqrt{d_k}$**.
As the head dimension $d_k$ grows large, the dot products also grow large in magnitude. When we apply Softmax to these large values, the output distribution becomes extremely peaked (one value close to $1.0$, others close to $0.0$). 
In these flat regions of Softmax, the derivative (gradient) is close to $0$. This causes the **vanishing gradient problem** during training. By dividing by $\sqrt{d_k}$, we scale the variance of the dot product back down to $1.0$, preserving healthy gradients.

---

## Question 2: Why do we use Softmax?

### Simple Answer
Softmax converts raw similarity scores into a valid probability distribution (values between $0$ and $1$ that sum to $1$). This allows the model to perform a "weighted average" over the values of all tokens.

### Mathematical Explanation
For a set of scores $z = [z_1, z_2, \dots, z_n]$, Softmax is defined as:
$$\text{Softmax}(z)_i = \frac{e^{z_i}}{\sum_{j=1}^n e^{z_j}}$$

### Architectural Benefits
1. **Differentiability**: Softmax is smooth and differentiable everywhere, making it compatible with backpropagation.
2. **Exponential Contrast**: The exponential function $e^x$ amplifies differences: it rewards tokens with slightly higher similarity scores with significantly higher attention weights, allowing the model to focus sharply on the most relevant tokens.
3. **Probability Bounds**: Normalizing the sum to $1.0$ ensures that the output context vector remains at the same scale as the input value vectors, preventing activations from exploding as they pass through multiple attention blocks.

---

## Question 3: Why is causal masking necessary?

### Simple Answer
Causal masking prevents a token from looking at future tokens (tokens to its right) during training and autoregressive generation. Without it, the model would simply "cheat" during training by copying the next word, and fail to learn how to predict the next word when deployed.

### Architectural Benefits
In autoregressive text generation (like GPT), we generate text one token at a time:
- Step 1: Input `"The"` -> Predict `"sky"`
- Step 2: Input `"The sky"` -> Predict `"is"`
- Step 3: Input `"The sky is"` -> Predict `"blue"`

If the model could attend to future tokens during training, it would use future words to predict the target word. However, at inference time, future words do not exist yet. Causal masking ensures that the training environment perfectly matches the inference environment.

---

## Question 4: Why is the complexity of Self-Attention $O(n^2)$?

### Simple Answer
Self-attention compares **every token in the sequence with every other token**. For a sequence of length $n$, this requires calculating $n \times n$ dot products, resulting in quadratic complexity with respect to sequence length.

### Complexity Breakdown
Let $n$ be the sequence length and $d$ be the embedding dimension.

1. **Projection to Q, K, V**:
   Multiplying input $X_{(n \times d)}$ by weight matrices $W_{(d \times d)}$ takes $O(n \cdot d^2)$ operations.
2. **Attention Scores ($Q K^T$)**:
   Multiplying $Q_{(n \times d_k)}$ by $K^T_{(d_k \times n)}$ requires $n^2$ dot products of vectors of size $d_k$. This takes $O(n^2 \cdot d)$ operations.
3. **Softmax**:
   Applying Softmax to an $n \times n$ matrix takes $O(n^2)$ operations.
4. **Context Output ($A V$)**:
   Multiplying Attention Matrix $A_{(n \times n)}$ by Value Matrix $V_{(n \times d_k)}$ takes $O(n^2 \cdot d)$ operations.

As $n$ grows (long context windows), the $n^2$ term dominates the computation. This is why processing extremely long books or documents in vanilla Transformers is computationally expensive compared to Recurrent Neural Networks (RNNs), which process sequences sequentially in $O(n \cdot d^2)$ time but cannot be parallelized.

---

## Question 5: Why do we use multiple heads instead of one large head?

### Simple Answer
Multiple heads allow the model to attend to different parts of the sentence at the same time, capturing different types of relationships (e.g., grammar rules vs. semantic pronoun references) in parallel.

### Architectural Benefits
If we had only one head of dimension $d_{model}$, the model would calculate a single attention probability map. If a word has multiple relationships (e.g. a verb relating to both its subject and its direct object), a single head must average its attention, diluting the signal.

By splitting $d_{model}$ into $h$ heads:
1. **Representational diversity**: Each head operates in a different subspace $d_k$, focusing on different linguistic properties.
2. **Ensemble effect**: It works like an ensemble of models, where each head learns to identify a specific type of connection.
3. **No extra parameters**: Because we split $d_{model}$ such that $d_k = d_{model} / h$, the parameter count and total floating-point operations remain almost identical to a single-head implementation.
