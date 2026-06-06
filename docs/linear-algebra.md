# Linear Algebra Foundation: Matrices, Shapes, and Projections

To understand how Transformers process text, you must first master the language of **Linear Algebra**. Don't worry—as your architect, I will break down the essential concepts using visual shapes, diagrams, and code snippets.

---

## 1. What are Vectors and Matrices?

In machine learning, we represent tokens (words/punctuation) as numeric lists.

* **Vector (1D Array)**: A single sequence of numbers.
  - *Example*: An embedding vector for the token `"sky"` of size $4$:
    $$\vec{x} = [0.17, 0.26, -0.11, 0.09]$$
* **Matrix (2D Array)**: A grid of numbers representing a collection of vectors stack-ordered.
  - *Example*: A sequence of 4 tokens, where each token has a 4-dimensional embedding vector, forms a **$4 \times 4$ Matrix**:
    $$X = \begin{pmatrix}
    0.21 & -0.41 & 0.33 & -0.15 \\
    0.17 & 0.26 & -0.11 & 0.09 \\
    -0.35 & 0.14 & 0.05 & -0.21 \\
    0.08 & -0.22 & 0.37 & 0.18
    \end{pmatrix}$$
    *Here, row index $i$ corresponds to the token index, and column index $j$ corresponds to the embedding dimension index.*

---

## 2. Matrix Multiplication Rules: The "Shape Match"

You cannot multiply any two random matrices. They must satisfy a dimensional constraint.

### The Rule
If you multiply matrix $A$ of shape $(M \times K)$ and matrix $B$ of shape $(K \times N)$, the resulting matrix $C$ will have the shape $(M \times N)$.
$$\mathbf{A}_{(M \times K)} \cdot \mathbf{B}_{(K \times N)} = \mathbf{C}_{(M \times N)}$$

> [!IMPORTANT]
> The **inner dimensions** must match: the number of columns in $A$ ($K$) must equal the number of rows in $B$ ($K$). The **outer dimensions** ($M \times N$) determine the shape of the output.

### Visualizing Matrix Multiplication
Let's multiply a sequence matrix $X$ ($4 \times 16$) by a projection weight matrix $W_Q$ ($16 \times 8$).
* Inner dimensions match ($16 = 16$).
* Output matrix $Q$ will be $4 \times 8$.

```
   X (4 x 16)                 WQ (16 x 8)             Q (4 x 8)
[ x x x ... x ]             [ w w ... w ]           [ q q ... q ]
[ x x x ... x ]      X      [ w w ... w ]     =     [ q q ... q ]
[ x x x ... x ]             [ ...       ]           [ q q ... q ]
[ x x x ... x ]             [ w w ... w ]           [ q q ... q ]
```

### Numerical Calculation
For any element $C_{i,j}$ in the output:
$$C_{i,j} = \sum_{k=1}^{K} A_{i,k} \cdot B_{k,j}$$
You take the **$i$-th row** of $A$, dot-product it with the **$j$-th column** of $B$.

---

## 3. The Transpose Operation ($A^T$)

Transposing a matrix means swapping its rows and columns. Visually, you reflect the matrix across its main diagonal.

If $A$ has shape $(M \times N)$, then $A^T$ has shape $(N \times M)$.
$$A_{i,j} = A^T_{j,i}$$

### Why do we need Transposition in Self-Attention?
To find how much every token attends to every other token, we need to compare all Query vectors with all Key vectors.
- Query matrix $Q$ has shape $(N \times d_k)$ (where $N$ is sequence length).
- Key matrix $K$ has shape $(N \times d_k)$.

We want a final raw attention score matrix of shape $(N \times N)$. How do we multiply them?
* We cannot multiply $(N \times d_k) \cdot (N \times d_k)$ directly (inner dimensions $d_k \neq N$ in general).
* Therefore, we transpose $K$ to get $K^T$ of shape $(d_k \times N)$.
* Now, we can compute:
  $$\text{Scores} = Q \cdot K^T$$
  $$\text{Shape: } (N \times d_k) \cdot (d_k \times N) = (N \times N)$$

---

## 4. Dot Product as a Similarity Metric

At the core of the self-attention formula is the dot product of two vectors: $\vec{u} \cdot \vec{v}$.

### Geometric Definition
$$\vec{u} \cdot \vec{v} = \|\vec{u}\| \|\vec{v}\| \cos(\theta)$$
where $\theta$ is the angle between the two vectors.

* If the vectors point in the **same direction** ($\theta = 0^\circ$), the cosine is $1.0$, resulting in a large positive dot product.
* If they are **orthogonal** ($\theta = 90^\circ$), the cosine is $0.0$, resulting in a dot product of $0.0$.
* If they point in **opposite directions** ($\theta = 180^\circ$), the cosine is $-1.0$, resulting in a large negative dot product.

```
   u
   ^               u               u
   |               |              /
   |               +----> v      /____> v
   +----> v 
 同向 (Angle 0)   正交 (Angle 90)  有夹角 (Angle 0-90)
 Max Similarity    No Similarity   Partial Similarity
```

### Self-Attention Application
When we compute $Q K^T$:
* We are calculating the dot product of every token's Query vector with every token's Key vector.
* A high score means the Query of token $A$ and the Key of token $B$ are aligned in the latent space—meaning token $A$ finds token $B$ highly relevant to its current context.
* We then run these similarity scores through a Softmax function to turn them into valid probabilities.
