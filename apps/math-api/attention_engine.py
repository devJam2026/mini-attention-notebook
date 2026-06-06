import numpy as np
from typing import Dict, Any, List

def tokenize(text: str) -> List[str]:
    """
    Splits text into a clean list of tokens, ignoring empty strings.
    """
    if not text:
        return []
    # Split on whitespace, strip characters
    words = text.strip().split()
    return [w for w in words if w]

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Numerically stable softmax calculation.
    """
    # Subtract max for numerical stability (prevents overflow)
    max_val = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - max_val)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def compute_attention(
    text: str,
    d_model: int,
    heads: int,
    apply_mask: bool,
    seed: int
) -> Dict[str, Any]:
    """
    Computes all intermediate steps of Transformer Self-Attention using NumPy.
    
    Args:
        text: Input string.
        d_model: Latent dimension size.
        heads: Number of attention heads.
        apply_mask: If True, applies causal mask.
        seed: Random seed for reproducibility.
        
    Returns:
        Dictionary of token list, dimensions, and serialized matrices.
    """
    tokens = tokenize(text)
    seq_len = len(tokens)
    
    if seq_len < 2 or seq_len > 12:
        raise ValueError(f"Sequence length must be between 2 and 12 tokens. Got {seq_len}.")
        
    if d_model % heads != 0:
        raise ValueError(f"d_model ({d_model}) must be divisible by heads ({heads}).")
        
    d_k = d_model // heads
    
    # Initialize random generator
    rng = np.random.default_rng(seed)
    
    # 1. Generate Input Embeddings X (seq_len, d_model)
    # Scaled to make visualization values readable (mean 0, std 0.5)
    X = rng.normal(loc=0.0, scale=0.5, size=(seq_len, d_model))
    
    # 2. Initialize Weight Matrices
    # WQ, WK, WV shape: (heads, d_model, d_k)
    # Using Xavier initialization scaling
    scale = 1.0 / np.sqrt(d_model)
    WQ = rng.normal(loc=0.0, scale=scale, size=(heads, d_model, d_k))
    WK = rng.normal(loc=0.0, scale=scale, size=(heads, d_model, d_k))
    WV = rng.normal(loc=0.0, scale=scale, size=(heads, d_model, d_k))
    
    # Pre-allocate collections for head calculations
    Q_heads = []
    K_heads = []
    V_heads = []
    raw_scores_heads = []
    scaled_scores_heads = []
    masked_scores_heads = []
    weights_heads = []
    context_heads = []
    
    # Causal Mask creation
    # 0 for active, -1e9 for masked out elements
    mask = np.triu(np.ones((seq_len, seq_len)), k=1) * -1e9
    
    for h in range(heads):
        # 3. Project to Q, K, V for this head
        # X: (seq_len, d_model) @ WQ[h]: (d_model, d_k) -> Q_h: (seq_len, d_k)
        Q_h = X @ WQ[h]
        K_h = X @ WK[h]
        V_h = X @ WV[h]
        
        Q_heads.append(Q_h)
        K_heads.append(K_h)
        V_heads.append(V_h)
        
        # 4. Attention Wizard Step 1: Raw Scores Q K^T (seq_len, seq_len)
        raw_scores_h = Q_h @ K_h.T
        raw_scores_heads.append(raw_scores_h)
        
        # Step 2: Scaled Scores (Q K^T) / sqrt(d_k)
        scaled_scores_h = raw_scores_h / np.sqrt(d_k)
        scaled_scores_heads.append(scaled_scores_h)
        
        # Step 3: Apply Causal Mask if requested
        if apply_mask:
            masked_scores_h = scaled_scores_h + mask
        else:
            masked_scores_h = scaled_scores_h.copy()
        masked_scores_heads.append(masked_scores_h)
        
        # Step 4: Softmax Attention Weights (seq_len, seq_len)
        weights_h = softmax(masked_scores_h, axis=-1)
        weights_heads.append(weights_h)
        
        # Step 5: Output Context matrix: Attention weights @ V (seq_len, d_k)
        context_h = weights_h @ V_h
        context_heads.append(context_h)
        
    # Stack heads to match (heads, ...) shapes
    Q = np.stack(Q_heads)
    K = np.stack(K_heads)
    V = np.stack(V_heads)
    raw_scores = np.stack(raw_scores_heads)
    scaled_scores = np.stack(scaled_scores_heads)
    masked_scores = np.stack(masked_scores_heads)
    attention_weights = np.stack(weights_heads)
    context_outputs = np.stack(context_heads)
    
    # 5. Merge Heads: Concatenate context matrices along head dimension
    # Transpose from (heads, seq_len, d_k) to (seq_len, heads, d_k)
    # Then reshape to (seq_len, heads * d_k) which is (seq_len, d_model)
    merged_output = np.transpose(context_outputs, (1, 0, 2)).reshape(seq_len, d_model)
    
    return {
        "tokens": tokens,
        "dimensions": {
            "seqLen": seq_len,
            "dModel": d_model,
            "heads": heads,
            "dKey": d_k
        },
        "matrices": {
            "X": X.tolist(),
            "WQ": WQ.tolist(),
            "WK": WK.tolist(),
            "WV": WV.tolist(),
            "Q": Q.tolist(),
            "K": K.tolist(),
            "V": V.tolist(),
            "rawScores": raw_scores.tolist(),
            "scaledScores": scaled_scores.tolist(),
            "maskedScores": masked_scores.tolist(),
            "attentionWeights": attention_weights.tolist(),
            "contextOutputs": context_outputs.tolist(),
            "mergedOutput": merged_output.tolist()
        }
    }
