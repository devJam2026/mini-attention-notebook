import pytest
import numpy as np
from attention_engine import compute_attention, tokenize, softmax

def test_tokenize():
    assert tokenize("The sky is blue") == ["The", "sky", "is", "blue"]
    assert tokenize("  Hello   World! ") == ["Hello", "World!"]
    assert tokenize("") == []

def test_softmax():
    # Simple array
    arr = np.array([[1.0, 2.0, 3.0]])
    res = softmax(arr)
    assert np.allclose(np.sum(res, axis=-1), 1.0)
    
    # Check stable behavior with large negative numbers (masking)
    arr_masked = np.array([[1.0, -1e9, 2.0, -1e9]])
    res_masked = softmax(arr_masked)
    assert np.allclose(np.sum(res_masked, axis=-1), 1.0)
    # Masked positions should have effectively zero probability
    assert res_masked[0, 1] < 1e-7
    assert res_masked[0, 3] < 1e-7

def test_attention_shapes_and_values():
    text = "The sky is blue"
    d_model = 16
    heads = 2
    seed = 42
    
    # 1. Test Mask ON
    res = compute_attention(text, d_model, heads, apply_mask=True, seed=seed)
    
    assert res["tokens"] == ["The", "sky", "is", "blue"]
    dims = res["dimensions"]
    assert dims["seqLen"] == 4
    assert dims["dModel"] == 16
    assert dims["heads"] == 2
    assert dims["dKey"] == 8
    
    matrices = res["matrices"]
    
    # Check matrix shapes
    assert np.shape(matrices["X"]) == (4, 16)
    assert np.shape(matrices["WQ"]) == (2, 16, 8)
    assert np.shape(matrices["Q"]) == (2, 4, 8)
    assert np.shape(matrices["K"]) == (2, 4, 8)
    assert np.shape(matrices["V"]) == (2, 4, 8)
    assert np.shape(matrices["rawScores"]) == (2, 4, 4)
    assert np.shape(matrices["attentionWeights"]) == (2, 4, 4)
    assert np.shape(matrices["mergedOutput"]) == (4, 16)
    
    # Check masking values (upper triangle should be 0 in attention weights)
    weights = np.array(matrices["attentionWeights"])
    for h in range(heads):
        # Row 0: attends only to token 0
        assert np.allclose(weights[h, 0, 1:], 0.0)
        # Row 1: attends only to tokens 0, 1
        assert np.allclose(weights[h, 1, 2:], 0.0)
        # Row 2: attends only to tokens 0, 1, 2
        assert np.allclose(weights[h, 2, 3:], 0.0)
        
        # Verify row sums
        assert np.allclose(np.sum(weights[h], axis=-1), 1.0)

def test_attention_mask_off():
    text = "The sky is blue"
    res = compute_attention(text, d_model=8, heads=1, apply_mask=False, seed=42)
    weights = np.array(res["matrices"]["attentionWeights"])
    
    # In Mask OFF, upper triangle elements should not be forced to 0
    # Let's verify row sums are still 1.0
    assert np.allclose(np.sum(weights[0], axis=-1), 1.0)
    # The first row shouldn't have zero attention weights on future tokens
    assert np.any(weights[0, 0, 1:] > 0.01)

def test_validation_errors():
    # Divisibility error
    with pytest.raises(ValueError, match="divisible by heads"):
        compute_attention("The sky is blue", d_model=16, heads=3, apply_mask=True, seed=42)
        
    # Sequence length too short
    with pytest.raises(ValueError, match="between 2 and 12"):
        compute_attention("Hello", d_model=8, heads=1, apply_mask=True, seed=42)
        
    # Sequence length too long
    long_sentence = " ".join(["word"] * 15)
    with pytest.raises(ValueError, match="between 2 and 12"):
        compute_attention(long_sentence, d_model=8, heads=1, apply_mask=True, seed=42)
