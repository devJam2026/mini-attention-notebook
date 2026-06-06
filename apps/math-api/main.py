from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from typing import Dict, Any, List

from attention_engine import compute_attention, tokenize

app = FastAPI(
    title="Mini Attention Notebook Math API",
    description="Backend math service for computing self-attention matrices.",
    version="1.0.0"
)

# Enable CORS for frontend web client interaction
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify local domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AttentionComputeRequest(BaseModel):
    text: str = Field(..., description="Sentence input of 2 to 12 tokens.")
    dModel: int = Field(default=16, description="Latent embedding dimension (8, 16, 32, 64, 128).")
    heads: int = Field(default=2, description="Number of attention heads (1, 2, 4, 8).")
    applyMask: bool = Field(default=True, description="Whether to apply causal masking.")
    seed: int = Field(default=42, description="Random seed for projection reproducibility.")

    @field_validator("text")
    def validate_token_length(cls, v):
        tokens = tokenize(v)
        num_tokens = len(tokens)
        if num_tokens < 2 or num_tokens > 12:
            raise ValueError(
                f"Token count must be between 2 and 12. Input contains {num_tokens} tokens."
            )
        return v

    @field_validator("dModel")
    def validate_d_model(cls, v):
        allowed = {8, 16, 32, 64, 128}
        if v not in allowed:
            raise ValueError(f"dModel must be one of {allowed}. Got {v}.")
        return v

    @field_validator("heads")
    def validate_heads(cls, v):
        allowed = {1, 2, 4, 8}
        if v not in allowed:
            raise ValueError(f"Heads must be one of {allowed}. Got {v}.")
        return v

@app.post("/api/attention/compute", status_code=status.HTTP_200_OK)
def compute_attention_matrices(request: AttentionComputeRequest):
    # Verify divisibility constraint
    if request.dModel % request.heads != 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"dModel ({request.dModel}) must be divisible by heads ({request.heads}) without remainder."
        )
        
    try:
        results = compute_attention(
            text=request.text,
            d_model=request.dModel,
            heads=request.heads,
            apply_mask=request.applyMask,
            seed=request.seed
        )
        return results
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.get("/api/health")
def health_check():
    return {"status": "healthy"}
