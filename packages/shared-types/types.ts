export interface AttentionComputeRequest {
  text: string;
  dModel: number;
  heads: number;
  applyMask: boolean;
  seed: number;
}

export interface AttentionDimensions {
  seqLen: number;
  dModel: number;
  heads: number;
  dKey: number;
}

export interface AttentionMatrices {
  X: number[][];                 // [seqLen, dModel]
  WQ: number[][][];              // [heads, dModel, dKey]
  WK: number[][][];              // [heads, dModel, dKey]
  WV: number[][][];              // [heads, dModel, dKey]
  Q: number[][][];               // [heads, seqLen, dKey]
  K: number[][][];               // [heads, seqLen, dKey]
  V: number[][][];               // [heads, seqLen, dKey]
  rawScores: number[][][];       // [heads, seqLen, seqLen]
  scaledScores: number[][][];    // [heads, seqLen, seqLen]
  maskedScores: number[][][];    // [heads, seqLen, seqLen]
  attentionWeights: number[][][];// [heads, seqLen, seqLen]
  contextOutputs: number[][][];  // [heads, seqLen, dKey]
  mergedOutput: number[][];      // [seqLen, dModel]
}

export interface AttentionComputeResponse {
  tokens: string[];
  dimensions: AttentionDimensions;
  matrices: AttentionMatrices;
}
