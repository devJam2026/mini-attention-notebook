"use client";

import React from "react";
import { useAttentionStore } from "../store/attentionStore";
import { MatrixGrid } from "./MatrixGrid";

export const AttentionWizard: React.FC = () => {
  const { attentionData, activeHead, loading } = useAttentionStore();

  if (loading) {
    return (
      <div className="bg-slate-900/40 border border-slate-800 rounded-xl p-8 flex flex-col items-center justify-center min-h-[300px]">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-500 mb-2"></div>
        <p className="text-slate-400 text-xs font-mono">Computing attention tensors...</p>
      </div>
    );
  }

  if (!attentionData) {
    return (
      <div className="bg-slate-900/40 border border-slate-800 rounded-xl p-8 text-center min-h-[300px] flex items-center justify-center">
        <p className="text-slate-500 italic text-xs">Enter a sentence and click Tokenize to begin.</p>
      </div>
    );
  }

  const { tokens, matrices, dimensions } = attentionData;
  const h = activeHead;

  // Generate column labels for the Key projections (0 to dKey - 1)
  const dKeyLabels = Array.from({ length: dimensions.dKey }, (_, idx) => `k_${idx}`);

  return (
    <div className="bg-slate-900/30 border border-slate-850 rounded-xl p-4 flex flex-col gap-4 shadow-lg">
      <div className="flex items-center justify-between border-b border-slate-800/80 pb-2">
        <div className="flex items-center gap-2">
          <span className="flex items-center justify-center bg-indigo-500/20 text-indigo-400 font-bold text-xs rounded-full w-5 h-5">
            5
          </span>
          <h3 className="text-sm font-bold uppercase tracking-wider text-slate-200">
            Attention Computation Steps (Active Head: {h + 1} of {dimensions.heads})
          </h3>
        </div>
        <div className="text-[10px] text-slate-500 font-mono">
          {"QKᵀ / √d_k -> Softmax -> Attn × V"}
        </div>
      </div>

      {/* Steps Horizontal Grid */}
      <div className="grid grid-cols-1 xl:grid-cols-5 gap-4">
        {/* Step 1: Raw Scores */}
        <div className="flex flex-col h-full">
          <div className="text-[10px] font-bold text-indigo-400 uppercase tracking-wide mb-1.5 px-1">
            Step 1: Raw Scores (QKᵀ)
          </div>
          <div className="flex-1">
            <MatrixGrid
              title="Raw Scores"
              subtitle="Q @ K.T Similarity"
              data={matrices.rawScores[h]}
              rowLabels={tokens}
              colLabels={tokens}
              colorType="diverging"
            />
          </div>
        </div>

        {/* Step 2: Scaled Scores */}
        <div className="flex flex-col h-full">
          <div className="text-[10px] font-bold text-indigo-400 uppercase tracking-wide mb-1.5 px-1">
            Step 2: Scaled Scores
          </div>
          <div className="flex-1">
            <MatrixGrid
              title="Scaled Scores"
              subtitle={`Divide by √d_k (√${dimensions.dKey})`}
              data={matrices.scaledScores[h]}
              rowLabels={tokens}
              colLabels={tokens}
              colorType="diverging"
            />
          </div>
        </div>

        {/* Step 3: Masked Scores */}
        <div className="flex flex-col h-full">
          <div className="text-[10px] font-bold text-indigo-400 uppercase tracking-wide mb-1.5 px-1">
            Step 3: Masked Scores
          </div>
          <div className="flex-1">
            <MatrixGrid
              title="Masked Scores"
              subtitle="Apply Causal Mask"
              data={matrices.maskedScores[h]}
              rowLabels={tokens}
              colLabels={tokens}
              colorType="diverging"
            />
          </div>
        </div>

        {/* Step 4: Attention Weights */}
        <div className="flex flex-col h-full">
          <div className="text-[10px] font-bold text-indigo-400 uppercase tracking-wide mb-1.5 px-1">
            Step 4: Softmax Weights
          </div>
          <div className="flex-1">
            <MatrixGrid
              title="Attention Weights"
              subtitle="Softmax row-wise"
              data={matrices.attentionWeights[h]}
              rowLabels={tokens}
              colLabels={tokens}
              colorType="sequential"
              minMax={{ min: 0.0, max: 1.0 }}
            />
          </div>
        </div>

        {/* Step 5: Output Context Matrix */}
        <div className="flex flex-col h-full">
          <div className="text-[10px] font-bold text-indigo-400 uppercase tracking-wide mb-1.5 px-1">
            Step 5: Output Context
          </div>
          <div className="flex-1">
            <MatrixGrid
              title="Output Vectors"
              subtitle="AttentionWeights @ V"
              data={matrices.contextOutputs[h]}
              rowLabels={tokens}
              colLabels={dKeyLabels}
              colorType="diverging"
            />
          </div>
        </div>
      </div>
    </div>
  );
};
