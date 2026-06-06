"use client";

import React, { useEffect } from "react";
import { useAttentionStore } from "../store/attentionStore";
import { ArrowRight, HelpCircle } from "lucide-react";

export const InspectCell: React.FC = () => {
  const {
    attentionData,
    activeHead,
    hoveredCell,
    activeCell,
    setActiveCell,
    precision,
  } = useAttentionStore();

  // If no cell is selected or hovered, default to cell (0, 0)
  const cell = hoveredCell || activeCell || { i: 0, j: 0 };

  useEffect(() => {
    // Keep activeCell in sync if it gets out of bounds
    if (attentionData) {
      const seqLen = attentionData.tokens.length;
      if (cell.i >= seqLen || cell.j >= seqLen) {
        setActiveCell({ i: 0, j: 0 });
      }
    }
  }, [attentionData, cell, setActiveCell]);

  if (!attentionData) {
    return (
      <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-5 h-full flex items-center justify-center text-slate-500 italic text-xs">
        Select a cell to inspect details
      </div>
    );
  }

  const { tokens, matrices, dimensions } = attentionData;
  const seqLen = tokens.length;
  const h = activeHead;

  // Safety bounds check
  const i = Math.min(cell.i, seqLen - 1);
  const j = Math.min(cell.j, seqLen - 1);

  const tokenI = tokens[i];
  const tokenJ = tokens[j];

  const rawScore = matrices.rawScores[h][i][j];
  const scaledScore = matrices.scaledScores[h][i][j];
  const maskedScore = matrices.maskedScores[h][i][j];
  const weight = matrices.attentionWeights[h][i][j];

  // Retrieve vectors for dot product breakdown
  const qVector = matrices.Q[h][i];
  const kVector = matrices.K[h][j];

  const formatNumber = (val: number): string => {
    if (val === -9e9 || val < -1e8) return "-∞";
    if (precision === ".2f") return val.toFixed(2);
    if (precision === ".4f") return val.toFixed(4);
    return val.toFixed(3);
  };

  const isMasked = maskedScore === -9e9 || maskedScore < -1e8;

  // Construct plain English translation
  let explanation = "";
  if (isMasked) {
    explanation = `Token "${tokenI}" cannot attend to token "${tokenJ}" because causal masking is enabled. GPT-style autoregressive decoders prevent looking at future tokens (${j} > ${i}) to avoid information leakage during training.`;
  } else {
    const percentage = (weight * 100).toFixed(1);
    if (i === j) {
      explanation = `Token "${tokenI}" attends to itself with ${percentage}% importance. This serves as its identity representation in Head ${h + 1}.`;
    } else {
      explanation = `Token "${tokenI}" attends to token "${tokenJ}" with ${percentage}% importance. This means when updating "${tokenI}"'s context vector, it extracts ${percentage}% of its value from "${tokenJ}"'s contents.`;
    }
  }

  return (
    <div className="bg-slate-900/60 backdrop-blur-md border border-slate-800 rounded-xl p-5 shadow-xl flex flex-col gap-4 text-slate-200 h-full">
      <div className="flex items-center gap-2 border-b border-slate-800 pb-2">
        <span className="flex items-center justify-center bg-indigo-500/20 text-indigo-400 font-bold text-xs rounded-full w-5 h-5">
          7
        </span>
        <h3 className="text-sm font-bold uppercase tracking-wider text-slate-200">
          Inspect Cell
        </h3>
      </div>

      {/* Position Selector readout */}
      <div className="flex items-center justify-between bg-slate-950 p-2.5 border border-slate-900 rounded-lg">
        <div className="flex items-center gap-2 text-xs">
          <span className="font-semibold text-indigo-400 font-mono">Row {i}</span>
          <span className="text-slate-500">({tokenI})</span>
          <ArrowRight size={12} className="text-slate-500" />
          <span className="font-semibold text-teal-400 font-mono">Col {j}</span>
          <span className="text-slate-500">({tokenJ})</span>
        </div>
        <span className="text-[10px] bg-slate-850 px-2 py-0.5 rounded font-mono text-slate-400">
          Head {h + 1}
        </span>
      </div>

      {/* Numeric Highlights */}
      <div className="grid grid-cols-2 gap-3">
        <div className="bg-slate-950/60 border border-slate-850 rounded p-2 text-center">
          <span className="text-[9px] text-slate-500 uppercase tracking-wide block">Raw Score</span>
          <span className="text-sm font-bold font-mono text-indigo-300">{formatNumber(rawScore)}</span>
        </div>
        <div className="bg-slate-950/60 border border-slate-850 rounded p-2 text-center">
          <span className="text-[9px] text-slate-500 uppercase tracking-wide block">Scaled Score</span>
          <span className="text-sm font-bold font-mono text-indigo-300">{formatNumber(scaledScore)}</span>
        </div>
        <div className="bg-slate-950/60 border border-slate-850 rounded p-2 text-center">
          <span className="text-[9px] text-slate-500 uppercase tracking-wide block">Masked Score</span>
          <span className="text-sm font-bold font-mono text-indigo-300">{formatNumber(maskedScore)}</span>
        </div>
        <div className="bg-slate-950/60 border border-slate-850 rounded p-2 text-center ring-1 ring-pink-500/25">
          <span className="text-[9px] text-pink-400 uppercase tracking-wide block font-semibold">Weight (Softmax)</span>
          <span className="text-sm font-bold font-mono text-pink-400">{formatNumber(weight)}</span>
        </div>
      </div>

      {/* Plain English Explanation */}
      <div className="bg-indigo-950/20 border border-indigo-900/50 rounded-lg p-3 text-xs leading-normal text-slate-300 shadow-inner">
        <div className="text-[10px] font-bold text-indigo-400 uppercase tracking-wide mb-1 flex items-center gap-1">
          <HelpCircle size={10} /> Interpretation
        </div>
        {explanation}
      </div>

      {/* Mathematical Vector Dot Product Breakdown */}
      <div className="flex-1 flex flex-col gap-2 mt-1">
        <h4 className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">
          Dot Product Breakdown: Q[{i}] · K[{j}]
        </h4>
        <div className="flex-1 overflow-y-auto max-h-[160px] custom-scrollbar bg-slate-950/80 border border-slate-900 rounded p-2 text-[9px] font-mono text-slate-400">
          <div className="mb-2">
            <span className="text-indigo-400 block font-semibold">Query Vector Q[{i}] (shape: {dimensions.dKey}):</span>
            <div className="flex flex-wrap gap-1 p-1 bg-slate-900/40 rounded mt-0.5 max-h-[50px] overflow-auto">
              {qVector.map((q, idx) => (
                <span key={idx} className="bg-slate-900 border border-slate-800 px-1 rounded">
                  {formatNumber(q)}
                </span>
              ))}
            </div>
          </div>

          <div className="mb-2">
            <span className="text-teal-400 block font-semibold">Key Vector K[{j}] (shape: {dimensions.dKey}):</span>
            <div className="flex flex-wrap gap-1 p-1 bg-slate-900/40 rounded mt-0.5 max-h-[50px] overflow-auto">
              {kVector.map((k, idx) => (
                <span key={idx} className="bg-slate-900 border border-slate-800 px-1 rounded">
                  {formatNumber(k)}
                </span>
              ))}
            </div>
          </div>

          <div className="border-t border-slate-900 pt-2 mt-2 flex flex-col gap-1">
            <div className="flex justify-between text-slate-500">
              <span>Formula:</span>
              <span>∑ (q_d * k_d)</span>
            </div>
            <div className="flex justify-between font-semibold text-slate-300">
              <span>Result:</span>
              <span>{formatNumber(rawScore)}</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
