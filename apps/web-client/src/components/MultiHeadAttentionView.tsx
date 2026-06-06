"use client";

import React from "react";
import { useAttentionStore } from "../store/attentionStore";
import { MatrixGrid } from "./MatrixGrid";

export const MultiHeadAttentionView: React.FC = () => {
  const { attentionData, activeHead, setActiveHead, loading } = useAttentionStore();

  if (loading || !attentionData) return null;

  const { tokens, matrices, dimensions } = attentionData;
  const numHeads = dimensions.heads;

  // Compute average attention weight matrix across all heads
  const seqLen = tokens.length;
  const averageWeights: number[][] = Array.from({ length: seqLen }, () =>
    Array.from({ length: seqLen }, () => 0)
  );

  for (let i = 0; i < seqLen; i++) {
    for (let j = 0; j < seqLen; j++) {
      let sum = 0;
      for (let h = 0; h < numHeads; h++) {
        sum += matrices.attentionWeights[h][i][j];
      }
      averageWeights[i][j] = sum / numHeads;
    }
  }

  // Find another head to display for side-by-side comparison (default to head 1 or 2)
  const compareHeadIndex = numHeads > 1 ? (activeHead === 0 ? 1 : 0) : 0;

  return (
    <div className="bg-slate-900/30 border border-slate-850 rounded-xl p-4 flex flex-col gap-4 shadow-lg text-slate-100">
      <div className="flex items-center justify-between border-b border-slate-800/80 pb-2">
        <div className="flex items-center gap-2">
          <span className="flex items-center justify-center bg-indigo-500/20 text-indigo-400 font-bold text-xs rounded-full w-5 h-5">
            6
          </span>
          <h3 className="text-sm font-bold uppercase tracking-wider text-slate-200">
            Multi-Head Attention View
          </h3>
        </div>
        <div className="text-[10px] text-slate-500 font-mono">
          Compare independent representation heads
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5 items-start">
        {/* Head Selector Diagram */}
        <div className="lg:col-span-3 bg-slate-950/60 border border-slate-850 rounded-xl p-4 flex flex-col gap-4 shadow-inner">
          <div>
            <label className="text-[10px] text-slate-400 font-bold uppercase tracking-wider block mb-1">
              Select Head
            </label>
            <select
              value={activeHead}
              onChange={(e) => setActiveHead(Number(e.target.value))}
              className="w-full bg-slate-900 border border-slate-805 rounded px-2.5 py-1.5 text-xs text-indigo-400 font-semibold focus:outline-none cursor-pointer"
            >
              {Array.from({ length: numHeads }, (_, idx) => (
                <option key={idx} value={idx}>
                  Head {idx + 1}
                </option>
              ))}
            </select>
            <p className="text-[9px] text-slate-500 mt-1.5 leading-normal">
              Each head attends to the same sequence but learns different projection matrices to capture different context aspects.
            </p>
          </div>

          {/* Simple Visual Flow Diagram */}
          <div className="flex flex-col gap-2 p-2 border border-slate-900 bg-slate-950/80 rounded-lg text-[10px] font-mono text-slate-400">
            <div className="text-center border-b border-slate-900 pb-1 text-slate-500 text-[9px]">PROJECTION FLOW</div>
            <div className="flex items-center justify-between gap-1">
              <span className="bg-slate-900 border border-slate-800 px-1.5 py-0.5 rounded text-[9px]">X</span>
              <span className="text-slate-600">→</span>
              <div className="flex flex-col gap-0.5 text-center shrink-0">
                <span className="bg-indigo-950 border border-indigo-900/60 px-1 py-0.5 rounded text-[8px] text-indigo-300">WQ {activeHead + 1}</span>
                <span className="bg-teal-950 border border-teal-900/60 px-1 py-0.5 rounded text-[8px] text-teal-300">WK {activeHead + 1}</span>
                <span className="bg-amber-950 border border-amber-900/60 px-1 py-0.5 rounded text-[8px] text-amber-300">WV {activeHead + 1}</span>
              </div>
              <span className="text-slate-600">→</span>
              <div className="flex flex-col gap-0.5 text-center shrink-0">
                <span className="bg-indigo-900/40 px-1.5 py-0.5 rounded text-[8px] text-indigo-200">Q</span>
                <span className="bg-teal-900/40 px-1.5 py-0.5 rounded text-[8px] text-teal-200">K</span>
                <span className="bg-amber-900/40 px-1.5 py-0.5 rounded text-[8px] text-amber-200">V</span>
              </div>
            </div>
          </div>
        </div>

        {/* Dynamic Head Maps Comparison */}
        <div className="lg:col-span-9 grid grid-cols-1 md:grid-cols-3 gap-4">
          {/* Selected Head */}
          <div>
            <div className="text-[10px] font-bold text-indigo-400 uppercase tracking-wide mb-1.5 text-center">
              HEAD {activeHead + 1} (d_k={dimensions.dKey})
            </div>
            <MatrixGrid
              title="Attention Weights"
              subtitle={`Head ${activeHead + 1} Softmax`}
              data={matrices.attentionWeights[activeHead]}
              rowLabels={tokens}
              colLabels={tokens}
              colorType="sequential"
              minMax={{ min: 0.0, max: 1.0 }}
            />
          </div>

          {/* Another Head (if available) */}
          <div>
            <div className="text-[10px] font-bold text-slate-400 uppercase tracking-wide mb-1.5 text-center">
              {numHeads > 1 ? `HEAD ${compareHeadIndex + 1} (d_k=${dimensions.dKey})` : "NO ALTERNATE HEAD"}
            </div>
            {numHeads > 1 ? (
              <MatrixGrid
                title="Attention Weights"
                subtitle={`Head ${compareHeadIndex + 1} Softmax`}
                data={matrices.attentionWeights[compareHeadIndex]}
                rowLabels={tokens}
                colLabels={tokens}
                colorType="sequential"
                minMax={{ min: 0.0, max: 1.0 }}
              />
            ) : (
              <div className="bg-slate-900/40 border border-slate-800 rounded-xl p-6 text-center text-xs text-slate-500 italic flex items-center justify-center h-[200px]">
                Increase heads in settings to compare.
              </div>
            )}
          </div>

          {/* Average Weight Overlay */}
          <div>
            <div className="text-[10px] font-bold text-pink-400 uppercase tracking-wide mb-1.5 text-center">
              WEIGHT OVERLAY (Average)
            </div>
            <MatrixGrid
              title="Average Weights"
              subtitle="Mean across all heads"
              data={averageWeights}
              rowLabels={tokens}
              colLabels={tokens}
              colorType="sequential"
              minMax={{ min: 0.0, max: 1.0 }}
            />
          </div>
        </div>
      </div>
    </div>
  );
};
