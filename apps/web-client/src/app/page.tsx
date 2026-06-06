"use client";

import React from "react";
import { useAttentionStore } from "../store/attentionStore";
import { ControlPanel } from "../components/ControlPanel";
import { AttentionWizard } from "../components/AttentionWizard";
import { MultiHeadAttentionView } from "../components/MultiHeadAttentionView";
import { InspectCell } from "../components/InspectCell";
import { MatrixGrid } from "../components/MatrixGrid";
import { FileText, Download, RotateCcw, Sparkles } from "lucide-react";

export default function Home() {
  const { attentionData, activeHead, loading, resetAll } = useAttentionStore();

  const handleExport = () => {
    if (!attentionData) return;
    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(
      JSON.stringify(attentionData, null, 2)
    );
    const downloadAnchor = document.createElement("a");
    downloadAnchor.setAttribute("href", dataStr);
    downloadAnchor.setAttribute("download", "mini_attention_matrices.json");
    document.body.appendChild(downloadAnchor);
    downloadAnchor.click();
    downloadAnchor.remove();
  };

  const handleOpenDocs = () => {
    // Open the github styled local markdown or general details
    alert(
      "Educational documentation guides are located inside the local project folder:\n\n" +
      "1. docs/linear-algebra.md - Linear Algebra Guide\n" +
      "2. docs/self-attention.md - Self-Attention Mechanics\n" +
      "3. docs/causal-masking.md - Causal Masking Details\n" +
      "4. docs/multi-head-attention.md - Multi-Head Details\n" +
      "5. docs/interview-defense.md - Interview Questions"
    );
  };

  const hasData = attentionData && !loading;
  const h = activeHead;

  // Generate labels for projection dimensions
  const dModelLabels = hasData
    ? Array.from({ length: attentionData.dimensions.dModel }, (_, idx) => `${idx}`)
    : [];
  const dKeyLabels = hasData
    ? Array.from({ length: attentionData.dimensions.dKey }, (_, idx) => `${idx}`)
    : [];

  return (
    <main className="min-h-screen bg-[#070b13] bg-radial-gradient text-slate-100 flex flex-col p-4 md:p-6 select-none">
      {/* Header Bar */}
      <header className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 border-b border-slate-800/80 pb-4 mb-6">
        <div className="flex items-center gap-3">
          <div className="bg-indigo-600/20 border border-indigo-500/40 p-2 rounded-xl text-indigo-400">
            <Sparkles size={24} />
          </div>
          <div>
            <h1 className="text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-indigo-200 via-indigo-400 to-pink-300 tracking-tight">
              Mini Attention Notebook
            </h1>
            <p className="text-xs text-slate-400 font-medium">
              Q / K / V Structural Mechanics & Self-Attention Visualizer
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={handleOpenDocs}
            className="flex items-center gap-1.5 bg-slate-900 hover:bg-slate-800 border border-slate-800 px-3 py-1.5 rounded-lg text-xs font-semibold text-slate-300 transition-colors"
          >
            <FileText size={14} />
            <span>Docs</span>
          </button>
          <button
            onClick={handleExport}
            disabled={!hasData}
            className="flex items-center gap-1.5 bg-slate-900 hover:bg-slate-800 border border-slate-800 px-3 py-1.5 rounded-lg text-xs font-semibold text-slate-300 transition-colors disabled:opacity-50"
          >
            <Download size={14} />
            <span>Export</span>
          </button>
          <button
            onClick={resetAll}
            className="flex items-center gap-1.5 bg-slate-900 hover:bg-rose-950/30 border border-slate-800 hover:border-rose-900/40 px-3 py-1.5 rounded-lg text-xs font-semibold text-rose-400 transition-colors"
          >
            <RotateCcw size={14} />
            <span>Reset All</span>
          </button>
        </div>
      </header>

      {/* Main Grid Workspace */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-stretch flex-1">
        {/* Left Column (Span 3): Settings and controls */}
        <div className="lg:col-span-3">
          <ControlPanel />
        </div>

        {/* Center Workspace (Span 6): Projections matrices and wizard steps */}
        <div className="lg:col-span-6 flex flex-col gap-6">
          {/* Embedding & Projection layer matrices (X, Q, K, V) */}
          {hasData ? (
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              <MatrixGrid
                title="Token Embeddings (X)"
                subtitle="Shape: seq_len × d_model"
                data={attentionData.matrices.X}
                rowLabels={attentionData.tokens}
                colLabels={dModelLabels}
              />
              <MatrixGrid
                title="Query (Q)"
                subtitle="Q = X * WQ"
                data={attentionData.matrices.Q[h]}
                rowLabels={attentionData.tokens}
                colLabels={dKeyLabels}
              />
              <MatrixGrid
                title="Key (K)"
                subtitle="K = X * WK"
                data={attentionData.matrices.K[h]}
                rowLabels={attentionData.tokens}
                colLabels={dKeyLabels}
              />
              <MatrixGrid
                title="Value (V)"
                subtitle="V = X * WV"
                data={attentionData.matrices.V[h]}
                rowLabels={attentionData.tokens}
                colLabels={dKeyLabels}
              />
            </div>
          ) : (
            <div className="grid grid-cols-4 gap-4 animate-pulse">
              {Array.from({ length: 4 }).map((_, idx) => (
                <div
                  key={idx}
                  className="bg-slate-900/40 border border-slate-850 h-[150px] rounded-xl"
                />
              ))}
            </div>
          )}

          {/* Attention Wizard Steps 1 to 5 */}
          <AttentionWizard />

          {/* Multi-head attention matrices comparison & average */}
          <MultiHeadAttentionView />
        </div>

        {/* Right Column (Span 3): Interactive cell inspector */}
        <div className="lg:col-span-3">
          <InspectCell />
        </div>
      </div>

      <footer className="mt-8 text-center text-[10px] text-slate-500 font-mono tracking-wide">
        DevJam LLM Learning Project Series • Visualizing latent representations and matrix dot-product routing.
      </footer>
    </main>
  );
}
