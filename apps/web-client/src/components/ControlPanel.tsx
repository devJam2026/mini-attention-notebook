"use client";

import React, { useEffect, useState } from "react";
import { useAttentionStore } from "../store/attentionStore";
import { Shuffle, HelpCircle, AlertCircle } from "lucide-react";

export const ControlPanel: React.FC = () => {
  const {
    text,
    setText,
    dModel,
    setDModel,
    heads,
    setHeads,
    applyMask,
    setApplyMask,
    seed,
    setSeed,
    precision,
    setPrecision,
    fetchAttention,
    attentionData,
    loading,
    error,
    resetAll,
  } = useAttentionStore();

  const [inputVal, setInputVal] = useState(text);

  useEffect(() => {
    fetchAttention();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setText(inputVal);
    // Trigger update
    setTimeout(() => {
      fetchAttention();
    }, 50);
  };

  const handleRandomSeed = () => {
    const newSeed = Math.floor(Math.random() * 1000);
    setSeed(newSeed);
    setTimeout(() => {
      fetchAttention();
    }, 50);
  };

  const handleDimensionChange = (val: number) => {
    setDModel(val);
    setTimeout(() => {
      fetchAttention();
    }, 50);
  };

  const handleHeadsChange = (val: number) => {
    setHeads(val);
    setTimeout(() => {
      fetchAttention();
    }, 50);
  };

  const handleMaskToggle = () => {
    setApplyMask(!applyMask);
    setTimeout(() => {
      fetchAttention();
    }, 50);
  };

  // Calculate total projection parameter count: 3 * d_model * d_model
  const totalParams = 3 * dModel * dModel;
  const dK = dModel / heads;

  return (
    <div className="flex flex-col gap-5 bg-slate-900/60 backdrop-blur-md border border-slate-800 rounded-xl p-5 shadow-xl h-full text-slate-100">
      {/* Step 1: Input Sequence */}
      <div>
        <div className="flex items-center gap-1.5 mb-2">
          <span className="flex items-center justify-center bg-indigo-500/20 text-indigo-400 font-bold text-xs rounded-full w-5 h-5">
            1
          </span>
          <h3 className="text-xs font-bold uppercase tracking-wider text-slate-300">
            Input Sequence
          </h3>
          <div className="relative group cursor-help text-slate-500 hover:text-slate-300">
            <HelpCircle size={14} />
            <span className="absolute left-6 top-0 hidden group-hover:block bg-slate-800 text-[10px] text-slate-300 p-2 rounded shadow-lg w-48 z-30">
              Input a short sentence between 2 and 12 tokens.
            </span>
          </div>
        </div>

        <form onSubmit={handleSubmit} className="flex gap-2">
          <input
            type="text"
            value={inputVal}
            onChange={(e) => setInputVal(e.target.value)}
            disabled={loading}
            className="flex-1 bg-slate-950 border border-slate-800 focus:border-indigo-500 rounded px-3 py-1.5 text-xs text-slate-100 focus:outline-none transition-colors"
            placeholder="The sky is blue"
          />
          <button
            type="submit"
            disabled={loading}
            className="bg-indigo-600 hover:bg-indigo-500 text-white font-semibold text-xs px-3 py-1.5 rounded disabled:opacity-50 transition-colors"
          >
            {loading ? "..." : "Tokenize"}
          </button>
        </form>

        {/* Tokens List */}
        {attentionData && (
          <div className="mt-3 flex flex-wrap gap-1.5 max-h-[100px] overflow-y-auto custom-scrollbar p-1">
            {attentionData.tokens.map((token, idx) => (
              <div
                key={idx}
                className="bg-slate-950 border border-slate-800/80 rounded px-2 py-1 text-[10px] font-mono flex items-center gap-1.5"
              >
                <span className="text-slate-500">{idx}</span>
                <span className="text-indigo-400 font-semibold">{token}</span>
              </div>
            ))}
          </div>
        )}
      </div>

      <hr className="border-slate-800/60" />

      {/* Step 2: Model Dimensions */}
      <div>
        <div className="flex items-center gap-1.5 mb-2.5">
          <span className="flex items-center justify-center bg-indigo-500/20 text-indigo-400 font-bold text-xs rounded-full w-5 h-5">
            2
          </span>
          <h3 className="text-xs font-bold uppercase tracking-wider text-slate-300">
            Model Dimensions
          </h3>
        </div>

        <div className="grid grid-cols-2 gap-3 mb-3">
          <div className="flex flex-col gap-1">
            <label className="text-[10px] text-slate-400 font-medium">d_model</label>
            <select
              value={dModel}
              onChange={(e) => handleDimensionChange(Number(e.target.value))}
              disabled={loading}
              className="bg-slate-950 border border-slate-850 focus:border-indigo-500 rounded px-2 py-1.5 text-xs focus:outline-none cursor-pointer"
            >
              {[8, 16, 32, 64, 128].map((d) => (
                <option key={d} value={d}>
                  {d}
                </option>
              ))}
            </select>
          </div>

          <div className="flex flex-col gap-1">
            <label className="text-[10px] text-slate-400 font-medium">Heads (h)</label>
            <select
              value={heads}
              onChange={(e) => handleHeadsChange(Number(e.target.value))}
              disabled={loading}
              className="bg-slate-950 border border-slate-850 focus:border-indigo-500 rounded px-2 py-1.5 text-xs focus:outline-none cursor-pointer"
            >
              {[1, 2, 4, 8].map((h) => (
                <option key={h} value={h} disabled={dModel % h !== 0}>
                  {h}
                </option>
              ))}
            </select>
          </div>
        </div>

        <div className="bg-slate-950/80 border border-slate-900 rounded p-2 text-[10px] font-mono text-slate-400 flex flex-col gap-1 shadow-inner">
          <div className="flex justify-between">
            <span>head_dim (d_k = d_model / h):</span>
            <span className="text-indigo-400 font-semibold">{dK}</span>
          </div>
          <div className="flex justify-between">
            <span>Total parameters (Wq, Wk, Wv):</span>
            <span className="text-indigo-400 font-semibold">{totalParams}</span>
          </div>
        </div>
      </div>

      <hr className="border-slate-800/60" />

      {/* Step 3: Initialization */}
      <div>
        <div className="flex items-center gap-1.5 mb-2.5">
          <span className="flex items-center justify-center bg-indigo-500/20 text-indigo-400 font-bold text-xs rounded-full w-5 h-5">
            3
          </span>
          <h3 className="text-xs font-bold uppercase tracking-wider text-slate-300">
            Initialization
          </h3>
        </div>

        <div className="flex flex-col gap-2">
          <div className="flex items-center gap-2">
            <div className="flex-1 flex flex-col gap-1">
              <label className="text-[10px] text-slate-400 font-medium">Random Seed</label>
              <input
                type="number"
                value={seed}
                onChange={(e) => {
                  setSeed(Number(e.target.value));
                  setTimeout(() => fetchAttention(), 50);
                }}
                disabled={loading}
                className="w-full bg-slate-950 border border-slate-850 focus:border-indigo-500 rounded px-2 py-1 text-xs focus:outline-none font-mono"
              />
            </div>
            <button
              onClick={handleRandomSeed}
              disabled={loading}
              title="Generate new seed"
              className="mt-5 bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700 p-1.5 rounded transition-colors disabled:opacity-50"
            >
              <Shuffle size={14} />
            </button>
          </div>

          <div className="grid grid-cols-2 gap-2 mt-1">
            <button
              onClick={() => fetchAttention()}
              disabled={loading}
              className="bg-slate-800 hover:bg-slate-750 border border-slate-750 text-slate-300 hover:text-white font-semibold text-[10px] py-1.5 rounded transition-all disabled:opacity-50"
            >
              Regenerate Weights
            </button>
            <button
              onClick={() => fetchAttention()}
              disabled={loading}
              className="bg-slate-800 hover:bg-slate-750 border border-slate-750 text-slate-300 hover:text-white font-semibold text-[10px] py-1.5 rounded transition-all disabled:opacity-50"
            >
              Regenerate Embeddings
            </button>
          </div>
        </div>
      </div>

      <hr className="border-slate-800/60" />

      {/* Step 4: View Options */}
      <div>
        <div className="flex items-center gap-1.5 mb-3">
          <span className="flex items-center justify-center bg-indigo-500/20 text-indigo-400 font-bold text-xs rounded-full w-5 h-5">
            4
          </span>
          <h3 className="text-xs font-bold uppercase tracking-wider text-slate-300">
            View Options
          </h3>
        </div>

        <div className="flex flex-col gap-3">
          <div className="flex justify-between items-center bg-slate-950/40 p-2 rounded border border-slate-900">
            <span className="text-xs text-slate-300">Causal Mask</span>
            <button
              onClick={handleMaskToggle}
              disabled={loading}
              className={`relative inline-flex h-5 w-9 items-center rounded-full transition-colors focus:outline-none
                ${applyMask ? "bg-indigo-600" : "bg-slate-800"}
              `}
            >
              <span
                className={`inline-block h-3.5 w-3.5 transform rounded-full bg-white transition-transform
                  ${applyMask ? "translate-x-4.5" : "translate-x-1"}
                `}
              />
            </button>
          </div>

          <div className="flex items-center justify-between">
            <span className="text-xs text-slate-300">Number Format</span>
            <select
              value={precision}
              onChange={(e) => setPrecision(e.target.value)}
              className="bg-slate-950 border border-slate-850 rounded px-2 py-1 text-xs focus:outline-none"
            >
              <option value=".2f">.2f (e.g. 0.42)</option>
              <option value=".3f">.3f (e.g. 0.420)</option>
              <option value=".4f">.4f (e.g. 0.4200)</option>
            </select>
          </div>
        </div>
      </div>

      {error && (
        <div className="mt-auto bg-rose-950/40 border border-rose-800/80 rounded-lg p-3 text-rose-300 flex items-start gap-2 shadow-md">
          <AlertCircle size={16} className="mt-0.5 shrink-0" />
          <div className="text-[10px] leading-tight font-mono">{error}</div>
        </div>
      )}
    </div>
  );
};
