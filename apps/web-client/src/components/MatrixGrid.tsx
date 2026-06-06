"use client";

import React from "react";
import { useAttentionStore } from "../store/attentionStore";

interface MatrixGridProps {
  data: number[][];
  rowLabels: string[];
  colLabels: string[];
  title: string;
  subtitle?: string;
  colorType?: "sequential" | "diverging";
  minMax?: { min: number; max: number };
  stepIndex?: number; // If part of wizard, step index (1-5)
  headIndex?: number; // Head index if multihead
  onCellClick?: (i: number, j: number) => void;
}

export const MatrixGrid: React.FC<MatrixGridProps> = ({
  data,
  rowLabels,
  colLabels,
  title,
  subtitle,
  colorType = "diverging",
  minMax,
  stepIndex,
  headIndex,
  onCellClick,
}) => {
  const { hoveredCell, setHoveredCell, activeCell, setActiveCell, precision } =
    useAttentionStore();

  if (!data || data.length === 0) {
    return <div className="text-slate-500 italic p-4">No matrix data</div>;
  }

  // Calculate local min/max if not provided
  let min = 0;
  let max = 1;
  if (minMax) {
    min = minMax.min;
    max = minMax.max;
  } else {
    const flat = data.flat().filter((v) => v !== -9e9 && v > -1e8);
    if (flat.length > 0) {
      min = Math.min(...flat);
      max = Math.max(...flat);
    }
  }

  const formatNumber = (val: number): string => {
    if (val === -9e9 || val < -1e8) {
      return "-∞";
    }
    if (precision === ".2f") return val.toFixed(2);
    if (precision === ".4f") return val.toFixed(4);
    return val.toFixed(3);
  };

  const getCellStyle = (val: number): React.CSSProperties => {
    if (val === -9e9 || val < -1e8) {
      return {
        backgroundColor: "rgba(15, 23, 42, 0.95)",
        color: "#64748b",
        borderColor: "#1e293b",
      };
    }

    if (colorType === "sequential") {
      // Scale 0.0 -> 1.0
      // Map 0 to dark purple, 1 to bright yellow/amber
      const ratio = Math.max(0, Math.min(1, (val - min) / (max - min || 1)));
      // Interpolate HSL: Purple (H=260) to Amber/Yellow (H=40)
      const h = 265 - ratio * 225; // 265 -> 40
      const s = 45 + ratio * 50;   // 45% -> 95%
      const l = 16 + ratio * 44;   // 16% -> 60%
      return {
        backgroundColor: `hsl(${h}, ${s}%, ${l}%)`,
        color: l > 42 ? "#000000" : "#ffffff",
      };
    } else {
      // Diverging Scale
      // Max absolute value determines scale
      const absMax = Math.max(Math.abs(min), Math.abs(max)) || 1;
      const ratio = Math.max(-1, Math.min(1, val / absMax)); // -1 to 1
      if (ratio < 0) {
        // Negative: Red/Crimson hue
        const r = Math.abs(ratio);
        return {
          backgroundColor: `rgba(239, 68, 68, ${r * 0.6})`,
          color: "#ffffff",
        };
      } else {
        // Positive: Cyan/Teal/Indigo hue
        const r = ratio;
        return {
          backgroundColor: `rgba(99, 102, 241, ${r * 0.6})`,
          color: "#ffffff",
        };
      }
    }
  };

  return (
    <div className="bg-slate-900/60 backdrop-blur-md border border-slate-800 rounded-xl p-4 flex flex-col h-full shadow-lg">
      <div className="mb-2 flex items-center justify-between">
        <div>
          <h4 className="text-sm font-semibold text-slate-200 uppercase tracking-wider">
            {title}
          </h4>
          {subtitle && <p className="text-xs text-slate-400">{subtitle}</p>}
        </div>
        <span className="text-[10px] bg-slate-800 text-slate-400 px-2 py-0.5 rounded font-mono">
          ({rowLabels.length} × {colLabels.length})
        </span>
      </div>

      <div className="overflow-auto flex-1 max-h-[300px] custom-scrollbar">
        <table className="w-full border-collapse text-left">
          <thead>
            <tr>
              <th className="sticky top-0 left-0 z-20 bg-slate-900/90 p-1.5 text-[10px] font-mono text-slate-500 border-b border-slate-800 min-w-[50px]">
                i \ j
              </th>
              {colLabels.map((col, idx) => (
                <th
                  key={idx}
                  className="sticky top-0 z-10 bg-slate-900/95 p-1.5 text-[10px] font-mono text-slate-400 border-b border-slate-800 text-center min-w-[55px]"
                >
                  <div className="truncate max-w-[60px]" title={col}>
                    {col}
                  </div>
                  <span className="text-[8px] text-slate-500 block font-normal">
                    {idx}
                  </span>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {data.map((row, i) => (
              <tr key={i} className="hover:bg-slate-800/30 transition-colors">
                <td className="sticky left-0 z-10 bg-slate-900/95 p-1.5 text-[10px] font-mono font-medium text-slate-400 border-r border-slate-850 truncate max-w-[80px]" title={rowLabels[i]}>
                  <span className="text-[9px] text-slate-500 mr-1">{i}</span>
                  {rowLabels[i]}
                </td>
                {row.map((val, j) => {
                  const isHovered =
                    hoveredCell?.i === i && hoveredCell?.j === j;
                  const isActive =
                    activeCell?.i === i && activeCell?.j === j;

                  return (
                    <td
                      key={j}
                      onMouseEnter={() => setHoveredCell({ i, j })}
                      onMouseLeave={() => setHoveredCell(null)}
                      onClick={() => {
                        setActiveCell({ i, j });
                        if (onCellClick) onCellClick(i, j);
                      }}
                      style={getCellStyle(val)}
                      className={`p-1.5 text-center text-[10px] font-mono border border-slate-800/40 cursor-pointer select-none transition-all duration-150 relative
                        ${isHovered ? "scale-105 z-10 ring-2 ring-indigo-400 shadow-md" : ""}
                        ${isActive ? "ring-2 ring-pink-500 z-10 shadow-lg" : ""}
                      `}
                    >
                      {formatNumber(val)}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};
