import { create } from "zustand";
import { AttentionComputeResponse } from "../../../../packages/shared-types/types";

interface AttentionState {
  // Inputs
  text: string;
  dModel: number;
  heads: number;
  applyMask: boolean;
  seed: number;
  precision: string;

  // Active view states
  activeStep: number; // Steps 1-5
  activeHead: number; // index 0 to heads-1
  hoveredCell: { i: number; j: number } | null;
  activeCell: { i: number; j: number } | null;

  // Compute responses
  attentionData: AttentionComputeResponse | null;
  loading: boolean;
  error: string | null;

  // Actions
  setText: (text: string) => void;
  setDModel: (d: number) => void;
  setHeads: (h: number) => void;
  setApplyMask: (m: boolean) => void;
  setSeed: (s: number) => void;
  setPrecision: (p: string) => void;
  setActiveStep: (step: number) => void;
  setActiveHead: (head: number) => void;
  setHoveredCell: (cell: { i: number; j: number } | null) => void;
  setActiveCell: (cell: { i: number; j: number } | null) => void;
  fetchAttention: () => Promise<void>;
  resetAll: () => void;
}

const DEFAULT_STATE = {
  text: "The sky is blue",
  dModel: 16,
  heads: 2,
  applyMask: true,
  seed: 42,
  precision: ".3f",
  activeStep: 1,
  activeHead: 0,
  hoveredCell: null,
  activeCell: null,
  attentionData: null,
  loading: false,
  error: null,
};

export const useAttentionStore = create<AttentionState>((set, get) => ({
  ...DEFAULT_STATE,

  setText: (text) => set({ text }),
  setDModel: (dModel) => {
    // If head division constraint is violated, try to auto-adjust heads
    const { heads } = get();
    if (dModel % heads !== 0) {
      // Find a valid head count
      const validHeads = [1, 2, 4, 8].find((h) => dModel % h === 0) || 1;
      set({ dModel, heads: validHeads, activeHead: 0 });
    } else {
      set({ dModel });
    }
  },
  setHeads: (heads) => set({ heads, activeHead: 0 }),
  setApplyMask: (applyMask) => set({ applyMask }),
  setSeed: (seed) => set({ seed }),
  setPrecision: (precision) => set({ precision }),
  setActiveStep: (activeStep) => set({ activeStep }),
  setActiveHead: (activeHead) => set({ activeHead }),
  setHoveredCell: (hoveredCell) => set({ hoveredCell }),
  setActiveCell: (activeCell) => set({ activeCell }),

  fetchAttention: async () => {
    set({ loading: true, error: null });
    const { text, dModel, heads, applyMask, seed } = get();

    try {
      const response = await fetch("http://localhost:8000/api/attention/compute", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          text,
          dModel,
          heads,
          applyMask,
          seed,
        }),
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(
          errData?.detail || errData?.message || `API error: ${response.statusText}`
        );
      }

      const data: AttentionComputeResponse = await response.json();
      
      // Safety bounds for activeHead selector
      let currentHead = get().activeHead;
      if (currentHead >= data.dimensions.heads) {
        currentHead = 0;
      }

      set({
        attentionData: data,
        activeHead: currentHead,
        loading: false,
      });
    } catch (err: any) {
      console.error("Fetch attention error:", err);
      set({
        error: err.message || "Failed to connect to math backend.",
        loading: false,
      });
    }
  },

  resetAll: () => {
    set(DEFAULT_STATE);
    get().fetchAttention();
  },
}));
