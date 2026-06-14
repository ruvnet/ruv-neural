import { create } from "zustand";
import type { EvidenceBundle } from "../schemas/evidence";
import { parseBundle } from "../schemas/evidence";
import { verifyBundle, type VerificationResult } from "../verifier/verify";
import { DEMO_PRESETS, presetById } from "../fixtures";

export type ScreenId =
  | "overview"
  | "session"
  | "stimulus"
  | "biosense"
  | "ruvector"
  | "safety"
  | "audit"
  | "witness";

export type Mode = "demo" | "replay";

interface State {
  mode: Mode;
  bundle: EvidenceBundle | null;
  verification: VerificationResult | null;
  sourceLabel: string | null;
  presetId: string | null;
  error: string | null;
  screen: ScreenId;
  step: number; // playback cursor
  playing: boolean;

  setMode: (m: Mode) => void;
  setScreen: (s: ScreenId) => void;
  loadPreset: (id: string) => void;
  importJson: (text: string, filename: string) => void;
  clear: () => void;
  setStep: (i: number) => void;
  play: () => void;
  pause: () => void;
  reset: () => void;
}

function applyBundle(
  bundle: EvidenceBundle,
  mode: Mode,
  sourceLabel: string,
  presetId: string | null,
): Partial<State> {
  return {
    bundle,
    verification: verifyBundle(bundle),
    sourceLabel,
    presetId,
    mode,
    error: null,
    step: Math.max(0, bundle.steps.length - 1),
    playing: false,
    screen: "overview",
  };
}

export const useStore = create<State>((set, get) => ({
  mode: "demo",
  bundle: null,
  verification: null,
  sourceLabel: null,
  presetId: null,
  error: null,
  screen: "overview",
  step: 0,
  playing: false,

  setMode: (m) => set({ mode: m }),
  setScreen: (s) => set({ screen: s }),

  loadPreset: (id) => {
    const preset = presetById(id);
    if (!preset) {
      set({ error: `Unknown demo preset: ${id}` });
      return;
    }
    set(applyBundle(preset.bundle, "demo", preset.label, id));
  },

  importJson: (text, filename) => {
    try {
      const bundle = parseBundle(text);
      set(applyBundle(bundle, "replay", filename, null));
    } catch (e) {
      set({ error: (e as Error).message, bundle: null, verification: null });
    }
  },

  clear: () =>
    set({ bundle: null, verification: null, sourceLabel: null, presetId: null, error: null, step: 0, playing: false }),

  setStep: (i) => {
    const b = get().bundle;
    if (!b) return;
    set({ step: Math.min(Math.max(0, i), b.steps.length - 1) });
  },
  play: () => set({ playing: true }),
  pause: () => set({ playing: false }),
  reset: () => set({ step: 0, playing: false }),
}));

export { DEMO_PRESETS };
