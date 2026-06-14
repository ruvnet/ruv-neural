import { create } from "zustand";
import type { EvidenceBundle } from "../schemas/evidence";
import { verifyBundle, type VerificationResult } from "../verifier/verify";
import { presetById } from "../fixtures";

/**
 * Phase 5 — guided research workflow (ADR-0014 §7.4). A non-medical, local-first
 * study flow: consent → contraindication → baseline → protocol → session →
 * survey → signed evidence export → local replay verification. The browser
 * cannot run the Rust controller, so the session step plays the deterministic
 * evidence bundle for the chosen protocol and wraps it with study metadata.
 */

export const STUDY_SCHEMA = "ruflo-study/1";

export type ResearchStep =
  | "consent"
  | "contraindication"
  | "baseline"
  | "protocol"
  | "session"
  | "survey"
  | "export";

export const STEP_ORDER: ResearchStep[] = [
  "consent",
  "contraindication",
  "baseline",
  "protocol",
  "session",
  "survey",
  "export",
];

export interface Consent {
  boundary: boolean; // not a medical device
  voluntary: boolean; // participation is voluntary, may stop anytime
  dataLocal: boolean; // data stays local unless exported
}

export interface Contraindication {
  epilepsy: boolean;
  photosensitivity: boolean;
  pregnancy: boolean;
  implant: boolean;
}

export interface Baseline {
  calm: number; // 0..10 subjective
  focus: number;
  energy: number;
}

export interface Survey {
  calm: number;
  focus: number;
  comfort: number; // tolerability
}

export interface StudyRecord {
  schemaVersion: typeof STUDY_SCHEMA;
  createdAt: string;
  consent: Consent;
  contraindication: Contraindication;
  baseline: Baseline;
  protocol: string;
  survey: Survey;
  bundle: EvidenceBundle;
}

interface ResearchState {
  step: ResearchStep;
  consent: Consent;
  contraindication: Contraindication;
  baseline: Baseline;
  protocol: "relaxed" | "focused" | "gamma";
  bundle: EvidenceBundle | null;
  verification: VerificationResult | null;
  survey: Survey;

  setConsent: (k: keyof Consent, v: boolean) => void;
  setContra: (k: keyof Contraindication, v: boolean) => void;
  setBaseline: (k: keyof Baseline, v: number) => void;
  setProtocol: (p: "relaxed" | "focused" | "gamma") => void;
  setSurvey: (k: keyof Survey, v: number) => void;
  next: () => void;
  back: () => void;
  reset: () => void;
  canProceed: () => boolean;
  hasContraindication: () => boolean;
  buildRecord: () => StudyRecord | null;
}

const presetForProtocol: Record<string, string> = {
  relaxed: "relaxed",
  focused: "focused",
  gamma: "gamma",
};

export const useResearchStore = create<ResearchState>((set, get) => ({
  step: "consent",
  consent: { boundary: false, voluntary: false, dataLocal: false },
  contraindication: { epilepsy: false, photosensitivity: false, pregnancy: false, implant: false },
  baseline: { calm: 5, focus: 5, energy: 5 },
  protocol: "relaxed",
  bundle: null,
  verification: null,
  survey: { calm: 5, focus: 5, comfort: 5 },

  setConsent: (k, v) => set((s) => ({ consent: { ...s.consent, [k]: v } })),
  setContra: (k, v) => set((s) => ({ contraindication: { ...s.contraindication, [k]: v } })),
  setBaseline: (k, v) => set((s) => ({ baseline: { ...s.baseline, [k]: v } })),
  setProtocol: (p) => set({ protocol: p }),
  setSurvey: (k, v) => set((s) => ({ survey: { ...s.survey, [k]: v } })),

  hasContraindication: () => {
    const c = get().contraindication;
    return c.epilepsy || c.photosensitivity || c.pregnancy || c.implant;
  },

  canProceed: () => {
    const s = get();
    switch (s.step) {
      case "consent":
        return s.consent.boundary && s.consent.voluntary && s.consent.dataLocal;
      case "contraindication":
        return !s.hasContraindication();
      default:
        return true;
    }
  },

  next: () => {
    const s = get();
    if (!s.canProceed()) return;
    const idx = STEP_ORDER.indexOf(s.step);
    const nextStep = STEP_ORDER[Math.min(idx + 1, STEP_ORDER.length - 1)];

    // Entering the session step loads + verifies the chosen protocol's bundle.
    if (nextStep === "session") {
      const preset = presetById(presetForProtocol[s.protocol]);
      const bundle = preset?.bundle ?? null;
      set({
        step: nextStep,
        bundle,
        verification: bundle ? verifyBundle(bundle) : null,
      });
      return;
    }
    set({ step: nextStep });
  },

  back: () => {
    const idx = STEP_ORDER.indexOf(get().step);
    set({ step: STEP_ORDER[Math.max(idx - 1, 0)] });
  },

  reset: () =>
    set({
      step: "consent",
      consent: { boundary: false, voluntary: false, dataLocal: false },
      contraindication: { epilepsy: false, photosensitivity: false, pregnancy: false, implant: false },
      baseline: { calm: 5, focus: 5, energy: 5 },
      protocol: "relaxed",
      bundle: null,
      verification: null,
      survey: { calm: 5, focus: 5, comfort: 5 },
    }),

  buildRecord: () => {
    const s = get();
    if (!s.bundle) return null;
    return {
      schemaVersion: STUDY_SCHEMA,
      createdAt: new Date().toISOString(),
      consent: s.consent,
      contraindication: s.contraindication,
      baseline: s.baseline,
      protocol: s.protocol,
      survey: s.survey,
      bundle: s.bundle,
    };
  },
}));
