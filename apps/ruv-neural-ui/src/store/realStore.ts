import { create } from "zustand";
import {
  detectCapabilities,
  MockLink,
  WebSerialLink,
  type Capabilities,
  type DeviceInfo,
  type DeviceLink,
  type Modality,
} from "../realmode/link";
import { DeviceAuditChain, type DeviceAuditRecord } from "../realmode/audit";
import { SAFETY } from "../util/constants";

export type RealStatus =
  | "locked" // not opted in
  | "disconnected"
  | "connecting"
  | "validating"
  | "ready"
  | "running"
  | "estopped"
  | "error";

export type Transport = "mock" | "webserial";

/** Hard cap on a single real-mode session (defense-in-depth, ADR-0014 §7.3). */
export const REAL_SESSION_LIMIT_S = 1200;

interface ConsentState {
  boundary: boolean;
  contraindication: boolean;
  photosensitivity: boolean;
}

interface RealState {
  caps: Capabilities;
  consent: ConsentState;
  gated: boolean;
  transport: Transport;
  status: RealStatus;
  info: DeviceInfo | null;
  link: DeviceLink | null;
  chain: DeviceAuditChain;
  events: DeviceAuditRecord[];
  chainValid: boolean;
  modality: Modality;
  intensity: number;
  error: string | null;

  setConsent: (k: keyof ConsentState, v: boolean) => void;
  setTransport: (t: Transport) => void;
  optIn: () => void;
  connect: () => Promise<void>;
  validate: () => Promise<void>;
  setModality: (m: Modality) => void;
  setIntensity: (v: number) => void;
  start: () => Promise<void>;
  stop: () => Promise<void>;
  estop: () => Promise<void>;
  disconnect: () => Promise<void>;
  reset: () => void;
}

/** The intensity the device may actually be commanded: the lower of the UI
 *  ceiling and the device-reported ceiling. */
export function effectiveCeiling(info: DeviceInfo | null): number {
  return Math.min(SAFETY.intensityCeiling, info?.maxIntensity ?? SAFETY.intensityCeiling);
}

export const useRealStore = create<RealState>((set, get) => ({
  caps: detectCapabilities(),
  consent: { boundary: false, contraindication: false, photosensitivity: false },
  gated: false,
  transport: "mock",
  status: "locked",
  info: null,
  link: null,
  chain: new DeviceAuditChain(),
  events: [],
  chainValid: true,
  modality: "haptic",
  intensity: 0.2,
  error: null,

  setConsent: (k, v) => set((s) => ({ consent: { ...s.consent, [k]: v } })),
  setTransport: (t) => set({ transport: t }),

  optIn: () => {
    const { consent } = get();
    if (!consent.boundary || !consent.contraindication || !consent.photosensitivity) {
      set({ error: "All acknowledgements are required before enabling real mode." });
      return;
    }
    set({ gated: true, status: "disconnected", error: null });
  },

  connect: async () => {
    const { transport } = get();
    const link: DeviceLink = transport === "webserial" ? new WebSerialLink() : new MockLink();
    const chain = new DeviceAuditChain();
    link.onEvent((e) => {
      chain.append(e);
      set({ events: [...chain.all], chainValid: chain.verify() });
    });
    set({ status: "connecting", link, chain, events: [], error: null, info: null });
    try {
      await link.connect();
      set({ status: "validating" });
    } catch (e) {
      set({ status: "error", error: (e as Error).message });
    }
  },

  validate: async () => {
    const { link } = get();
    if (!link) return;
    set({ status: "validating", error: null });
    try {
      const result = await link.validate();
      if (result.ok && result.info) {
        set({ status: "ready", info: result.info });
      } else {
        set({ status: "error", error: result.reason ?? "validation failed" });
      }
    } catch (e) {
      set({ status: "error", error: (e as Error).message });
    }
  },

  setModality: (m) => set({ modality: m }),
  setIntensity: (v) => set({ intensity: Math.max(0, Math.min(1, v)) }),

  start: async () => {
    const { link, info, intensity, modality, status } = get();
    if (!link || status !== "ready") return;
    const capped = Math.min(intensity, effectiveCeiling(info));
    set({ status: "running", intensity: capped });
    try {
      await link.startStimulus({ modality, frequencyHz: 40, intensity: capped });
    } catch (e) {
      set({ status: "error", error: (e as Error).message });
    }
  },

  stop: async () => {
    const { link } = get();
    if (!link) return;
    await link.stopStimulus();
    if (get().status === "running") set({ status: "ready" });
  },

  estop: async () => {
    const { link } = get();
    if (!link) return;
    await link.emergencyStop();
    // No auto-restart: an emergency stop is terminal until reconnect.
    set({ status: "estopped" });
  },

  disconnect: async () => {
    const { link } = get();
    if (link) await link.disconnect();
    set({ status: "disconnected", link: null, info: null });
  },

  reset: () =>
    set({
      gated: false,
      status: "locked",
      link: null,
      info: null,
      chain: new DeviceAuditChain(),
      events: [],
      chainValid: true,
      consent: { boundary: false, contraindication: false, photosensitivity: false },
      error: null,
    }),
}));
