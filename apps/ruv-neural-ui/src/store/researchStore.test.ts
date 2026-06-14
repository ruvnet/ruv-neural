import { describe, it, expect, beforeEach } from "vitest";
import { useResearchStore, STEP_ORDER } from "./researchStore";

describe("research workflow store", () => {
  beforeEach(() => useResearchStore.getState().reset());

  it("blocks consent until all three acknowledgements", () => {
    const s = useResearchStore.getState();
    expect(s.canProceed()).toBe(false);
    s.setConsent("boundary", true);
    s.setConsent("voluntary", true);
    expect(useResearchStore.getState().canProceed()).toBe(false);
    s.setConsent("dataLocal", true);
    expect(useResearchStore.getState().canProceed()).toBe(true);
  });

  it("blocks on any contraindication", () => {
    const s = useResearchStore.getState();
    // advance to contraindication
    s.setConsent("boundary", true);
    s.setConsent("voluntary", true);
    s.setConsent("dataLocal", true);
    s.next();
    expect(useResearchStore.getState().step).toBe("contraindication");
    useResearchStore.getState().setContra("epilepsy", true);
    expect(useResearchStore.getState().hasContraindication()).toBe(true);
    expect(useResearchStore.getState().canProceed()).toBe(false);
    useResearchStore.getState().setContra("epilepsy", false);
    expect(useResearchStore.getState().canProceed()).toBe(true);
  });

  it("loads and verifies a bundle when entering the session step", () => {
    const s = useResearchStore.getState();
    s.setConsent("boundary", true);
    s.setConsent("voluntary", true);
    s.setConsent("dataLocal", true);
    s.next(); // contraindication
    useResearchStore.getState().next(); // baseline
    useResearchStore.getState().next(); // protocol
    useResearchStore.getState().setProtocol("relaxed");
    useResearchStore.getState().next(); // session
    const st = useResearchStore.getState();
    expect(st.step).toBe("session");
    expect(st.bundle).not.toBeNull();
    expect(st.verification?.ok).toBe(true);
  });

  it("builds a complete study record", () => {
    const s = useResearchStore.getState();
    s.setConsent("boundary", true);
    s.setConsent("voluntary", true);
    s.setConsent("dataLocal", true);
    for (let i = 0; i < STEP_ORDER.length - 1; i++) useResearchStore.getState().next();
    const record = useResearchStore.getState().buildRecord();
    expect(record).not.toBeNull();
    expect(record?.schemaVersion).toBe("ruflo-study/1");
    expect(record?.bundle.acceptance.passed).toBe(true);
    expect(record?.consent.boundary).toBe(true);
  });
});
