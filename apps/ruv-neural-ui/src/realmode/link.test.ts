import { describe, it, expect } from "vitest";
import { MockLink, detectCapabilities, type DeviceEvent } from "./link";
import { DeviceAuditChain } from "./audit";

describe("MockLink device flow", () => {
  it("connects, validates, stimulates, and emergency-stops", async () => {
    const link = new MockLink();
    const events: DeviceEvent[] = [];
    link.onEvent((e) => events.push(e));

    await link.connect();
    const v = await link.validate();
    expect(v.ok).toBe(true);
    expect(v.info?.maxIntensity).toBe(0.6);
    expect(v.info?.safetyInterlock).toBe(true);

    await link.startStimulus({ modality: "haptic", frequencyHz: 40, intensity: 0.3 });
    await link.emergencyStop();

    // After an emergency stop, stimulation is refused until reconnect.
    await expect(
      link.startStimulus({ modality: "haptic", frequencyHz: 40, intensity: 0.3 }),
    ).rejects.toThrow(/emergency-stopped/);

    const kinds = events.map((e) => e.kind);
    expect(kinds).toEqual(["connect", "validate", "stimulate", "estop"]);
  });

  it("detects capabilities without throwing", () => {
    const caps = detectCapabilities();
    expect(typeof caps.webSerial).toBe("boolean");
  });
});

describe("DeviceAuditChain", () => {
  it("chains events and verifies", () => {
    const chain = new DeviceAuditChain();
    chain.append({ t: 0, kind: "connect", detail: "ok" });
    chain.append({ t: 10, kind: "stimulate", detail: "haptic 40 Hz @ 0.30" });
    chain.append({ t: 20, kind: "estop", detail: "EMERGENCY STOP" });
    expect(chain.all.length).toBe(3);
    expect(chain.verify()).toBe(true);
  });

  it("detects tampering", () => {
    const chain = new DeviceAuditChain();
    chain.append({ t: 0, kind: "connect", detail: "ok" });
    chain.append({ t: 10, kind: "stimulate", detail: "haptic 40 Hz @ 0.30" });
    // Mutate a recorded event payload (the records are read-only by contract,
    // but we force it to prove the chain catches it).
    (chain.all[0] as { event: DeviceEvent }).event.detail = "tampered";
    expect(chain.verify()).toBe(false);
  });
});
