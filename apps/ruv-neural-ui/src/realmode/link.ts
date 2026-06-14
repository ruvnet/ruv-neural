/**
 * Real-mode device bridge (ADR-0014 §7.3, Phase 4).
 *
 * Real mode is **explicit and gated**: nothing here runs until the operator
 * opts in. Two transports implement one `DeviceLink` interface:
 *   - `WebSerialLink` — best-effort `navigator.serial` bridge to firmware.
 *   - `MockLink` — an in-browser simulated device so the gated flow (validate,
 *     stimulate, emergency-stop, audit) is demonstrable and testable without
 *     hardware.
 *
 * Safety is enforced above the transport (intensity ceiling, validate-before-
 * stimulate, no auto-restart after an emergency stop).
 */

export type Modality = "light" | "audio" | "haptic";

export interface DeviceInfo {
  name: string;
  firmware: string;
  /** Device-reported maximum normalized intensity. */
  maxIntensity: number;
  /** Hardware safety interlock present and armed. */
  safetyInterlock: boolean;
  channels: Modality[];
}

export interface ValidationResult {
  ok: boolean;
  info?: DeviceInfo;
  reason?: string;
}

export interface StimCommand {
  modality: Modality;
  frequencyHz: number;
  intensity: number;
}

export interface DeviceEvent {
  /** Session-relative ms. */
  t: number;
  kind: string;
  detail: string;
}

export interface DeviceLink {
  readonly kind: "webserial" | "mock";
  connect(): Promise<void>;
  validate(): Promise<ValidationResult>;
  startStimulus(cmd: StimCommand): Promise<void>;
  stopStimulus(): Promise<void>;
  emergencyStop(): Promise<void>;
  disconnect(): Promise<void>;
  onEvent(cb: (e: DeviceEvent) => void): void;
}

export interface Capabilities {
  webSerial: boolean;
  webUsb: boolean;
  webBluetooth: boolean;
}

export function detectCapabilities(): Capabilities {
  const nav = (typeof navigator !== "undefined" ? navigator : {}) as Record<string, unknown>;
  return {
    webSerial: "serial" in nav,
    webUsb: "usb" in nav,
    webBluetooth: "bluetooth" in nav,
  };
}

/** Shared base: timestamps + event fan-out. */
abstract class BaseLink implements DeviceLink {
  abstract readonly kind: "webserial" | "mock";
  protected started = Date.now();
  private listeners: ((e: DeviceEvent) => void)[] = [];

  onEvent(cb: (e: DeviceEvent) => void): void {
    this.listeners.push(cb);
  }
  protected emit(kind: string, detail: string): void {
    const e: DeviceEvent = { t: Date.now() - this.started, kind, detail };
    for (const l of this.listeners) l(e);
  }

  abstract connect(): Promise<void>;
  abstract validate(): Promise<ValidationResult>;
  abstract startStimulus(cmd: StimCommand): Promise<void>;
  abstract stopStimulus(): Promise<void>;
  abstract emergencyStop(): Promise<void>;
  abstract disconnect(): Promise<void>;
}

/**
 * A deterministic in-browser mock device. It models a compliant Ruflo firmware:
 * a validation handshake reporting a 0.6 intensity ceiling and an armed safety
 * interlock, and command acknowledgements.
 */
export class MockLink extends BaseLink {
  readonly kind = "mock" as const;
  private estopped = false;

  async connect(): Promise<void> {
    this.started = Date.now();
    this.estopped = false;
    this.emit("connect", "mock device connected");
  }

  async validate(): Promise<ValidationResult> {
    const info: DeviceInfo = {
      name: "Ruflo Mock Stimulator",
      firmware: "ruflo-fw/1.0.0-mock",
      maxIntensity: 0.6,
      safetyInterlock: true,
      channels: ["light", "audio", "haptic"],
    };
    this.emit("validate", `handshake ok · fw ${info.firmware} · ceiling ${info.maxIntensity}`);
    return { ok: true, info };
  }

  async startStimulus(cmd: StimCommand): Promise<void> {
    if (this.estopped) throw new Error("device is emergency-stopped; reconnect required");
    this.emit(
      "stimulate",
      `${cmd.modality} ${cmd.frequencyHz} Hz @ ${cmd.intensity.toFixed(2)}`,
    );
  }

  async stopStimulus(): Promise<void> {
    this.emit("stop", "stimulus stopped (intensity 0)");
  }

  async emergencyStop(): Promise<void> {
    this.estopped = true;
    this.emit("estop", "EMERGENCY STOP — all channels forced to zero");
  }

  async disconnect(): Promise<void> {
    this.emit("disconnect", "mock device disconnected");
  }
}

/**
 * Best-effort Web Serial bridge. Uses a simple newline-delimited JSON protocol
 * (`{cmd:"HELLO"}` → `{ok,info}`; `{cmd:"STIM",...}`; `{cmd:"ESTOP"}`). Requires
 * compatible firmware and a user gesture for `requestPort`. Untestable without
 * hardware; the Mock is the demonstrable path.
 */
export class WebSerialLink extends BaseLink {
  readonly kind = "webserial" as const;
  private port: unknown = null;
  private writer: WritableStreamDefaultWriter<Uint8Array> | null = null;

  async connect(): Promise<void> {
    const serial = (navigator as unknown as { serial?: { requestPort(): Promise<unknown> } }).serial;
    if (!serial) throw new Error("Web Serial API not available in this browser");
    const port = (await serial.requestPort()) as {
      open(opts: { baudRate: number }): Promise<void>;
      writable: WritableStream<Uint8Array>;
    };
    await port.open({ baudRate: 115200 });
    this.port = port;
    this.writer = port.writable.getWriter();
    this.started = Date.now();
    this.emit("connect", "serial port opened @ 115200");
  }

  private async send(obj: Record<string, unknown>): Promise<void> {
    if (!this.writer) throw new Error("not connected");
    await this.writer.write(new TextEncoder().encode(JSON.stringify(obj) + "\n"));
  }

  async validate(): Promise<ValidationResult> {
    await this.send({ cmd: "HELLO" });
    // A production bridge reads the firmware's reply here; without a confirmed
    // protocol we report that validation is pending operator confirmation.
    this.emit("validate", "HELLO sent — awaiting firmware handshake");
    return { ok: false, reason: "firmware handshake not confirmed (connect real hardware)" };
  }

  async startStimulus(cmd: StimCommand): Promise<void> {
    await this.send({ cmd: "STIM", ...cmd });
    this.emit("stimulate", `${cmd.modality} ${cmd.frequencyHz} Hz @ ${cmd.intensity.toFixed(2)}`);
  }

  async stopStimulus(): Promise<void> {
    await this.send({ cmd: "STOP" });
    this.emit("stop", "stop sent");
  }

  async emergencyStop(): Promise<void> {
    await this.send({ cmd: "ESTOP" });
    this.emit("estop", "EMERGENCY STOP sent");
  }

  async disconnect(): Promise<void> {
    try {
      this.writer?.releaseLock();
      await (this.port as { close?: () => Promise<void> })?.close?.();
    } finally {
      this.emit("disconnect", "serial port closed");
      this.port = null;
      this.writer = null;
    }
  }
}
