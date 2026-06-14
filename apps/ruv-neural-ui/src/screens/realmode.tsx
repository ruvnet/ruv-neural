import { useRealStore, effectiveCeiling, REAL_SESSION_LIMIT_S } from "../store/realStore";
import { Card, Stat, StatusPill, Mono } from "../components/common";
import { Bar } from "../components/charts";
import { fmt, hashShort, titleCase } from "../util/format";
import { SAFETY } from "../util/constants";
import type { Modality } from "../realmode/link";

/**
 * Real mode (ADR-0014 §7.3, Phase 4) — explicit, gated local-hardware control.
 * Excluded from the public default flow; every safety control is visible and a
 * hash-chained device-event log is kept locally.
 */
export function RealMode() {
  const s = useRealStore();
  const connected = ["validating", "ready", "running", "estopped"].includes(s.status);

  return (
    <div className="real">
      {connected && <EStopBar />}
      <div className="grid">
        <Card
          title="Real mode — local hardware"
          subtitle="Experimental · gated · local-only"
          right={<StatusPill ok={s.status === "estopped" ? false : s.status === "ready" || s.status === "running" ? true : null}>{s.status.toUpperCase()}</StatusPill>}
        >
          <p className="muted">
            Real mode connects to a local Ruflo stimulator over Web Serial (or an in-browser mock
            device). It is <strong>off by default</strong> and never uploads anything. Intensity is
            capped at {SAFETY.intensityCeiling.toFixed(2)} and a single session is limited to{" "}
            {Math.round(REAL_SESSION_LIMIT_S / 60)} minutes.
          </p>
          {!s.gated ? <ConsentGate /> : <ConnectAndControl />}
          {s.error && <div className="breach-box" data-testid="real-error">{s.error}</div>}
        </Card>

        <DeviceLog />
      </div>
    </div>
  );
}

function EStopBar() {
  const { estop, status } = useRealStore();
  return (
    <div className="estop-bar" data-testid="estop-bar">
      <span>{status === "estopped" ? "EMERGENCY STOP ENGAGED — reconnect to resume" : "Stimulation can be halted instantly"}</span>
      <button className="estop-btn" onClick={() => estop()} disabled={status === "estopped"} data-testid="estop-btn">
        ⛔ EMERGENCY STOP
      </button>
    </div>
  );
}

function ConsentGate() {
  const { caps, consent, setConsent, optIn, transport } = useRealStore();
  const webSerialMissing = transport === "webserial" && !caps.webSerial;
  return (
    <div className="gate" data-testid="consent-gate">
      <h4>Before enabling real mode</h4>
      <label>
        <input type="checkbox" checked={consent.boundary} onChange={(e) => setConsent("boundary", e.target.checked)} data-testid="ack-boundary" />
        I understand this is wellness/cognitive-state research on safe external sensory channels and{" "}
        <strong>not a medical device</strong>; it does not diagnose, treat, cure, or prevent disease.
      </label>
      <label>
        <input type="checkbox" checked={consent.contraindication} onChange={(e) => setConsent("contraindication", e.target.checked)} data-testid="ack-contra" />
        I have completed a contraindication screen (epilepsy / photosensitivity / pregnancy /
        implanted devices) and have no contraindication to sensory stimulation.
      </label>
      <label>
        <input type="checkbox" checked={consent.photosensitivity} onChange={(e) => setConsent("photosensitivity", e.target.checked)} data-testid="ack-photo" />
        I acknowledge photosensitivity, comfortable-listening, and vibration-intensity cautions, and
        that I can stop at any time.
      </label>
      <div className="caps-row">
        <Stat label="Web Serial" value={caps.webSerial ? "available" : "unavailable"} />
        <Stat label="Web USB" value={caps.webUsb ? "available" : "unavailable"} />
        <Stat label="Web Bluetooth" value={caps.webBluetooth ? "available" : "unavailable"} />
      </div>
      {webSerialMissing && <p className="muted">Web Serial is unavailable here — use the mock device.</p>}
      <button className="primary" onClick={() => optIn()} data-testid="optin-btn">Enable real mode</button>
    </div>
  );
}

function ConnectAndControl() {
  const s = useRealStore();
  return (
    <div className="control">
      <div className="transport-row">
        <span className="muted">Transport:</span>
        {(["mock", "webserial"] as const).map((t) => (
          <button
            key={t}
            className={`chip ${s.transport === t ? "active" : ""}`}
            disabled={t === "webserial" && !s.caps.webSerial}
            onClick={() => s.setTransport(t)}
            data-testid={`transport-${t}`}
          >
            {t === "mock" ? "Mock device" : "Web Serial"}
          </button>
        ))}
      </div>

      <div className="btn-row">
        <button className="primary" disabled={s.status !== "disconnected"} onClick={() => s.connect()} data-testid="connect-btn">Connect</button>
        <button disabled={s.status !== "validating"} onClick={() => s.validate()} data-testid="validate-btn">Validate hardware</button>
        <button disabled={!["disconnected", "validating", "ready", "running", "estopped"].includes(s.status) || s.status === "disconnected"} onClick={() => s.disconnect()}>Disconnect</button>
      </div>

      {s.info && (
        <div className="validation" data-testid="validation-panel">
          <div className="stat-row">
            <Stat label="Device" value={s.info.name} />
            <Stat label="Firmware" value={<Mono>{s.info.firmware}</Mono>} />
            <Stat label="Device ceiling" value={fmt(s.info.maxIntensity, 2)} />
            <Stat label="Interlock" value={<StatusPill ok={s.info.safetyInterlock}>{s.info.safetyInterlock ? "armed" : "absent"}</StatusPill>} />
          </div>
        </div>
      )}

      {(s.status === "ready" || s.status === "running") && (
        <div className="stim" data-testid="stim-controls">
          <div className="transport-row">
            <span className="muted">Modality:</span>
            {(["light", "audio", "haptic"] as Modality[]).map((m) => (
              <button key={m} className={`chip ${s.modality === m ? "active" : ""}`} onClick={() => s.setModality(m)}>{m}</button>
            ))}
          </div>
          <Bar value={s.intensity} color="#5b8cff" label={<span>intensity {fmt(s.intensity, 2)} · effective ceiling {fmt(effectiveCeiling(s.info), 2)} · 40 Hz</span>} />
          <input
            type="range"
            min={0}
            max={effectiveCeiling(s.info)}
            step={0.01}
            value={s.intensity}
            onChange={(e) => s.setIntensity(Number(e.target.value))}
            data-testid="intensity-slider"
          />
          <div className="btn-row">
            <button className="primary" disabled={s.status === "running"} onClick={() => s.start()} data-testid="start-btn">Start 40 Hz</button>
            <button disabled={s.status !== "running"} onClick={() => s.stop()} data-testid="stop-btn">Stop</button>
          </div>
        </div>
      )}

      {s.status === "estopped" && (
        <p className="decision decision-stop">Emergency stop engaged. No auto-restart — disconnect and reconnect to run again.</p>
      )}
    </div>
  );
}

function DeviceLog() {
  const { events, chainValid, chain } = useRealStore();
  return (
    <Card
      title="Device audit log"
      subtitle="Hash-chained, local — every device event"
      right={<StatusPill ok={events.length ? chainValid : null}>{events.length ? (chainValid ? "CHAIN VALID" : "INVALID") : "EMPTY"}</StatusPill>}
    >
      {events.length === 0 ? (
        <div className="empty">No device events yet. Connect a device to begin.</div>
      ) : (
        <>
          <div className="stat-row">
            <Stat label="Events" value={events.length} />
            <Stat label="Chain head" value={<Mono title={chain.head}>{hashShort(chain.head)}</Mono>} />
          </div>
          <div className="timeline" data-testid="device-log">
            {events.map((r, i) => (
              <div key={i} className="tl-row">
                <span className="tl-idx">{r.event.t}ms</span>
                <span className={`tl-kind kind-${r.event.kind === "estop" ? "safe_stop" : r.event.kind}`}>{titleCase(r.event.kind)}</span>
                <span className="tl-meta">{r.event.detail}</span>
                <Mono title={`prev ${r.prevHash}\nhash ${r.hash}`}>{hashShort(r.hash, 8, 6)}</Mono>
              </div>
            ))}
          </div>
        </>
      )}
    </Card>
  );
}
