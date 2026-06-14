import { useStore, type ScreenId } from "../store/store";
import { Card, Stat, StatusPill, Mono, Empty } from "../components/common";
import { LineChart, Bar, Legend } from "../components/charts";
import { PlaybackControls } from "../components/PlaybackControls";
import { fmt, hashShort, titleCase } from "../util/format";
import { SAFETY, PLATFORM, BOUNDARY } from "../util/constants";
import type { EvidenceBundle, Step } from "../schemas/evidence";

function useBundle(): { bundle: EvidenceBundle; step: number; cur: Step } | null {
  const { bundle, step } = useStore();
  if (!bundle || bundle.steps.length === 0) return null;
  const idx = Math.min(step, bundle.steps.length - 1);
  return { bundle, step: idx, cur: bundle.steps[idx] };
}

function phaseTone(phase: string): boolean | null {
  if (phase === "Completed") return true;
  if (phase === "SafeStopped" || phase === "Aborted") return false;
  return null;
}

function NoData() {
  return (
    <Empty>
      No session loaded. Pick a <strong>Demo</strong> scenario or import a Ruflo evidence bundle in{" "}
      <strong>Replay</strong>.
    </Empty>
  );
}

function stopIndices(b: EvidenceBundle): number[] {
  return b.steps.map((s, i) => (s.phase === "SafeStopped" ? i : -1)).filter((i) => i >= 0);
}

// ── Overview ────────────────────────────────────────────────────────────────

export function Overview() {
  const data = useBundle();
  return (
    <div className="grid">
      <Card title="What is Ruflo?" subtitle="Closed-loop sensory neuromodulation — the public proof surface">
        <p>
          Ruflo drives cognitive state with a <strong>closed loop</strong> on safe external sensory
          channels: detect the state, deliver a <em>verified</em> 40&nbsp;Hz stimulus, measure the
          physiological response, adapt conservatively, and <strong>stop safely</strong> the moment
          the response leaves an allowed envelope.
        </p>
        <pre className="diagram">{`observe ─▶ embed (ruVector) ─▶ estimate ─▶ SAFETY ENVELOPE
                                              │
                  within ────────────────────┴───── breach
                     │                                  │
           select protocol & dose               fail-safe STOP
                     │                            (intensity 0)
           deliver VERIFIED stimulus ─▶ audit (hash-chained, signed)`}</pre>
        <div className="boundary-inline">{BOUNDARY}</div>
      </Card>

      <Card title="Platform" subtitle="Workspace attestation (static)">
        <div className="stat-row">
          <Stat label="Version" value={PLATFORM.version} />
          <Stat label="Crates" value={PLATFORM.crates} />
          <Stat label="Tests" value={PLATFORM.workspaceTests} />
          <Stat label="Attestations" value={PLATFORM.attestations} />
          <Stat label="Entrainment" value={PLATFORM.gammaHz} unit="Hz" />
        </div>
        <p className="muted">
          Channels: 40&nbsp;Hz light / audio / haptic. Response: HRV · respiration · motion · sleep
          proxy. Control: ruVector embedding, safety envelope, conservative dosing, hash-chained
          audit.
        </p>
      </Card>

      {data ? <SessionSummaryCard /> : <Card title="Get started">{<NoData />}</Card>}
      <VerificationCard />
    </div>
  );
}

function SessionSummaryCard() {
  const { bundle, verification } = useStore();
  if (!bundle) return null;
  const a = bundle.acceptance;
  return (
    <Card
      title="Session summary"
      subtitle={`${bundle.targetState} · ${bundle.protocol}`}
      right={<StatusPill ok={a.passed}>{a.passed ? "ACCEPTANCE PASS" : "FAIL"}</StatusPill>}
    >
      <div className="stat-row">
        <Stat label="Outcome" value={bundle.report.final_phase} />
        <Stat label="Steps" value={bundle.report.total_steps} />
        <Stat label="Stim time" value={fmt(bundle.report.total_stimulation_s, 0)} unit="s" />
        <Stat label="Peak intensity" value={fmt(bundle.report.peak_intensity)} />
        <Stat label="Receipts" value={bundle.report.num_receipts} />
      </div>
      <div className="clause-grid">
        <Clause ok={a.targetStateIdentified} label="Target identified" />
        <Clause ok={a.verifiedStimulusDelivered} label="Verified stimulus delivered" />
        <Clause ok={a.responseMeasured} label="Response measured" />
        <Clause
          ok={a.stoppedSafelyOutsideEnvelope || a.goalReached}
          label={a.goalReached ? "Target reached" : "Stopped safely outside envelope"}
        />
      </div>
      {verification && (
        <p className="muted">
          Local verification: <StatusPill ok={verification.ok}>{verification.ok ? "ALL CHECKS PASS" : "SEE DETAILS"}</StatusPill>
        </p>
      )}
    </Card>
  );
}

function Clause({ ok, label }: { ok: boolean; label: string }) {
  return (
    <div className={`clause ${ok ? "clause-ok" : "clause-bad"}`}>
      <span className="clause-mark">{ok ? "✓" : "✗"}</span> {label}
    </div>
  );
}

function VerificationCard() {
  const { verification } = useStore();
  if (!verification) return null;
  return (
    <Card title="Local verification" subtitle="Recomputed in your browser — nothing is trusted blindly" right={<StatusPill ok={verification.ok}>{verification.ok ? "VERIFIED" : "FAILED"}</StatusPill>}>
      <ul className="checklist" data-testid="checklist">
        {verification.checks.map((c) => (
          <li key={c.id} className={c.ok ? "check-ok" : "check-bad"}>
            <span className="check-mark">{c.ok ? "✓" : "✗"}</span>
            <span className="check-label">{c.label}</span>
            <span className="check-detail">{c.detail}</span>
          </li>
        ))}
      </ul>
    </Card>
  );
}

// ── Live Session ─────────────────────────────────────────────────────────────

export function LiveSession() {
  const data = useBundle();
  if (!data) return <Card title="Live session"><NoData /></Card>;
  const { bundle, step, cur } = data;
  const distances = bundle.steps.map((s) => s.distanceToTarget);
  const intensities = bundle.steps.map((s) => s.intensity);
  const modalities = cur.receipts.map((r) => r.modality);

  return (
    <div className="grid">
      <Card
        title="Live session"
        subtitle={`${bundle.targetState} · ${bundle.protocol}`}
        right={<StatusPill ok={phaseTone(cur.phase)}>{cur.phase}</StatusPill>}
      >
        <PlaybackControls />
        <div className="stat-row">
          <Stat label="Phase" value={cur.phase} />
          <Stat label="Intensity" value={fmt(cur.intensity)} />
          <Stat label="Distance → target" value={fmt(cur.distanceToTarget, 3)} />
          <Stat label="Arousal" value={fmt(cur.biosense.arousalScore)} />
          <Stat label="Relaxation" value={fmt(cur.biosense.relaxationScore)} />
        </div>
        <div className="modality-row">
          {(["light", "audio", "haptic"] as const).map((m) => (
            <span key={m} className={`mod-chip ${modalities.includes(m) ? "mod-on" : "mod-off"}`}>
              {m} {modalities.includes(m) ? "● 40 Hz" : "○"}
            </span>
          ))}
        </div>
        <p className={`decision ${cur.envelope.within ? "" : "decision-stop"}`}>
          {cur.envelope.within
            ? "Within safety envelope — control proceeding."
            : `FAIL-SAFE STOP — ${cur.envelope.breaches.join(", ")}`}
        </p>
      </Card>

      <Card title="Convergence" subtitle="Distance-to-target & commanded intensity over the session">
        <LineChart
          series={[
            { label: "distance", color: "#5b8cff", values: distances },
            { label: "intensity", color: "#42d6a4", values: intensities },
          ]}
          yDomain={[0, 1]}
          cursor={step}
          stops={stopIndices(bundle)}
          thresholds={[{ value: SAFETY.intensityCeiling, color: "#f7b955" }]}
        />
        <Legend
          items={[
            { label: "distance → target", color: "#5b8cff" },
            { label: "intensity", color: "#42d6a4" },
            { label: "intensity ceiling", color: "#f7b955" },
            { label: "safe stop", color: "#ff5d6c" },
          ]}
        />
      </Card>
    </div>
  );
}

// ── Stimulus Verifier ────────────────────────────────────────────────────────

export function StimulusVerifier() {
  const data = useBundle();
  if (!data) return <Card title="Stimulus verifier"><NoData /></Card>;
  const { bundle, cur } = data;
  const totalVerified = bundle.steps.flatMap((s) => s.receipts).filter((r) => r.verified).length;
  const total = bundle.steps.flatMap((s) => s.receipts).length;

  return (
    <div className="grid">
      <Card
        title="Stimulus verifier"
        subtitle={`Receipts at step ${cur.index} · tolerance ±${SAFETY.frequencyToleranceHz} Hz`}
        right={<StatusPill ok={total > 0 && totalVerified === total}>{totalVerified}/{total} verified</StatusPill>}
      >
        <PlaybackControls />
        {cur.receipts.length === 0 ? (
          <Empty>No active stimulation this step (rest / hold / safe-stop).</Empty>
        ) : (
          <table className="table" data-testid="receipt-table">
            <thead>
              <tr>
                <th>Modality</th>
                <th>Intended Hz</th>
                <th>Measured Hz</th>
                <th>Error Hz</th>
                <th>Duty</th>
                <th>Intensity</th>
                <th>Verified</th>
                <th>Waveform SHA-256</th>
              </tr>
            </thead>
            <tbody>
              {cur.receipts.map((r, i) => (
                <tr key={i}>
                  <td>{r.modality}</td>
                  <td>{fmt(r.intendedFrequencyHz, 1)}</td>
                  <td>{fmt(r.measuredFrequencyHz, 1)}</td>
                  <td>{fmt(r.frequencyErrorHz, 2)}</td>
                  <td>{fmt(r.dutyCycle, 2)}</td>
                  <td>{fmt(r.intensity, 2)}</td>
                  <td><StatusPill ok={r.verified}>{r.verified ? "yes" : "no"}</StatusPill></td>
                  <td><Mono title={r.waveformSha256}>{hashShort(r.waveformSha256)}</Mono></td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
        <p className="muted">
          A receipt is <em>verified</em> only when its waveform hash is present and the measured
          entrainment frequency is within tolerance of the command.
        </p>
      </Card>
    </div>
  );
}

// ── Biosense ─────────────────────────────────────────────────────────────────

export function Biosense() {
  const data = useBundle();
  if (!data) return <Card title="Biosense"><NoData /></Card>;
  const { bundle, step, cur } = data;
  const b = cur.biosense;
  return (
    <div className="grid">
      <Card title="Biosense" subtitle={`Physiological response at step ${cur.index}`}>
        <PlaybackControls />
        <div className="stat-row wrap">
          <Stat label="Heart rate" value={fmt(b.heartRateBpm, 1)} unit="bpm" />
          <Stat label="SDNN" value={fmt(b.sdnnMs, 1)} unit="ms" />
          <Stat label="RMSSD" value={fmt(b.rmssdMs, 1)} unit="ms" />
          <Stat label="pNN50" value={fmt(b.pnn50, 2)} />
          <Stat label="LF/HF" value={fmt(b.lfHfRatio, 2)} />
          <Stat label="Respiration" value={fmt(b.respirationRateBpm, 1)} unit="bpm" />
          <Stat label="Motion" value={fmt(b.motionIndex, 3)} unit="g" />
          <Stat label="Stillness" value={fmt(b.stillness, 2)} />
        </div>
      </Card>
      <Card title="Autonomic state" subtitle="Arousal vs. relaxation over the session">
        <LineChart
          series={[
            { label: "arousal", color: "#ff8c5a", values: bundle.steps.map((s) => s.biosense.arousalScore) },
            { label: "relaxation", color: "#42d6a4", values: bundle.steps.map((s) => s.biosense.relaxationScore) },
          ]}
          yDomain={[0, 1]}
          cursor={step}
          stops={stopIndices(bundle)}
          thresholds={[{ value: SAFETY.maxArousal, color: "#ff5d6c", label: "arousal ceiling" }]}
        />
        <Legend
          items={[
            { label: "arousal", color: "#ff8c5a" },
            { label: "relaxation", color: "#42d6a4" },
            { label: "arousal ceiling", color: "#ff5d6c" },
          ]}
        />
      </Card>
    </div>
  );
}

// ── ruVector ─────────────────────────────────────────────────────────────────

export function RuVector() {
  const data = useBundle();
  if (!data) return <Card title="ruVector state"><NoData /></Card>;
  const { bundle, step, cur } = data;
  return (
    <div className="grid">
      <Card title="ruVector personal state embedding" subtitle={`9-D fusion at step ${cur.index}`}>
        <PlaybackControls />
        <div className="embedding">
          {cur.embedding.map((v, i) => (
            <Bar
              key={i}
              value={v}
              color="#7c9cff"
              label={
                <span className="emb-label">
                  {cur.featureNames[i]} <span className="muted">{fmt(v, 2)}</span>
                </span>
              }
            />
          ))}
        </div>
      </Card>
      <Card title="Convergence" subtitle="Distance-to-target trajectory">
        <LineChart
          series={[{ label: "distance", color: "#5b8cff", values: bundle.steps.map((s) => s.distanceToTarget) }]}
          yDomain={[0, 1]}
          cursor={step}
          stops={stopIndices(bundle)}
        />
        <div className="stat-row">
          <Stat label="Best distance" value={fmt(bundle.report.best_distance, 3)} />
          <Stat label="Final distance" value={fmt(bundle.report.final_distance, 3)} />
        </div>
      </Card>
    </div>
  );
}

// ── Safety Envelope ──────────────────────────────────────────────────────────

export function SafetyEnvelope() {
  const data = useBundle();
  if (!data) return <Card title="Safety envelope"><NoData /></Card>;
  const { bundle, step, cur } = data;
  const hr = bundle.steps.map((s) => s.biosense.heartRateBpm ?? SAFETY.minHrBpm);

  return (
    <div className="grid">
      <Card
        title="Safety envelope"
        subtitle="Stimulation continues only while proved safe"
        right={<StatusPill ok={cur.envelope.within}>{cur.envelope.within ? "WITHIN" : "BREACH"}</StatusPill>}
      >
        <PlaybackControls />
        <div className="stat-row">
          <Stat label="HR ceiling" value={SAFETY.maxHrBpm} unit="bpm" />
          <Stat label="HR floor" value={SAFETY.minHrBpm} unit="bpm" />
          <Stat label="Arousal ceiling" value={SAFETY.maxArousal} />
          <Stat label="Motion ceiling" value={SAFETY.maxMovementIndex} unit="g" />
          <Stat label="Divergence tol." value={SAFETY.divergenceTolerance} />
        </div>
        {cur.envelope.within ? (
          <p className="decision">Current response is inside the allowed envelope.</p>
        ) : (
          <div className="breach-box" data-testid="breach-box">
            <strong>Fail-safe stop — intensity forced to 0.</strong>
            <ul>
              {cur.envelope.breaches.map((br, i) => (
                <li key={i}>{br}</li>
              ))}
            </ul>
          </div>
        )}
      </Card>
      <Card title="Heart rate vs. envelope" subtitle="Shaded band = allowed HR range">
        <LineChart
          series={[{ label: "HR", color: "#ff8c5a", values: hr }]}
          yDomain={[40, 110]}
          cursor={step}
          stops={stopIndices(bundle)}
          band={{ from: SAFETY.minHrBpm, to: SAFETY.maxHrBpm, color: "#42d6a4" }}
          thresholds={[
            { value: SAFETY.maxHrBpm, color: "#ff5d6c" },
            { value: SAFETY.minHrBpm, color: "#ff5d6c" },
          ]}
        />
      </Card>
    </div>
  );
}

// ── Audit Trail ──────────────────────────────────────────────────────────────

export function AuditTrail() {
  const data = useBundle();
  const { verification } = useStore();
  if (!data) return <Card title="Audit trail"><NoData /></Card>;
  const { bundle } = data;
  const chainCheck = verification?.checks.find((c) => c.id === "chain");

  return (
    <div className="grid">
      <Card
        title="Audit trail"
        subtitle="Tamper-evident, hash-chained — recomputed locally"
        right={<StatusPill ok={chainCheck?.ok ?? null}>{chainCheck?.ok ? "CHAIN VALID" : "INVALID"}</StatusPill>}
      >
        <div className="stat-row">
          <Stat label="Bundle chain head" value={<Mono title={bundle.bundleChainHead}>{hashShort(bundle.bundleChainHead)}</Mono>} />
          <Stat label="Internal audit head" value={<Mono title={bundle.auditHeadHash}>{hashShort(bundle.auditHeadHash)}</Mono>} />
          <Stat label="Audit records" value={bundle.auditRecords} />
          <Stat label="Signature" value={bundle.signature ? "Ed25519" : "none"} />
        </div>
        <div className="timeline" data-testid="timeline">
          {bundle.steps.map((s) => (
            <div key={s.index} className="tl-row">
              <span className="tl-idx">#{s.index}</span>
              <span className={`tl-kind kind-${s.auditKind}`}>{titleCase(s.auditKind)}</span>
              <span className="tl-meta">t={fmt(s.timestampS, 0)}s · I={fmt(s.intensity)} · d={fmt(s.distanceToTarget, 3)}</span>
              <Mono title={`prev ${s.prevHash}\nhash ${s.hash}`}>{hashShort(s.hash, 8, 6)}</Mono>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}

// ── Witness ──────────────────────────────────────────────────────────────────

export function Witness() {
  const { bundle, verification } = useStore();
  return (
    <div className="grid">
      <Card title="Witness & evidence" subtitle="Acceptance clauses and attestation">
        <div className="stat-row">
          <Stat label="Workspace tests" value={PLATFORM.workspaceTests} />
          <Stat label="Attestations" value={PLATFORM.attestations} />
          <Stat label="Crates" value={PLATFORM.crates} />
          <Stat label="Version" value={PLATFORM.version} />
        </div>
        {bundle ? (
          <div className="clause-grid">
            <Clause ok={bundle.acceptance.targetStateIdentified} label="Identify a target state" />
            <Clause ok={bundle.acceptance.verifiedStimulusDelivered} label="Deliver a verified stimulus" />
            <Clause ok={bundle.acceptance.responseMeasured} label="Measure a response" />
            <Clause
              ok={bundle.acceptance.stoppedSafelyOutsideEnvelope || bundle.acceptance.goalReached}
              label="Stop safely outside envelope (or reach target)"
            />
          </div>
        ) : (
          <NoData />
        )}
      </Card>
      {verification && <VerificationCard />}
    </div>
  );
}

// ── dispatcher ───────────────────────────────────────────────────────────────

const SCREENS: Record<ScreenId, () => JSX.Element> = {
  overview: Overview,
  session: LiveSession,
  stimulus: StimulusVerifier,
  biosense: Biosense,
  ruvector: RuVector,
  safety: SafetyEnvelope,
  audit: AuditTrail,
  witness: Witness,
};

export function ActiveScreen() {
  const screen = useStore((s) => s.screen);
  const Comp = SCREENS[screen];
  return <Comp />;
}
