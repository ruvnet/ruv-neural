import { useResearchStore, STEP_ORDER, type ResearchStep } from "../store/researchStore";
import { Card, Stat, StatusPill } from "../components/common";
import { titleCase } from "../util/format";
import { BOUNDARY } from "../util/constants";

/** Phase 5 — guided, non-medical, local-first research workflow (ADR-0014 §7.4). */
export function Research() {
  const s = useResearchStore();
  return (
    <div className="research">
      <Stepper current={s.step} />
      <div className="boundary-inline">{BOUNDARY}</div>
      <Card title={`Step ${STEP_ORDER.indexOf(s.step) + 1} of ${STEP_ORDER.length} — ${stepTitle(s.step)}`}>
        <StepBody />
        <Nav />
      </Card>
    </div>
  );
}

function stepTitle(step: ResearchStep): string {
  return {
    consent: "Consent & boundary",
    contraindication: "Contraindication screen",
    baseline: "Baseline capture",
    protocol: "Protocol selection",
    session: "Verified session",
    survey: "Post-session survey",
    export: "Signed evidence export",
  }[step];
}

function Stepper({ current }: { current: ResearchStep }) {
  const idx = STEP_ORDER.indexOf(current);
  return (
    <div className="stepper" data-testid="stepper">
      {STEP_ORDER.map((s, i) => (
        <div key={s} className={`step-dot ${i < idx ? "done" : i === idx ? "active" : ""}`}>
          <span className="dot">{i < idx ? "✓" : i + 1}</span>
          <span className="dot-label">{titleCase(s)}</span>
        </div>
      ))}
    </div>
  );
}

function Nav() {
  const { step, next, back, canProceed } = useResearchStore();
  const isFirst = STEP_ORDER.indexOf(step) === 0;
  const isLast = step === "export";
  return (
    <div className="btn-row" style={{ marginTop: 14 }}>
      {!isFirst && <button onClick={back} data-testid="research-back">← Back</button>}
      {!isLast && (
        <button className="primary" disabled={!canProceed()} onClick={next} data-testid="research-next">
          Continue →
        </button>
      )}
    </div>
  );
}

function Slider({ label, value, onChange }: { label: string; value: number; onChange: (v: number) => void }) {
  return (
    <label className="slider-row">
      <span>{label}</span>
      <input type="range" min={0} max={10} step={1} value={value} onChange={(e) => onChange(Number(e.target.value))} />
      <span className="slider-val">{value}/10</span>
    </label>
  );
}

function StepBody() {
  const s = useResearchStore();
  switch (s.step) {
    case "consent":
      return (
        <div className="gate" data-testid="research-consent">
          <label><input type="checkbox" checked={s.consent.boundary} onChange={(e) => s.setConsent("boundary", e.target.checked)} data-testid="rc-boundary" /> I understand this is wellness / cognitive-state research on safe external sensory channels and <strong>not a medical device</strong>.</label>
          <label><input type="checkbox" checked={s.consent.voluntary} onChange={(e) => s.setConsent("voluntary", e.target.checked)} data-testid="rc-voluntary" /> My participation is voluntary and I may stop at any time.</label>
          <label><input type="checkbox" checked={s.consent.dataLocal} onChange={(e) => s.setConsent("dataLocal", e.target.checked)} data-testid="rc-local" /> I understand all data stays in this browser unless I choose to export it.</label>
        </div>
      );
    case "contraindication":
      return (
        <div className="gate" data-testid="research-contra">
          <p className="muted">Select any that apply. If any apply, the workflow will not proceed.</p>
          {(["epilepsy", "photosensitivity", "pregnancy", "implant"] as const).map((k) => (
            <label key={k}><input type="checkbox" checked={s.contraindication[k]} onChange={(e) => s.setContra(k, e.target.checked)} data-testid={`rk-${k}`} /> {titleCase(k)} {k === "implant" ? "(implanted electronic / neurostimulation device)" : ""}</label>
          ))}
          {s.hasContraindication() && (
            <div className="breach-box" data-testid="contra-block">A contraindication is selected. This workflow is not appropriate; please consult a qualified professional.</div>
          )}
        </div>
      );
    case "baseline":
      return (
        <div data-testid="research-baseline">
          <p className="muted">Rate your current state to anchor the session.</p>
          <Slider label="Calm" value={s.baseline.calm} onChange={(v) => s.setBaseline("calm", v)} />
          <Slider label="Focus" value={s.baseline.focus} onChange={(v) => s.setBaseline("focus", v)} />
          <Slider label="Energy" value={s.baseline.energy} onChange={(v) => s.setBaseline("energy", v)} />
        </div>
      );
    case "protocol":
      return (
        <div className="transport-row" data-testid="research-protocol">
          {(["relaxed", "focused", "gamma"] as const).map((p) => (
            <button key={p} className={`chip ${s.protocol === p ? "active" : ""}`} onClick={() => s.setProtocol(p)} data-testid={`rp-${p}`}>{p}</button>
          ))}
          <p className="muted" style={{ width: "100%" }}>Selected: <strong>{s.protocol}</strong> — a verified 40 Hz session toward this target.</p>
        </div>
      );
    case "session":
      return <SessionStep />;
    case "survey":
      return (
        <div data-testid="research-survey">
          <p className="muted">Rate your state now, after the session.</p>
          <Slider label="Calm" value={s.survey.calm} onChange={(v) => s.setSurvey("calm", v)} />
          <Slider label="Focus" value={s.survey.focus} onChange={(v) => s.setSurvey("focus", v)} />
          <Slider label="Comfort / tolerability" value={s.survey.comfort} onChange={(v) => s.setSurvey("comfort", v)} />
        </div>
      );
    case "export":
      return <ExportStep />;
  }
}

function SessionStep() {
  const { bundle, verification } = useResearchStore();
  if (!bundle) return <div className="empty">No session loaded.</div>;
  return (
    <div data-testid="research-session">
      <div className="stat-row">
        <Stat label="Outcome" value={bundle.report.final_phase} />
        <Stat label="Receipts" value={bundle.report.num_receipts} />
        <Stat label="Acceptance" value={<StatusPill ok={bundle.acceptance.passed}>{bundle.acceptance.passed ? "PASS" : "FAIL"}</StatusPill>} />
        <Stat label="Local verify" value={<StatusPill ok={verification?.ok ?? null}>{verification?.ok ? "VERIFIED" : "—"}</StatusPill>} />
      </div>
      <p className="muted">A deterministic, verified Ruflo session for the chosen protocol. All receipts and the hash chain are verified locally before you continue.</p>
    </div>
  );
}

function ExportStep() {
  const s = useResearchStore();
  const record = s.buildRecord();

  function download() {
    if (!record) return;
    const blob = new Blob([JSON.stringify(record, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `ruflo-study-${s.protocol}-${record.bundle.sessionId}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }

  return (
    <div data-testid="research-export">
      <div className="clause-grid">
        <Stat label="Protocol" value={s.protocol} />
        <Stat label="Baseline (calm/focus/energy)" value={`${s.baseline.calm}/${s.baseline.focus}/${s.baseline.energy}`} />
        <Stat label="Survey (calm/focus/comfort)" value={`${s.survey.calm}/${s.survey.focus}/${s.survey.comfort}`} />
        <Stat label="Session verified" value={<StatusPill ok={s.verification?.ok ?? null}>{s.verification?.ok ? "yes" : "—"}</StatusPill>} />
      </div>
      <p className="muted">
        The study record bundles consent, contraindication answers, baseline, the verified session
        evidence, and your survey into a single local-first JSON artifact — re-verifiable in Replay
        mode. Nothing is uploaded.
      </p>
      <div className="btn-row">
        <button className="primary" onClick={download} disabled={!record} data-testid="study-export-btn">⬇ Export study record</button>
        <button onClick={() => s.reset()} data-testid="study-reset-btn">Start over</button>
      </div>
    </div>
  );
}
