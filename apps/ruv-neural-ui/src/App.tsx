import { useEffect, useRef, type ChangeEvent } from "react";
import { useStore, DEMO_PRESETS, type ScreenId } from "./store/store";
import { ActiveScreen } from "./screens";
import { StatusPill } from "./components/common";
import { BOUNDARY } from "./util/constants";

const NAV: { id: ScreenId; label: string }[] = [
  { id: "overview", label: "Overview" },
  { id: "session", label: "Live Session" },
  { id: "stimulus", label: "Stimulus Verifier" },
  { id: "biosense", label: "Biosense" },
  { id: "ruvector", label: "ruVector State" },
  { id: "safety", label: "Safety Envelope" },
  { id: "audit", label: "Audit Trail" },
  { id: "witness", label: "Witness" },
];

function ModeBar() {
  const { mode, setMode, loadPreset, importJson, presetId, bundle, clear } = useStore();
  const fileRef = useRef<HTMLInputElement>(null);

  function onFile(e: ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => importJson(String(reader.result ?? ""), file.name);
    reader.readAsText(file);
    e.target.value = "";
  }

  function exportBundle() {
    if (!bundle) return;
    const blob = new Blob([JSON.stringify(bundle, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `ruflo-${bundle.targetState}-${bundle.sessionId}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }

  return (
    <div className="modebar">
      <div className="mode-toggle" role="tablist" aria-label="Mode">
        <button className={mode === "demo" ? "active" : ""} onClick={() => setMode("demo")} role="tab" aria-selected={mode === "demo"}>
          Demo
        </button>
        <button className={mode === "replay" ? "active" : ""} onClick={() => setMode("replay")} role="tab" aria-selected={mode === "replay"}>
          Replay
        </button>
      </div>

      {mode === "demo" ? (
        <div className="preset-row">
          {DEMO_PRESETS.map((p) => (
            <button
              key={p.id}
              className={`preset ${presetId === p.id ? "active" : ""}`}
              onClick={() => loadPreset(p.id)}
              title={p.description}
              data-testid={`preset-${p.id}`}
            >
              {p.label}
            </button>
          ))}
        </div>
      ) : (
        <div className="preset-row">
          <button className="preset" onClick={() => fileRef.current?.click()} data-testid="import-btn">
            ⬆ Import evidence bundle (.json)
          </button>
          <input ref={fileRef} type="file" accept="application/json,.json" onChange={onFile} hidden />
        </div>
      )}

      <div className="modebar-right">
        {bundle && (
          <>
            <button className="ghost" onClick={exportBundle} data-testid="export-btn">⬇ Export</button>
            <button className="ghost" onClick={clear}>Clear</button>
          </>
        )}
      </div>
    </div>
  );
}

export default function App() {
  const { screen, setScreen, bundle, playing, step, setStep, pause, error } = useStore();

  // Playback timer: advance the cursor while playing.
  useEffect(() => {
    if (!playing || !bundle) return;
    const id = window.setInterval(() => {
      const { step: s, bundle: b } = useStore.getState();
      if (!b) return;
      if (s >= b.steps.length - 1) {
        pause();
      } else {
        setStep(s + 1);
      }
    }, 650);
    return () => window.clearInterval(id);
  }, [playing, bundle, pause, setStep, step]);

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="logo">◓</span>
          <div>
            <h1>rUv Neural UI</h1>
            <p className="tagline">Ruflo closed-loop sensory neuromodulation · local-first console</p>
          </div>
        </div>
        <StatusPill ok={null}>Not a medical device</StatusPill>
      </header>

      <div className="boundary-banner" role="note">{BOUNDARY}</div>

      <ModeBar />

      {error && <div className="error-banner" role="alert" data-testid="error">⚠ {error}</div>}

      <div className="layout">
        <nav className="sidenav" aria-label="Screens">
          {NAV.map((n) => (
            <button
              key={n.id}
              className={screen === n.id ? "active" : ""}
              onClick={() => setScreen(n.id)}
              data-testid={`nav-${n.id}`}
            >
              {n.label}
            </button>
          ))}
        </nav>
        <main className="content">
          <ActiveScreen />
        </main>
      </div>

      <footer className="footer">
        <span>
          Verification runs entirely in your browser. No backend, no accounts, no health data leaves
          this page.
        </span>
        <span>
          <a href="https://github.com/ruvnet/ruv-neural" rel="noreferrer">ruvnet/ruv-neural</a> · ADR-0014
        </span>
      </footer>
    </div>
  );
}
