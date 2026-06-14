import { useStore } from "../store/store";

export function PlaybackControls() {
  const { bundle, step, playing, setStep, play, pause, reset } = useStore();
  if (!bundle) return null;
  const last = bundle.steps.length - 1;
  const cur = bundle.steps[step];

  return (
    <div className="playback" data-testid="playback">
      <button onClick={() => (playing ? pause() : play())} aria-label={playing ? "Pause" : "Play"}>
        {playing ? "⏸ Pause" : "▶ Play"}
      </button>
      <button onClick={reset} aria-label="Reset">⏮ Reset</button>
      <input
        type="range"
        min={0}
        max={last}
        value={step}
        onChange={(e) => setStep(Number(e.target.value))}
        aria-label="Step"
      />
      <span className="playback-pos">
        step {step + 1}/{last + 1} · {cur ? cur.phase : ""}
      </span>
    </div>
  );
}
