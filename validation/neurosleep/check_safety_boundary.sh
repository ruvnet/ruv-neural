#!/usr/bin/env bash
# ADR-0015 / Helix ADR-051 upstream static safety boundary.
#
# Fails CLOSED: a missing search tool, an absent expected path, or a search
# error is a failure, never a pass. An earlier version shelled out to `rg`
# inside an `if` condition, so on a host without ripgrep every check evaluated
# false and this script printed "clean" while testing nothing.
set -euo pipefail

repo_root=$(git rev-parse --show-toplevel)
cd "$repo_root"

if ! command -v grep >/dev/null 2>&1; then
  echo "FATAL: grep is unavailable; cannot evaluate the safety boundary" >&2
  exit 2
fi

# Three-state search: 0 = match (violation), 1 = no match (clean), >=2 = error.
# A tool error aborts rather than reading as "clean".
search() {
  local pattern=$1
  shift
  local status=0
  grep -rEn --include='*.rs' --include='Cargo.toml' -- "$pattern" "$@" || status=$?
  if ((status >= 2)); then
    echo "FATAL: grep failed (exit $status) while scanning: $*" >&2
    exit 2
  fi
  return $status
}

violations=0

# 1. NeuroSleep must never reach a closed-loop actuation crate. ADR-051 makes a
#    NeuroSleep-driven stimulation decision a hard release stop.
actuation_paths=()
for path in ruv-neural-loop ruv-neural-stim; do
  [[ -d $path ]] && actuation_paths+=("$path")
done
if ((${#actuation_paths[@]} == 0)); then
  echo "FATAL: no actuation crate found; refusing to pass vacuously" >&2
  exit 2
fi
if search 'NeuroSleep|neurosleep|RUVN-QEEG' "${actuation_paths[@]}"; then
  echo "VIOLATION: a NeuroSleep identifier reached an actuation crate" >&2
  violations=1
fi

# 2. The bounded parser must not depend on an actuation crate.
if [[ -f ruv-neural-io/Cargo.toml ]] &&
  search 'ruv-neural-(loop|stim)' ruv-neural-io/Cargo.toml; then
  echo "VIOLATION: the bounded I/O crate depends on an actuation crate" >&2
  violations=1
fi

# 3. The bounded parser must not write to the filesystem.
if [[ -d ruv-neural-io/src ]] &&
  search 'File::create|OpenOptions|fs::write|write_all|remove_file' ruv-neural-io/src; then
  echo "VIOLATION: the bounded parser contains a filesystem write path" >&2
  violations=1
fi

# 4. No raw-waveform-shaped field may appear in the derived evidence contract.
contract_files=()
for file in ruv-neural-core/src/neurosleep.rs ruv-neural-core/src/attestation.rs; do
  if [[ ! -f $file ]]; then
    echo "FATAL: expected contract file $file is missing" >&2
    exit 2
  fi
  contract_files+=("$file")
done
if search 'Vec<f(32|64)>|raw_(eeg|eog|emg)|waveform(_bytes|_samples)?' "${contract_files[@]}"; then
  echo "VIOLATION: a raw waveform-shaped field reached the derived evidence contract" >&2
  violations=1
fi

if ((violations != 0)); then
  echo "upstream NeuroSleep static safety boundary: FAILED" >&2
  exit 1
fi

echo "upstream NeuroSleep static safety boundary: clean"
