#!/usr/bin/env bash
# Val-selected baseline: sweep val_mix, freeze the choice, score it on trend/full_{n}.
#
# The reporting sweep (raw/rows.jsonl) is NOT re-run -- this stage only adds a selection
# split and re-reads the existing test rows through it. Safe to re-run.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
PY="$REPO/.venv/bin/python"
WORKERS="${WORKERS:-22}"
cd "$HERE"

# --- preflight: the three splits must be disjoint -------------------------------
# val_mix is the selection set only if no instance in it also appears in a training set or
# in the reporting set. Generated separately, but "generated separately" is not a proof.
"$PY" - "$REPO" <<'PYEOF'
import hashlib, json, sys
from pathlib import Path
REPO = Path(sys.argv[1])
def sig(p):
    return {hashlib.sha1(json.dumps(json.loads(l)["jobs"], sort_keys=True).encode()).hexdigest()
            for l in Path(p).read_text().splitlines() if l.strip()}
bad = []
for n in (20, 40, 60, 80, 100):
    val = sig(REPO / f"test_case/v3/mix/val_mix_{n}.jsonl")
    test = sig(REPO / f"test_case/v3/trend/full_{n}.jsonl")
    tr_mix = sig(REPO / f"test_case/v3/mix/train_mix_{n}.jsonl")
    tr_60 = sig(REPO / "test_case/v3/train_60.jsonl") if n == 60 else set()
    for label, other in (("test", test), ("train_mix", tr_mix), ("train_60", tr_60)):
        if val & other:
            bad.append(f"n={n}: val_mix shares {len(val & other)} instances with {label}")
    if test & tr_mix or (test & tr_60):
        bad.append(f"n={n}: test overlaps a training set")
if bad:
    sys.exit("REFUSING -- splits are not disjoint:\n  " + "\n  ".join(bad))
print("preflight: val_mix / train_* / trend disjoint at every size")
PYEOF

[ -s raw/rows.jsonl ] || { echo "REFUSING: raw/rows.jsonl (test rows) missing" >&2; exit 1; }

# --- stage 1: sweep the selection split -----------------------------------------
# Rules and policies only. The GA has nothing being selected, and at 15-145 s/instance it
# would cost more than the rest of the sweep combined.
if [ -s raw/rows_val.jsonl ] && [ -z "${FORCE:-}" ]; then
  echo "raw/rows_val.jsonl exists -- reusing (FORCE=1 to re-sweep)"
else
  FB_SPLIT=val "$PY" sweep_all.py --workers "$WORKERS" --what models,rules \
    --out raw/rows_val.jsonl
fi

# --- stage 2: freeze the choice --------------------------------------------------
"$PY" select_on_val.py

# --- stage 3: report it on test --------------------------------------------------
"$PY" compare_selected.py

# --- stage 4: matched-seed variant ----------------------------------------------
# Every recipe at s44, the only seed the v10 cells have. Same frozen rule baseline.
"$PY" compare_fixed_seed.py --seed "${FIXED_SEED:-s44}"

echo
echo "selection -> $HERE/selection_val.json"
echo "results   -> $HERE/val_selected.csv, $HERE/val_selected_summary.txt"
echo "matched   -> $HERE/fixed_s44.csv, $HERE/fixed_s44_summary.txt"
