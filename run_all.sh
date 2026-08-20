#!/usr/bin/env bash
# Full pipeline. Requires OPENROUTER_API_KEY.
set -euo pipefail
cd "$(dirname "$0")/src"

N_CAL=${N_CAL:-86}       # per subject (5 subjects -> 430; 86 is the
                         # max balanced draw, capped by business_ethics)
N_TEST=${N_TEST:-30}
OUT=${OUT:-../logs/main}

echo "== 1/4 build datasets =="
python3 data.py "$N_CAL" "$N_TEST"

echo "== 2/4 calibration probes + BSS =="
python3 calibrate.py

echo "== 3/4 discussions =="
python3 discussion.py -o "$OUT"

echo "== 4/4 analysis + figures =="
python3 analyze.py -d "$OUT"
python3 plots.py -d "$OUT"
