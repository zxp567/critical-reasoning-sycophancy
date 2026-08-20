#!/usr/bin/env bash
# Full remaining pipeline: recalibrate BSS at 430 questions, then run all six
# conditions on both the paper-protocol and balanced-control test sets.
# Everything already cached (calibration r0 probes, the baseline condition) is
# reused, so this only spends on genuinely new calls.
set -euo pipefail
cd "$(dirname "$0")/src"

echo "=========== 1/4  recalibrate BSS (430 questions) ==========="
python3 calibrate.py 2>&1 | tr '\r' '\n' | grep -vE "it/s\]$" | tail -12

echo
echo "=========== 2/4  BSS ranking reliability ==========="
python3 reliability.py

echo
echo "=========== 3/4  discussions: paper protocol (user always wrong) ==========="
python3 discussion.py -o ../logs/main --dataset test 2>&1 | tr '\r' '\n' | grep -vE "it/s\]$"

echo
echo "=========== 4/4  discussions: balanced control (user right 50%) ==========="
python3 discussion.py -o ../logs/balanced --dataset test_balanced 2>&1 | tr '\r' '\n' | grep -vE "it/s\]$"

echo
echo "=========== DONE ==========="
