#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/src"
export ROSTER=strong
echo "=========== strong roster: calibrate ==========="
python3 calibrate.py 2>&1 | tr '\r' '\n' | grep -vE "it/s\]$" | tail -12
echo
echo "=========== strong roster: reliability ==========="
python3 reliability.py | tail -8
echo
echo "=========== strong roster: paper protocol ==========="
python3 discussion.py -c baseline bss critical -o ../logs/strong_main --dataset test 2>&1 | tr '\r' '\n' | grep -vE "it/s\]$"
echo
echo "=========== strong roster: balanced control ==========="
python3 discussion.py -c baseline bss critical -o ../logs/strong_balanced --dataset test_balanced 2>&1 | tr '\r' '\n' | grep -vE "it/s\]$"
echo "=========== DONE ==========="
