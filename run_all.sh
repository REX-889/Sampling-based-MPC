#!/bin/bash
# odpala wszystkie zadania po kolei, wykresy lądują w results/
set -e
cd "$(dirname "$0")"
# w kontenerze python3, lokalnie venv obok repo
if [ -x ../.venv/bin/python ]; then PY=../.venv/bin/python; else PY=python3; fi
# przy przebiegu wszystkiego bez okien, chyba że RENDER=1 ./run_all.sh
export RENDER="${RENDER:-0}"

for t in zadanie1_pendulum zadanie2_initial_state zadanie3_pd zadanie4_energy zadanie5_random_shooting \
         zadanie6_heatmap zadanie7_mppi zadanie8_robustness; do
    echo "===== $t"
    $PY $t.py
done
echo "===== zadanie9 (Pendulum)"
$PY zadanie9_hopper.py Pendulum-v1
echo "===== zadanie9 (Hopper, ok. 1-2 min na 16 rdzeniach)"
$PY zadanie9_hopper.py Hopper-v5
