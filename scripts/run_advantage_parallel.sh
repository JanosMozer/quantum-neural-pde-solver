#!/usr/bin/env bash
# Parallel quantum-advantage sweep on this machine (RTX 5090 + 20 cores).
set -euo pipefail
cd /home/mozer/qt-pinn
PY=.venv/bin/python
LOG=blog/checkpoint/v4/parallel_logs
mkdir -p "$LOG" \
  blog/checkpoint/v4/advantage_A/A_q8l8 \
  blog/checkpoint/v4/advantage_A/A_q6l6 \
  blog/checkpoint/v4/advantage_A/A_q8l4 \
  blog/checkpoint/v4/advantage_A/A_cgen \
  blog/checkpoint/v4/advantage_A/A_cdirect \
  blog/checkpoint/v4/advantage_B

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2

echo "=== launching parallel jobs $(date -Is) ==="

# --- Experiment A arms (circuit on CPU; MLP on shared GPU) ---
$PY scripts/exp_quantum_advantage_A.py --arm quantum --n-qubits 8 --n-layers 8 \
  --steps 20000 --seed 0 --out blog/checkpoint/v4/advantage_A/A_q8l8 \
  >"$LOG/A_q8l8.log" 2>&1 &
PID_AQ1=$!

$PY scripts/exp_quantum_advantage_A.py --arm quantum --n-qubits 6 --n-layers 6 \
  --steps 20000 --seed 0 --out blog/checkpoint/v4/advantage_A/A_q6l6 \
  >"$LOG/A_q6l6.log" 2>&1 &
PID_AQ2=$!

$PY scripts/exp_quantum_advantage_A.py --arm quantum --n-qubits 8 --n-layers 4 \
  --steps 20000 --seed 1 --out blog/checkpoint/v4/advantage_A/A_q8l4 \
  >"$LOG/A_q8l4.log" 2>&1 &
PID_AQ3=$!

$PY scripts/exp_quantum_advantage_A.py --arm classical_gen --n-qubits 8 --n-layers 8 \
  --steps 20000 --seed 0 --out blog/checkpoint/v4/advantage_A/A_cgen \
  >"$LOG/A_cgen.log" 2>&1 &
PID_ACG=$!

$PY scripts/exp_quantum_advantage_A.py --arm classical_direct \
  --steps 20000 --seed 0 --out blog/checkpoint/v4/advantage_A/A_cdirect \
  >"$LOG/A_cdirect.log" 2>&1 &
PID_ACD=$!

# --- Multi-ν DNS family (GPU sequential in one process) ---
$PY scripts/gen_merger_dns_family.py --all \
  >"$LOG/dns_family.log" 2>&1 &
PID_DNS=$!

echo "PIDs: A_q8l8=$PID_AQ1 A_q6l6=$PID_AQ2 A_q8l4=$PID_AQ3 A_cgen=$PID_ACG A_cdirect=$PID_ACD DNS=$PID_DNS"
echo "$PID_AQ1 $PID_AQ2 $PID_AQ3 $PID_ACG $PID_ACD $PID_DNS" >"$LOG/pids.txt"

# Wait for DNS first so B can start as soon as family is ready
wait $PID_DNS
echo "DNS family done $(date -Is)"

# --- Experiment B: held-out ν=0.008 interpolation (physics-recommended protocol) ---
$PY scripts/exp_quantum_advantage_B.py --arm quantum --n-qubits 8 --n-layers 8 \
  --holdout-nu 0.008 --steps 25000 --seed 0 --out blog/checkpoint/v4/advantage_B \
  >"$LOG/B_q8l8.log" 2>&1 &
PID_BQ1=$!

$PY scripts/exp_quantum_advantage_B.py --arm quantum --n-qubits 6 --n-layers 6 \
  --holdout-nu 0.008 --steps 25000 --seed 0 --out blog/checkpoint/v4/advantage_B \
  >"$LOG/B_q6l6.log" 2>&1 &
PID_BQ2=$!

$PY scripts/exp_quantum_advantage_B.py --arm classical --n-qubits 8 --n-layers 8 \
  --holdout-nu 0.008 --steps 25000 --seed 0 --out blog/checkpoint/v4/advantage_B \
  >"$LOG/B_cgen.log" 2>&1 &
PID_BC=$!

$PY scripts/exp_quantum_advantage_B.py --arm classical --n-qubits 6 --n-layers 6 \
  --holdout-nu 0.008 --steps 25000 --seed 0 --out blog/checkpoint/v4/advantage_B \
  >"$LOG/B_cgen_q6.log" 2>&1 &
PID_BC2=$!

echo "B PIDs: q8=$PID_BQ1 q6=$PID_BQ2 c8=$PID_BC c6=$PID_BC2"
echo "$PID_BQ1 $PID_BQ2 $PID_BC $PID_BC2" >>"$LOG/pids.txt"

wait || true
echo "=== all launched jobs finished $(date -Is) ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv
