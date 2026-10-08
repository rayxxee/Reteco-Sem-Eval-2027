#!/usr/bin/env bash
# run_all.sh  --  Reproducible run order for all phases
# Run from the project root: bash run_all.sh
# Prerequisites: py -3.12 in PATH; see README.md for full setup

set -euo pipefail
PYTHON="py -3.12"

echo "=== RETECO SemEval-2027 Sub-track 1a ==="
echo "Python: $($PYTHON --version)"
echo "Data:   data/raw/track1_tempo"
echo

# ── Phase 1: Setup & BM25 baseline ───────────────────────────────────────────
echo "[Phase 1] Running setup notebook..."
$PYTHON -m jupyter nbconvert --to notebook --execute \
    --output notebooks/01_phase1_setup_executed.ipynb \
    notebooks/01_phase1_setup.ipynb

# ── Phase 2: EDA ──────────────────────────────────────────────────────────────
echo "[Phase 2] Running EDA notebook..."
$PYTHON -m jupyter nbconvert --to notebook --execute \
    --output notebooks/02_phase2_eda_executed.ipynb \
    notebooks/02_phase2_eda.ipynb

# ── Phase 3: Classical baselines ─────────────────────────────────────────────
echo "[Phase 3] Running classical baselines notebook..."
$PYTHON -m jupyter nbconvert --to notebook --execute \
    --output notebooks/03_phase3_classical_executed.ipynb \
    notebooks/03_phase3_classical.ipynb

# ── Phase 4: Dense retrieval ──────────────────────────────────────────────────
echo "[Phase 4] Running dense retrieval notebook..."
$PYTHON -m jupyter nbconvert --to notebook --execute \
    --output notebooks/04_phase4_dense_executed.ipynb \
    notebooks/04_phase4_dense.ipynb

# ── Phase 5: Error analysis ───────────────────────────────────────────────────
echo "[Phase 5] Running error analysis notebook..."
$PYTHON -m jupyter nbconvert --to notebook --execute \
    --output notebooks/05_phase5_error_analysis_executed.ipynb \
    notebooks/05_phase5_error_analysis.ipynb

echo "=== All phases complete. Outputs in outputs/ ==="
