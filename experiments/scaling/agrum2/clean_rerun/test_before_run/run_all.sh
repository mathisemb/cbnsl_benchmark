#!/usr/bin/env bash
#
# Run the 5 CPC/CMIIC scaling notebooks (agrum3) sequentially.
# Results are saved incrementally to agrum3/<timestamp>_<generator>/ (crash-safe).
# Executed notebooks (with outputs) are saved as *_executed.ipynb.
#
# Usage, dans agrum3/:  bash run_all.sh
#
# En arriere-plan, dans agrum3/:  nohup bash run_all.sh > run_all.log 2>&1 &
#          Suivre, dans agrum3/:  tail -f run_all.log

cd "$(dirname "$0")"

# Verification de la stack : ce run doit tourner sous agrum2 (venv).
python - <<'EOF' || exit 1
import sys
import pyagrum, otagrum
print(f"python  : {sys.executable}")
print(f"pyagrum {pyagrum.__version__} | otagrum {otagrum.__version__}")
if not pyagrum.__version__.startswith("2."):
    sys.exit("ERREUR : ce run doit tourner sous agrum2 (venv actif ?). Abandon.")
EOF

NOTEBOOKS=(
    scaling_study_sem_laplace.ipynb
    scaling_study_cbn_unif_mixture.ipynb
)

failed=()

for nb in "${NOTEBOOKS[@]}"; do
    echo "========================================"
    echo "Starting: $nb"
    echo "$(date)"
    echo "========================================"

    if python -m nbconvert \
        --to notebook \
        --execute \
        --ExecutePreprocessor.timeout=-1 \
        --output "${nb%.ipynb}_executed.ipynb" \
        "$nb"; then
        echo "Done: $nb  ($(date))"
    else
        echo "FAILED: $nb  ($(date))"
        failed+=("$nb")
    fi
    echo ""
done

echo "========================================"
if [ ${#failed[@]} -eq 0 ]; then
    echo "All notebooks finished successfully at $(date)"
else
    echo "Finished at $(date) with ${#failed[@]} failure(s):"
    for nb in "${failed[@]}"; do
        echo "  - $nb"
    done
fi
echo "========================================"
