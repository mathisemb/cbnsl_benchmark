#!/usr/bin/env bash
#
# Run all 4 scaling notebooks sequentially.
# Results are saved incrementally to scaling/results/ (crash-safe).
# Executed notebooks (with outputs) are saved as *_executed.ipynb.
#
# Usage:  cd scaling && bash run_all.sh
#    or:  bash scaling/run_all.sh
#
# En arriere-plan:  cd scaling && nohup bash run_all.sh > run_all.log 2>&1 &
#          Suivre:  tail -f scaling/run_all.log

cd "$(dirname "$0")"

NOTEBOOKS=(
    scaling_study_sem_gaussian.ipynb
    scaling_study_cbn_unif_gauss.ipynb
    scaling_study_cbn_exp_clayton.ipynb
    scaling_study_cbn_unif_mixture.ipynb
)

failed=()

for nb in "${NOTEBOOKS[@]}"; do
    echo "========================================"
    echo "Starting: $nb"
    echo "$(date)"
    echo "========================================"

    if jupyter nbconvert \
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
