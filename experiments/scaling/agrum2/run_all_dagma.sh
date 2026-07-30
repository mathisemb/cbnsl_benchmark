#!/usr/bin/env bash
#
# Run all 5 DAGMA-only scaling notebooks sequentially.
# Results are saved incrementally to scaling/results/ (crash-safe).
# Executed notebooks (with outputs) are saved as *_executed.ipynb.
#
# Usage, dans scaling/:  bash run_all_dagma.sh
#
# En arriere-plan, dans scaling/:  nohup bash run_all_dagma.sh > run_all_dagma.log 2>&1 &
#          Suivre, dans scaling/:  tail -f run_all_dagma.log

cd "$(dirname "$0")"

NOTEBOOKS=(
    scaling_study_sem_gaussian_dagma.ipynb
    scaling_study_sem_laplace_dagma.ipynb
    scaling_study_cbn_unif_gauss_dagma.ipynb
    scaling_study_cbn_exp_clayton_dagma.ipynb
    scaling_study_cbn_unif_mixture_dagma.ipynb
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
