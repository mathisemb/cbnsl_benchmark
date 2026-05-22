"""
Sanity checks for the AlphaPrime KMode added to CorrectedMutualInformation.

Compares Naive (constant α) vs AlphaPrime (Taamouti 2014 formula) on
4 canonical cases, for several sample sizes.

The CMIIC 2-pt decision rule is: reject independence iff
    compute2PtCorrectedInformation(X, Y, U) = Î(X;Y|U) - penalty > 0

Run from the cbnsl_benchmark root, after the venv is activated and
PYTHONPATH points to the local otagrum fork:

    python tests/sanity_alpha_prime.py
"""

import numpy as np
import openturns as ot
import otagrum

ALPHA_LEVEL = 0.05
SAMPLE_SIZES = [200, 1000, 5000]
SEED = 42

CASES = ["marginal_indep", "marginal_dep", "conditional_indep", "conditional_dep"]
EXPECTED_REJECT = {
    "marginal_indep": False,
    "marginal_dep": True,
    "conditional_indep": False,
    "conditional_dep": True,
}


def generate(case, n, rng):
    """Return (data, U_indices) where columns are X, Y, (Z)."""
    if case == "marginal_indep":
        X = rng.uniform(0, 1, n)
        Y = rng.uniform(0, 1, n)
        return np.column_stack([X, Y]), ot.Indices()

    if case == "marginal_dep":
        XY = rng.multivariate_normal([0.0, 0.0], [[1.0, 0.7], [0.7, 1.0]], size=n)
        return XY, ot.Indices()

    if case == "conditional_indep":
        # Chain X ← Z → Y : X and Y are independent given Z.
        Z = rng.normal(0, 1, n)
        X = Z + rng.normal(0, 0.5, n)
        Y = Z + rng.normal(0, 0.5, n)
        return np.column_stack([X, Y, Z]), ot.Indices([2])

    if case == "conditional_dep":
        # V-structure X → Z ← Y : X, Y marginally independent
        # but become dependent when conditioning on Z.
        X = rng.normal(0, 1, n)
        Y = rng.normal(0, 1, n)
        Z = X + Y + rng.normal(0, 0.5, n)
        return np.column_stack([X, Y, Z]), ot.Indices([2])

    raise ValueError(case)


def evaluate(case, n, rng):
    data, U = generate(case, n, rng)
    sample = ot.Sample(data)
    cmi = otagrum.CorrectedMutualInformation(sample)
    cmi.setAlpha(ALPHA_LEVEL)

    X_idx, Y_idx = 0, 1

    cmi.setKMode(otagrum.CorrectedMutualInformation.KModeTypes_Naive)
    info_naive = cmi.compute2PtCorrectedInformation(X_idx, Y_idx, U)

    cmi.setKMode(otagrum.CorrectedMutualInformation.KModeTypes_AlphaPrime)
    info_aprime = cmi.compute2PtCorrectedInformation(X_idx, Y_idx, U)

    return info_naive, info_aprime


def main():
    rng = np.random.default_rng(SEED)
    rows = []
    for case in CASES:
        for n in SAMPLE_SIZES:
            info_naive, info_aprime = evaluate(case, n, rng)
            rows.append((case, n, info_naive, info_aprime))

    print(f"\nSanity checks for α' (Taamouti 2014) — test level α = {ALPHA_LEVEL}")
    print("Decision: reject independence iff 2PtCorrectedInformation > 0")
    print("=" * 105)
    header = "{:<22} {:>6} {:>15} {:>15} {:>8} {:>8} {:>10} {:>10}"
    print(header.format("Case", "n", "I_Naive", "I_AlphaPrime",
                        "RejN", "RejAP", "Expected", "Status"))
    print("-" * 105)
    for case, n, i_naive, i_aprime in rows:
        rej_n = i_naive > 0
        rej_ap = i_aprime > 0
        expected = EXPECTED_REJECT[case]
        statuses = []
        statuses.append("Nv:OK" if rej_n == expected else "Nv:KO")
        statuses.append("AP:OK" if rej_ap == expected else "AP:KO")
        print(header.format(
            case, n,
            f"{i_naive:+.4e}", f"{i_aprime:+.4e}",
            str(rej_n), str(rej_ap),
            str(expected), " ".join(statuses),
        ))
    print("=" * 105)


if __name__ == "__main__":
    main()
