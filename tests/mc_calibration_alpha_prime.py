"""
Monte Carlo calibration check for α' (Taamouti 2014) under H0 of conditional
independence (chain X ← Z → Y, so true CMI = 0), as a function of n.

For each n in SAMPLE_SIZES:
  - Run B_REPLICATIONS Monte Carlo draws.
  - For each, compute Î_n(X;Y|Z) via NoCorr (no penalty, raw estimator).
  - Compare empirical mean / sd / q95 to the Taamouti theoretical values
    (μ_n = K^(D/2)/(2n) · ξ ,   σ_n = K^(D/2)/(2n) · σ_0 ,   α' = μ_n + z · σ_n).

Outputs a 2-panel plot (means and 95% thresholds vs n) and a stats table.

Run from cbnsl_benchmark root:
    python tests/mc_calibration_alpha_prime.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import openturns as ot
import otagrum
from scipy.stats import norm

SAMPLE_SIZES = [1000, 5000, 20000]
B_REPLICATIONS = 50
ALPHA_LEVEL = 0.05
SEED = 42


def K_lasserre(n, dim):
    return int(1 + n ** (2 / (4 + dim)))


def taamouti_decomposition(n, K, p_X, p_Y, p_Z, alpha=ALPHA_LEVEL):
    """Returns (mu_theo, sd_theo, alpha_prime) for the Bernstein-CMI under H0."""
    D = p_X + p_Y + p_Z
    z = norm.ppf(1 - alpha)
    sigma0 = np.sqrt(2) * (np.pi / 4) ** (D / 2)
    xi = (
        -(2.0 ** (-D)) * (np.pi ** (D / 2))
        + (2.0 ** (-(p_Y + p_Z))) * (np.pi ** ((p_Y + p_Z) / 2)) * (K ** (-p_X / 2))
        + (2.0 ** (-(p_X + p_Z))) * (np.pi ** ((p_X + p_Z) / 2)) * (K ** (-p_Y / 2))
        + 2.0 * (2.0 ** (-p_Z) - 1) * (np.pi ** (p_Z / 2)) * (K ** (-(p_X + p_Y) / 2))
    )
    scale = (K ** (D / 2)) / (2 * n)
    mu_theo = scale * xi
    sd_theo = scale * sigma0
    return mu_theo, sd_theo, mu_theo + z * sd_theo


def generate_chain(n, rng):
    Z = rng.normal(0, 1, n)
    X = Z + rng.normal(0, 0.5, n)
    Y = Z + rng.normal(0, 0.5, n)
    return np.column_stack([X, Y, Z])


def estimate_cmi(data):
    sample = ot.Sample(data)
    cmi = otagrum.CorrectedMutualInformation(sample)
    cmi.setKMode(otagrum.CorrectedMutualInformation.KModeTypes_NoCorr)
    U = ot.Indices([2])
    return cmi.compute2PtCorrectedInformation(0, 1, U)


def run_for_n(n, rng):
    values = np.empty(B_REPLICATIONS)
    for b in range(B_REPLICATIONS):
        data = generate_chain(n, rng)
        values[b] = estimate_cmi(data)
    return values


def main():
    rng = np.random.default_rng(SEED)
    rows = []
    for n in SAMPLE_SIZES:
        print(f"[n={n}] running {B_REPLICATIONS} replications...")
        values = run_for_n(n, rng)
        K = K_lasserre(n, dim=3)
        mu_th, sd_th, ap = taamouti_decomposition(n, K, 1, 1, 1)
        rows.append({
            "n": n, "K": K,
            "mu_emp": values.mean(), "sd_emp": values.std(ddof=1),
            "q95_emp": np.quantile(values, 0.95),
            "mu_th": mu_th, "sd_th": sd_th, "alpha_prime": ap,
            "rej_naive": float(np.mean(values > ALPHA_LEVEL)),
            "rej_ap": float(np.mean(values > ap)),
        })

    print()
    print("=" * 100)
    print(f"Calibration scaling under H0 (chain X←Z→Y) — B={B_REPLICATIONS}, α={ALPHA_LEVEL}")
    print("=" * 100)
    fmt = "{:>6} {:>4} {:>11} {:>11} {:>11} {:>11} {:>11} {:>11} {:>8} {:>8}"
    print(fmt.format("n", "K", "μ_emp", "μ_th", "sd_emp", "sd_th", "q95_emp", "α'",
                     "rejN%", "rejAP%"))
    print("-" * 100)
    for r in rows:
        print(fmt.format(
            r["n"], r["K"],
            f"{r['mu_emp']:+.3e}", f"{r['mu_th']:+.3e}",
            f"{r['sd_emp']:+.3e}", f"{r['sd_th']:+.3e}",
            f"{r['q95_emp']:+.3e}", f"{r['alpha_prime']:+.3e}",
            f"{r['rej_naive']:.0%}", f"{r['rej_ap']:.0%}",
        ))
    print("=" * 100)

    # Plot
    ns = np.array([r["n"] for r in rows])
    mu_emp = np.array([r["mu_emp"] for r in rows])
    mu_th = np.array([r["mu_th"] for r in rows])
    sd_emp = np.array([r["sd_emp"] for r in rows])
    q95_emp = np.array([r["q95_emp"] for r in rows])
    aprime = np.array([r["alpha_prime"] for r in rows])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.errorbar(ns, mu_emp, yerr=sd_emp / np.sqrt(B_REPLICATIONS),
                 fmt="o-", color="steelblue", label="μ empirical (±SE)", capsize=4)
    ax1.plot(ns, mu_th, "s--", color="crimson", label="μ theoretical (Taamouti)")
    ax1.axhline(0, color="black", linewidth=0.5)
    ax1.set_xscale("log")
    ax1.set_xlabel("n (log scale)")
    ax1.set_ylabel(r"Mean of $\hat I_n(X;Y|Z)$ under $H_0$")
    ax1.set_title("Centering: empirical vs theoretical")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(ns, q95_emp, "o-", color="steelblue", label="q95 empirical")
    ax2.plot(ns, aprime, "s--", color="crimson", label="α' (theoretical)")
    ax2.axhline(ALPHA_LEVEL, color="forestgreen", linestyle=":", label="α-Naive (const)")
    ax2.set_xscale("log")
    ax2.set_xlabel("n (log scale)")
    ax2.set_ylabel("Threshold value")
    ax2.set_title("95% threshold: empirical vs α'")
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.suptitle(
        "Convergence of Bernstein-CMI estimator and Taamouti α' under conditional independence",
        fontsize=11,
    )
    fig.tight_layout()

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "mc_calibration_alpha_prime.png")
    fig.savefig(out_path, dpi=120)
    print(f"\nPlot saved to: {out_path}")


if __name__ == "__main__":
    main()
