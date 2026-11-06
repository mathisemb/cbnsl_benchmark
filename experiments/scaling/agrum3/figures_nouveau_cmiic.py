"""
Export des figures du scaling pour meeting/nouveau_cmiic.tex (cmiic_wip).

Trois courbes, CPC et CMIIC du run de juillet 2026 et le nouveau CMIIC d'un run donné,
mêmes conventions que les figures du rapport cbnsl_benchmark_report : une figure par
(générateur, n_vars, métrique), police Computer Modern, couleurs du rapport pour CPC
et CMIIC, tab:cyan pour le nouveau.

Usage, depuis agrum3/, env cbnsl actif :
    python figures_nouveau_cmiic.py 3pt_sans_penalite /chemin/vers/figures
    python figures_nouveau_cmiic.py orient005 /chemin/vers/figures
    python figures_nouveau_cmiic.py pc_kl /chemin/vers/figures "PC KL"
Le premier argument est le suffixe des dossiers de résultats du run, le second le dossier
de sortie, dans lequel un sous-dossier par suffixe est créé, le troisième, facultatif,
l'étiquette de la troisième courbe, "nouveau CMIIC" par défaut. Les générateurs sans
dossier de résultats sont ignorés.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.scaling.io import load_results

HERE = Path(__file__).resolve().parent

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.unicode_minus": False,
    "font.size": 12,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
})

JULY = {
    "sem_gaussian":     "2026-07-30_14h34_sem_gaussian",
    "sem_laplace":      "2026-07-30_23h47_sem_laplace",
    "cbn_unif_gauss":   "2026-07-31_02h24_cbn_unif_gauss",
    "cbn_exp_clayton":  "2026-07-31_06h33_cbn_exp_clayton",
    "cbn_unif_mixture": "2026-07-31_10h35_cbn_unif_mixture",
}
GENERATORS = list(JULY)

# (colonne, étiquette, nom de fichier)
METRICS = [
    ("time_s", "Time (seconds)", "time"),
    ("SHD", "SHD (on CPDAGs)", "cpdags_shd"),
    ("F1-Score", "F1 (on CPDAGs)", "cpdags_f1"),
    ("TPR", "TPR (on CPDAGs)", "cpdags_tpr"),
    ("SHD_skeleton", "SHD (on skeletons)", "skeletons_shd"),
    ("F1-Score_skeleton", "F1 (on skeletons)", "skeletons_f1"),
    ("TPR_skeleton", "TPR (on skeletons)", "skeletons_tpr"),
]


def new_run_dir(gen, suffix):
    """Le dossier de résultats du run `suffix` pour ce générateur, None s'il n'a pas tourné."""
    dirs = sorted(HERE.glob(f"2026-*_{gen}_{suffix}"))
    if len(dirs) > 1:
        sys.exit(f"{gen}, suffixe {suffix} : {len(dirs)} dossiers trouvés, il en faut un seul")
    return dirs[0] if dirs else None


def load(gen, suffix, label):
    old = load_results(HERE / JULY[gen] / "results.csv")
    old = old[old.algo.isin(["CPC", "CMIIC"])]
    new = load_results(new_run_dir(gen, suffix) / "results.csv")
    new["algo"] = label
    return pd.concat([old, new], ignore_index=True)


def plot(data, y, ylabel, path, algos, palette):
    fig, ax = plt.subplots(figsize=(5, 3.5))
    sns.lineplot(
        data=data, x="n_samples", y=y,
        hue="algo", hue_order=algos, palette=palette,
        errorbar="se", marker="o", ax=ax,
    )
    ax.set_xlabel("Number of samples")
    ax.set_ylabel(ylabel)
    ax.legend(title=None, frameon=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    suffix, out = sys.argv[1], Path(sys.argv[2]) / sys.argv[1]
    label = sys.argv[3] if len(sys.argv) > 3 else "nouveau CMIIC"
    algos = ["CPC", "CMIIC", label]
    palette = {"CPC": "tab:blue", "CMIIC": "tab:orange", label: "tab:cyan"}
    out.mkdir(parents=True, exist_ok=True)
    for gen in GENERATORS:
        if new_run_dir(gen, suffix) is None:
            print(gen, "pas de run, ignoré", flush=True)
            continue
        df = load(gen, suffix, label)
        for n_vars in (10, 20):
            data = df[df.n_vars == n_vars]
            for y, ylabel, name in METRICS:
                plot(data, y, ylabel, out / f"{gen}_{n_vars}v_{name}.png", algos, palette)
        print(gen, "fait", flush=True)
    print(len(list(out.glob("*.png"))), "figures dans", out)
