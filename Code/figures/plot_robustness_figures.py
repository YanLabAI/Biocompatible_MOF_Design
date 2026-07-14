from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageStat


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "Results" / "robustness_current1000"
RANDOM_DIR = ROOT / "random_10_splits"
GROUP_DIR = ROOT / "groupkfold_dominant_metal"
OOD_DIR = ROOT / "leave_one_metal_out_ood"

# ColorHunt-inspired: #27374D #526D82 #9DB2BF #DDE6ED, with a softened orange contrast.
NAVY = "#27374D"
STEEL = "#526D82"
ICE = "#DDE6ED"
ORANGE = "#C97942"
DOT = "#2F3437"


def style() -> None:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
    plt.rcParams["svg.fonttype"] = "none"
    mpl.rcParams.update({
        "pdf.fonttype": 42,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#222222",
        "axes.linewidth": 0.65,
        "axes.labelsize": 7.2,
        "axes.titlesize": 7.2,
        "xtick.labelsize": 6.2,
        "ytick.labelsize": 6.2,
        "legend.fontsize": 6.4,
    })


def savefig(fig: plt.Figure, out_base: Path) -> None:
    for suffix in ("svg", "pdf", "tiff"):
        p = out_base.with_suffix(f".{suffix}")
        if suffix == "tiff":
            fig.savefig(p, dpi=600, bbox_inches="tight", pad_inches=0.04, pil_kwargs={"compression": "tiff_lzw"})
        else:
            fig.savefig(p, bbox_inches="tight", pad_inches=0.04)


def plot_random() -> None:
    res = pd.read_csv(RANDOM_DIR / "random_10_split_results.csv")
    fig, ax = plt.subplots(figsize=(3.8, 2.65))
    colors = {"benzene": STEEL, "toluene": ORANGE}
    for i, ads in enumerate(["benzene", "toluene"]):
        d = res[res["adsorbate"].eq(ads)]
        mean, sd = d["r2"].mean(), d["r2"].std(ddof=1)
        ax.bar(i, mean, yerr=sd, color=colors[ads], alpha=0.88, width=0.52, capsize=2.4, edgecolor="none",
               error_kw={"elinewidth": 0.8, "capthick": 0.8, "ecolor": "#202020"})
        ax.scatter(np.full(len(d), i) + np.linspace(-.065, .065, len(d)), d["r2"], s=10, color=DOT, alpha=.58, zorder=3)
        ax.text(i, mean + sd + .008, f"{mean:.3f} ± {sd:.3f}", ha="center", fontsize=6.2, color="#333333")
    ax.set_xticks([0, 1], ["Benzene", "Toluene"])
    ax.set_ylabel(r"Test $R^2$")
    ax.set_title("10 independent random 80/20 splits", fontweight="semibold")
    ax.set_ylim(0.90, 1.005)
    savefig(fig, RANDOM_DIR / "random_10_split_r2")
    plt.close(fig)


def plot_group() -> None:
    res = pd.read_csv(GROUP_DIR / "groupkfold_fold_results.csv")
    fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.7))
    cv_colors = {"KFold": STEEL, "GroupKFold": ORANGE}
    for ax, ads, panel in zip(axes, ["benzene", "toluene"], ["a", "b"]):
        d = res[res["adsorbate"].eq(ads)]
        for i, cv in enumerate(["KFold", "GroupKFold"]):
            v = d[d["cv"].eq(cv)]["r2"]
            mean, sd = v.mean(), v.std(ddof=1)
            ax.bar(i, mean, yerr=sd, color=cv_colors[cv], alpha=0.88, width=0.52, capsize=2.4, edgecolor="none",
                   error_kw={"elinewidth": 0.8, "capthick": 0.8, "ecolor": "#202020"})
            ax.scatter(np.full(len(v), i) + np.linspace(-.04, .04, len(v)), v, s=10, color=DOT, alpha=.58, zorder=3)
            ax.text(i, mean + sd + .009, f"{mean:.3f} ± {sd:.3f}", ha="center", fontsize=6.0, color="#333333")
        ax.set_xticks([0, 1], ["KFold (5)", "GroupKFold (5)"])
        ax.set_ylabel(r"$R^2$ score")
        ax.set_title(f"KFold vs GroupKFold - {ads}", fontweight="semibold")
        ax.text(-0.15, 1.05, panel, transform=ax.transAxes, fontweight="bold", fontsize=9)
        ax.set_ylim((0.88, 1.005) if ads == "toluene" else (0.88, 0.99))
    fig.subplots_adjust(wspace=.34)
    savefig(fig, GROUP_DIR / "kfold_vs_groupkfold_r2")
    plt.close(fig)


def plot_ood() -> None:
    res = pd.read_csv(OOD_DIR / "leave_one_metal_out_results_n20.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.65))
    for ax, ads, panel in zip(axes, ["benzene", "toluene"], ["a", "b"]):
        d = res[res["adsorbate"].eq(ads)].sort_values("r2")
        y = np.arange(len(d))
        ax.barh(y, d["r2"], color=STEEL if ads == "benzene" else ORANGE, alpha=.88, height=.62, edgecolor="none")
        ax.set_yticks(y, d["dominant_metal"])
        ax.set_xlabel(r"OOD $R^2$ (leave-one-metal-out)")
        ax.set_ylabel("Held-out dominant metal")
        ax.set_title(f"Leave-one-metal-out OOD - {ads} (n ≥ 20)", fontweight="semibold")
        ax.grid(False)
        ax.set_axisbelow(False)
        xmin = max(0.0, float(d["r2"].min()) - 0.015)
        ax.set_xlim(xmin, 1.005)
        for yi, (_, r) in enumerate(d.iterrows()):
            ax.text(r["r2"] + .0025, yi, f"n={int(r['n_test'])}", va="center", fontsize=5.4, color="#333333")
        ax.text(-0.14, 1.04, panel, transform=ax.transAxes, fontweight="bold", fontsize=9)
    fig.subplots_adjust(wspace=.42)
    savefig(fig, OOD_DIR / "leave_one_metal_out_r2_n20")
    plt.close(fig)


def qc() -> None:
    rows = []
    for p in [RANDOM_DIR / "random_10_split_r2.tiff", GROUP_DIR / "kfold_vs_groupkfold_r2.tiff", OOD_DIR / "leave_one_metal_out_r2_n20.tiff"]:
        im = Image.open(p).convert("RGB")
        stat = ImageStat.Stat(im)
        rows.append({"file": str(p), "width": im.width, "height": im.height, "mean_rgb": ";".join(f"{x:.2f}" for x in stat.mean), "nonblank": bool(any(lo < hi for lo, hi in im.getextrema()))})
    pd.DataFrame(rows).to_csv(ROOT / "tiff_qc.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    style()
    plot_random()
    plot_group()
    plot_ood()
    qc()
    print(f"Replotted robustness figures in {ROOT}")


if __name__ == "__main__":
    main()
