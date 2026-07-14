from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, Polygon
from PIL import Image, ImageStat


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "Figures" / "screening_workflow"
OUT_DIR.mkdir(parents=True, exist_ok=True)
STEM = OUT_DIR / "Figure_3a_screening_workflow"

STAGES = [
    {
        "stage": "Descriptor QC",
        "retained_n": 162_985,
        "rule": "Complete Sine Coulomb matrix\ndescriptors",
    },
    {
        "stage": "Adsorption",
        "retained_n": 32_597,
        "rule": "Top 20% by harmonic-mean\nbalanced score",
    },
    {
        "stage": "Metal safety",
        "retained_n": 26_034,
        "rule": "Exclude Pb, Hg, Cd, Tl, Be\nand other hazardous metals",
    },
    {
        "stage": "Synthesizability",
        "retained_n": 6_000,
        "rule": "Ligand SA score < 5.0",
    },
    {
        "stage": "Aquatic toxicity",
        "retained_n": 1_000,
        "rule": "Top 1,000 with lowest predicted\naquatic toxicity",
    },
    {
        "stage": "PMT/vPvM",
        "retained_n": None,
        "rule": "Exclude predicted PMT or\nvPvM ligands",
    },
    {
        "stage": "Cost",
        "retained_n": 500,
        "rule": "Top 500 with lowest predicted\nlinker cost (CoPriNet)",
    },
    {
        "stage": "Expert review",
        "retained_n": 150,
        "rule": "Precursor availability,\nreproducibility and scale-up",
    },
]

COLORS = [
    "#3F6FA3",
    "#3C82A4",
    "#319594",
    "#4FAE9E",
    "#76BFA9",
    "#9CCFB7",
    "#C3DDC7",
    "#E9B949",
]


def width_for_count(count: float, minimum: float = 0.95, maximum: float = 3.75) -> float:
    lower, upper = math.log10(150), math.log10(168_524)
    fraction = (math.log10(count) - lower) / (upper - lower)
    return minimum + fraction * (maximum - minimum)


def write_source_data() -> None:
    with (STEM.parent / f"{STEM.name}_source_data.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=("stage", "retained_n", "selection_rule"))
        writer.writeheader()
        writer.writerow(
            {
                "stage": "DB-MOF database",
                "retained_n": 168_524,
                "selection_rule": "Initial database",
            }
        )
        for stage in STAGES:
            writer.writerow(
                {
                    "stage": stage["stage"],
                    "retained_n": "" if stage["retained_n"] is None else stage["retained_n"],
                    "selection_rule": stage["rule"].replace("\n", " "),
                }
            )


def build_figure() -> plt.Figure:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7.0,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.linewidth": 0.8,
        }
    )

    fig, ax = plt.subplots(figsize=(7.2, 5.35), facecolor="white")
    fig.subplots_adjust(left=0.015, right=0.99, bottom=0.025, top=0.985)
    ax.set_xlim(0, 7.2)
    ax.set_ylim(0, 5.55)
    ax.axis("off")

    ax.text(0.06, 5.39, "a", fontsize=9, fontweight="bold", va="top")
    ax.text(
        0.35,
        5.39,
        "Hierarchical screening of DB-MOF candidates",
        fontsize=10.2,
        fontweight="bold",
        color="#1F2933",
        va="top",
    )

    center_x = 2.08
    initial_width = width_for_count(168_524)
    cap_height = 0.48
    cap_y = 4.66
    cap = FancyBboxPatch(
        (center_x - initial_width / 2, cap_y),
        initial_width,
        cap_height,
        boxstyle="round,pad=0.015,rounding_size=0.09",
        linewidth=0,
        facecolor="#274C77",
    )
    ax.add_patch(cap)
    ax.text(
        center_x,
        cap_y + 0.30,
        "DB-MOF database",
        ha="center",
        va="center",
        fontsize=8.3,
        fontweight="bold",
        color="white",
    )
    ax.text(
        center_x,
        cap_y + 0.13,
        "n = 168,524",
        ha="center",
        va="center",
        fontsize=7.3,
        color="white",
    )

    # The PMT/vPvM retained count was not reported; use an intermediate width only for layout.
    visual_counts = [162_985, 32_597, 26_034, 6_000, 1_000, math.sqrt(1_000 * 500), 500, 150]
    widths = [initial_width] + [width_for_count(value) for value in visual_counts]
    band_height = 0.47
    gap = 0.045
    top_y = 4.56

    for index, (stage, color) in enumerate(zip(STAGES, COLORS)):
        top_width, bottom_width = widths[index], widths[index + 1]
        bottom_y = top_y - band_height
        polygon = Polygon(
            [
                (center_x - top_width / 2, top_y),
                (center_x + top_width / 2, top_y),
                (center_x + bottom_width / 2, bottom_y),
                (center_x - bottom_width / 2, bottom_y),
            ],
            closed=True,
            facecolor=color,
            edgecolor="none",
        )
        ax.add_patch(polygon)

        midpoint_y = (top_y + bottom_y) / 2
        dark_text = index >= 5
        text_color = "#1F2933" if dark_text else "white"
        if stage["retained_n"] is None:
            ax.text(
                center_x,
                midpoint_y,
                stage["stage"],
                ha="center",
                va="center",
                fontsize=6.8,
                fontweight="bold",
                color=text_color,
            )
        else:
            ax.text(
                center_x,
                midpoint_y + 0.075,
                stage["stage"],
                ha="center",
                va="center",
                fontsize=6.7,
                fontweight="bold",
                color=text_color,
            )
            ax.text(
                center_x,
                midpoint_y - 0.095,
                f"n = {stage['retained_n']:,}",
                ha="center",
                va="center",
                fontsize=6.3,
                color=text_color,
            )

        right_edge = center_x + max(top_width, bottom_width) / 2
        circle_x = 4.18
        ax.plot(
            [right_edge + 0.08, circle_x - 0.12],
            [midpoint_y, midpoint_y],
            color="#A7B0B7",
            linewidth=0.65,
            solid_capstyle="round",
            zorder=0,
        )
        marker = Circle((circle_x, midpoint_y), 0.095, facecolor=color, edgecolor="white", linewidth=0.8)
        ax.add_patch(marker)
        ax.text(
            circle_x,
            midpoint_y,
            str(index + 1),
            ha="center",
            va="center",
            fontsize=6.1,
            fontweight="bold",
            color=text_color,
        )
        ax.text(
            4.38,
            midpoint_y,
            stage["rule"],
            ha="left",
            va="center",
            fontsize=7.0,
            color="#27333D",
            linespacing=1.2,
        )
        top_y = bottom_y - gap

    ax.text(
        0.35,
        0.08,
        "Funnel widths are illustrative; n values denote retained candidates.",
        fontsize=5.8,
        color="#7B8790",
        va="bottom",
    )
    return fig


def save_figure(fig: plt.Figure) -> None:
    fig.savefig(f"{STEM}.svg", facecolor="white")
    fig.savefig(f"{STEM}.pdf", facecolor="white")
    fig.savefig(f"{STEM}.png", dpi=300, facecolor="white")
    fig.savefig(
        f"{STEM}.tiff",
        dpi=600,
        facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )


def verify_outputs() -> None:
    with Image.open(f"{STEM}.tiff") as image:
        assert image.width >= 4_000 and image.height >= 2_800
        grayscale = image.convert("L")
        assert ImageStat.Stat(grayscale).var[0] > 100
    svg_text = Path(f"{STEM}.svg").read_text(encoding="utf-8")
    assert "DB-MOF database" in svg_text
    assert "n = 150" in svg_text


def main() -> None:
    write_source_data()
    figure = build_figure()
    save_figure(figure)
    plt.close(figure)
    verify_outputs()
    print(STEM)


if __name__ == "__main__":
    main()
