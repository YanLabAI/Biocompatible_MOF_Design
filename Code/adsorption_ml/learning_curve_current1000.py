from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBRegressor
except Exception as exc:  # pragma: no cover
    XGBRegressor = None
    XGB_IMPORT_ERROR = exc
else:
    XGB_IMPORT_ERROR = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "Data" / "adsorption"
OUT_DIR = PROJECT_ROOT / "Results" / "learning_curve" / "rerun"
SOURCE_DIR = OUT_DIR / "source_data"

TASKS = [
    {
        "adsorbate": "benzene",
        "title": "Benzene",
        "data_file": DATA_ROOT / "sine_matrix" / "sine_matrix_benzene_1000.xlsx",
        "target_col": "benzene_adsorption",
        "model_name": "GradientBoosting",
        "color": "#4C78A8",
        "fill": "#D8E7F5",
    },
    {
        "adsorbate": "toluene",
        "title": "Toluene",
        "data_file": DATA_ROOT / "sine_matrix" / "sine_matrix_toluene_1000.xlsx",
        "target_col": "toluene_adsorption",
        "model_name": "XGBoost",
        "color": "#F28E2B",
        "fill": "#FCE0C1",
    },
]

SAMPLE_SIZES = [200, 400, 600, 800, 1000]
SPLIT_SEEDS = [0, 1, 2, 3, 4]
MODEL_RANDOM_STATE = 42


def apply_style() -> None:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
    plt.rcParams["svg.fonttype"] = "none"
    mpl.rcParams.update(
        {
            "pdf.fonttype": 42,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#222222",
            "axes.linewidth": 0.75,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "axes.labelsize": 7.3,
            "axes.titlesize": 8.0,
            "xtick.labelsize": 6.4,
            "ytick.labelsize": 6.4,
            "legend.fontsize": 5.8,
        }
    )


def load_xy(task: dict) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    df = pd.read_excel(task["data_file"])
    y = pd.to_numeric(df[task["target_col"]], errors="coerce")
    valid = y.notna()
    df = df.loc[valid].reset_index(drop=True)
    y = y.loc[valid].reset_index(drop=True)
    feature_cols = [
        c for c in df.columns
        if c not in {"MOF", task["target_col"]} and pd.to_numeric(df[c], errors="coerce").notna().sum() > 0
    ]
    X = pd.DataFrame({c: pd.to_numeric(df[c], errors="coerce") for c in feature_cols})
    return X, y, feature_cols


def build_model(model_name: str):
    if model_name == "GradientBoosting":
        return GradientBoostingRegressor(random_state=MODEL_RANDOM_STATE)
    if model_name == "XGBoost":
        if XGBRegressor is None:
            raise RuntimeError(f"xgboost is unavailable: {XGB_IMPORT_ERROR}")
        return XGBRegressor(
            n_estimators=160,
            max_depth=6,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="reg:squarederror",
            random_state=MODEL_RANDOM_STATE,
            n_jobs=-1,
            verbosity=0,
        )
    raise ValueError(f"Unsupported model: {model_name}")


def build_pipeline(model_name: str, feature_cols: list[str]) -> Pipeline:
    preprocess = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                feature_cols,
            )
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return Pipeline(steps=[("preprocess", preprocess), ("model", build_model(model_name))])


def evaluate_task(task: dict) -> pd.DataFrame:
    X_all, y_all, feature_cols = load_xy(task)
    n_total = len(y_all)
    if n_total != 1000:
        raise ValueError(f"{task['adsorbate']} expected 1000 rows, found {n_total}")

    records = []
    for sample_size in SAMPLE_SIZES:
        r2_values: list[float] = []
        for seed in SPLIT_SEEDS:
            if sample_size < n_total:
                rng = np.random.default_rng(seed)
                idx = rng.choice(n_total, size=sample_size, replace=False)
                X = X_all.iloc[idx].reset_index(drop=True)
                y = y_all.iloc[idx].reset_index(drop=True)
            else:
                X = X_all
                y = y_all

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=seed,
            )
            pipe = build_pipeline(task["model_name"], feature_cols)
            pipe.fit(X_train, y_train)
            pred = pipe.predict(X_test)
            r2_values.append(float(r2_score(y_test, pred)))

        records.append(
            {
                "dataset": task["adsorbate"],
                "model": task["model_name"],
                "data_file": str(task["data_file"]),
                "sample_size": int(sample_size),
                "repeats": len(SPLIT_SEEDS),
                "r2_mean": float(np.mean(r2_values)),
                "r2_std": float(np.std(r2_values, ddof=1)),
                "r2_values": ";".join(f"{v:.6f}" for v in r2_values),
            }
        )
    return pd.DataFrame(records)


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.13,
        1.06,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.6,
        fontweight="bold",
    )


def save_figure(fig: plt.Figure, stem: str) -> None:
    for suffix in ("svg", "pdf", "tiff"):
        out = OUT_DIR / f"{stem}.{suffix}"
        if suffix == "tiff":
            fig.savefig(
                out,
                dpi=600,
                format="tiff",
                bbox_inches="tight",
                pad_inches=0.04,
                pil_kwargs={"compression": "tiff_lzw"},
            )
        else:
            fig.savefig(out, bbox_inches="tight", pad_inches=0.04)


def plot_s8(curves: dict[str, pd.DataFrame]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.75))
    for ax, task, panel in zip(axes, TASKS, ["a", "b"]):
        df = curves[task["adsorbate"]].sort_values("sample_size")
        x = df["sample_size"].to_numpy(dtype=float)
        y = df["r2_mean"].to_numpy(dtype=float)
        sd = df["r2_std"].to_numpy(dtype=float)
        ax.fill_between(x, y - sd, y + sd, color=task["fill"], alpha=0.75, linewidth=0, label="mean ± SD")
        ax.errorbar(
            x,
            y,
            yerr=sd,
            color=task["color"],
            marker="o",
            markersize=3.0,
            lw=1.1,
            elinewidth=0.8,
            capsize=2.0,
        )
        best_idx = int(np.argmax(y))
        ax.scatter([x[best_idx]], [y[best_idx]], color="#D62728", s=20, zorder=4, label="best point")
        ymin = max(0.0, float(np.min(y - sd)) - 0.035)
        ymax = min(1.0, float(np.max(y + sd)) + 0.025)
        ax.set_ylim(ymin, ymax)
        ax.set_xticks(SAMPLE_SIZES)
        ax.set_title(task["title"], pad=5)
        ax.set_xlabel("Sample size")
        ax.set_ylabel(r"Test $R^2$")
        ax.grid(axis="y", color="#EEEEEE", linewidth=0.5)
        ax.legend(loc="lower right", frameon=False)
        add_panel_label(ax, panel)
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.19, top=0.81, wspace=0.30)
    save_figure(fig, "Figure_S8_learning_curve")
    plt.close(fig)


def main() -> None:
    apply_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)

    curves = {task["adsorbate"]: evaluate_task(task) for task in TASKS}
    combined = pd.concat(curves.values(), ignore_index=True)
    curves["benzene"].to_csv(SOURCE_DIR / "learning_curve_benzene.csv", index=False, encoding="utf-8-sig")
    curves["toluene"].to_csv(SOURCE_DIR / "learning_curve_toluene.csv", index=False, encoding="utf-8-sig")
    combined.to_csv(SOURCE_DIR / "learning_curve_combined.csv", index=False, encoding="utf-8-sig")
    meta = {
        "note": "Corrected Figure S8 using current 1000-row sine_matrix modeling datasets.",
        "sample_sizes": SAMPLE_SIZES,
        "split_seeds": SPLIT_SEEDS,
        "test_size": 0.2,
        "tasks": [
            {
                "adsorbate": task["adsorbate"],
                "data_file": str(task["data_file"]),
                "model": task["model_name"],
                "n_rows": 1000,
            }
            for task in TASKS
        ],
    }
    (SOURCE_DIR / "learning_curve_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    plot_s8(curves)
    print(combined.to_string(index=False))
    print(f"Updated Figure S8 in: {OUT_DIR}")


if __name__ == "__main__":
    main()
