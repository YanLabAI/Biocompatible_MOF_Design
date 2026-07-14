from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageStat
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "Data" / "adsorption"
OUT = PROJECT_ROOT / "Results" / "robustness_current1000" / "rerun"
RANDOM_DIR = OUT / "random_10_splits"
GROUP_DIR = OUT / "groupkfold_dominant_metal"
OOD_DIR = OUT / "leave_one_metal_out_ood"

TASKS = {
    "benzene": {
        "data": DATA_ROOT / "sine_matrix" / "sine_matrix_benzene_1000.xlsx",
        "stoich": DATA_ROOT / "stoichiometric_120" / "stoichiometric_120_benzene_1000.xlsx",
        "target": "benzene_adsorption",
        "model": "GradientBoosting",
        "color": "#4C78A8",
    },
    "toluene": {
        "data": DATA_ROOT / "sine_matrix" / "sine_matrix_toluene_1000.xlsx",
        "stoich": DATA_ROOT / "stoichiometric_120" / "stoichiometric_120_toluene_1000.xlsx",
        "target": "toluene_adsorption",
        "model": "XGBoost",
        "color": "#F28E2B",
    },
}

NON_METALS = {
    "H", "He", "B", "C", "N", "O", "F", "Ne", "Si", "P", "S", "Cl", "Ar",
    "Ge", "As", "Se", "Br", "Kr", "Sb", "Te", "I", "Xe", "At", "Rn", "Ts", "Og",
}
SEEDS_10 = list(range(10))
CV_SEED = 42
MIN_OOD_N = 20


def style() -> None:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
    plt.rcParams["svg.fonttype"] = "none"
    mpl.rcParams.update({
        "pdf.fonttype": 42, "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.75,
        "axes.labelsize": 7.5, "axes.titlesize": 7.5, "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5, "legend.fontsize": 6.2,
    })


def savefig(fig: plt.Figure, out_base: Path) -> None:
    for suffix in ("svg", "pdf", "tiff"):
        p = out_base.with_suffix(f".{suffix}")
        if suffix == "tiff":
            fig.savefig(p, dpi=600, bbox_inches="tight", pad_inches=0.04, pil_kwargs={"compression": "tiff_lzw"})
        else:
            fig.savefig(p, bbox_inches="tight", pad_inches=0.04)


def model(name: str):
    if name == "GradientBoosting":
        return GradientBoostingRegressor(random_state=42)
    return XGBRegressor(
        n_estimators=160, max_depth=6, learning_rate=0.08, subsample=0.85,
        colsample_bytree=0.85, objective="reg:squarederror", random_state=42,
        n_jobs=-1, verbosity=0,
    )


def load_task(task: dict) -> tuple[pd.DataFrame, pd.Series, list[str], pd.Series, pd.DataFrame]:
    df = pd.read_excel(task["data"])
    sto = pd.read_excel(task["stoich"])
    assert len(df) == 1000, f"{task['data']} rows != 1000"
    assert len(sto) == 1000, f"{task['stoich']} rows != 1000"

    target = task["target"]
    features = [c for c in df.columns if c not in {"MOF", target}]
    assert len(features) == 584, f"{task['data']} feature count != 584"
    X = pd.DataFrame({c: pd.to_numeric(df[c], errors="coerce") for c in features})
    y = pd.to_numeric(df[target], errors="coerce")
    valid = y.notna()
    df, X, y = df.loc[valid].reset_index(drop=True), X.loc[valid].reset_index(drop=True), y.loc[valid].reset_index(drop=True)

    groups = dominant_metal(sto).set_index("MOF").reindex(df["MOF"])["dominant_metal"].fillna("Unknown").reset_index(drop=True)
    return X, y, features, groups, df[["MOF"]].copy()


def dominant_metal(sto: pd.DataFrame) -> pd.DataFrame:
    frac_cols = [c for c in sto.columns if str(c).endswith(" fraction")]
    metal_cols = [c for c in frac_cols if str(c).replace(" fraction", "") not in NON_METALS]
    rows = []
    for _, r in sto.iterrows():
        vals = pd.to_numeric(r[metal_cols], errors="coerce").fillna(0.0)
        mx = float(vals.max())
        if mx <= 0:
            metal = "Unknown"
        else:
            tied = sorted(str(c).replace(" fraction", "") for c, v in vals.items() if np.isclose(float(v), mx))
            metal = tied[0]
        rows.append({"MOF": r["MOF"], "dominant_metal": metal, "dominant_metal_fraction": mx})
    return pd.DataFrame(rows)


def pipe(model_name: str, features: list[str]) -> Pipeline:
    prep = ColumnTransformer([("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), features)], verbose_feature_names_out=False)
    return Pipeline([("preprocess", prep), ("model", model(model_name))])


def metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def summarize(df: pd.DataFrame, group_cols: list[str], metric_cols=("r2", "rmse", "mae")) -> pd.DataFrame:
    out = df.groupby(group_cols)[list(metric_cols)].agg(["mean", "std"]).reset_index()
    out.columns = [
        "_".join(str(x) for x in col if str(x))
        if isinstance(col, tuple) else str(col)
        for col in out.columns
    ]
    return out


def run_random(all_data: dict) -> None:
    RANDOM_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for ads, d in all_data.items():
        X, y, features = d["X"], d["y"], d["features"]
        for seed in SEEDS_10:
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed)
            m = pipe(TASKS[ads]["model"], features).fit(Xtr, ytr)
            rows.append({"adsorbate": ads, "seed": seed, "model": TASKS[ads]["model"], **metrics(yte, m.predict(Xte))})
    res = pd.DataFrame(rows)
    summ = summarize(res, ["adsorbate", "model"])
    res.to_csv(RANDOM_DIR / "random_10_split_results.csv", index=False, encoding="utf-8-sig")
    summ.to_csv(RANDOM_DIR / "random_10_split_summary.csv", index=False, encoding="utf-8-sig")
    plot_random(res)
    write_meta(RANDOM_DIR, "10 independent random 80/20 splits", {"seeds": SEEDS_10, "test_size": 0.2})
    assert res.groupby("adsorbate").size().eq(10).all()


def run_groupkfold(all_data: dict) -> None:
    GROUP_DIR.mkdir(parents=True, exist_ok=True)
    rows, missing = [], []
    for ads, d in all_data.items():
        X, y, features, groups = d["X"], d["y"], d["features"], d["groups"]
        missing += d["mofs"].loc[groups.eq("Unknown"), "MOF"].astype(str).tolist()
        for cv_name, splitter in [("KFold", KFold(5, shuffle=True, random_state=CV_SEED)), ("GroupKFold", GroupKFold(5))]:
            splits = splitter.split(X, y, groups) if cv_name == "GroupKFold" else splitter.split(X, y)
            for fold, (tr, te) in enumerate(splits, start=1):
                m = pipe(TASKS[ads]["model"], features).fit(X.iloc[tr], y.iloc[tr])
                rows.append({"adsorbate": ads, "cv": cv_name, "fold": fold, "model": TASKS[ads]["model"], **metrics(y.iloc[te], m.predict(X.iloc[te]))})
    res = pd.DataFrame(rows)
    summ = summarize(res, ["adsorbate", "cv", "model"])
    res.to_csv(GROUP_DIR / "groupkfold_fold_results.csv", index=False, encoding="utf-8-sig")
    summ.to_csv(GROUP_DIR / "groupkfold_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"MOF": missing}).to_csv(GROUP_DIR / "missing_dominant_metal_mofs.csv", index=False, encoding="utf-8-sig")
    plot_group(res)
    write_meta(GROUP_DIR, "KFold vs GroupKFold by dominant metal", {"n_splits": 5, "missing_group_count": len(missing)})
    assert res.groupby(["adsorbate", "cv"]).size().eq(5).all()


def run_ood(all_data: dict) -> None:
    OOD_DIR.mkdir(parents=True, exist_ok=True)
    rows, preds = [], []
    for ads, d in all_data.items():
        X, y, features, groups, mofs = d["X"], d["y"], d["features"], d["groups"], d["mofs"]
        for metal, n_test in groups.value_counts().items():
            if metal == "Unknown":
                status = "skipped_unknown_group"
            else:
                status = "evaluated"
            test = groups.eq(metal).to_numpy()
            train = ~test
            row = {"adsorbate": ads, "dominant_metal": metal, "n_train": int(train.sum()), "n_test": int(test.sum()), "model": TASKS[ads]["model"], "status": status}
            if status == "evaluated" and train.sum() and test.sum() >= 2:
                m = pipe(TASKS[ads]["model"], features).fit(X.loc[train], y.loc[train])
                pred = m.predict(X.loc[test])
                row.update(metrics(y.loc[test], pred))
                for mof, yt, yp in zip(mofs.loc[test, "MOF"], y.loc[test], pred):
                    preds.append({"adsorbate": ads, "dominant_metal": metal, "MOF": mof, "actual": float(yt), "predicted": float(yp)})
            else:
                row.update({"r2": np.nan, "rmse": np.nan, "mae": np.nan})
            rows.append(row)
    res = pd.DataFrame(rows)
    pred_df = pd.DataFrame(preds)
    summary = ood_summary(res)
    res.to_csv(OOD_DIR / "leave_one_metal_out_results_all_groups.csv", index=False, encoding="utf-8-sig")
    res[res["n_test"] >= MIN_OOD_N].to_csv(OOD_DIR / "leave_one_metal_out_results_n20.csv", index=False, encoding="utf-8-sig")
    pred_df.to_csv(OOD_DIR / "leave_one_metal_out_predictions.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OOD_DIR / "leave_one_metal_out_summary.csv", index=False, encoding="utf-8-sig")
    plot_ood(res)
    write_meta(OOD_DIR, "Leave-one-metal-out OOD", {"min_n_for_main_plot": MIN_OOD_N})
    assert (res.query("status == 'evaluated' and n_test >= @MIN_OOD_N").groupby("adsorbate").size() > 0).all()


def ood_summary(res: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (ads, model_name), df in res[res["status"].eq("evaluated")].groupby(["adsorbate", "model"]):
        shown = df[df["n_test"] >= MIN_OOD_N]
        for label, part in [("all_evaluated", df), ("n_test_ge_20", shown)]:
            part = part.dropna(subset=["r2", "rmse", "mae"])
            w = part["n_test"].to_numpy(float)
            rows.append({
                "adsorbate": ads, "model": model_name, "scope": label, "n_groups": int(len(part)),
                "weighted_r2": float(np.average(part["r2"], weights=w)) if len(part) else np.nan,
                "mean_r2": float(part["r2"].mean()) if len(part) else np.nan,
                "weighted_rmse": float(np.average(part["rmse"], weights=w)) if len(part) else np.nan,
                "weighted_mae": float(np.average(part["mae"], weights=w)) if len(part) else np.nan,
            })
    return pd.DataFrame(rows)


def plot_random(res: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    x = np.arange(2)
    for i, ads in enumerate(["benzene", "toluene"]):
        d = res[res["adsorbate"].eq(ads)]
        ax.bar(x[i], d["r2"].mean(), yerr=d["r2"].std(ddof=1), color=TASKS[ads]["color"], alpha=0.85, capsize=3)
        ax.scatter(np.full(len(d), x[i]) + np.linspace(-.08, .08, len(d)), d["r2"], s=14, color="#444", zorder=3)
        ax.text(x[i], d["r2"].mean() + d["r2"].std(ddof=1) + .01, f"{d['r2'].mean():.3f} ± {d['r2'].std(ddof=1):.3f}", ha="center", fontsize=6)
    ax.set_xticks(x, ["Benzene", "Toluene"])
    ax.set_ylabel(r"Test $R^2$")
    ax.set_title("10 independent random 80/20 splits")
    ax.grid(axis="y", color="#EEE")
    savefig(fig, RANDOM_DIR / "random_10_split_r2")
    plt.close(fig)


def plot_group(res: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.8))
    for ax, ads, label in zip(axes, ["benzene", "toluene"], ["a", "b"]):
        d = res[res["adsorbate"].eq(ads)]
        for i, cv in enumerate(["KFold", "GroupKFold"]):
            v = d[d["cv"].eq(cv)]["r2"]
            ax.bar(i, v.mean(), yerr=v.std(ddof=1), color=[TASKS[ads]["color"], "#F28E2B"][i], alpha=0.85, capsize=3)
            ax.scatter(np.full(len(v), i) + np.linspace(-.04, .04, len(v)), v, s=14, color="#444", zorder=3)
            ax.text(i, v.mean() + v.std(ddof=1) + .01, f"{v.mean():.3f} ± {v.std(ddof=1):.3f}", ha="center", fontsize=6)
        ax.set_xticks([0, 1], ["KFold (5)", "GroupKFold (5)"])
        ax.set_ylabel(r"$R^2$ score")
        ax.set_title(f"KFold vs GroupKFold - {ads}")
        ax.grid(axis="y", color="#EEE")
        ax.text(-0.18, 1.05, label, transform=ax.transAxes, fontweight="bold")
    fig.subplots_adjust(wspace=.32)
    savefig(fig, GROUP_DIR / "kfold_vs_groupkfold_r2")
    plt.close(fig)


def plot_ood(res: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.8))
    for ax, ads, label in zip(axes, ["benzene", "toluene"], ["a", "b"]):
        d = res.query("adsorbate == @ads and status == 'evaluated' and n_test >= @MIN_OOD_N").sort_values("r2")
        y = np.arange(len(d))
        ax.barh(y, d["r2"], color=TASKS[ads]["color"], alpha=.9)
        ax.set_yticks(y, d["dominant_metal"])
        ax.set_xlabel(r"OOD $R^2$ (leave-one-metal-out)")
        ax.set_ylabel("Held-out dominant metal")
        ax.set_title(f"OOD by dominant metal - {ads} (n ≥ 20)")
        ax.grid(axis="x", color="#EEE")
        for yi, (_, r) in enumerate(d.iterrows()):
            ax.text(r["r2"] + .005, yi, f"n={int(r['n_test'])}", va="center", fontsize=5.5)
        ax.text(-0.18, 1.04, label, transform=ax.transAxes, fontweight="bold")
    fig.subplots_adjust(wspace=.42)
    savefig(fig, OOD_DIR / "leave_one_metal_out_r2_n20")
    plt.close(fig)


def write_meta(folder: Path, analysis: str, extra: dict) -> None:
    meta = {
        "analysis": analysis,
        "project_root": str(PROJECT_ROOT),
        "tasks": {k: {"data": str(v["data"]), "stoich": str(v["stoich"]), "target": v["target"], "model": v["model"]} for k, v in TASKS.items()},
        "dominant_metal_rule": "maximum metal fraction from current stoichiometric_120_1000 table; ties sorted alphabetically",
        **extra,
    }
    (folder / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def tiff_check(folder: Path) -> pd.DataFrame:
    rows = []
    for p in folder.glob("*.tiff"):
        im = Image.open(p).convert("RGB")
        stat = ImageStat.Stat(im)
        rows.append({"file": str(p), "width": im.width, "height": im.height, "mean_rgb": ";".join(f"{x:.2f}" for x in stat.mean), "nonblank": bool(any(lo < hi for lo, hi in im.getextrema()))})
    return pd.DataFrame(rows)


def main() -> None:
    style()
    all_data = {}
    for ads, task in TASKS.items():
        X, y, features, groups, mofs = load_task(task)
        all_data[ads] = {"X": X, "y": y, "features": features, "groups": groups, "mofs": mofs}
    run_random(all_data)
    run_groupkfold(all_data)
    run_ood(all_data)
    qc = pd.concat([tiff_check(RANDOM_DIR), tiff_check(GROUP_DIR), tiff_check(OOD_DIR)], ignore_index=True)
    qc.to_csv(OUT / "tiff_qc.csv", index=False, encoding="utf-8-sig")
    assert qc["nonblank"].all()
    print(f"Saved robustness outputs to: {OUT}")
    print(qc.to_string(index=False))


if __name__ == "__main__":
    main()
