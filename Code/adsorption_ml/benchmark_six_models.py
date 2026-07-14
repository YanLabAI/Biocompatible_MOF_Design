from __future__ import annotations

import argparse
import json
import math
import shutil
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    BaggingRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42
TEST_SIZE = 0.2
N_SPLITS = 5
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class TaskConfig:
    descriptor: str
    adsorbate: str
    data_file: Path
    target_col: str
    id_cols: tuple[str, ...]
    exclude_cols: tuple[str, ...] = ()


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(math.sqrt(mean_squared_error(y_true, y_pred)))


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": rmse(y_true, y_pred),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
    }


def nature_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "font.size": 8,
            "axes.linewidth": 0.8,
            "axes.edgecolor": "black",
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "legend.frameon": False,
        }
    )


def sanitize_name(name: str) -> str:
    return (
        name.replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("+", "plus")
        .replace("-", "_")
    )


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".csv":
        return pd.read_csv(path, encoding="utf-8-sig")
    raise ValueError(f"Unsupported file type: {path}")


def build_models() -> dict[str, Any]:
    models: dict[str, Any] = {
        "RandomForest": RandomForestRegressor(
            n_estimators=120,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            max_features="sqrt",
        ),
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=120,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            max_features="sqrt",
        ),
        "GradientBoosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
        "HistGradientBoosting": HistGradientBoostingRegressor(random_state=RANDOM_STATE),
        "Bagging": BaggingRegressor(n_estimators=80, random_state=RANDOM_STATE, n_jobs=-1),
    }
    try:
        from xgboost import XGBRegressor

        models["XGBoost"] = XGBRegressor(
            n_estimators=160,
            max_depth=6,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0,
        )
    except Exception:
        pass
    return models


def prepare_features(df: pd.DataFrame, task: TaskConfig) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    if task.target_col not in df.columns:
        raise ValueError(f"Target column {task.target_col!r} not found in {task.data_file}")

    df = df.copy()
    df = df.dropna(subset=[task.target_col])
    y = pd.to_numeric(df[task.target_col], errors="coerce")
    valid_y = y.notna()
    df = df.loc[valid_y].reset_index(drop=True)
    y = y.loc[valid_y].reset_index(drop=True)

    drop_cols = set(task.id_cols) | {task.target_col} | set(task.exclude_cols)
    drop_cols = {col for col in drop_cols if col in df.columns}
    feature_df = df.drop(columns=sorted(drop_cols))

    numeric_features: list[str] = []
    numeric_series: dict[str, pd.Series] = {}
    for col in feature_df.columns:
        converted = pd.to_numeric(feature_df[col], errors="coerce")
        if converted.notna().sum() > 0:
            numeric_series[col] = converted
            numeric_features.append(col)

    if not numeric_features:
        raise ValueError(f"No numeric features found for {task.descriptor} {task.adsorbate}")

    numeric_df = pd.DataFrame(numeric_series, index=feature_df.index)
    return numeric_df, y, numeric_features


def make_pipeline(model: Any, numeric_features: list[str]) -> Pipeline:
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
                numeric_features,
            )
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return Pipeline(steps=[("preprocess", preprocess), ("model", model)])


def save_fit_plot(
    y_train: np.ndarray,
    train_pred: np.ndarray,
    y_test: np.ndarray,
    test_pred: np.ndarray,
    cv_metrics: dict[str, float],
    test_metrics: dict[str, float],
    title: str,
    out_base: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(3.45, 3.2))
    ax.scatter(
        y_train,
        train_pred,
        s=14,
        c="#4C78A8",
        edgecolors="none",
        alpha=0.58,
        label="Train",
    )
    ax.scatter(
        y_test,
        test_pred,
        s=18,
        c="#F58518",
        edgecolors="black",
        linewidths=0.25,
        alpha=0.78,
        label="Test",
    )

    values = np.concatenate([np.asarray(y_train), np.asarray(train_pred), np.asarray(y_test), np.asarray(test_pred)])
    finite = values[np.isfinite(values)]
    lower = float(np.min(finite))
    upper = float(np.max(finite))
    pad = (upper - lower) * 0.05 if upper > lower else 1.0
    lower -= pad
    upper += pad
    ax.plot([lower, upper], [lower, upper], linestyle="--", color="#C44E52", linewidth=1.0)
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_xlabel("Actual values", fontsize=9)
    ax.set_ylabel("Predicted values", fontsize=9)
    ax.set_title(title, fontsize=9.5, pad=5)
    ax.tick_params(labelsize=8)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.8)
    ax.legend(loc="lower right", fontsize=7)

    text = (
        "5-fold CV\n"
        f"R2 = {cv_metrics['R2']:.3f}\n"
        f"RMSE = {cv_metrics['RMSE']:.3f}\n"
        f"MAE = {cv_metrics['MAE']:.3f}\n\n"
        "Test\n"
        f"R2 = {test_metrics['R2']:.3f}\n"
        f"RMSE = {test_metrics['RMSE']:.3f}\n"
        f"MAE = {test_metrics['MAE']:.3f}"
    )
    ax.text(
        0.04,
        0.96,
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=7,
        bbox={"facecolor": "white", "edgecolor": "black", "linewidth": 0.35, "alpha": 0.84, "pad": 2.5},
    )
    fig.tight_layout()
    fig.savefig(out_base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def run_task(task: TaskConfig, descriptor_result_dir: Path) -> pd.DataFrame:
    task_dir = descriptor_result_dir / task.adsorbate
    figures_dir = task_dir / "figures"
    weights_dir = task_dir / "model_weights"
    predictions_dir = task_dir / "predictions"
    for path in [figures_dir, weights_dir, predictions_dir]:
        path.mkdir(parents=True, exist_ok=True)

    df = read_table(task.data_file)
    X, y, numeric_features = prepare_features(df, task)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    models = build_models()
    cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    rows: list[dict[str, Any]] = []

    task_meta = {
        "descriptor": task.descriptor,
        "adsorbate": task.adsorbate,
        "data_file": str(task.data_file),
        "target_col": task.target_col,
        "id_cols": list(task.id_cols),
        "excluded_cols": list(task.exclude_cols),
        "n_samples": int(len(y)),
        "n_features": int(len(numeric_features)),
        "train_samples": int(len(y_train)),
        "test_samples": int(len(y_test)),
        "test_size": TEST_SIZE,
        "cv_folds": N_SPLITS,
        "random_state": RANDOM_STATE,
        "feature_columns": numeric_features,
    }
    (task_dir / "task_metadata.json").write_text(json.dumps(task_meta, indent=2), encoding="utf-8")

    best_model_name = None
    best_test_r2 = -np.inf
    best_test_rmse = np.inf

    for model_name, model in models.items():
        safe_model_name = sanitize_name(model_name)
        pipeline = make_pipeline(clone(model), numeric_features)
        print(f"[RUN] {task.descriptor} / {task.adsorbate} / {model_name}", flush=True)

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cv_pred = cross_val_predict(pipeline, X_train, y_train, cv=cv, n_jobs=None)
                cv_m = regression_metrics(y_train.to_numpy(), cv_pred)
                pipeline.fit(X_train, y_train)
                train_pred = pipeline.predict(X_train)
                test_pred = pipeline.predict(X_test)
        except Exception as exc:
            rows.append(
                {
                    "descriptor": task.descriptor,
                    "adsorbate": task.adsorbate,
                    "model": model_name,
                    "status": "failed",
                    "error": repr(exc),
                    "n_samples": len(y),
                    "n_features": len(numeric_features),
                }
            )
            continue

        train_m = regression_metrics(y_train.to_numpy(), train_pred)
        test_m = regression_metrics(y_test.to_numpy(), test_pred)

        model_file = weights_dir / f"{safe_model_name}.joblib"
        joblib.dump(pipeline, model_file)

        pred_df = pd.DataFrame(
            {
                "split": ["train"] * len(y_train) + ["test"] * len(y_test),
                "actual": np.concatenate([y_train.to_numpy(), y_test.to_numpy()]),
                "predicted": np.concatenate([train_pred, test_pred]),
            },
            index=list(y_train.index) + list(y_test.index),
        )
        pred_df.index.name = "original_row_index_after_cleaning"
        pred_df.to_csv(predictions_dir / f"{safe_model_name}_predictions.csv", encoding="utf-8-sig")

        save_fit_plot(
            y_train.to_numpy(),
            train_pred,
            y_test.to_numpy(),
            test_pred,
            cv_m,
            test_m,
            f"{model_name} - {task.adsorbate}",
            figures_dir / f"{safe_model_name}_fit",
        )

        row = {
            "descriptor": task.descriptor,
            "adsorbate": task.adsorbate,
            "model": model_name,
            "status": "ok",
            "n_samples": len(y),
            "n_features": len(numeric_features),
            "train_samples": len(y_train),
            "test_samples": len(y_test),
            "cv_folds": N_SPLITS,
            "random_state": RANDOM_STATE,
            "CV_R2": cv_m["R2"],
            "CV_RMSE": cv_m["RMSE"],
            "CV_MAE": cv_m["MAE"],
            "Train_R2": train_m["R2"],
            "Train_RMSE": train_m["RMSE"],
            "Train_MAE": train_m["MAE"],
            "Test_R2": test_m["R2"],
            "Test_RMSE": test_m["RMSE"],
            "Test_MAE": test_m["MAE"],
            "model_file": str(model_file),
            "figure_tiff": str(figures_dir / f"{safe_model_name}_fit.tiff"),
        }
        rows.append(row)

        if test_m["R2"] > best_test_r2 or (math.isclose(test_m["R2"], best_test_r2) and test_m["RMSE"] < best_test_rmse):
            best_model_name = model_name
            best_test_r2 = test_m["R2"]
            best_test_rmse = test_m["RMSE"]

    results = pd.DataFrame(rows)
    results.to_csv(task_dir / "model_metrics.csv", index=False, encoding="utf-8-sig")
    if not results.empty:
        ok = results[results["status"] == "ok"].copy()
        if not ok.empty:
            ok.sort_values(["Test_R2", "Test_RMSE"], ascending=[False, True]).to_csv(
                task_dir / "model_metrics_ranked.csv", index=False, encoding="utf-8-sig"
            )
            try:
                with pd.ExcelWriter(task_dir / "model_metrics_summary.xlsx") as writer:
                    ok.to_excel(writer, sheet_name="all_metrics", index=False)
                    ok.sort_values(["Test_R2", "Test_RMSE"], ascending=[False, True]).to_excel(
                        writer, sheet_name="ranked_by_test_R2", index=False
                    )
            except Exception as exc:
                (task_dir / "excel_export_error.txt").write_text(repr(exc), encoding="utf-8")

            if best_model_name is not None:
                safe_best = sanitize_name(best_model_name)
                shutil.copy2(weights_dir / f"{safe_best}.joblib", task_dir / "best_model.joblib")
                (task_dir / "best_model.json").write_text(
                    json.dumps(
                        {
                            "best_model": best_model_name,
                            "selection_rule": "highest Test_R2, lower Test_RMSE as tie-breaker",
                            "Test_R2": best_test_r2,
                            "Test_RMSE": best_test_rmse,
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
    return results


def build_tasks(root: Path) -> list[TaskConfig]:
    return [
        TaskConfig(
            descriptor="stoichiometric_120",
            adsorbate="benzene",
            data_file=root / "stoichiometric_120" / "stoichiometric_120_benzene_1000.xlsx",
            target_col="benzene_adsorption",
            id_cols=("MOF",),
        ),
        TaskConfig(
            descriptor="stoichiometric_120",
            adsorbate="toluene",
            data_file=root / "stoichiometric_120" / "stoichiometric_120_toluene_1000.xlsx",
            target_col="toluene_adsorption",
            id_cols=("MOF",),
        ),
        TaskConfig(
            descriptor="orbital_field_matrix",
            adsorbate="benzene",
            data_file=root / "orbital_field_matrix" / "orbital_field_matrix_benzene_1000.xlsx",
            target_col="benzene_adsorption",
            id_cols=("MOF",),
        ),
        TaskConfig(
            descriptor="orbital_field_matrix",
            adsorbate="toluene",
            data_file=root / "orbital_field_matrix" / "orbital_field_matrix_toluene_1000.xlsx",
            target_col="toluene_adsorption",
            id_cols=("MOF",),
        ),
        TaskConfig(
            descriptor="sine_matrix",
            adsorbate="benzene",
            data_file=root / "sine_matrix" / "sine_matrix_benzene_1000.xlsx",
            target_col="benzene_adsorption",
            id_cols=("MOF",),
        ),
        TaskConfig(
            descriptor="sine_matrix",
            adsorbate="toluene",
            data_file=root / "sine_matrix" / "sine_matrix_toluene_1000.xlsx",
            target_col="toluene_adsorption",
            id_cols=("MOF",),
        ),
        TaskConfig(
            descriptor="structural",
            adsorbate="benzene",
            data_file=root / "structural" / "structural_benzene_679.csv",
            target_col="Absorb",
            id_cols=("filename",),
        ),
        TaskConfig(
            descriptor="structural",
            adsorbate="toluene",
            data_file=root / "structural" / "structural_toluene_1000.csv",
            target_col="ABS",
            id_cols=("filename",),
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run adsorption regression experiments for MOF descriptors.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT_ROOT / "Data" / "adsorption",
        help="Root folder containing descriptor subfolders.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "Results" / "model_benchmark" / "rerun",
        help="Directory for metrics, models, predictions and plots.",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Optional task filters such as stoichiometric_120:benzene.",
    )
    args = parser.parse_args()

    nature_style()
    data_root = args.data_root
    output_root = args.output_root
    all_results: list[pd.DataFrame] = []
    filters = set(args.only or [])

    for task in build_tasks(data_root):
        key = f"{task.descriptor}:{task.adsorbate}"
        if filters and key not in filters:
            continue
        descriptor_result_dir = output_root / task.descriptor
        descriptor_result_dir.mkdir(parents=True, exist_ok=True)
        results = run_task(task, descriptor_result_dir)
        all_results.append(results)

    if all_results:
        summary = pd.concat(all_results, ignore_index=True)
        summary_dir = output_root / "summary"
        summary_dir.mkdir(parents=True, exist_ok=True)
        summary.to_csv(summary_dir / "all_model_metrics.csv", index=False, encoding="utf-8-sig")
        ok = summary[summary["status"] == "ok"].copy()
        if not ok.empty:
            best_by_task = (
                ok.sort_values(["descriptor", "adsorbate", "Test_R2", "Test_RMSE"], ascending=[True, True, False, True])
                .groupby(["descriptor", "adsorbate"], as_index=False)
                .first()
            )
            best_by_task.to_csv(summary_dir / "best_models_by_task.csv", index=False, encoding="utf-8-sig")
            try:
                with pd.ExcelWriter(summary_dir / "all_model_metrics_summary.xlsx") as writer:
                    ok.to_excel(writer, sheet_name="all_metrics", index=False)
                    best_by_task.to_excel(writer, sheet_name="best_by_task", index=False)
            except Exception as exc:
                (summary_dir / "excel_export_error.txt").write_text(repr(exc), encoding="utf-8")

if __name__ == "__main__":
    main()
