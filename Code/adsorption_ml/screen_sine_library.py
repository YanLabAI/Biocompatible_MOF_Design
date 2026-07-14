from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def load_feature_columns(metadata_path: Path) -> list[str]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return metadata["feature_columns"]


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def predict_screening_library(
    data_file: Path,
    out_dir: Path,
    chunksize: int = 20000,
    top_fraction: float = 0.20,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    benzene_model_path = PROJECT_ROOT / "Models" / "sine_matrix" / "benzene_gradient_boosting.joblib"
    toluene_model_path = PROJECT_ROOT / "Models" / "sine_matrix" / "toluene_xgboost.joblib"
    feature_metadata = PROJECT_ROOT / "Models" / "sine_matrix" / "benzene_task_metadata.json"

    feature_cols = load_feature_columns(feature_metadata)
    usecols = ["MOF_Name"] + feature_cols

    benzene_model = joblib.load(benzene_model_path)
    toluene_model = joblib.load(toluene_model_path)

    prediction_parts: list[pd.DataFrame] = []
    total_rows = 0
    for i, chunk in enumerate(pd.read_csv(data_file, usecols=usecols, chunksize=chunksize), start=1):
        names = chunk["MOF_Name"].astype(str)
        X = chunk[feature_cols]
        benzene_pred = benzene_model.predict(X)
        toluene_pred = toluene_model.predict(X)
        part = pd.DataFrame(
            {
                "MOF_Name": names,
                "predicted_benzene_adsorption": benzene_pred,
                "predicted_toluene_adsorption": toluene_pred,
            }
        )
        part["mean_predicted_adsorption"] = part[
            ["predicted_benzene_adsorption", "predicted_toluene_adsorption"]
        ].mean(axis=1)
        part["min_predicted_adsorption"] = part[
            ["predicted_benzene_adsorption", "predicted_toluene_adsorption"]
        ].min(axis=1)
        prediction_parts.append(part)
        total_rows += len(part)
        print(f"Processed chunk {i}: total rows={total_rows}", flush=True)

    predictions = pd.concat(prediction_parts, ignore_index=True)
    top_n = int(np.ceil(len(predictions) * top_fraction))

    full_predictions_file = out_dir / "all_mofs_sine_best_models_predictions.csv"
    predictions.to_csv(full_predictions_file, index=False, encoding="utf-8-sig")

    sorted_specs = [
        ("benzene", "predicted_benzene_adsorption"),
        ("toluene", "predicted_toluene_adsorption"),
        ("mean_adsorption", "mean_predicted_adsorption"),
        ("min_adsorption", "min_predicted_adsorption"),
    ]
    output_files: dict[str, str] = {"full_predictions": str(full_predictions_file)}
    for label, col in sorted_specs:
        sorted_df = predictions.sort_values(col, ascending=False, kind="mergesort").reset_index(drop=True)
        sorted_df.insert(0, "rank", np.arange(1, len(sorted_df) + 1))
        sorted_file = out_dir / f"all_mofs_sorted_by_{label}.csv"
        top_file = out_dir / f"top20_percent_mofs_by_{label}.csv"
        sorted_df.to_csv(sorted_file, index=False, encoding="utf-8-sig")
        sorted_df.head(top_n).to_csv(top_file, index=False, encoding="utf-8-sig")
        output_files[f"sorted_by_{label}"] = str(sorted_file)
        output_files[f"top20_by_{label}"] = str(top_file)

    metadata = {
        "input_file": str(data_file),
        "output_dir": str(out_dir),
        "n_rows_screened": int(len(predictions)),
        "top_fraction": top_fraction,
        "top_n": int(top_n),
        "feature_count": len(feature_cols),
        "feature_columns_first_5": feature_cols[:5],
        "feature_columns_last_5": feature_cols[-5:],
        "benzene_model": str(benzene_model_path),
        "benzene_model_name": "GradientBoosting",
        "toluene_model": str(toluene_model_path),
        "toluene_model_name": "XGBoost",
        "output_files": output_files,
        "prediction_summary": {
            "predicted_benzene_adsorption": {
                "min": float(predictions["predicted_benzene_adsorption"].min()),
                "max": float(predictions["predicted_benzene_adsorption"].max()),
                "mean": float(predictions["predicted_benzene_adsorption"].mean()),
                "median": float(predictions["predicted_benzene_adsorption"].median()),
            },
            "predicted_toluene_adsorption": {
                "min": float(predictions["predicted_toluene_adsorption"].min()),
                "max": float(predictions["predicted_toluene_adsorption"].max()),
                "mean": float(predictions["predicted_toluene_adsorption"].mean()),
                "median": float(predictions["predicted_toluene_adsorption"].median()),
            },
        },
    }
    (out_dir / "screening_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Screen all MOFs using best sine_matrix adsorption models.")
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="CSV containing MOF_Name and the 584 Sine Coulomb matrix eigenvalue columns.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "Results" / "screening" / "rerun",
    )
    parser.add_argument("--chunksize", type=int, default=20000)
    parser.add_argument("--top-fraction", type=float, default=0.20)
    args = parser.parse_args()
    predict_screening_library(
        args.input,
        args.output_dir,
        chunksize=args.chunksize,
        top_fraction=args.top_fraction,
    )


if __name__ == "__main__":
    main()
