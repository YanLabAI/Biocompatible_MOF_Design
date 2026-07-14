from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


BENZENE = "predicted_benzene_adsorption"
TOLUENE = "predicted_toluene_adsorption"


def rank_balanced_score(input_csv: Path, output_dir: Path, top_fraction: float) -> None:
    data = pd.read_csv(input_csv)
    required = {"MOF_Name", BENZENE, TOLUENE}
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    data["benzene_percentile"] = data[BENZENE].rank(method="average", pct=True)
    data["toluene_percentile"] = data[TOLUENE].rank(method="average", pct=True)
    p_b = data["benzene_percentile"].clip(lower=np.finfo(float).eps)
    p_t = data["toluene_percentile"].clip(lower=np.finfo(float).eps)
    data["balanced_score"] = 2.0 * p_b * p_t / (p_b + p_t)
    data["min_percentile"] = data[["benzene_percentile", "toluene_percentile"]].min(axis=1)

    ranked = data.sort_values("balanced_score", ascending=False, kind="mergesort").reset_index(drop=True)
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))
    top_n = int(np.ceil(len(ranked) * top_fraction))

    output_dir.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(output_dir / "all_mofs_ranked_by_balanced_score.csv", index=False, encoding="utf-8-sig")
    ranked.head(top_n).to_csv(
        output_dir / "top20_percent_by_balanced_score.csv",
        index=False,
        encoding="utf-8-sig",
    )
    metadata = {
        "input_file": str(input_csv),
        "n_candidates": int(len(ranked)),
        "top_fraction": float(top_fraction),
        "top_n": top_n,
        "formula": "2 * benzene_percentile * toluene_percentile / (benzene_percentile + toluene_percentile)",
        "percentile_method": "pandas rank(method='average', pct=True)",
    }
    (output_dir / "balanced_score_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank MOFs by the harmonic mean of adsorption percentiles.")
    parser.add_argument("--input", type=Path, required=True, help="Prediction CSV from screen_sine_library.py")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-fraction", type=float, default=0.20)
    args = parser.parse_args()
    rank_balanced_score(args.input, args.output_dir, args.top_fraction)


if __name__ == "__main__":
    main()
