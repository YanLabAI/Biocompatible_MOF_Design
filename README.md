# Biocompatible MOF design for benzene-series adsorption

Data, trained models, analysis code and source results for the manuscript on machine-learning-assisted screening of environmentally compatible MOFs for benzene and toluene adsorption.

The interactive MOFScreen-Agent platform is available at <https://web-phi-peach-77.vercel.app/>.

## Repository contents

- `Code/adsorption_ml/`: six-model benchmarking, robustness analyses, learning curves, full-library prediction and balanced-score ranking.
- `Code/figures/`: Python scripts for the robustness panels and hierarchical screening workflow.
- `Data/adsorption/`: processed descriptor datasets used in the current analyses.
- `Data/*.csv`: aquatic-toxicity and PMT/vPvM datasets retained from the original repository.
- `Models/sine_matrix/`: fitted optimal Sine Coulomb matrix models used for virtual screening.
- `Results/model_benchmark/`: the complete six-algorithm benchmark and task-wise best models.
- `Results/robustness_current1000/`: per-seed, per-fold and leave-one-metal-out numerical results.
- `Results/learning_curve/`: source values for the 1,000-sample learning-curve analysis.
- `Results/screening/`: all 162,985 ranked candidates and the top 20% selected by balanced score.
- `Results/economic_screening/`: CoPriNet linker-cost ranking and the 150 candidates retained before expert review.
- `Figures/`: publication figures and machine-readable source data.

## Current adsorption datasets

| Descriptor | Benzene | Toluene |
|---|---:|---:|
| Sine Coulomb matrix | 1,000 | 1,000 |
| Stoichiometric-120 | 1,000 | 1,000 |
| Orbital field matrix | 1,000 | 1,000 |
| Structural features | 679 | 1,000 |

The six benchmarked regressors are Random Forest, Gradient Boosting Regressor, XGBoost, Histogram-based Gradient Boosting Regressor, Extra Trees and Bagging Regressor. The Sine Coulomb matrix screening models are Gradient Boosting for benzene and XGBoost for toluene.

## Reproduction

Create a Python 3.11 environment and install the recorded dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the six-model benchmark:

```bash
python Code/adsorption_ml/benchmark_six_models.py
```

Run the three robustness analyses and the learning curve:

```bash
python Code/adsorption_ml/robustness_current1000.py
python Code/adsorption_ml/learning_curve_current1000.py
python Code/figures/plot_robustness_figures.py
```

Screen a precomputed Sine Coulomb matrix library and calculate the balanced score:

```bash
python Code/adsorption_ml/screen_sine_library.py --input PATH_TO_SINE_MATRIX_LIBRARY.csv
python Code/adsorption_ml/rank_balanced_score.py \
  --input Results/screening/rerun/all_mofs_sine_best_models_predictions.csv \
  --output-dir Results/screening/rerun
```

For candidate *i*, the balanced score is the harmonic mean of the benzene and toluene adsorption percentiles:

```text
balanced_score_i = 2 P_benzene,i P_toluene,i / (P_benzene,i + P_toluene,i)
```

The complete precomputed screening descriptor matrix is approximately 909 MB and is not stored as a single GitHub file. The repository instead provides the trained models, screening code, complete ranked predictions and top-20% output needed to audit the reported screening result.

## Reproducibility settings

- Hold-out split: 80/20, `random_state=42` for the primary benchmark.
- Cross-validation: shuffled five-fold CV, `random_state=42`.
- Random-split robustness: ten independent 80/20 splits with seeds 0-9.
- Grouped CV and OOD grouping variable: dominant metal from Stoichiometric-120 fractions.
- Leave-one-metal-out plots: groups with `n_test >= 20`; CSV files retain all evaluated groups.
