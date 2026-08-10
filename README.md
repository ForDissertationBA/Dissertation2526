# Dissertation2526: A Systematic Evaluation of Machine Learning Models for Equity Prediction: Ranking Performance, Temporal Stability, and Generalisability

This repository contains the notebooks used for the empirical analysis of machine-learning benchmarks for equity prediction, with emphasis on predictive level performance, within-ticker timing, cross-sectional ranking, temporal stability, and generalisability across equity universes.

## Repository contents

- `EDA_for_SP100.ipynb`  
  Exploratory and data-quality analysis for the final feature set. Feature diagnostics are based on the training data; validation/test target distributions are used only for post-hoc descriptive stability checks.

- `Integrated_Experiment_Runner_SP100_main_experiment.ipynb`  
  Main S&P 100-derived experiment. Includes rolling two-year and expanding-window experiments across four targets: RL current PnL, RL long-action quality, RL long action, and perfect-hindsight positive long entry. Additional target variables retained in the notebook are from earlier experiments and are not used in the final reported analysis.

- `Integrated_Experiment_Runner_SP100_fixed_temporal_split.ipynb`  
  Fixed chronological train/validation/test evaluation for the current-PnL target. This notebook is used as a diagnostic fixed-split comparison rather than the main walk-forward evaluation.

- `Integrated_Experiment_Runner_SP100_signedlog_PnL.ipynb`  
  Robustness check using a signed-log-transformed current-PnL target with a rolling two-year training window.

- `Integrated_Experiment_Runner_rolling_1_2_3yr_time_sensitivity.ipynb`  
  Training-window sensitivity analysis using rolling one-, two-, and three-year windows. All schemes use the same frozen feature specification and model set (268 features); only the training-window length varies. Like-for-like comparisons use the common complete-year overlap (2023-2025).

- `Integrated_Experiment_Runner_NASDAQ.ipynb`  
  Generalisability check on the Nasdaq 100-derived universe using the frozen feature specification (268 features) and the current-PnL target.


## Environment setup

Python 3.10+ is recommended.

Create and activate a virtual environment, then install the required packages:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Then launch Jupyter:

```bash
jupyter lab
```

## Data and configuration

The datasets are not included in this repository for the privacy concerns. The notebooks expect local train/validation/test data files and the final feature inventory.

Before running a notebook, update the paths in its dataset/configuration section, including:

- train / validation / test data paths;
- output directory;
- `final_feature_B.txt` or the equivalent final feature-list path.

The loaders support the formats implemented in the notebooks, including JSONL, CSV and Parquet where applicable.

## Final model specifications

The shared modelling utilities retain several additional candidate models for compatibility with earlier experiments. The final reported experiments are restricted through `EXPERIMENT_CONFIG`.

For continuous targets, the reported benchmark set is:

- Dummy mean baseline;
- Elastic Net;
- Random Forest;
- LightGBM.

For binary targets, the reported benchmark set is:

- Majority-class dummy baseline;
- Logistic Regression;
- Random Forest;
- LightGBM.

Definitions for Ridge, Lasso, histogram gradient boosting, stratified dummy models and multiclass models may remain in shared utility cells, but they are not part of the final reported experiment 

## Confidence intervals

The final dissertation confidence intervals for within-ticker Spearman use a **21-trading-day, fold-stratified moving-block bootstrap**. Contiguous date blocks are resampled within each out-of-sample fold and do not cross retraining boundaries.

Some notebooks retain earlier or alternative inference utilities, including other bootstrap constructions and Newey-West calculations, for diagnostic comparison and sensitivity checks. These are not the confidence intervals used for the final reported within-ticker results unless explicitly stated.

## Reproducibility notes

- Random seeds are fixed in the experiment notebooks.
- Preprocessing is fitted within the training portion of each split/fold.
- Walk-forward training uses only observations preceding the corresponding test year.
- The final modelling experiments use the frozen feature specification (268 features) rather than re-selecting predictors on the test universe for consistent comparisons.
- Partial 2026 results are excluded from complete-year aggregate comparisons where indicated in the notebooks.

## Outputs

The experiment notebooks register result tables during execution and export consolidated Excel workbooks and, where applicable, row-level prediction files and figures to the configured output directories.


