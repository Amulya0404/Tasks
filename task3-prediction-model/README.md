# Task 3 — Prediction Model: Diamond Price Regression

CodeNova AI/ML Internship · Batch 01

Predicts diamond price from carat, cut, colour, clarity and dimensions.
Dataset: `diamonds` from [`seaborn-data`](https://github.com/mwaskom/seaborn-data) (53,940 rows → 53,772 usable).

## Methodology
- **Cleaning:** removed 146 exact duplicates and 22 rows with impossible dimensions (x/y/z = 0, y ≥ 20, z ≥ 10) — physical data errors.
- **Features:** ordinal encoding of cut/colour/clarity using domain order (not derived from the target), standard scaling of numerics, engineered `volume`, `log_carat`, `aspect_xy` (row-wise only → no target leakage).
- **Target:** `log(price)`, because errors are naturally relative; metrics are also reported back in dollars.
- **Validation:** 80/20 split; 5-fold CV on the training set; all preprocessing inside a `Pipeline` so scalers/encoders are fit on training folds only.
- **Models:** baseline (predict the mean), Ridge, Random Forest, HistGradientBoosting.

## Results
| Model | CV R² | Test R² | Test RMSE ($) | Test MAE ($) | Test MAPE |
|---|---|---|---|---|---|
| Baseline (mean) | −0.000 | 0.000 | 4,214 | 2,772 | 110.1% |
| Ridge | 0.982 | 0.981 | 971 | 437 | 10.9% |
| Random Forest | 0.993 | 0.992 | 523 | 261 | 6.1% |
| **Gradient Boosting** | **0.993** | **0.992** | **521** | 263 | 6.1% |

Gradient Boosting reduces dollar RMSE by **87.6%** versus the baseline. Random Forest is statistically indistinguishable (RMSE within 0.5%).

Charts: `residual_plots.png` (predicted vs actual, residuals, residual histogram) · `feature_importance.png` (permutation importance).

## Feature importance (permutation, test set)
Top: `y` (0.51), `carat` (0.38), `clarity` (0.22), `volume` (0.20), `color` (0.13). Size features (`carat`, `x`, `y`, `z`, `volume`) are highly correlated, so permutation importance splits credit among them — read it as "size matters most, then clarity, then colour", not as exact per-feature effects. `cut` contributes very little (0.008).

## Known limitations
- **Error by price band** (`error_by_price_band.csv`): relative error is lowest for cheap stones (MAPE 4.6% under $1k) and ~6–7% elsewhere; absolute error grows with price (MAE ≈ $938 above $10k).
- **Underprediction at the top end:** the predicted-vs-actual chart shows the largest stones being predicted below their true price (tree models cannot extrapolate beyond the training range).
- **A few large outliers** (log-residuals up to ≈ 1.4, i.e. predictions off by ~4×) probably reflect data-entry errors or unusual stones.
- Prices are from a single historical snapshot (US dollars) and ignore factors such as certification, fluorescence or market shifts; do not use for real valuation.
- Random split: near-identical stones can appear in both train and test, so performance on genuinely new kinds of stones may be slightly worse.

## Usage
```bash
pip install -r ../requirements.txt
python diamond_price_model.py      # prints metrics, writes charts + CSVs
```
