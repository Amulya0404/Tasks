"""Task 3 - Prediction Model: diamond price regression (CodeNova AI/ML Batch 01).

Baseline (predict mean) vs Ridge vs Random Forest vs Gradient Boosting, with
5-fold CV, a leakage-safe sklearn Pipeline (scaling/encoding fit on train only),
and evaluation charts.  Target = log(price) so errors are relative.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

HERE = Path(__file__).parent
SEED = 42
# Ordinal categories (worst -> best); order is domain knowledge, NOT derived from the target.
CUT = ["Fair", "Good", "Very Good", "Premium", "Ideal"]
COLOR = ["J", "I", "H", "G", "F", "E", "D"]
CLARITY = ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]
CATS = ["cut", "color", "clarity"]


def load():
    df = pd.read_csv(HERE / "data" / "diamonds.csv").drop_duplicates()
    # x,y,z == 0 are impossible physical dimensions -> data errors; extreme y/z are typos
    df = df[(df[["x", "y", "z"]] > 0).all(axis=1) & (df.y < 20) & (df.z < 10)]
    return df.reset_index(drop=True)


def engineer(df):
    """Row-wise features only (no statistics from other rows => no leakage; target never used)."""
    d = df.copy()
    d["volume"] = d.x * d.y * d.z
    d["log_carat"] = np.log(d.carat)
    d["aspect_xy"] = d.x / d.y
    return d.drop(columns=["price"])


def make_pipe(model):
    num = ["carat", "depth", "table", "x", "y", "z", "volume", "log_carat", "aspect_xy"]
    pre = ColumnTransformer([
        ("cat", OrdinalEncoder(categories=[CUT, COLOR, CLARITY]), CATS),
        ("num", StandardScaler(), num),
    ])
    return Pipeline([("pre", pre), ("model", model)])


def rmse(y, p): return float(np.sqrt(mean_squared_error(y, p)))


def main():
    df = load()
    X, y = engineer(df), np.log(df.price)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED)
    print(f"rows={len(df)} train={len(X_tr)} test={len(X_te)}")

    models = {
        "Baseline (mean)": DummyRegressor(strategy="mean"),
        "Ridge": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(n_estimators=150, min_samples_leaf=2,
                                               n_jobs=-1, random_state=SEED),
        "Gradient Boosting": HistGradientBoostingRegressor(max_iter=400, learning_rate=0.08,
                                                           random_state=SEED),
    }
    cv = KFold(5, shuffle=True, random_state=SEED)
    rows, fitted = [], {}
    for name, m in models.items():
        pipe = make_pipe(m)
        s = cross_validate(pipe, X_tr, y_tr, cv=cv, n_jobs=-1,
                           scoring=("neg_root_mean_squared_error", "r2"))
        pipe.fit(X_tr, y_tr)
        p_log = pipe.predict(X_te)
        p_usd, y_usd = np.exp(p_log), np.exp(y_te)       # back to dollars
        rows.append({
            "model": name,
            "cv_rmse_log": round(-s["test_neg_root_mean_squared_error"].mean(), 4),
            "cv_r2": round(s["test_r2"].mean(), 4),
            "test_rmse_log": round(rmse(y_te, p_log), 4),
            "test_r2": round(r2_score(y_te, p_log), 4),
            "test_rmse_usd": round(rmse(y_usd, p_usd), 1),
            "test_mae_usd": round(mean_absolute_error(y_usd, p_usd), 1),
            "test_mape_pct": round(float(np.mean(np.abs(p_usd - y_usd) / y_usd) * 100), 2),
        })
        fitted[name] = (pipe, p_log)
        print(rows[-1])

    res = pd.DataFrame(rows).set_index("model")
    res.to_csv(HERE / "results.csv")
    best = res.drop("Baseline (mean)").test_rmse_log.idxmin()
    pipe, p_log = fitted[best]
    base_rmse = res.loc["Baseline (mean)", "test_rmse_usd"]
    gain = 1 - res.loc[best, "test_rmse_usd"] / base_rmse
    print(f"Best: {best}; RMSE ($) improvement over baseline: {gain:.1%}")

    # ---- charts ----
    resid = y_te.values - p_log
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    ax[0].scatter(np.exp(y_te), np.exp(p_log), s=4, alpha=.3)
    lim = [0, 19000]; ax[0].plot(lim, lim, "r--", label="perfect prediction")
    ax[0].set(xlim=lim, ylim=lim, xlabel="Actual price ($)", ylabel="Predicted price ($)",
              title=f"{best}: predicted vs actual (test)"); ax[0].legend()
    ax[1].scatter(p_log, resid, s=4, alpha=.3); ax[1].axhline(0, c="r", ls="--")
    ax[1].set(xlabel="Predicted log(price)", ylabel="Residual (log scale)",
              title="Residuals vs prediction")
    ax[2].hist(resid, bins=80, color="steelblue")
    ax[2].set(xlabel="Residual (log scale)", ylabel="Count",
              title=f"Residual distribution (std={resid.std():.3f})")
    fig.tight_layout(); fig.savefig(HERE / "residual_plots.png", dpi=200)

    pi = permutation_importance(pipe, X_te, y_te, n_repeats=5, random_state=SEED, n_jobs=-1,
                                scoring="neg_root_mean_squared_error")
    imp = pd.Series(pi.importances_mean, index=X_te.columns).sort_values()
    fig, ax = plt.subplots(figsize=(7, 5))
    imp.plot.barh(ax=ax, color="darkorange")
    ax.set(xlabel="Increase in test RMSE when feature is shuffled",
           title=f"Permutation feature importance ({best})")
    fig.tight_layout(); fig.savefig(HERE / "feature_importance.png", dpi=200)

    # error by price band -> limitation analysis
    bands = pd.cut(np.exp(y_te), [0, 1000, 2500, 5000, 10000, 20000],
                   labels=["<1k", "1-2.5k", "2.5-5k", "5-10k", ">10k"])
    err = pd.DataFrame({"band": bands.values,
                        "ape": np.abs(np.exp(p_log) - np.exp(y_te)) / np.exp(y_te) * 100,
                        "ae": np.abs(np.exp(p_log) - np.exp(y_te))})
    band_tbl = err.groupby("band", observed=True).agg(n=("ape", "size"),
                MAPE_pct=("ape", "mean"), MAE_usd=("ae", "mean")).round(2)
    band_tbl.to_csv(HERE / "error_by_price_band.csv")
    print(band_tbl)
    pd.DataFrame({"feature": imp.index[::-1], "importance": imp.values[::-1]}).to_csv(
        HERE / "feature_importance.csv", index=False)
    (HERE / "summary.json").write_text(json.dumps(
        {"best_model": best, "rmse_gain_vs_baseline": round(gain, 4),
         "n_rows": len(df)}, indent=2))


if __name__ == "__main__":
    main()
