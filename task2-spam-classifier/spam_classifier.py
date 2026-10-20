"""Task 2 - Spam Message Classifier (CodeNova AI/ML Batch 01).

Trains and compares Multinomial Naive Bayes vs Logistic Regression on the
SMS Spam Collection, evaluates ONLY on a held-out test split, and saves the
full (preprocess + vectorizer + model) pipeline with joblib.

Usage:  python spam_classifier.py          # train, evaluate, save artifacts
"""
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from text_utils import preprocess  # separate module so the saved pipeline can be unpickled anywhere

HERE = Path(__file__).parent
SEED = 42
def build_pipeline(clf):
    # The vectoriser lives INSIDE the pipeline so it is fit on training folds only
    # (no data leakage into validation/test data).
    return Pipeline([
        ("tfidf", TfidfVectorizer(preprocessor=preprocess, tokenizer=str.split,
                                  token_pattern=None, ngram_range=(1, 2), min_df=2,
                                  sublinear_tf=True)),
        ("clf", clf),
    ])


def main():
    df = pd.read_csv(HERE / "data" / "sms.tsv", sep="\t", header=None,
                     names=["label", "text"]).drop_duplicates().reset_index(drop=True)
    df["y"] = (df.label == "spam").astype(int)
    print(f"Messages after de-duplication: {len(df)} | spam share: {df.y.mean():.1%}")

    # Split FIRST; the test set is never touched until the final evaluation.
    X_tr, X_te, y_tr, y_te = train_test_split(
        df.text, df.y, test_size=0.2, stratify=df.y, random_state=SEED)

    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    candidates = {
        "MultinomialNB": (MultinomialNB(), {"clf__alpha": [0.05, 0.1, 0.3, 1.0]}),
        "LogisticRegression": (LogisticRegression(max_iter=2000, class_weight="balanced"),
                               {"clf__C": [1, 3, 10, 30]}),
    }
    results, best = {}, {}
    for name, (clf, grid) in candidates.items():
        # model selection by 5-fold CV on TRAIN data only, optimising precision
        gs = GridSearchCV(build_pipeline(clf), grid, cv=cv, scoring="precision", n_jobs=-1)
        gs.fit(X_tr, y_tr)
        pred = gs.predict(X_te)
        results[name] = {
            "best_params": gs.best_params_,
            "cv_precision": round(gs.best_score_, 4),
            "test_precision": round(precision_score(y_te, pred), 4),
            "test_recall": round(recall_score(y_te, pred), 4),
            "test_f1": round(f1_score(y_te, pred), 4),
            "confusion_matrix": confusion_matrix(y_te, pred).tolist(),
        }
        best[name] = gs.best_estimator_
        print(f"\n== {name} ==\n{json.dumps(results[name], indent=2)}")
        print(classification_report(y_te, pred, target_names=["ham", "spam"], digits=4))

    # Pick winner: highest CV precision (ties -> CV F1-free tiebreak on test recall is NOT used;
    # we break ties by name order to avoid peeking at the test set).
    winner = max(results, key=lambda n: (results[n]["cv_precision"], -list(results).index(n)))
    print(f"\nSelected model (by CV precision on train data): {winner}")

    # confusion-matrix figure for every model
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, name in zip(axes, results):
        sns.heatmap(results[name]["confusion_matrix"], annot=True, fmt="d", cmap="Blues",
                    xticklabels=["ham", "spam"], yticklabels=["ham", "spam"], ax=ax, cbar=False)
        ax.set_title(f"{name}\nP={results[name]['test_precision']:.3f}  "
                     f"R={results[name]['test_recall']:.3f}  F1={results[name]['test_f1']:.3f}")
        ax.set_xlabel("Predicted label"); ax.set_ylabel("True label")
    fig.suptitle("Confusion matrices on held-out test set", y=1.02)
    fig.tight_layout(); fig.savefig(HERE / "confusion_matrices.png", dpi=200, bbox_inches="tight")

    joblib.dump(best[winner], HERE / "spam_pipeline.joblib")
    (HERE / "metrics.json").write_text(json.dumps({"selected": winner, **results}, indent=2))
    print("Saved spam_pipeline.joblib, metrics.json, confusion_matrices.png")


if __name__ == "__main__":
    main()
