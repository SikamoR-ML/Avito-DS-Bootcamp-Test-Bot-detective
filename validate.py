"""Reproduce forward validation for the baseline and submitted blend."""
from __future__ import annotations
import catboost
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from extra_features import make_ua_query_features
from metric import precision_at_recall
from solution import ROOT, SEED, as_catboost_features, make_features

FOLDS = [
    ("2026-04-10", "2026-04-13", "3 days"),
    ("2026-04-12", "2026-04-16", "4 days"),
    ("2026-04-13", "2026-04-20", "7 days"),
    ("2026-04-16", "2026-04-20", "recent 4 days"),
]

def metrics(y: np.ndarray, score: np.ndarray) -> dict:
    order = np.argsort(-score, kind="mergesort")
    yy, ss = y[order], score[order]
    ends = np.r_[ss[1:] != ss[:-1], True]
    tp = np.cumsum(yy)[ends]
    fp = np.cumsum(1 - yy)[ends]
    allowed = np.flatnonzero(tp >= .70 * y.sum())
    best = allowed[np.argmax(tp[allowed] / (tp[allowed] + fp[allowed]))]
    p70 = tp[best] / (tp[best] + fp[best])
    assert np.isclose(p70, precision_at_recall(y, score))
    return {"p70": round(float(p70), 5), "ap": round(average_precision_score(y, score), 5),
            "auc": round(roc_auc_score(y, score), 5), "tp": int(tp[best]), "fp": int(fp[best])}

def ensemble(x: pd.DataFrame, y: pd.Series, fit: pd.Index, val: pd.Index) -> np.ndarray:
    cats = x.select_dtypes(include=["object", "str"]).columns.tolist()
    x = x.copy()
    for col in cats:
        x[col] = x[col].astype("category")
    predictions = []
    for leaves, rounds, child in ((15, 274, 45), (31, 220, 30)):
        model = lgb.LGBMClassifier(n_estimators=rounds, learning_rate=.035, num_leaves=leaves,
            min_child_samples=child, colsample_bytree=.85, subsample=.85, subsample_freq=1,
            reg_lambda=5., verbosity=-1, random_state=SEED, n_jobs=4)
        model.fit(x.loc[fit], y.loc[fit], categorical_feature=cats)
        predictions.append(model.predict_proba(x.loc[val])[:, 1])
    xc = as_catboost_features(x, cats)
    model = catboost.CatBoostClassifier(iterations=650, depth=5, learning_rate=.04,
        l2_leaf_reg=5, loss_function="Logloss", verbose=False, thread_count=4,
        random_seed=SEED, allow_writing_files=False, cat_features=cats)
    model.fit(xc.loc[fit], y.loc[fit])
    predictions.append(model.predict_proba(xc.loc[val])[:, 1])
    return .35 * predictions[0] + .35 * predictions[1] + .30 * predictions[2]

def main() -> None:
    train = pd.read_csv(ROOT / "data/train.csv")
    test = pd.read_csv(ROOT / "data/test.csv")
    events = pd.read_csv(ROOT / "data/events.csv.gz")
    cookies = pd.concat([train.drop(columns="target"), test], ignore_index=True)
    base = make_features(cookies, events)
    plus = base.join(make_ua_query_features(cookies, events))
    y = train.set_index("cookie_id").target
    dates = train.set_index("cookie_id").window_start_ts
    rows = []
    for start, end, label in FOLDS:
        fit = dates.index[dates < start]
        val = dates.index[(dates >= start) & (dates < end)]
        base_score = ensemble(base, y, fit, val)
        plus_score = ensemble(plus, y, fit, val)
        actual = y.loc[val].to_numpy()
        for name, score in (("baseline", base_score), ("submitted", .4 * base_score + .6 * plus_score)):
            row = {"fold": label, "model": name, "train": len(fit), "validation": len(val),
                   "positives": int(actual.sum()), **metrics(actual, score)}
            rows.append(row)
            print(row, flush=True)
    result = pd.DataFrame(rows)
    print(result.groupby("model").p70.agg(["mean", "std", "min"]).to_string(), flush=True)

if __name__ == "__main__":
    main()
