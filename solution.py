"""Train a cookie-level bot detector and write submission.csv.

Python 3.12; run from the project directory: python solution.py
"""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from metric import precision_at_recall


ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
SEED = 2026
EVENT_TYPES = [
    "search_results_view", "item_view", "photo_swipe", "seller_page_view",
    "contact_phone_show", "contact_chat_open", "contact_message_sent",
    "favorite_add", "login", "captcha_shown",
]


def browser_family(ua: str) -> str:
    if "HeadlessChrome" in ua:
        return "headless"
    if "YaBrowser" in ua:
        return "yandex"
    if "Firefox" in ua:
        return "firefox"
    if "Chrome" in ua:
        return "chrome"
    if "Safari" in ua:
        return "safari"
    return "other"


def os_family(ua: str) -> str:
    if "Android" in ua:
        return "android"
    if "iPhone" in ua or "iPad" in ua:
        return "ios"
    if "Windows" in ua:
        return "windows"
    if "Macintosh" in ua:
        return "mac"
    if "Linux" in ua:
        return "linux"
    return "other"


def make_features(cookies: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    meta = cookies.set_index("cookie_id")
    assert meta.index.is_unique
    meta = meta.copy()
    for col in ("cookie_created_at", "window_start_ts", "window_end_ts"):
        meta[col] = pd.to_datetime(meta[col])
    e = events.join(meta[["window_start_ts", "window_end_ts"]], on="cookie_id", how="inner")
    # Only observations known by the end of the cookie's specified day are legal.
    e["event_ts"] = pd.to_datetime(e["event_ts"])
    e = e.loc[(e.event_ts >= e.window_start_ts) & (e.event_ts < e.window_end_ts)].copy()
    e["platform_norm"] = e.platform.str.lower().replace({"web": "desktop", "iphone": "ios"})
    e["browser"] = e.user_agent.map(browser_family)
    e["os"] = e.user_agent.map(os_family)
    e["hour"] = e.event_ts.dt.hour
    e["minute"] = e.event_ts.dt.floor("min")
    e["second_of_day"] = (e.event_ts - e.window_start_ts).dt.total_seconds()
    e["query_length"] = e.search_query.str.len()
    e["query_words"] = e.search_query.str.split().str.len()
    e["item_id"] = e.item_id.astype("Int64")
    e = e.sort_values(["cookie_id", "event_ts"], kind="stable")
    g = e.groupby("cookie_id", sort=False)
    f = pd.DataFrame(index=meta.index)
    f["cookie_age_days"] = (meta.window_start_ts - meta.cookie_created_at).dt.total_seconds() / 86400
    f["created_hour"] = meta.cookie_created_at.dt.hour
    f["created_weekday"] = meta.cookie_created_at.dt.dayofweek
    f["n_events"] = g.size()
    for col in ("event_name", "platform_norm", "browser", "os", "user_agent", "item_id",
                "item_category", "item_location", "seller_type", "search_query", "search_page", "hour", "minute"):
        f[f"unique_{col}"] = g[col].nunique()
    for col in ("event_name", "platform_norm", "browser", "os", "seller_type"):
        counts = pd.crosstab(e.cookie_id, e[col]).add_prefix(f"count_{col}_")
        f = f.join(counts)
    for name in EVENT_TYPES:
        col = f"count_event_name_{name}"
        if col not in f:
            f[col] = 0
        f[f"share_{name}"] = f[col] / f.n_events
    for col in ("item_id", "item_category", "item_location", "search_query", "user_agent"):
        f[f"top_share_{col}"] = g[col].agg(lambda s: s.value_counts(normalize=True).iloc[0] if s.notna().any() else np.nan)
    for col in ("item_id", "item_category", "item_location", "search_query", "seller_type", "pointer_x"):
        f[f"fraction_filled_{col}"] = g[col].count() / f.n_events
    for col in ("search_page", "query_length", "query_words", "pointer_x", "pointer_y", "second_of_day"):
        stats = g[col].agg(["mean", "std", "min", "max", "median"])
        stats.columns = [f"{col}_{s}" for s in stats.columns]
        f = f.join(stats)
    f["span_seconds"] = f.second_of_day_max - f.second_of_day_min
    f["events_per_active_minute"] = f.n_events / f.unique_minute.clip(lower=1)
    f["events_per_span_hour"] = f.n_events / (f.span_seconds / 3600 + 1)
    f["items_per_view"] = f.unique_item_id / f.count_event_name_item_view.clip(lower=1)
    f["queries_per_search"] = f.unique_search_query / f.count_event_name_search_results_view.clip(lower=1)
    f["contact_per_item"] = (f.count_event_name_contact_phone_show + f.count_event_name_contact_chat_open) / f.count_event_name_item_view.clip(lower=1)
    f["photo_per_item"] = f.count_event_name_photo_swipe / f.count_event_name_item_view.clip(lower=1)
    f["captcha_per_event"] = f.count_event_name_captcha_shown / f.n_events
    f["late_night_share"] = e.hour.between(0, 5).groupby(e.cookie_id).mean()
    f["business_hours_share"] = e.hour.between(9, 18).groupby(e.cookie_id).mean()
    f["weekend"] = meta.window_start_ts.dt.dayofweek.isin([5, 6]).astype(int)

    # Gaps are computed only between consecutive in-window events for the same cookie.
    e["gap"] = g.event_ts.diff().dt.total_seconds()
    gaps = e.groupby("cookie_id").gap.agg(["mean", "std", "min", "max", "median"])
    gaps.columns = [f"gap_{s}" for s in gaps.columns]
    f = f.join(gaps)
    for threshold in (0, 1, 2, 5, 10, 30, 60, 300):
        f[f"gap_le_{threshold}"] = e.gap.le(threshold).groupby(e.cookie_id).sum() / (f.n_events - 1).clip(lower=1)
    f["gap_cv"] = f.gap_std / f.gap_mean.replace(0, np.nan)
    f["duplicate_event_share"] = e.duplicated(subset=["cookie_id", "event_ts", "eid", "item_id", "search_query"]).groupby(e.cookie_id).mean()

    # Repeated search pages and repeated item views reveal crawl patterns.
    e["prev_event"] = g.event_name.shift()
    e["prev_item"] = g.item_id.shift()
    e["prev_page"] = g.search_page.shift()
    for left, right in (("search_results_view", "item_view"), ("item_view", "item_view"),
                        ("item_view", "photo_swipe"), ("item_view", "contact_phone_show"),
                        ("search_results_view", "search_results_view")):
        f[f"transition_{left}_to_{right}"] = ((e.prev_event == left) & (e.event_name == right)).groupby(e.cookie_id).sum()
    f["same_item_in_row"] = ((e.item_id == e.prev_item) & e.item_id.notna()).fillna(False).groupby(e.cookie_id).sum()
    f["page_step_one"] = ((e.search_page - e.prev_page == 1) & (e.event_name == "search_results_view")).groupby(e.cookie_id).sum()
    f["deep_search_share"] = e.search_page.ge(5).groupby(e.cookie_id).sum() / f.count_event_name_search_results_view.clip(lower=1)
    f["pointer_unique_x"] = g.pointer_x.nunique()
    f["pointer_unique_y"] = g.pointer_y.nunique()

    # Stable, low-cardinality categories; no cookie identifier or absolute date is a feature.
    for col in ("browser", "os", "platform_norm"):
        f[f"mode_{col}"] = g[col].agg(lambda s: s.mode().iloc[0] if len(s) else "missing")
    f = f.replace([np.inf, -np.inf], np.nan)
    return f


def score_report(label: str, y: np.ndarray, p: np.ndarray) -> None:
    official = precision_at_recall(y, p)
    precision, recall, _ = precision_recall_curve(y, p)
    assert np.isclose(official, precision[recall >= 0.70].max())
    print(f"{label}: P@R70={official:.5f} "
          f"PR-AUC={average_precision_score(y, p):.5f} ROC-AUC={roc_auc_score(y, p):.5f}")


def report_operating_point(y: np.ndarray, p: np.ndarray, event_counts: np.ndarray) -> None:
    """Show the best local threshold, only for interpretation; submission has scores."""
    order = np.argsort(-p, kind="mergesort")
    sorted_p, sorted_y = p[order], y[order]
    ends = np.r_[sorted_p[1:] != sorted_p[:-1], True]
    tp = np.cumsum(sorted_y)[ends]
    selected_n = np.arange(1, len(y) + 1)[ends]
    precision = tp / selected_n
    recall = tp / y.sum()
    eligible = np.flatnonzero(recall >= 0.70)
    best = eligible[np.argmax(precision[eligible])]
    threshold = sorted_p[ends][best]
    selected = p >= threshold
    bot_events = event_counts[y == 1].sum()
    human_events = event_counts[y == 0].sum()
    removed_bot_events = event_counts[(y == 1) & selected].sum()
    removed_human_events = event_counts[(y == 0) & selected].sum()
    print(f"Validation operating point: threshold={threshold:.6f}, TP={tp[best]}, "
          f"FP={selected_n[best] - tp[best]}, recall={recall[best]:.5f}")
    print(f"Within-window events removed: bot={removed_bot_events}/{bot_events}, "
          f"human={removed_human_events}/{human_events}")


def main() -> None:
    train = pd.read_csv(ROOT / "data/train.csv")
    test = pd.read_csv(ROOT / "data/test.csv")
    events = pd.read_csv(ROOT / "data/events.csv.gz")
    cookies = pd.concat([train.drop(columns="target"), test], ignore_index=True)
    x = make_features(cookies, events)
    assert x.index.is_unique and not x.index.hasnans
    train_x = x.loc[train.cookie_id].copy()
    test_x = x.loc[test.cookie_id].copy()
    cat_cols = train_x.select_dtypes(include=["object", "str"]).columns.tolist()
    for col in cat_cols:
        categories = pd.Index(x[col].dropna().unique())
        train_x[col] = pd.Categorical(train_x[col], categories=categories)
        test_x[col] = pd.Categorical(test_x[col], categories=categories)
    y = train.target.to_numpy()
    valid = train.window_start_ts >= "2026-04-16"
    fit = ~valid
    fit_ids = train.loc[fit, "cookie_id"]
    valid_ids = train.loc[valid, "cookie_id"]
    print(f"Features: {train_x.shape[1]}; train={fit.sum()}, validation={valid.sum()}, positives={y[valid].sum()}")

    # Transparent baseline: event counts, age and volume with logistic regression.
    basic = ["cookie_age_days", "n_events", "unique_item_id", "unique_search_query"] + [f"count_event_name_{name}" for name in EVENT_TYPES]
    baseline = make_pipeline(SimpleImputer(strategy="constant", fill_value=0), StandardScaler(), LogisticRegression(max_iter=1000, random_state=SEED))
    baseline.fit(train_x.loc[fit_ids, basic], y[fit])
    score_report("Baseline", y[valid], baseline.predict_proba(train_x.loc[valid_ids, basic])[:, 1])

    params = dict(n_estimators=500, learning_rate=0.035, num_leaves=15, max_depth=-1,
                  min_child_samples=45, colsample_bytree=0.85, subsample=0.85,
                  subsample_freq=1, reg_lambda=5.0, verbosity=-1, random_state=SEED,
                  n_jobs=4)
    model = lgb.LGBMClassifier(**params)
    model.fit(train_x.loc[fit_ids], y[fit], categorical_feature=cat_cols,
              eval_X=train_x.loc[valid_ids], eval_y=y[valid], eval_metric="average_precision",
              callbacks=[lgb.early_stopping(50, verbose=False)])
    best_rounds = model.best_iteration_
    primary_score = model.predict_proba(train_x.loc[valid_ids])[:, 1]
    score_report(f"LightGBM ({best_rounds} rounds)", y[valid], primary_score)
    # Two nearby model capacities diversify errors; both are checked on rolling folds.
    larger_params = {**params, "n_estimators": 220, "num_leaves": 31, "min_child_samples": 30}
    larger = lgb.LGBMClassifier(**larger_params)
    larger.fit(train_x.loc[fit_ids], y[fit], categorical_feature=cat_cols)
    larger_score = larger.predict_proba(train_x.loc[valid_ids])[:, 1]
    ensemble_score = (primary_score + larger_score) / 2
    score_report("Two-model mean", y[valid], ensemble_score)
    report_operating_point(y[valid], ensemble_score,
                           train_x.loc[valid_ids, "n_events"].fillna(0).to_numpy())
    for train_end, valid_end in (("2026-04-10", "2026-04-13"), ("2026-04-12", "2026-04-16")):
        earlier_fit = train.window_start_ts < train_end
        earlier_valid = (train.window_start_ts >= train_end) & (train.window_start_ts < valid_end)
        earlier_fit_x = train_x.loc[train.loc[earlier_fit, "cookie_id"]]
        earlier_valid_x = train_x.loc[train.loc[earlier_valid, "cookie_id"]]
        earlier_model = lgb.LGBMClassifier(**{**params, "n_estimators": best_rounds})
        earlier_model.fit(earlier_fit_x, y[earlier_fit], categorical_feature=cat_cols)
        earlier_larger = lgb.LGBMClassifier(**larger_params)
        earlier_larger.fit(earlier_fit_x, y[earlier_fit], categorical_feature=cat_cols)
        earlier_score = (earlier_model.predict_proba(earlier_valid_x)[:, 1]
                         + earlier_larger.predict_proba(earlier_valid_x)[:, 1]) / 2
        score_report(f"Temporal fold {train_end} to {valid_end}, two-model mean", y[earlier_valid], earlier_score)
    importance = pd.Series(model.feature_importances_, index=train_x.columns).sort_values(ascending=False)
    print("Top features:", importance.head(20).to_dict())

    final = lgb.LGBMClassifier(**{**params, "n_estimators": best_rounds})
    final.fit(train_x, y, categorical_feature=cat_cols)
    final_larger = lgb.LGBMClassifier(**larger_params)
    final_larger.fit(train_x, y, categorical_feature=cat_cols)
    ARTIFACTS.mkdir(exist_ok=True)
    model_files = ["model_15_leaves.txt", "model_31_leaves.txt"]
    final.booster_.save_model(str(ARTIFACTS / model_files[0]))
    final_larger.booster_.save_model(str(ARTIFACTS / model_files[1]))
    schema = {
        "model_type": "mean_of_two_lightgbm_classifiers",
        "model_files": model_files,
        "weights": [0.5, 0.5],
        "feature_columns": train_x.columns.tolist(),
        "categorical_levels": {col: train_x[col].cat.categories.tolist() for col in cat_cols},
        "random_state": SEED,
        "n_estimators": [best_rounds, larger_params["n_estimators"]],
    }
    (ARTIFACTS / "feature_schema.json").write_text(
        json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    scores = (final.predict_proba(test_x)[:, 1] + final_larger.predict_proba(test_x)[:, 1]) / 2
    saved_scores = sum(
        weight * lgb.Booster(model_file=str(ARTIFACTS / filename)).predict(test_x)
        for weight, filename in zip(schema["weights"], model_files)
    )
    assert np.allclose(scores, saved_scores, rtol=0, atol=1e-12)
    submission = pd.DataFrame({"cookie_id": test.cookie_id, "score": scores})
    assert submission.cookie_id.equals(test.cookie_id)
    assert submission.cookie_id.is_unique and submission.score.between(0, 1).all()
    assert submission.notna().all().all()
    submission.to_csv(ROOT / "submission.csv", index=False)
    print(f"Saved {ROOT / 'submission.csv'}: {len(submission)} rows")
    print(f"Saved fitted ensemble and feature schema to {ARTIFACTS}")


if __name__ == "__main__":
    main()
