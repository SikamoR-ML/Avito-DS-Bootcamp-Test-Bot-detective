"""Predict cookie scores with the saved tree ensemble, without retraining."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import catboost
import lightgbm as lgb
import numpy as np
import pandas as pd

from solution import ROOT, as_catboost_features, make_features


def predict(test_path: Path, events_path: Path, artifacts_dir: Path, output_path: Path) -> None:
    schema = json.loads((artifacts_dir / "feature_schema.json").read_text(encoding="utf-8"))
    test = pd.read_csv(test_path)
    if test.cookie_id.isna().any() or test.cookie_id.duplicated().any():
        raise ValueError("test.csv has missing or duplicate cookie_id")
    events = pd.read_csv(events_path)
    features = make_features(test, events).loc[test.cookie_id]
    missing = set(schema["feature_columns"]) - set(features.columns)
    for col in missing:
        features[col] = 0 if col.startswith("count_") else np.nan
    features = features.reindex(columns=schema["feature_columns"])
    for col, levels in schema["categorical_levels"].items():
        features[col] = pd.Categorical(features[col], categories=levels)
    cat_features = as_catboost_features(features, list(schema["categorical_levels"]))
    scores = np.zeros(len(test), dtype=float)
    for family, weight, filename in zip(
        schema["model_families"], schema["weights"], schema["model_files"]
    ):
        model_path = artifacts_dir / filename
        if family == "lightgbm":
            scores += weight * lgb.Booster(model_file=str(model_path)).predict(features)
        elif family == "catboost":
            model = catboost.CatBoostClassifier()
            model.load_model(str(model_path))
            scores += weight * model.predict_proba(cat_features)[:, 1]
        else:
            raise ValueError(f"Unknown model family: {family}")
    if not np.isfinite(scores).all() or not np.all((0 <= scores) & (scores <= 1)):
        raise ValueError("Predictions must be finite and within [0, 1]")
    pd.DataFrame({"cookie_id": test.cookie_id, "score": scores}).to_csv(output_path, index=False)
    print(f"Saved {len(test)} predictions to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", type=Path, default=ROOT / "data/test.csv")
    parser.add_argument("--events", type=Path, default=ROOT / "data/events.csv.gz")
    parser.add_argument("--artifacts", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--output", type=Path, default=ROOT / "submission.csv")
    args = parser.parse_args()
    predict(args.test, args.events, args.artifacts, args.output)
