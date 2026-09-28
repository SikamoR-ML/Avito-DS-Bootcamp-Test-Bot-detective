"""Predict cookie scores with the saved LightGBM ensemble, without retraining."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from solution import ROOT, make_features


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
    scores = sum(
        weight * lgb.Booster(model_file=str(artifacts_dir / filename)).predict(features)
        for weight, filename in zip(schema["weights"], schema["model_files"])
    )
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
