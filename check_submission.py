"""Independently check the submission schema against the supplied test.csv."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent


def check_submission(submission_path: Path, test_path: Path) -> None:
    submission = pd.read_csv(submission_path)
    test = pd.read_csv(test_path, usecols=["cookie_id"])
    if submission.columns.tolist() != ["cookie_id", "score"]:
        raise ValueError("Submission must contain exactly cookie_id,score in that order")
    if len(submission) != len(test) or not submission.cookie_id.equals(test.cookie_id):
        raise ValueError("Submission cookie_id rows must match test.csv exactly")
    if submission.cookie_id.isna().any() or submission.cookie_id.duplicated().any():
        raise ValueError("Missing or duplicate cookie_id")
    if not pd.api.types.is_numeric_dtype(submission.score):
        raise ValueError("score must be numeric")
    if not np.isfinite(submission.score.to_numpy()).all() or not submission.score.between(0, 1).all():
        raise ValueError("score must be finite and between 0 and 1")
    print(f"OK: {len(submission)} rows; columns cookie_id,score; scores in [0, 1]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", type=Path, default=ROOT / "submission.csv")
    parser.add_argument("--test", type=Path, default=ROOT / "data/test.csv")
    args = parser.parse_args()
    check_submission(args.submission, args.test)
