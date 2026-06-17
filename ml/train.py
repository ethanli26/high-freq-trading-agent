"""Train and evaluate the signal-quality model WITHOUT cheating on time.

Leakage guards (each is enforced in code below):
  * Chronological split — train on the earlier ~70% of signals by date, test on the
    most recent ~30%. No random shuffling across time.
  * Standardization stats come from the TRAIN set only (fit on train, apply to test).
  * Assertion that every train date strictly precedes every test date.

Logistic regression only; no hyperparameter tuning in this pass. Read-only research.
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow running this file directly from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler  # noqa: E402

from ml.dataset import DATASET_PATH, FEATURE_COLUMNS, build_dataset  # noqa: E402

log = logging.getLogger(__name__)

TRAIN_FRAC = 0.70


def load_dataset() -> pd.DataFrame:
    """Load the cached dataset, or build it if missing."""
    if DATASET_PATH.exists():
        return pd.read_parquet(DATASET_PATH).sort_values("date").reset_index(drop=True)
    log.info("Dataset cache not found; building it now...")
    return build_dataset()


def temporal_split(dataset: pd.DataFrame, train_frac: float = TRAIN_FRAC):
    """Split by DATE, earliest first. Returns ``(train, test, split_date)``.

    The split date is the threshold; rows before it are train, on/after are test, so
    every train date strictly precedes every test date (no temporal leakage).
    """
    ordered = dataset.sort_values("date").reset_index(drop=True)
    split_date = ordered["date"].iloc[int(len(ordered) * train_frac)]
    train = ordered[ordered["date"] < split_date].reset_index(drop=True)
    test = ordered[ordered["date"] >= split_date].reset_index(drop=True)
    # LEAKAGE GUARD: the model must never see a test-period (or later) date in train.
    assert train["date"].max() < test["date"].min(), "temporal split leaked dates!"
    return train, test, split_date


def fit_model(train: pd.DataFrame) -> tuple[StandardScaler, LogisticRegression]:
    """Fit the scaler and logistic regression on the TRAIN set only."""
    x_train = train[FEATURE_COLUMNS].to_numpy(dtype=float)
    y_train = train["label"].to_numpy(dtype=int)

    # LEAKAGE GUARD: standardization stats come only from train.
    scaler = StandardScaler().fit(x_train)
    model = LogisticRegression(class_weight="balanced", max_iter=1000)
    model.fit(scaler.transform(x_train), y_train)
    return scaler, model


def _print_metrics(test: pd.DataFrame, probs: np.ndarray, train: pd.DataFrame) -> None:
    """Print held-out metrics, base rates, calibration, and coefficients."""
    y_test = test["label"].to_numpy(dtype=int)
    preds = (probs >= 0.5).astype(int)

    base_rate_test = y_test.mean()
    base_rate_train = train["label"].mean()

    print("\n=== Held-out test metrics (most recent ~30% of signals) ===")
    print(f"  Test signals       : {len(test)}")
    print(f"  ROC-AUC            : {roc_auc_score(y_test, probs):.4f}   <-- the number that matters")
    print(f"  Accuracy (@0.5)    : {accuracy_score(y_test, preds):.4f}")
    print(f"  Precision (@0.5)   : {precision_score(y_test, preds, zero_division=0):.4f}")
    print(f"  Recall (@0.5)      : {recall_score(y_test, preds, zero_division=0):.4f}")
    print(f"  Base rate (test)   : {base_rate_test:.4f}   (win rate if you take EVERY signal)")
    print(f"  Base rate (train)  : {base_rate_train:.4f}")

    print("\n  Calibration (quantile bins): predicted vs actual win rate")
    frac_pos, mean_pred = calibration_curve(y_test, probs, n_bins=10, strategy="quantile")
    for predicted, actual in zip(mean_pred, frac_pos):
        print(f"    predicted {predicted:.3f} -> actual {actual:.3f}")


def _print_coefficients(model: LogisticRegression) -> None:
    """Print standardized coefficients so the signs can be sanity-checked."""
    coefs = sorted(zip(FEATURE_COLUMNS, model.coef_[0]), key=lambda kv: abs(kv[1]), reverse=True)
    print("\n=== Logistic coefficients (standardized; + => more likely to win) ===")
    for name, coef in coefs:
        print(f"  {name:<13}: {coef:+.4f}")
    print(f"  intercept    : {model.intercept_[0]:+.4f}")


def main() -> int:
    """Train, evaluate on the held-out test set, and report honestly."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    logging.getLogger("yfinance").setLevel(logging.WARNING)

    dataset = load_dataset()
    train, test, split_date = temporal_split(dataset)
    print(f"\nChronological split at {split_date.date()}: "
          f"train={len(train)} ({train['date'].min().date()}..{train['date'].max().date()}), "
          f"test={len(test)} ({test['date'].min().date()}..{test['date'].max().date()})")

    scaler, model = fit_model(train)
    probs = model.predict_proba(scaler.transform(test[FEATURE_COLUMNS].to_numpy(dtype=float)))[:, 1]

    _print_metrics(test, probs, train)
    _print_coefficients(model)

    print("\n=== ML Limitations ===")
    for note in _LIMITATIONS:
        print(f"  - {note}")
    return 0


_LIMITATIONS = [
    "Small, biased universe: today's known constituents only (survivorship) — the "
    "model learns from names that survived, which inflates apparent edge.",
    "Single train/test split, not yet walk-forward: one held-out period can be lucky "
    "or unlucky; a rolling/expanding-window evaluation is the honest next step.",
    "In-sample feature choices: the feature set and the strategies were designed with "
    "hindsight over this same history.",
    "Labels assume the exact current exit rules and frictions; different exits would "
    "relabel many signals.",
    "Good test metrics still require LIVE out-of-sample confirmation before trusting "
    "the model for filtering or sizing. A simple model scoring >~0.60 AUC here would "
    "be a leakage red flag to investigate, not a success.",
]


if __name__ == "__main__":
    sys.exit(main())
