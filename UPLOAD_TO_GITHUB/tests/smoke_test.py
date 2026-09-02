"""
SMOKE TEST - Urban Company Promotion Intelligence Platform

PURPOSE
    Prove the deployed app can actually do its job: load its data, load its
    model, and score one customer end to end.

INPUT   nothing - reads sample_data/ via config.py
OUTPUT  exit code 0 if everything works, 1 if not. Prints what it checked.

WHERE IT SITS
    Run by .github/workflows/ci.yml on every push, and runnable by hand:

        python tests/smoke_test.py

WHY THIS EXISTS
    Streamlit failures are silent in the worst way. The app deploys, the URL
    works, and the page shows a red traceback only once somebody clicks the
    thing you forgot to test. This runs the risky path - encode a row, call
    the model - without a browser, so a broken push fails in CI instead of
    in front of whoever you sent the link to.
"""

import json
import sys
from pathlib import Path

# The repo root is one level up from tests/. Added to the path so `config`
# and `services` import the same way they do when Streamlit runs app.py.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

failures: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    """Record one assertion. Never raises - we want the full report, not the
    first failure. A CI run that surfaces four problems at once beats four
    runs that each surface one."""
    if condition:
        print(f"  PASS  {label}")
    else:
        print(f"  FAIL  {label}  {detail}")
        failures.append(label)


# ============================================================
# 1. Config imports, and its paths point at real files
# ============================================================

print("\n1. Config and data files")

import config  # noqa: E402  - deliberately after sys.path is set up

import re

config_source = (REPO_ROOT / "config.py").read_text(encoding="utf-8")

# Pulled by regex rather than by importing the dict, so this test does not
# break if the dict is ever renamed.
declared_parquet = sorted(set(re.findall(r'"([A-Za-z0-9_]+\.parquet)"', config_source)))

check("config.py imports", True)
check("DATA_DIR exists", config.DATA_DIR.is_dir(), str(config.DATA_DIR))

for filename in declared_parquet:
    path = config.DATA_DIR / filename
    check(f"{filename}", path.is_file(), "declared in config.py but not on disk")


# ============================================================
# 2. Model artifacts are present and loadable
# ============================================================

print("\n2. Model artifacts")

check("model.joblib exists", config.MODEL_FILE.is_file(), str(config.MODEL_FILE))
check("encoders.json exists", config.ENCODERS_FILE.is_file(), str(config.ENCODERS_FILE))
check("feature_order.json exists", config.FEATURE_ORDER_FILE.is_file(), str(config.FEATURE_ORDER_FILE))

model = None
feature_order: list[str] = []
encoders: dict = {}

if not failures:
    import joblib

    model = joblib.load(config.MODEL_FILE)
    check("model unpickles", model is not None, "")
    check(
        "model has predict_proba",
        hasattr(model, "predict_proba"),
        "a classifier is expected here",
    )

    feature_order = json.loads(config.FEATURE_ORDER_FILE.read_text(encoding="utf-8"))
    if isinstance(feature_order, dict):
        # Tolerate either a bare list or {"features": [...]}
        feature_order = feature_order.get("features", feature_order.get("feature_order", []))

    check("feature_order is a non-empty list", len(feature_order) > 0, f"got {type(feature_order)}")

    encoders = json.loads(config.ENCODERS_FILE.read_text(encoding="utf-8"))
    check("encoders.json parses", isinstance(encoders, dict), f"got {type(encoders)}")


# ============================================================
# 3. The real test - score one row end to end
# ============================================================
#
# This is the check worth having. Everything above proves files exist;
# this proves they FIT TOGETHER. The model wants 61 encoded integers in a
# fixed order. If notebook 17 ever exports an encoder set that disagrees
# with the model, or the feature order shifts, only this catches it.

print("\n3. End-to-end scoring")

if not failures and model is not None:
    import pandas as pd

    frame = pd.read_parquet(config.DATA_DIR / "training_features.parquet")
    check("training_features loaded", len(frame) > 0, "")

    # Two features are NOT stored in the parquet. They are interactions with
    # Discount_Percent, which changes for every offer being trialled, so
    # notebook 13 rebuilds them per candidate rather than persisting them.
    # The app does the same. Reproducing that here is the point: if these
    # formulas ever drift from notebook 13, the app scores differently from
    # the pipeline and nothing else would tell us.
    discount = frame["Discount_Percent"].fillna(0)

    frame["Discount_x_Response"] = discount * frame["Promotion_Response_Rate"].fillna(0)
    frame["Discount_x_Loyalty"] = discount * frame["Loyalty_Score"].fillna(0) / 100

    missing = [column for column in feature_order if column not in frame.columns]
    check(
        "every model feature is available (stored or derived)",
        not missing,
        f"missing: {missing[:5]}",
    )

    if not missing:
        row = frame.head(1)[feature_order].copy()

        # Apply the same encoding notebook 12 used. Text columns must become
        # the integers the model was trained on - "Loyal" -> 3, not "Loyal".
        #
        # Test for "not numeric" rather than "dtype == object". Under pandas 2
        # a text column is dtype object; under pandas 3 it is dtype str. An
        # `== object` check silently skips every text column on pandas 3, the
        # encoding never happens, and LightGBM raises far away from the cause.
        for column in row.columns:
            if column in encoders and not pd.api.types.is_numeric_dtype(row[column]):
                mapping = encoders[column]
                if isinstance(mapping, dict):
                    row[column] = row[column].map(mapping)

        # Anything still text would crash predict_proba with an unhelpful
        # error, so say plainly which column it was.
        still_text = [c for c in row.columns if not pd.api.types.is_numeric_dtype(row[c])]
        check(
            "all features are numeric after encoding",
            not still_text,
            f"still text: {still_text[:5]}",
        )

        if not still_text:
            row = row.fillna(0)
            probability = float(model.predict_proba(row)[:, 1][0])

            check(
                "model returns a probability",
                0.0 <= probability <= 1.0,
                f"got {probability}",
            )
            print(f"        scored one customer: P(book) = {probability:.4f}")


# ============================================================
# 4. Result
# ============================================================

print()
if failures:
    print(f"SMOKE TEST FAILED - {len(failures)} check(s) did not pass:")
    for name in failures:
        print(f"  - {name}")
    sys.exit(1)

print("SMOKE TEST PASSED - the app can load its data, its model, and score.")
sys.exit(0)
