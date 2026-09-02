"""
The model layer: loading, encoding, and building rows to score.

THIS IS THE RISKIEST FILE IN THE APPLICATION.

The model consumes encoded integers, not text. It expects
``Persona = 3``, not ``"Loyal"``. If a category maps to a different
integer here than it did during training, the model returns confident
predictions computed from a customer who does not exist. Nothing errors.
Nothing looks wrong. Every number on every page is quietly incorrect.

Two things prevent that:

1. The encodings are LOADED from `encoders.json`, exported by notebook
   17. They are never re-derived here. Deriving them independently is
   exactly how two codebases drift apart.

2. Notebook 17 verified those encodings reproduce Databricks scoring to
   within 1e-6 before writing them. `verify_against_recommendations()`
   below re-runs that check inside the app.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st

import config


# ============================================================
# OFFER
# ============================================================

@dataclass(frozen=True)
class Offer:
    """A promotion the platform can put in front of a customer.

    Attributes:
        promotion_type: One of `config.PROMOTION_TYPES`, or
            `config.NO_PROMOTION` for sending nothing.
        discount_percent: 0 for no promotion, otherwise one of
            `config.DISCOUNT_LEVELS`.
    """

    promotion_type: str
    discount_percent: int

    @property
    def is_promotion(self) -> bool:
        return self.discount_percent > 0

    @property
    def label(self) -> str:
        if not self.is_promotion:
            return "No promotion"
        return f"{self.promotion_type} {self.discount_percent}%"

    def margin_after_discount(self, gross_margin: float | None = None) -> float:
        """Margin left once this discount is applied.

        Goes negative above the break-even discount, which is the point
        at which extra bookings destroy value rather than create it.
        """
        base = config.GROSS_MARGIN if gross_margin is None else gross_margin
        return base - self.discount_percent / 100.0


def all_offers(include_no_promotion: bool = True) -> list[Offer]:
    """Every offer the platform can choose between.

    Four promotion types at four discount levels, plus the option of
    sending nothing - which wins roughly seven times out of ten and is
    a real decision, not an absence of one.
    """

    offers: list[Offer] = []

    if include_no_promotion:
        offers.append(Offer(config.NO_PROMOTION, 0))

    for promotion_type in config.PROMOTION_TYPES:
        for discount in config.DISCOUNT_LEVELS:
            offers.append(Offer(promotion_type, discount))

    return offers


# ============================================================
# ARTIFACTS
# ============================================================

class ModelNotFoundError(RuntimeError):
    """Raised when model artifacts are missing, with instructions."""


@st.cache_resource(show_spinner=False)
def load_model() -> Any:
    """Load the trained LightGBM classifier.

    Cached as a *resource* rather than data: it is an unserialisable
    object shared across sessions, not a value to copy per user.
    """

    if not config.MODEL_FILE.exists():
        raise ModelNotFoundError(
            f"Model not found at `{config.MODEL_FILE}`.\n\n"
            f"Run `17_Export_For_App` in Databricks and place "
            f"`model.joblib`, `encoders.json` and `feature_order.json` "
            f"in `streamlit_app/sample_data/model/`."
        )

    return joblib.load(config.MODEL_FILE)


@st.cache_data(show_spinner=False)
def load_encoders() -> dict[str, dict[str, int]]:
    """Category-to-integer mappings, exactly as used during training.

    Shape is ``{column: {category: code}}``. Loaded, never derived.
    """

    if not config.ENCODERS_FILE.exists():
        raise ModelNotFoundError(
            f"Encoders not found at `{config.ENCODERS_FILE}`. "
            f"The model cannot score without them."
        )

    with open(config.ENCODERS_FILE) as handle:
        return json.load(handle)


@st.cache_data(show_spinner=False)
def load_feature_order() -> list[str]:
    """Feature names in the order the model expects them.

    LightGBM validates both the count and the order, so this cannot be
    reconstructed from a set.
    """

    if not config.FEATURE_ORDER_FILE.exists():
        raise ModelNotFoundError(
            f"Feature order not found at `{config.FEATURE_ORDER_FILE}`."
        )

    with open(config.FEATURE_ORDER_FILE) as handle:
        return json.load(handle)


# ============================================================
# ENCODING
# ============================================================

NULL_CATEGORY = "None"
"""How a missing category was spelled at training time.

Spark nulls became Python None in `toPandas()`, and notebook 12's
`.astype(str)` turned them into this literal string before fitting the
encoder. It is a real category with a real code, not a missing value.
"""

UNSEEN_CODE = -1
"""Code for a category absent from training.

Should never occur, since the app scores rows drawn from the same table
the encoders were fitted on. Kept explicit so an unexpected value fails
visibly rather than raising a KeyError mid-render.
"""


def _rebuild_derived_features(
    frame: pd.DataFrame,
    feature_order: list[str],
) -> pd.DataFrame:
    """Recreate features notebook 12 built in memory and never stored.

    Two interaction terms were engineered during training and used by the
    model, but they live only in the training script - the Gold table
    holds the raw columns they were built from, so the export cannot
    contain them.

    Rebuilt here rather than at export time so the app works against any
    snapshot of `gold_training_dataset`, including one produced before
    this was understood. The formulas must match notebook 12 exactly; a
    different scaling would shift every prediction quietly rather than
    raising anything.

    Returns the frame unchanged when the model does not use them.
    """

    derivations = {
        "Discount_x_Response": (
            ("Discount_Percent", "Promotion_Response_Rate"), 1.0,
        ),
        "Discount_x_Loyalty": (
            ("Discount_Percent", "Loyalty_Score"), 100.0,
        ),
    }

    needed = [
        name for name in derivations
        if name in feature_order and name not in frame.columns
    ]

    if not needed:
        return frame

    frame = frame.copy()

    for name in needed:

        (left, right), divisor = derivations[name]

        if left not in frame.columns or right not in frame.columns:
            # Leave it absent - encode() reports it with the full list.
            continue

        frame[name] = (
            frame[left].fillna(0) * frame[right].fillna(0) / divisor
        )

    return frame


def encode(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the training encodings to a feature frame.

    Text columns become their integer codes, booleans become 0/1, and
    columns are returned in the model's expected order.

    Args:
        frame: Rows containing at least every model feature.

    Returns:
        A numeric frame ready for `predict_proba`.
    """

    encoders = load_encoders()
    feature_order = load_feature_order()

    frame = _rebuild_derived_features(frame, feature_order)

    missing = [f for f in feature_order if f not in frame.columns]

    if missing:
        raise ValueError(
            f"These model features are missing from the frame: {missing}. "
            f"They were most likely engineered inside notebook 12 and are "
            f"not stored in the exported table."
        )

    encoded = frame[feature_order].copy()

    for column, mapping in encoders.items():

        if column not in encoded.columns:
            continue

        # Notebook 12 encoded `X[column].astype(str)` on a frame that came
        # from `toPandas()`, where a SQL null arrives as Python None and
        # stringifies to "None". Reading the same data from Parquet gives
        # NaN, which stringifies to "nan" and matches no key - so 98% of
        # rows would silently score with Holiday_Name = -1.
        as_text = encoded[column].astype(object).where(
            encoded[column].notna(), NULL_CATEGORY
        ).astype(str)

        encoded[column] = (
            as_text
            .map(mapping)
            .fillna(UNSEEN_CODE)
            .astype("int32")
        )

    for column in encoded.columns:
        if encoded[column].dtype == bool:
            encoded[column] = encoded[column].astype(int)

    # Safety net for single-row frames. Transposing a Series turns every
    # column into `object`, and LightGBM rejects the whole frame rather
    # than coercing. Callers that already pass typed frames skip this.
    #
    # Selected by what it is NOT, rather than `include="object"`. Under
    # pandas 2 a text column is dtype object; under pandas 3 it is dtype
    # str, which `include="object"` does not match - so this net caught
    # nothing and LightGBM raised "pandas dtypes must be int, float or
    # bool" from deep inside its predict path, naming columns rather
    # than the cause.
    leftover = encoded.select_dtypes(exclude=["number", "bool"]).columns

    for column in leftover:
        encoded[column] = pd.to_numeric(encoded[column], errors="coerce")

    return encoded


def _apply_offer(
    encoded: pd.DataFrame,
    offer: Offer,
    seasonal_price: pd.Series,
    promotion_response: pd.Series | None = None,
    loyalty_score: pd.Series | None = None,
) -> pd.DataFrame:
    """Overwrite the promotion columns to describe a specific offer.

    All promotion columns move together. A row claiming no promotion
    while carrying a 20% discount is incoherent - no such row existed in
    training, so the model's answer for it would be meaningless.

    The two interaction features are recomputed here rather than carried
    over, because both are built from the discount. Leaving them stale
    would describe whatever promotion the customer originally received
    instead of the one being evaluated.
    """

    encoders = load_encoders()

    row = encoded.copy()

    row["Promotion_Sent"] = 1 if offer.is_promotion else 0

    if "Promotion_Type" in row.columns:
        type_mapping = encoders.get("Promotion_Type", {})
        row["Promotion_Type"] = type_mapping.get(
            offer.promotion_type, UNSEEN_CODE
        )

    if "Discount_Percent" in row.columns:
        row["Discount_Percent"] = offer.discount_percent

    if "Discount_Value" in row.columns:
        row["Discount_Value"] = (
            seasonal_price.values * offer.discount_percent / 100.0
        )

    if "Discount_x_Response" in row.columns and promotion_response is not None:
        row["Discount_x_Response"] = (
            offer.discount_percent * promotion_response.fillna(0).values
        )

    if "Discount_x_Loyalty" in row.columns and loyalty_score is not None:
        row["Discount_x_Loyalty"] = (
            offer.discount_percent * loyalty_score.fillna(0).values / 100.0
        )

    return row


# ============================================================
# SCORING
# ============================================================

def score_offer(
    rows: pd.DataFrame,
    offer: Offer,
) -> np.ndarray:
    """Booking probability for these customers under one specific offer.

    Args:
        rows: Raw (unencoded) feature rows from `training_features`.
            Must include `Seasonal_Price`, and `Promotion_Response_Rate`
            and `Loyalty_Score` if the model uses the interactions.
        offer: The offer to evaluate.

    Returns:
        One probability per row, in the same order.
    """

    model = load_model()

    encoded = encode(rows)

    with_offer = _apply_offer(
        encoded,
        offer,
        seasonal_price=rows["Seasonal_Price"],
        promotion_response=rows.get("Promotion_Response_Rate"),
        loyalty_score=rows.get("Loyalty_Score"),
    )

    return model.predict_proba(with_offer)[:, 1]


def score_all_offers(
    rows: pd.DataFrame,
    offers: list[Offer] | None = None,
) -> pd.DataFrame:
    """Score every offer against every customer.

    This is the core operation of the platform. The model only ever
    answers one question - *how likely is a booking?* - so a
    recommendation is produced by asking it once per candidate offer and
    comparing the answers.

    Args:
        rows: Raw feature rows, one per customer.
        offers: Offers to evaluate. Defaults to all 17.

    Returns:
        Long-format frame with one row per customer-offer pair:
        ``row_index``, ``offer``, ``promotion_type``,
        ``discount_percent``, ``booking_probability``, ``uplift``,
        ``revenue_if_booked``, ``expected_profit``.

        ``uplift`` is measured against the no-promotion baseline for
        that same customer.
    """

    if offers is None:
        offers = all_offers()

    rows = rows.reset_index(drop=True)

    price = rows["Seasonal_Price"].astype(float)

    frames: list[pd.DataFrame] = []

    for offer in offers:

        probability = score_offer(rows, offer)

        margin = offer.margin_after_discount()

        frames.append(pd.DataFrame({
            "row_index": rows.index,
            "offer": offer.label,
            "promotion_type": offer.promotion_type,
            "discount_percent": offer.discount_percent,
            "booking_probability": probability,
            "revenue_if_booked": price * (1 - offer.discount_percent / 100.0),
            "expected_profit": probability * price * margin,
        }))

    scored = pd.concat(frames, ignore_index=True)

    baseline = (
        scored[scored["discount_percent"] == 0]
        .set_index("row_index")["booking_probability"]
    )

    if not baseline.empty:
        scored["baseline_probability"] = scored["row_index"].map(baseline)
        scored["uplift"] = (
            scored["booking_probability"] - scored["baseline_probability"]
        )
    else:
        scored["baseline_probability"] = np.nan
        scored["uplift"] = np.nan

    return scored


def best_offer_per_customer(scored: pd.DataFrame) -> pd.DataFrame:
    """Pick the highest expected profit offer for each customer.

    Expected profit, not booking probability. Ranking on probability
    alone would recommend the largest discount to everybody, because a
    bigger discount always raises the chance of a booking - which is
    not a decision, it is an identity.

    Adds ``edge_over_next_best``: how much better the winner is than the
    runner-up. A thin edge means the choice is close and either offer
    would do, which is more useful to a human than a confidence score
    a single prediction cannot support.
    """

    ordered = scored.sort_values(
        ["row_index", "expected_profit"], ascending=[True, False]
    )

    best = ordered.groupby("row_index", as_index=False).first()

    second = (
        ordered.groupby("row_index", as_index=False)
        .nth(1)[["row_index", "expected_profit"]]
        .rename(columns={"expected_profit": "runner_up_profit"})
    )

    best = best.merge(second, on="row_index", how="left")

    best["edge_over_next_best"] = (
        best["expected_profit"] - best["runner_up_profit"]
    ).fillna(0.0)

    return best


# ============================================================
# INTEGRITY
# ============================================================

def verify_against_recommendations(
    training_features: pd.DataFrame,
    recommendations: pd.DataFrame,
    sample_size: int = 200,
) -> tuple[bool, float, str]:
    """Confirm the app scores identically to Databricks.

    `gold_promotion_recommendations` holds the probability the model
    produced inside Databricks for each winning offer. Re-scoring those
    same rows here must reproduce them. A mismatch means the encodings
    disagree, and every prediction in the app is wrong.

    Notebook 17 runs this check before exporting. Running it again here
    catches the case where somebody swaps a Parquet file for an older
    one - the failure mode that produces no error and no visible
    symptom.

    Returns:
        ``(passed, largest_deviation, message)``.
    """

    sample = recommendations.head(sample_size)

    merged = sample.merge(
        training_features,
        on="Activity_ID",
        how="inner",
        suffixes=("_rec", ""),
    )

    if merged.empty:
        return False, float("nan"), (
            "Could not match any recommendation to a feature row. The "
            "two exports are probably from different pipeline runs."
        )

    deviations: list[float] = []

    for (promotion_type, discount), group in merged.groupby(
        ["Offer_Type", "Offer_Discount"]
    ):
        offer = Offer(str(promotion_type), int(discount))

        predicted = score_offer(group, offer)

        deviations.extend(
            np.abs(predicted - group["Booking_Probability"].values).tolist()
        )

    largest = float(np.max(deviations)) if deviations else float("nan")

    passed = bool(largest < 1e-6)






