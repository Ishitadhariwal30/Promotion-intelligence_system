"""
Single-customer recommendation: which offer, and why.

Wraps `model_service` and adds the layer a business user needs - the
reasoning, the risks, and the alternatives that were considered and
rejected.

The explanation is generated from the customer's own numbers, never from
a template with blanks. "Booked this service 12 days ago, inside its
30-day repeat window" is checkable. "High propensity customer" is not.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

import config
from services import model_service
from services.model_service import Offer


# ============================================================
# RESULT
# ============================================================

@dataclass
class Recommendation:
    """Everything the platform can say about one customer's best offer."""

    customer_id: str
    activity_id: str
    context: pd.Series

    offer: Offer
    booking_probability: float
    baseline_probability: float
    uplift: float
    revenue_if_booked: float
    expected_profit: float
    baseline_expected_profit: float
    edge_over_next_best: float

    alternatives: pd.DataFrame
    reasons: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)

    @property
    def is_promotion(self) -> bool:
        return self.offer.is_promotion

    @property
    def profit_gain(self) -> float:
        """Expected profit gained over sending nothing.

        Zero when the recommendation *is* to send nothing, which is the
        correct answer roughly seven times in ten.
        """
        return self.expected_profit - self.baseline_expected_profit

    @property
    def discount_cost(self) -> float:
        """Rupees surrendered if this offer converts."""
        return (
            self.context["Seasonal_Price"]
            * self.offer.discount_percent / 100.0
        )

    @property
    def decision_confidence(self) -> str:
        """How clear-cut the choice was.

        Deliberately not a probability. A single prediction has no
        meaningful confidence interval, so this describes the *decision*
        instead: how far ahead the winner finished.
        """
        if self.edge_over_next_best > 20:
            return "Clear"
        if self.edge_over_next_best > 5:
            return "Moderate"
        return "Close call"


# ============================================================
# LOOKUP
# ============================================================

def available_customers(training_features: pd.DataFrame) -> list[str]:
    """Customers that can be scored, from the model's test period.

    Recommendations only exist out of time - the months the model never
    trained on. Offering earlier customers would mean scoring rows the
    model has already memorised.
    """

    test = training_features[training_features["Data_Split"] == "test"]

    return sorted(test["Customer_ID"].unique().tolist())


def customer_row(
    training_features: pd.DataFrame,
    customer_id: str,
) -> pd.Series:
    """The customer's most recent scoreable event.

    Raises:
        ValueError: If the customer has no event in the test period.
    """

    rows = training_features[
        (training_features["Customer_ID"] == customer_id)
        & (training_features["Data_Split"] == "test")
    ]

    if rows.empty:
        raise ValueError(
            f"{customer_id} has no event in the model's test period "
            f"(October to December). Only customers active in those "
            f"months can be scored."
        )

    return rows.sort_values("Activity_Date").iloc[-1]


def filter_customers(
    training_features: pd.DataFrame,
    city: str | None = None,
    persona: str | None = None,
    segment: str | None = None,
    service: str | None = None,
    membership: str | None = None,
) -> pd.DataFrame:
    """Test-period events matching a scenario description.

    Used by the recommendation page so a manager can describe the
    customer they care about rather than knowing an ID.

    Returns real rows rather than constructing a synthetic customer. A
    made-up row would need values for all 57 features, and any
    combination we invented might be one the model has never seen - it
    would answer confidently regardless.
    """

    frame = training_features[training_features["Data_Split"] == "test"]

    for column, value in [
        ("City", city), ("Persona", persona),
        ("Customer_Segment", segment), ("Service_Name", service),
        ("Membership", membership),
    ]:
        if value and value != "Any" and column in frame.columns:
            frame = frame[frame[column].astype(str) == value]

    return frame


# ============================================================
# RECOMMEND
# ============================================================

def recommend(row: pd.Series) -> Recommendation:
    """Score all 17 offers for one customer and pick the best.

    The model only answers *how likely is a booking?*. A recommendation
    comes from asking it once per candidate offer and comparing on
    expected profit - not on probability, which would always favour the
    largest discount.
    """

    # Transposing a Series makes every column `object`, because a Series
    # holds one dtype for all its values. infer_objects() puts the real
    # numeric and boolean types back - without it the model receives a
    # frame of Python objects and refuses to score.
    frame = row.to_frame().T.reset_index(drop=True).infer_objects()

    scored = model_service.score_all_offers(frame)

    scored = scored.sort_values("expected_profit", ascending=False)

    best = scored.iloc[0]
    runner_up = scored.iloc[1] if len(scored) > 1 else best

    baseline = scored[scored["discount_percent"] == 0].iloc[0]

    offer = Offer(str(best["promotion_type"]), int(best["discount_percent"]))

    recommendation = Recommendation(
        customer_id=str(row["Customer_ID"]),
        activity_id=str(row["Activity_ID"]),
        context=row,
        offer=offer,
        booking_probability=float(best["booking_probability"]),
        baseline_probability=float(baseline["booking_probability"]),
        uplift=float(best["uplift"]),
        revenue_if_booked=float(best["revenue_if_booked"]),
        expected_profit=float(best["expected_profit"]),
        baseline_expected_profit=float(baseline["expected_profit"]),
        edge_over_next_best=float(
            best["expected_profit"] - runner_up["expected_profit"]
        ),
        alternatives=scored.head(5).reset_index(drop=True),
    )

    recommendation.reasons = _build_reasons(recommendation)
    recommendation.risks = _build_risks(recommendation)

    return recommendation


# ============================================================
# EXPLANATION
# ============================================================

def _build_reasons(rec: Recommendation) -> list[str]:
    """Why this offer won, in business language.

    Every line cites the customer's own numbers so a manager can check
    it against the profile shown beside it.
    """

    context = rec.context
    reasons: list[str] = []

    # --- the headline decision --------------------------------

    if not rec.is_promotion:
        reasons.append(
            f"This customer already books at "
            f"{config.percent(rec.baseline_probability)} without any offer. "
            f"A discount would cost more than the extra bookings it buys."
        )
    else:
        reasons.append(
            f"A {rec.offer.discount_percent}% discount lifts booking "
            f"probability from {config.percent(rec.baseline_probability)} to "
            f"{config.percent(rec.booking_probability)} — enough to cover the "
            f"{config.money(rec.discount_cost)} given up."
        )

    # --- who they are ------------------------------------------

    reasons.append(
        f"{context['Persona']} persona in the {context['Customer_Segment']} "
        f"segment, loyalty score {context.get('Loyalty_Score', 0):.0f} of 100."
    )

    # --- promotion history --------------------------------------

    received = context.get("Promotions_Received_To_Date", 0)
    response = context.get("Promotion_Response_Rate")

    if received and pd.notna(response):
        reasons.append(
            f"Redeemed {config.percent(response)} of the {int(received)} "
            f"offers sent so far — the strongest available signal of how "
            f"they respond to promotions."
        )
    elif received:
        reasons.append(
            f"Sent {int(received)} offers previously and redeemed none."
        )
    else:
        reasons.append("Has never been sent a promotion before.")

    # --- repeat gap ---------------------------------------------

    if context.get("Within_Repeat_Gap"):
        days = context.get("Days_Since_Same_Service")
        gap = context.get("Repeat_Gap_Days")
        reasons.append(
            f"Booked {context['Service_Name']} {int(days)} days ago, inside "
            f"its {int(gap)}-day repeat window. Unlikely to need it again "
            f"yet regardless of price."
        )

    # --- value ---------------------------------------------------

    aov = context.get("Avg_Order_Value")
    ratio = context.get("Price_To_AOV_Ratio")

    if pd.notna(aov) and pd.notna(ratio) and ratio > 1.4:
        reasons.append(
            f"At {config.money(context['Seasonal_Price'])} this job is "
            f"{ratio:.1f}× their usual {config.money(aov)} order — a bigger "
            f"commitment than they typically make."
        )

    # --- context --------------------------------------------------

    if pd.notna(context.get("Holiday_Name")):
        reasons.append(
            f"Falls on {context['Holiday_Name']}, when demand for this "
            f"service runs above normal."
        )

    if context.get("Is_Service_In_Season"):
        reasons.append(
            f"{context['Service_Name']} is in season during "
            f"{context['Season']}, lifting baseline demand."
        )

    return reasons


def _build_risks(rec: Recommendation) -> list[str]:
    """What could go wrong with this recommendation.

    Stated plainly rather than buried. A recommendation engine that
    never expresses doubt trains people to either over-trust it or stop
    using it.
    """

    context = rec.context
    risks: list[str] = []

    if rec.decision_confidence == "Close call":
        second = rec.alternatives.iloc[1] if len(rec.alternatives) > 1 else None
        if second is not None:
            risks.append(
                f"Close call — '{second['offer']}' is only "
                f"{config.money(rec.edge_over_next_best)} behind. Either "
                f"would be defensible."
            )

    if rec.offer.discount_percent >= config.DISCOUNT_CAUTION_THRESHOLD:
        risks.append(
            f"At {rec.offer.discount_percent}% this is an aggressive "
            f"discount. Margin falls to "
            f"{config.percent(rec.offer.margin_after_discount())} — it only "
            f"pays if the customer genuinely would not have booked."
        )

    cancellation = context.get("Cancellation_Rate")

    if pd.notna(cancellation) and cancellation > 0.2:
        risks.append(
            f"Cancels {config.percent(cancellation)} of bookings. A "
            f"converted booking here may not become revenue, and it "
            f"consumes provider capacity either way."
        )

    if context.get("Persona") == "Premium" and rec.is_promotion:
        risks.append(
            "Premium customers respond least to promotions across the "
            "customer base. The model over-ranks this segment — treat a "
            "promotion here with more scepticism than the number suggests."
        )

    if pd.notna(context.get("Days_Since_Last_Booking")):
        if context["Days_Since_Last_Booking"] > 60:
            risks.append(
                f"Last booked {int(context['Days_Since_Last_Booking'])} days "
                f"ago. Reactivation is harder than retention, and a single "
                f"offer may not be enough."
            )

    if not risks:
        risks.append(
            "No material risks flagged. The customer's history, timing and "
            "the offer economics all point the same way."
        )

    return risks


# ============================================================
# BATCH
# ============================================================

def recommend_batch(rows: pd.DataFrame) -> pd.DataFrame:
    """Best offer for many customers at once.

    Used by the cohort views. Scores every offer against every row in
    one vectorised pass rather than looping, which matters when the
    Strategy Lab re-runs on every slider move.
    """

    if rows.empty:
        return pd.DataFrame()

    scored = model_service.score_all_offers(rows.reset_index(drop=True))

    best = model_service.best_offer_per_customer(scored)

    context_columns = [
        c for c in [
            "Customer_ID", "Activity_ID", "Activity_Date", "City", "Persona",
            "Customer_Segment", "Membership", "Service_Name", "Seasonal_Price",
        ] if c in rows.columns
    ]

    context = rows.reset_index(drop=True)[context_columns]

    return pd.concat(
        [context, best.drop(columns=["row_index"])], axis=1
    )
