"""
The recommendation panel: what to offer, and the argument for it.

A recommendation with no reasoning is an instruction, and people do not
follow instructions from software they cannot interrogate. Everything
here exists to let a manager check the machine's work: the numbers it
used, the alternatives it rejected, and the risks it sees in its own
answer.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

import config
from components import charts
from services.recommendation_service import Recommendation


# ============================================================
# HEADLINE
# ============================================================

def render(recommendation: Recommendation) -> None:
    """The full recommendation panel."""

    _headline(recommendation)

    left, right = st.columns([3, 2], gap="large")

    with left:
        _reasoning(recommendation)

    with right:
        _risks(recommendation)

    _alternatives(recommendation)


def _headline(rec: Recommendation) -> None:
    """The decision, with the four numbers behind it.

    When the answer is *send nothing*, the panel says so as
    affirmatively as it would announce a discount. Restraint is a
    decision, and presenting it apologetically teaches users that the
    tool only "works" when it recommends spending money.
    """

    if rec.is_promotion:
        offer_text = f"{rec.offer.discount_percent}% OFF"
        sub_text = f"{rec.offer.promotion_type} on {rec.context['Service_Name']}"
        tone = "promote"
    else:
        offer_text = "No promotion"
        sub_text = "This customer does not need one"
        tone = "hold"

    st.markdown(
        f"""
        <div class="rec-hero rec-{tone}">
            <div class="rec-offer">
                <div class="rec-offer-main">{offer_text}</div>
                <div class="rec-offer-sub">{sub_text}</div>
            </div>
            <div class="rec-stats">
                <div class="rec-stat">
                    <div class="rec-stat-label">Booking probability</div>
                    <div class="rec-stat-value">
                        {config.percent(rec.booking_probability)}
                    </div>
                    <div class="rec-stat-note">
                        {config.percent(rec.baseline_probability)} without an offer
                    </div>
                </div>
                <div class="rec-stat">
                    <div class="rec-stat-label">Uplift</div>
                    <div class="rec-stat-value">
                        {rec.uplift:+.1%}
                    </div>
                    <div class="rec-stat-note">
                        {"what the discount buys" if rec.is_promotion
                         else "no change, as intended"}
                    </div>
                </div>
                <div class="rec-stat">
                    <div class="rec-stat-label">Expected profit</div>
                    <div class="rec-stat-value">
                        {config.money(rec.expected_profit)}
                    </div>
                    <div class="rec-stat-note">
                        {config.money(rec.profit_gain)} better than the alternative
                    </div>
                </div>
                <div class="rec-stat">
                    <div class="rec-stat-label">Decision</div>
                    <div class="rec-stat-value">{rec.decision_confidence}</div>
                    <div class="rec-stat-note">
                        {config.money(rec.edge_over_next_best)} ahead of next best
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _reasoning(rec: Recommendation) -> None:
    """Why this offer won.

    Each line cites the customer's own numbers, so a manager can check
    them against the profile beside it. "Booked 12 days ago, inside a
    30-day repeat window" is verifiable. "High propensity customer" is
    not.
    """

    st.markdown("##### Why this offer")

    for reason in rec.reasons:
        st.markdown(
            f'<div class="reason"><span class="reason-tick">✓</span>'
            f'<span>{reason}</span></div>',
            unsafe_allow_html=True,
        )


def _risks(rec: Recommendation) -> None:
    """What could go wrong.

    Stated rather than buried. A recommendation engine that never
    expresses doubt teaches people either to over-trust it or to stop
    using it, and both failures are expensive.
    """

    st.markdown("##### Risks and caveats")

    for risk in rec.risks:
        st.markdown(
            f'<div class="risk"><span class="risk-mark">!</span>'
            f'<span>{risk}</span></div>',
            unsafe_allow_html=True,
        )


def _alternatives(rec: Recommendation) -> None:
    """What else was considered.

    Showing the rejected options turns a recommendation into an
    argument. A manager who can see the runner-up was ₹4 behind knows
    to treat the choice as close, which no confidence score would have
    told them as clearly.
    """

    st.markdown("##### Considered and rejected")

    left, right = st.columns([3, 2], gap="large")

    with left:
        st.plotly_chart(
            charts.offer_comparison(rec.alternatives),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    with right:

        table = rec.alternatives.head(5)[[
            "offer", "booking_probability", "uplift", "expected_profit"
        ]].copy()

        table.columns = ["Offer", "Books", "Uplift", "Profit"]

        table["Books"] = table["Books"].map(lambda v: f"{v:.0%}")
        table["Uplift"] = table["Uplift"].map(lambda v: f"{v:+.1%}")
        table["Profit"] = table["Profit"].map(config.money)

        st.dataframe(
            table,
            hide_index=True,
            use_container_width=True,
        )

        st.caption(
            f"All {len(rec.alternatives)} shown are the top of 17 scored. "
            f"Ranked on expected profit, not booking probability — ranking "
            f"on probability would always pick the largest discount."
        )


# ============================================================
# COMPACT
# ============================================================

def compact(recommendation: Recommendation) -> None:
    """One-line summary, for lists and cohort tables."""

    rec = recommendation

    badge = "promote" if rec.is_promotion else "hold"

    label = (
        f"{rec.offer.discount_percent}% off" if rec.is_promotion
        else "No promotion"
    )

    st.markdown(
        f"""
        <div class="rec-compact">
            <span class="rec-badge rec-badge-{badge}">{label}</span>
            <span class="rec-compact-customer">{rec.customer_id}</span>
            <span class="rec-compact-detail">
                {rec.context['Persona']} · {rec.context['City']} ·
                {rec.context['Service_Name']}
            </span>
            <span class="rec-compact-profit">
                {config.money(rec.expected_profit)}
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def cohort_summary(frame: pd.DataFrame) -> None:
    """Headline numbers for a batch of recommendations.

    The share receiving nothing is shown first and deliberately. A
    recommender that promotes everybody has learned nothing, so that
    figure is the fastest read on whether the model is discriminating.
    """

    if frame.empty:
        st.info("No customers match this selection.")
        return

    promoted = frame[frame["discount_percent"] > 0]

    hold_share = 1 - len(promoted) / len(frame)

    cols = st.columns(4)

    cols[0].metric(
        "Send nothing",
        config.percent(hold_share),
        help=(
            "Customers who book anyway, or where no discount large "
            "enough to move them would pay for itself."
        ),
    )

    cols[1].metric("Promote", f"{len(promoted):,}")

    cols[2].metric(
        "Average discount",
        f"{promoted['discount_percent'].mean():.1f}%" if len(promoted) else "—",
    )

    cols[3].metric(
        "Total expected profit",
        config.money(frame["expected_profit"].sum()),
    )
