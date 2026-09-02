"""
Promotion Performance — which promotions work, and what are they
costing?

The uplift chart is the centre of this page and the centre of the whole
platform. If those gaps were equal across personas, targeting could not
beat a blanket policy and the product would have no reason to exist.

Uplift is measured on the randomised group by default. Everywhere else
the comparison is confounded — promoted customers were chosen *because*
they looked responsive, so some of the apparent lift is selection rather
than treatment. The toggle exists so the difference is visible.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

import config
from components import charts, copilot, filters as filter_bar
from components.metric_cards import (
    metric_row, page_header, recommended_action, section,
)
from services import analytics_service as analytics
from services import data_loader

st.set_page_config(page_title="Promotion Performance", page_icon="◆", layout="wide")

stylesheet = config.APP_ROOT / "assets" / "styles.css"
if stylesheet.exists():
    st.markdown(f"<style>{stylesheet.read_text(encoding='utf-8')}</style>",
                unsafe_allow_html=True)


# ============================================================
# LOAD
# ============================================================

bookings_all = data_loader.load_bookings()
promotions_all = data_loader.load_promotions()
training = data_loader.load_training_features()

active = filter_bar.render(show_segments=False)

bookings = analytics.apply_filters(bookings_all, active, "Booking_Date")
promotions = analytics.apply_filters(promotions_all, active, "Promotion_Date")


page_header(
    "Promotion Performance",
    "Which promotions work, and what are they costing?",
    f"Showing {active.describe()}.",
)

if promotions.empty:
    st.warning("No promotion decisions match this selection.")
    st.stop()


# ============================================================
# HEADLINE
# ============================================================

sent = promotions[promotions["Promotion_Sent"] == True]

promoted_bookings = bookings[bookings["Promotion_Offered"] == True]

completed_promoted = promoted_bookings[
    promoted_bookings["Booking_Status"] == "Completed"
]

discount_spend = float(
    (completed_promoted["Seasonal_Price"] - completed_promoted["Final_Price"]).sum()
) if not completed_promoted.empty else 0.0

waste = analytics.promotion_waste(promotions, bookings)

metric_row([
    {
        "label": "Offers sent",
        "value": config.count(len(sent)),
        "caption": f"of {config.count(len(promotions))} eligible",
    },
    {
        "label": "Discount given",
        "value": config.money(discount_spend),
        "caption": "revenue surrendered",
        "severity": "warn",
    },
    {
        "label": "Estimated waste",
        "value": config.money(waste["wasted_spend"]),
        "caption": "to customers who'd have booked anyway",
        "severity": "bad",
        "help_text": (
            "Baseline conversion measured on the randomised control "
            "group, applied to total discount spend."
        ),
    },
    {
        "label": "Redemption",
        "value": config.percent(
            completed_promoted["Promotion_Redeemed"].mean()
            if not completed_promoted.empty else 0
        ),
        "caption": "offers that led to a completed job",
    },
], columns=4)


# ============================================================
# UPLIFT
# ============================================================

section("What a promotion is worth", "And to whom?")

honest_only = st.toggle(
    "Measure on the randomised group only",
    value=True,
    help=(
        "Inside Targeting_Mode = Exploration, promotions were assigned at "
        "random, so promoted and unpromoted customers are genuinely "
        "comparable and the gap between them is the causal effect. "
        "Everywhere else, promoted customers were chosen because they "
        "looked responsive — some of the apparent uplift is selection."
    ),
)

uplift = analytics.promotion_uplift(
    promotions, bookings, by="Persona", exploration_only=honest_only,
)

if uplift.empty:
    st.info("Not enough data in this selection to measure uplift.")
else:
    st.plotly_chart(
        charts.uplift_chart(uplift),
        use_container_width=True,
        config={"displayModeBar": False},
    )

    best = uplift.iloc[0]
    worst = uplift.iloc[-1]

    ratio = (
        best["uplift"] / worst["uplift"]
        if worst["uplift"] and worst["uplift"] > 0 else float("nan")
    )

    copilot.inline_insight(
        f"<b>{best['Persona']}</b> responds most — a promotion lifts them "
        f"from {config.percent(best['not_promoted'])} to "
        f"{config.percent(best['promoted'])}. "
        f"<b>{worst['Persona']}</b> responds least at "
        f"{worst['uplift']:+.1%}."
        + (
            f" That is a <b>{ratio:.1f}× difference</b>, and it is the entire "
            f"basis for targeting: the same discount is worth several times "
            f"more on one persona than another."
            if pd.notna(ratio) and ratio > 1 else ""
        ),
        source=(
            f"silver_promotions · "
            f"{'randomised subset only' if honest_only else 'all promotion decisions'} · "
            f"{int(uplift['n_promoted'].sum() + uplift['n_not_promoted'].sum()):,} customers"
        ),
    )

    if not honest_only:
        st.warning(
            "These figures include targeted promotions, where promoted "
            "customers were selected *because* they looked responsive. "
            "Some of this uplift is selection, not treatment. Toggle above "
            "for the unconfounded measure."
        )


# ============================================================
# RESPONSE CURVE
# ============================================================

section("How response scales with discount", "Where does more stop being better?")

curve = (
    training[training["Promotion_Sent"] == True]
    .groupby(["Persona", "Discount_Percent"], as_index=False)
    .agg(conversion=("Booked", "mean"), customers=("Activity_ID", "count"))
)

st.plotly_chart(
    charts.discount_response_curve(curve),
    use_container_width=True,
    config={"displayModeBar": False},
)

st.caption(
    "Different slopes are the basis for targeting. Parallel lines would "
    "mean everyone responds the same, and a single blanket discount would "
    "be optimal."
)


# ============================================================
# ECONOMICS
# ============================================================

section("What each promotion returns", "Which ones pay for themselves?")

roi = analytics.promotion_roi(bookings)

if roi.empty:
    st.info("No promoted bookings in this selection.")
else:
    display = roi.copy()

    for column in ["revenue", "discount_cost", "net_contribution"]:
        display[column] = display[column].map(config.money)

    display["roi"] = display["roi"].map(
        lambda v: f"{v:.2f}" if pd.notna(v) else "—"
    )

    display = display[[
        "Promotion_Type", "Discount_Percent", "bookings",
        "revenue", "discount_cost", "net_contribution", "roi",
    ]]

    display.columns = [
        "Type", "Discount", "Bookings", "Revenue",
        "Discount cost", "Net contribution", "Return per rupee",
    ]

    st.dataframe(display, hide_index=True, use_container_width=True)

    st.download_button(
        "Download",
        roi.to_csv(index=False).encode(),
        "promotion_roi.csv",
        "text/csv",
    )


# ============================================================
# WHERE IT GOES
# ============================================================

section("Who receives the discount", "Is the budget going to the right people?")

left, right = st.columns(2, gap="large")

with left:
    by_persona = (
        completed_promoted.groupby("Persona", as_index=False)
        .apply(lambda g: pd.Series({
            "discount": (g["Seasonal_Price"] - g["Final_Price"]).sum()
        }), include_groups=False)
        .reset_index(drop=True)
        if not completed_promoted.empty else pd.DataFrame()
    )

    if not by_persona.empty:
        by_persona["Persona"] = (
            completed_promoted.groupby("Persona").size().index
        )
        st.plotly_chart(
            charts.ranked_bar(
                by_persona, "Persona", "discount",
                value_format=",.0f", color_by_persona=True,
            ),
            use_container_width=True,
            config={"displayModeBar": False},
        )
        st.caption("Discount spend by persona")

with right:
    if not uplift.empty:
        st.plotly_chart(
            charts.ranked_bar(
                uplift, "Persona", "uplift",
                value_format=".1%", color_by_persona=True,
            ),
            use_container_width=True,
            config={"displayModeBar": False},
        )
        st.caption("Uplift by persona — compare against spend on the left")


# ============================================================
# ACTION
# ============================================================

if not uplift.empty:
    best = uplift.iloc[0]
    worst = uplift.iloc[-1]

    recommended_action(
        f"Move budget from <b>{worst['Persona']}</b> toward "
        f"<b>{best['Persona']}</b>. The same rupee of discount buys "
        f"{best['uplift']:+.1%} of extra conversion on one and "
        f"{worst['uplift']:+.1%} on the other. About "
        f"{config.money(waste['wasted_spend'])} of current spend is going to "
        f"customers who would have booked without any offer — model the "
        f"reallocation in the Strategy Lab before committing to it.",
        severity="warn",
    )
