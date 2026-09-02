"""
Customer Intelligence — who are our customers, and what has happened to
them?

Two tabs. **Portfolio** looks across the base; **Individual** looks at
one person, including their full lifecycle timeline.

The Journey lives here rather than on its own page because both start
with "select a customer", and splitting them would mean selecting the
same customer twice on two different screens.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

import config
from components import charts, customer_profile, filters as filter_bar
from components.metric_cards import (
    metric_row, page_header, recommended_action, section,
)
from services import analytics_service as analytics
from services import data_loader

st.set_page_config(page_title="Customer Intelligence", page_icon="◆", layout="wide")

stylesheet = config.APP_ROOT / "assets" / "styles.css"
if stylesheet.exists():
    st.markdown(f"<style>{stylesheet.read_text(encoding='utf-8')}</style>",
                unsafe_allow_html=True)


# ============================================================
# LOAD
# ============================================================

state_year_end = data_loader.load_customer_state_year_end()
bookings = data_loader.load_bookings()
promotions = data_loader.load_promotions()
recommendations = data_loader.load_recommendations()

active = filter_bar.render(show_date=False)

state = analytics.apply_filters(state_year_end, active, "State_Date")


page_header(
    "Customer Intelligence",
    "Who are our customers?",
    "Customer state as at 31 December 2025. "
    f"Showing {active.describe()}.",
)

if state.empty:
    st.warning("No customers match this selection.")
    st.stop()


portfolio_tab, individual_tab = st.tabs(["Portfolio", "Individual customer"])


# ============================================================
# PORTFOLIO
# ============================================================

with portfolio_tab:

    metric_row([
        {
            "label": "Customers",
            "value": config.count(len(state)),
        },
        {
            "label": "Total lifetime value",
            "value": config.money(state["Lifetime_Spend"].sum()),
        },
        {
            "label": "Average per customer",
            "value": config.money(state["Lifetime_Spend"].mean()),
        },
        {
            "label": "Gone quiet",
            "value": config.percent(
                (state["Customer_Segment"] == "Dormant").mean()
            ),
            "caption": "no attempt in 45+ days",
            "severity": "warn",
        },
    ], columns=4)

    section("Segment mix", "Who is worth investing in?")

    left, right = st.columns([2, 3], gap="large")

    segments = analytics.segment_summary(state)

    with left:
        st.plotly_chart(
            charts.donut(segments, "Customer_Segment", "customers"),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    with right:
        display = segments.copy()
        display["avg_spend"] = display["avg_spend"].map(config.money)
        display["total_spend"] = display["total_spend"].map(config.money)
        display["avg_promo_response"] = display["avg_promo_response"].map(
            lambda v: config.percent(v) if pd.notna(v) else "—"
        )
        display.columns = [
            "Segment", "Customers", "Avg spend", "Total spend",
            "Avg bookings", "Avg loyalty", "Promo response",
        ]
        st.dataframe(display, hide_index=True, use_container_width=True)

    section("Value by persona", "Which behavioural types drive revenue?")

    by_persona = (
        state.groupby("Persona", as_index=False)
        .agg(
            customers=("Customer_ID", "count"),
            total_spend=("Lifetime_Spend", "sum"),
            avg_spend=("Lifetime_Spend", "mean"),
            avg_response=("Promotion_Response_Rate", "mean"),
        )
    )

    left, right = st.columns(2, gap="large")

    with left:
        st.plotly_chart(
            charts.ranked_bar(
                by_persona, "Persona", "total_spend",
                value_format=",.0f", color_by_persona=True,
            ),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    with right:
        st.plotly_chart(
            charts.ranked_bar(
                by_persona, "Persona", "avg_response",
                value_format=".1%", color_by_persona=True,
            ),
            use_container_width=True,
            config={"displayModeBar": False},
        )
        st.caption(
            "Promotion response rate — the share of offers each persona "
            "has historically redeemed. The strongest available predictor "
            "of how they will respond next time."
        )

    section("Who we are about to lose", "Which valuable customers have gone quiet?")

    at_risk = analytics.at_risk_customers(state)

    if at_risk.empty:
        st.success("No high-value customers have gone quiet in this selection.")
    else:
        st.markdown(
            f"**{len(at_risk):,} customers** worth "
            f"{config.money(at_risk['Lifetime_Spend'].sum())} have not "
            f"attempted a booking in over 45 days."
        )

        display = at_risk.head(20)[[
            "Customer_ID", "City", "Persona", "Customer_Segment",
            "Lifetime_Spend", "Completed_To_Date",
            "Days_Since_Last_Attempt", "Promotion_Response_Rate",
        ]].copy()

        display["Lifetime_Spend"] = display["Lifetime_Spend"].map(config.money)
        display["Promotion_Response_Rate"] = display["Promotion_Response_Rate"].map(
            lambda v: config.percent(v) if pd.notna(v) else "never offered"
        )
        display.columns = [
            "Customer", "City", "Persona", "Segment", "Lifetime spend",
            "Bookings", "Days quiet", "Past response",
        ]

        st.dataframe(display, hide_index=True, use_container_width=True)

        st.download_button(
            "Download the full list",
            at_risk.to_csv(index=False).encode(),
            "at_risk_customers.csv",
            "text/csv",
        )

        recommended_action(
            f"Prioritise by <b>spend</b>, not by how long they have been "
            f"away. A customer worth {config.money(at_risk['Lifetime_Spend'].max())} "
            f"who left three months ago matters more than one worth "
            f"{config.money(at_risk['Lifetime_Spend'].min())} who left six. "
            f"Check past response rate before discounting — some of these "
            f"have never redeemed an offer, and a discount is not what will "
            f"bring them back.",
            severity="warn",
        )


# ============================================================
# INDIVIDUAL
# ============================================================

with individual_tab:

    customers = sorted(state["Customer_ID"].unique().tolist())

    chosen = customer_profile.selector(customers, key="ci_customer")

    row = state[state["Customer_ID"] == chosen].iloc[0]

    customer_profile.header(row)

    customer_profile.metrics(row)

    customer_profile.engagement_gap(row)

    section("Lifecycle", "What has happened to this customer?")

    st.caption(
        "Promotions, bookings and platform recommendations merged into one "
        "sequence — the pipeline read as a story rather than as tables."
    )

    customer_profile.timeline(
        chosen, bookings, promotions, recommendations,
    )

    with st.expander("How their standing accumulated"):

        progression = customer_profile.state_progression(
            chosen, data_loader.load_customer_state_daily()
        )

        if progression.empty:
            st.info("No daily state recorded for this customer.")
        else:
            st.plotly_chart(
                charts.daily_trend(
                    progression, "State_Date", "Loyalty_Score", "Loyalty score"
                ),
                use_container_width=True,
                config={"displayModeBar": False},
            )
            st.caption(
                "The line only ever moves forward. That is what "
                "point-in-time features mean: a prediction made in March "
                "could not have seen November."
            )
