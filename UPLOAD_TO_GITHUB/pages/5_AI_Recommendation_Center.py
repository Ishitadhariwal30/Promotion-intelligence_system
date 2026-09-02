"""
AI Recommendation Center — what should we offer this specific customer?

The heart of the platform. Describe a customer or pick one by ID, and
the model scores all 17 offers and explains its choice.

The filters select a **real** customer rather than constructing one. A
synthetic row would need values for all 57 features, and any combination
we invented might be one the model has never seen — it would answer
confidently regardless. Every prediction here is on somebody who
actually existed on that day.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

import config
from components import charts, customer_profile, recommendation_card
from components.metric_cards import page_header, recommended_action, section
from services import data_loader, model_service, recommendation_service

st.set_page_config(page_title="AI Recommendation Center", page_icon="◆", layout="wide")

stylesheet = config.APP_ROOT / "assets" / "styles.css"
if stylesheet.exists():
    st.markdown(f"<style>{stylesheet.read_text(encoding='utf-8')}</style>",
                unsafe_allow_html=True)


# ============================================================
# LOAD
# ============================================================

try:
    training = data_loader.load_training_features()
    state_year_end = data_loader.load_customer_state_year_end()
    bookings = data_loader.load_bookings()
    promotions = data_loader.load_promotions()
    recommendations = data_loader.load_recommendations()
except Exception as error:
    st.error(str(error))
    st.stop()


page_header(
    "AI Recommendation Center",
    "What should we offer this customer?",
    "All 17 possible offers scored, ranked on expected profit rather "
    "than booking probability — ranking on probability would always pick "
    "the largest discount.",
)


# ============================================================
# SELECT
# ============================================================

single_tab, cohort_tab = st.tabs(["One customer", "A whole segment"])


with single_tab:

    st.markdown("##### Describe the customer")

    cols = st.columns(5)

    city = cols[0].selectbox("City", ["Any"] + config.CITIES, key="rc_city")
    persona = cols[1].selectbox("Persona", ["Any"] + config.PERSONAS, key="rc_persona")
    segment = cols[2].selectbox("Segment", ["Any"] + config.CUSTOMER_SEGMENTS, key="rc_segment")
    membership = cols[3].selectbox("Membership", ["Any"] + config.MEMBERSHIPS, key="rc_member")

    services = sorted(training["Service_Name"].dropna().unique().tolist())
    service = cols[4].selectbox("Service", ["Any"] + services, key="rc_service")

    matches = recommendation_service.filter_customers(
        training, city=city, persona=persona, segment=segment,
        service=service, membership=membership,
    )

    if matches.empty:
        st.warning(
            "No customer matches that description. Set one or more "
            "dropdowns back to **Any**."
        )
        st.stop()

    st.caption(
        f"{len(matches):,} customers match. Showing the first — pick a "
        f"specific one below."
    )

    customer_ids = sorted(matches["Customer_ID"].unique().tolist())

    chosen = st.selectbox(
        "Customer",
        options=customer_ids,
        key="rc_customer",
        help="Type to search.",
    )

    row = matches[matches["Customer_ID"] == chosen].sort_values(
        "Activity_Date"
    ).iloc[-1]

    # --- profile ------------------------------------------------

    state_rows = state_year_end[state_year_end["Customer_ID"] == chosen]

    if not state_rows.empty:
        with st.expander("Customer profile", expanded=False):
            customer_profile.header(state_rows.iloc[0])
            customer_profile.metrics(state_rows.iloc[0])

    # --- the recommendation --------------------------------------

    st.divider()

    with st.spinner("Scoring 17 offers…"):
        result = recommendation_service.recommend(row)

    recommendation_card.render(result)

    # --- context --------------------------------------------------

    with st.expander("What the model saw"):

        left, right = st.columns(2)

        left.markdown(f"""
        **Event**
        - Date: {pd.Timestamp(row['Activity_Date']):%d %b %Y}
        - Service: {row['Service_Name']} ({config.money(row['Seasonal_Price'])})
        - City: {row['City']}
        - Season: {row['Season']}, {row['Weather_Condition']}
        - Holiday: {row['Holiday_Name'] if pd.notna(row.get('Holiday_Name')) else 'none'}
        """)

        right.markdown(f"""
        **Customer standing that day**
        - Segment: {row.get('Customer_Segment', '—')}
        - Loyalty: {row.get('Loyalty_Score', 0):.0f}
        - Completed bookings: {row.get('Completed_To_Date', 0):.0f}
        - Promotions received: {row.get('Promotions_Received_To_Date', 0):.0f}
        - Within repeat gap: {'yes' if row.get('Within_Repeat_Gap') else 'no'}
        """)

        st.caption(
            "Every value is as at the morning of that date. Nothing that "
            "happened afterwards is visible to the model."
        )


# ============================================================
# COHORT
# ============================================================

with cohort_tab:

    st.markdown("##### Describe the segment")

    cols = st.columns(3)

    c_city = cols[0].multiselect("Cities", config.CITIES, key="rcc_city")
    c_persona = cols[1].multiselect("Personas", config.PERSONAS, key="rcc_persona")
    c_segment = cols[2].multiselect("Segments", config.CUSTOMER_SEGMENTS, key="rcc_segment")

    cohort = training[training["Data_Split"] == "test"]

    for column, values in [
        ("City", c_city), ("Persona", c_persona),
        ("Customer_Segment", c_segment),
    ]:
        if values:
            cohort = cohort[cohort[column].astype(str).isin(values)]

    if cohort.empty:
        st.warning("No customers match. Clear a filter.")
        st.stop()

    limit = st.slider(
        "Customers to score",
        min_value=100, max_value=min(5000, len(cohort)),
        value=min(1000, len(cohort)), step=100,
        help="Scoring is fast, but a smaller sample keeps the page responsive.",
    )

    sample = cohort.head(limit)

    with st.spinner(f"Scoring {len(sample):,} customers against 17 offers each…"):
        batch = recommendation_service.recommend_batch(sample)

    recommendation_card.cohort_summary(batch)

    section("What it recommends", "Which offers, and to how many?")

    by_offer = (
        batch.groupby("offer", as_index=False)
        .agg(
            customers=("Customer_ID", "count"),
            avg_uplift=("uplift", "mean"),
            total_profit=("expected_profit", "sum"),
        )
        .sort_values("customers", ascending=False)
    )

    left, right = st.columns([3, 2], gap="large")

    with left:
        st.plotly_chart(
            charts.ranked_bar(
                by_offer, "offer", "customers", value_format=",.0f", top_n=10,
            ),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    with right:
        display = by_offer.copy()
        display["avg_uplift"] = display["avg_uplift"].map(lambda v: f"{v:+.1%}")
        display["total_profit"] = display["total_profit"].map(config.money)
        display.columns = ["Offer", "Customers", "Avg uplift", "Expected profit"]
        st.dataframe(display, hide_index=True, use_container_width=True)

    section("By persona", "Is it treating segments differently?")

    by_persona = (
        batch.groupby("Persona", as_index=False)
        .agg(
            customers=("Customer_ID", "count"),
            promoted_share=("discount_percent", lambda s: (s > 0).mean()),
            avg_discount=("discount_percent", "mean"),
            avg_uplift=("uplift", "mean"),
        )
        .sort_values("promoted_share", ascending=False)
    )

    st.plotly_chart(
        charts.ranked_bar(
            by_persona, "Persona", "promoted_share",
            value_format=".1%", color_by_persona=True,
        ),
        use_container_width=True,
        config={"displayModeBar": False},
    )

    top = by_persona.iloc[0]
    bottom = by_persona.iloc[-1]

    recommended_action(
        f"The engine treats these segments differently — "
        f"{config.percent(top['promoted_share'])} of <b>{top['Persona']}</b> "
        f"warrant an offer against {config.percent(bottom['promoted_share'])} "
        f"of <b>{bottom['Persona']}</b>. That difference is where the value "
        f"is. A blanket campaign spends the same on both and captures the "
        f"uplift of neither.",
    )

    st.download_button(
        "Download these recommendations",
        batch.to_csv(index=False).encode(),
        "recommendations.csv",
        "text/csv",
    )


# ============================================================
# INTEGRITY
# ============================================================

with st.expander("Is the model scoring correctly?"):

    st.caption(
        "The model reads encoded integers, not text. If a category mapped "
        "to a different number here than during training, every prediction "
        "would be confidently wrong with nothing to notice. This re-scores "
        "customers the app's way and compares against what Databricks "
        "already saved."
    )

    if st.button("Run the check"):
        with st.spinner("Re-scoring 200 customers…"):
            passed, deviation, message = (
                model_service.verify_against_recommendations(
                    training, recommendations
                )
            )

        if passed:
            st.success(message)
        else:
            st.error(message)
