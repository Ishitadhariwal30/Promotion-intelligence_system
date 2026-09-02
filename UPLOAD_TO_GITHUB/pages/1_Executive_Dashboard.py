"""
Executive Dashboard — how is the business performing, and what needs
attention?

Alerts sit above the metrics deliberately. Nine healthy-looking numbers
at the top of a page bury the one thing worth acting on, and a manager
with three minutes reads top to bottom.

This page is summary only. Drill-down lives on Demand & Bookings, so
nothing is duplicated between the two.
"""

from __future__ import annotations

import streamlit as st

import config
from components import charts, copilot, filters as filter_bar
from components.metric_cards import (
    alert_strip, data_quality_panel, metric_row,
    page_header, recommended_action, section,
)
from services import analytics_service as analytics
from services import data_loader

st.set_page_config(page_title="Executive Dashboard", page_icon="◆", layout="wide")

stylesheet = config.APP_ROOT / "assets" / "styles.css"
if stylesheet.exists():
    st.markdown(f"<style>{stylesheet.read_text(encoding='utf-8')}</style>",
                unsafe_allow_html=True)


# ============================================================
# LOAD
# ============================================================

bookings_all = data_loader.load_bookings()
activity_all = data_loader.load_activity_daily()
promotions_all = data_loader.load_promotions()
providers = data_loader.load_providers()
state_year_end = data_loader.load_customer_state_year_end()
manifest = data_loader.load_manifest()

active = filter_bar.render(show_weather=False)

bookings = analytics.apply_filters(bookings_all, active, "Booking_Date")
activity = analytics.apply_filters(activity_all, active, "Activity_Date")
promotions = analytics.apply_filters(promotions_all, active, "Promotion_Date")


# ============================================================
# HEADER
# ============================================================

page_header(
    "Executive Dashboard",
    "How is the business performing?",
    f"Showing {active.describe()}.",
)

if bookings.empty:
    st.warning("No bookings match this selection. Widen the filters.")
    st.stop()


# ============================================================
# ALERTS
# ============================================================

alert_strip(
    analytics.build_alerts(bookings, providers, promotions, state_year_end)
)


# ============================================================
# METRICS
# ============================================================

metrics = analytics.headline_metrics(bookings, activity, promotions)

metric_row([
    {
        "label": "Revenue",
        "value": config.money(metrics["revenue"]),
        "caption": "completed jobs only",
        "help_text": "Cancelled bookings were refunded and No_Provider never transacted.",
    },
    {
        "label": "Completed bookings",
        "value": config.count(metrics["bookings_completed"]),
        "caption": f"of {config.count(metrics['booking_attempts'])} attempts",
    },
    {
        "label": "Average order value",
        "value": config.money(metrics["avg_order_value"]),
        "caption": "after discount",
    },
    {
        "label": "Customers",
        "value": config.count(metrics["customers"]),
        "caption": "with at least one attempt",
    },
], columns=4)

metric_row([
    {
        "label": "Conversion",
        "value": config.percent(metrics["conversion_rate"]),
        "caption": "of customers who wanted a service",
        "help_text": "Booking attempts divided by service-need events.",
    },
    {
        "label": "Completion",
        "value": config.percent(metrics["completion_rate"]),
        "caption": "of attempts delivered",
        "severity": "good" if metrics["completion_rate"] > 0.88 else "warn",
    },
    {
        "label": "Cancellations",
        "value": config.percent(metrics["cancellation_rate"]),
        "caption": "customer changed their mind",
        "severity": "warn" if metrics["cancellation_rate"] > config.ALERT_CANCELLATION_RATE else "neutral",
    },
    {
        "label": "Turned away",
        "value": config.percent(metrics["no_provider_rate"]),
        "caption": "no provider available",
        "severity": "bad" if metrics["no_provider_rate"] > config.ALERT_NO_PROVIDER_RATE else "neutral",
        "help_text": "A supply failure, not a customer decision. No promotion fixes these.",
    },
], columns=4)

metric_row([
    {
        "label": "Promotions sent",
        "value": config.count(metrics["promotions_sent"]),
    },
    {
        "label": "Redemption",
        "value": config.percent(metrics["promotion_redemption_rate"]),
        "caption": "of offers led to a completed job",
    },
    {
        "label": "Discount given",
        "value": config.money(metrics["discount_given"]),
        "caption": "revenue surrendered",
        "severity": "warn",
    },
    {
        "label": "Average rating",
        "value": f"{metrics['avg_rating']:.2f}",
        "caption": "on completed jobs",
    },
], columns=4)


# ============================================================
# TREND
# ============================================================

section("Revenue and volume", "Is the business growing, and when did it spike?")

st.plotly_chart(
    charts.revenue_trend(analytics.revenue_trend(bookings, freq="M")),
    use_container_width=True,
    config={"displayModeBar": False},
)


# ============================================================
# FUNNEL AND MIX
# ============================================================

left, right = st.columns([2, 3], gap="large")

with left:
    section("Where we lose people", "At which stage do customers drop out?")

    funnel_frame = analytics.booking_funnel(activity, bookings)

    st.plotly_chart(
        charts.funnel(funnel_frame),
        use_container_width=True,
        config={"displayModeBar": False},
    )

with right:
    section("Where the money comes from", "Which cities and services carry revenue?")

    tab_city, tab_service = st.tabs(["By city", "By service"])

    with tab_city:
        st.plotly_chart(
            charts.ranked_bar(
                analytics.revenue_by_dimension(bookings, "City"),
                "City", "revenue", value_format=",.0f",
            ),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    with tab_service:
        st.plotly_chart(
            charts.ranked_bar(
                analytics.revenue_by_dimension(bookings, "Service_Name"),
                "Service_Name", "revenue", value_format=",.0f", top_n=10,
            ),
            use_container_width=True,
            config={"displayModeBar": False},
        )


# ============================================================
# ACTION
# ============================================================

gaps = analytics.supply_gaps(bookings, providers)

waste = analytics.promotion_waste(promotions, bookings)

if not gaps.empty and gaps["lost_revenue"].sum() > waste["wasted_spend"]:
    worst = gaps.iloc[0]
    recommended_action(
        f"The largest recoverable loss is supply, not targeting. "
        f"{config.money(gaps['lost_revenue'].sum())} of demand went unserved, "
        f"worst in <b>{worst['City']} — {worst['Service_Name']}</b> with "
        f"{int(worst['active_providers'])} active providers. Recruiting there "
        f"returns more than any promotional change.",
        severity="bad",
    )
elif waste["wasted_spend"] > 0:
    recommended_action(
        f"About {config.money(waste['wasted_spend'])} of discount went to "
        f"customers who would have booked anyway. Move that budget toward "
        f"the personas with genuine uplift — the Strategy Lab quantifies the "
        f"trade.",
        severity="warn",
    )
else:
    recommended_action(
        "No supply gaps or promotional waste above threshold in this "
        "selection. Promotion Performance shows where the remaining upside is.",
        severity="good",
    )


# ============================================================
# COPILOT AND PROVENANCE
# ============================================================

st.divider()

copilot.render({
    "bookings": bookings,
    "promotions": promotions,
    "recommendations": data_loader.load_recommendations(),
    "providers": providers,
    "activity": activity,
    "state_year_end": state_year_end,
}, compact=True)

data_quality_panel(manifest)
