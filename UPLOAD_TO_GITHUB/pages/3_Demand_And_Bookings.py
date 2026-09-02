"""
Demand & Bookings — how is demand behaving, and can we serve it?

Three tabs: demand patterns, the conditions driving them, and supply.

Weather sits here as a *dimension* rather than as its own page. Every
weather chart is an existing metric sliced by weather, so a separate
page would duplicate this one while being thinner than it.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

import config
from components import charts, filters as filter_bar
from components.metric_cards import (
    metric_row, page_header, recommended_action, section,
)
from services import analytics_service as analytics
from services import data_loader

st.set_page_config(page_title="Demand & Bookings", page_icon="◆", layout="wide")

stylesheet = config.APP_ROOT / "assets" / "styles.css"
if stylesheet.exists():
    st.markdown(f"<style>{stylesheet.read_text(encoding='utf-8')}</style>",
                unsafe_allow_html=True)


# ============================================================
# HELPERS
# ============================================================

def weekend_lift(frame: pd.DataFrame) -> str:
    """Weekend bookings per day against weekday bookings per day.

    Per day rather than in total, because there are five weekdays for
    every two weekend days and a raw comparison would say weekends are
    quiet when they are the opposite.
    """

    completed = frame[frame["Booking_Status"] == "Completed"]

    if completed.empty or "Is_Weekend" not in completed.columns:
        return "—"

    grouped = completed.groupby("Is_Weekend").agg(
        bookings=("Booking_ID", "count"),
        days=("Booking_Date", "nunique"),
    )

    if len(grouped) < 2:
        return "—"

    per_day = grouped["bookings"] / grouped["days"]

    weekday = per_day.get(False)
    weekend = per_day.get(True)

    if not weekday:
        return "—"

    return f"{weekend / weekday:.2f}×"


# ============================================================
# LOAD
# ============================================================

bookings_all = data_loader.load_bookings()
activity_all = data_loader.load_activity_daily()
providers = data_loader.load_providers()

active = filter_bar.render(show_weather=True, show_segments=False)

bookings = analytics.apply_filters(bookings_all, active, "Booking_Date")
activity = analytics.apply_filters(activity_all, active, "Activity_Date")


page_header(
    "Demand & Bookings",
    "How is demand behaving, and can we serve it?",
    f"Showing {active.describe()}.",
)

if bookings.empty:
    st.warning("No bookings match this selection. Widen the filters.")
    st.stop()


demand_tab, conditions_tab, supply_tab = st.tabs(
    ["Demand patterns", "What drives demand", "Supply"]
)


# ============================================================
# DEMAND
# ============================================================

with demand_tab:

    completed = bookings[bookings["Booking_Status"] == "Completed"]

    days = pd.to_datetime(bookings["Booking_Date"]).dt.normalize().nunique() or 1

    metric_row([
        {
            "label": "Bookings per day",
            "value": f"{len(completed) / days:,.0f}",
            "caption": f"across {days} days",
        },
        {
            "label": "Peak day",
            "value": (
                completed.groupby("Booking_Date").size().max()
                if not completed.empty else 0
            ),
            "caption": "highest single day",
        },
        {
            "label": "Weekend lift",
            "value": weekend_lift(bookings),
            "caption": "vs a weekday",
        },
        {
            "label": "Services booked",
            "value": config.count(completed["Service_Name"].nunique()),
            "caption": "distinct",
        },
    ], columns=4)

    section("Daily volume", "When did demand spike?")

    daily = (
        completed.groupby("Booking_Date", as_index=False)
        .agg(bookings=("Booking_ID", "count"))
    )

    holiday_days = set(
        completed[completed["Holiday_Name"].notna()]["Booking_Date"]
    )

    st.plotly_chart(
        charts.daily_trend(
            daily, "Booking_Date", "bookings", "Bookings",
            highlight=daily["Booking_Date"].isin(holiday_days),
        ),
        use_container_width=True,
        config={"displayModeBar": False},
    )

    left, right = st.columns(2, gap="large")

    with left:
        section("Top services", "What are people actually buying?")
        st.plotly_chart(
            charts.ranked_bar(
                analytics.revenue_by_dimension(bookings, "Service_Name"),
                "Service_Name", "bookings", value_format=",.0f", top_n=10,
            ),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    with right:
        section("When they book", "Which slots are busiest?")
        heat = analytics.demand_heatmap(bookings)
        st.plotly_chart(
            charts.heatmap(heat, "Booking_Window", "Day_Of_Week", "bookings"),
            use_container_width=True,
            config={"displayModeBar": False},
        )


# ============================================================
# CONDITIONS
# ============================================================

with conditions_tab:

    section("Seasonality", "What should we push, and when?")

    seasonal = analytics.seasonality(bookings)

    st.plotly_chart(
        charts.heatmap(seasonal, "Season", "Service_Name", "bookings"),
        use_container_width=True,
        config={"displayModeBar": False},
    )

    st.caption(
        "AC Repair concentrates in Summer, Pest Control in Monsoon, Deep "
        "Home Cleaning across Festival and Winter. These are demand "
        "patterns the marketplace should staff against, not discount into."
    )

    section("Weather", "Which services should we staff up when it rains?")

    weather = analytics.weather_impact(bookings)

    if weather.empty:
        st.info("No weather data in this selection.")
    else:
        top_services = (
            weather.groupby("Service_Name")["bookings"].sum()
            .nlargest(6).index.tolist()
        )

        st.plotly_chart(
            charts.grouped_bar(
                weather[weather["Service_Name"].isin(top_services)],
                "Weather_Condition", "bookings", "Service_Name",
                value_format=",.0f",
            ),
            use_container_width=True,
            config={"displayModeBar": False},
        )

        st.caption(
            "Rain creates the problems Plumbing and Pest Control solve, and "
            "suppresses discretionary bookings like Salon at Home. Supply "
            "should follow, not price."
        )

    section("Holidays", "Which days deserve extra capacity?")

    holidays = analytics.holiday_impact(bookings)

    if holidays.empty:
        st.info("No holidays fall inside this selection.")
    else:
        st.plotly_chart(
            charts.ranked_bar(
                holidays, "day_type", "avg_bookings",
                value_format=",.0f", horizontal=True,
            ),
            use_container_width=True,
            config={"displayModeBar": False},
        )

        peak = holidays[holidays["day_type"] != "Ordinary day"]

        if not peak.empty:
            top = peak.iloc[0]
            recommended_action(
                f"<b>{top['day_type']}</b> runs at {top['vs_ordinary']:.1f}× an "
                f"ordinary day. Demand concentrates rather than spreading, so "
                f"capacity is the constraint, not marketing. Discounting into "
                f"a spike is the least efficient possible use of budget — "
                f"those customers were coming anyway. Spend it on the quiet "
                f"weeks instead.",
                severity="neutral",
            )


# ============================================================
# SUPPLY
# ============================================================

with supply_tab:

    utilisation = analytics.provider_utilisation(bookings, providers)

    active_providers = utilisation[utilisation["Provider_Status"] == "Active"]

    metric_row([
        {
            "label": "Active providers",
            "value": config.count(len(active_providers)),
        },
        {
            "label": "Average utilisation",
            "value": config.percent(active_providers["utilisation"].mean()),
            "caption": "of daily capacity",
        },
        {
            "label": "Idle providers",
            "value": config.count((active_providers["utilisation"] < 0.1).sum()),
            "caption": "under 10% utilised",
            "severity": "warn",
        },
        {
            "label": "Customers turned away",
            "value": config.count(
                (bookings["Booking_Status"] == "No_Provider").sum()
            ),
            "caption": "nobody available",
            "severity": "bad",
        },
    ], columns=4)

    section("Utilisation spread", "Are we short of providers, or carrying idle ones?")

    st.plotly_chart(
        charts.utilisation_distribution(active_providers),
        use_container_width=True,
        config={"displayModeBar": False},
    )

    st.caption(
        "Shown as a distribution, not an average. A marketplace where half "
        "the providers are saturated and half idle averages to 'fine' and is "
        "nothing of the sort."
    )

    section("Where supply fails", "Where should we recruit?")

    gaps = analytics.supply_gaps(bookings, providers)

    if gaps.empty:
        st.success("No bookings were lost to missing providers in this selection.")
    else:
        display = gaps.head(15).copy()
        display["lost_revenue"] = display["lost_revenue"].map(config.money)
        display.columns = [
            "City", "Service", "Lost bookings", "Lost revenue", "Active providers",
        ]

        st.dataframe(display, hide_index=True, use_container_width=True)

        uncovered = gaps[gaps["active_providers"] == 0]

        recommended_action(
            f"{config.money(gaps['lost_revenue'].sum())} of demand went "
            f"unserved, with <b>{len(uncovered)} city/service combinations "
            f"having no active provider at all</b>. These are not a marketing "
            f"problem — the customer wanted to book and the marketplace could "
            f"not serve them. No promotion, targeting change or discount "
            f"recovers them. The fix is recruitment.",
            severity="bad",
        )

        st.download_button(
            "Download the gap list",
            gaps.to_csv(index=False).encode(),
            "supply_gaps.csv",
            "text/csv",
        )
