"""
Every number behind every chart in the platform.

One rule governs this module: a function belongs here only if some page
asks a business question it answers. There are no generic "describe the
data" helpers, because a chart nobody can act on is a chart that should
not be drawn.

Contains no Streamlit imports. Everything takes DataFrames and returns
DataFrames or plain values, which keeps it testable and means the
Strategy Lab can reuse the same maths the Executive Dashboard uses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

import config


# ============================================================
# FILTERING
# ============================================================

@dataclass
class Filters:
    """Global filter state, shared across every page.

    ``None`` means no filter on that dimension. Empty list means the
    same thing, so a user who clears a multiselect sees everything
    rather than nothing - which is what they expect and almost never
    what an unguarded filter does.
    """

    date_from: pd.Timestamp | None = None
    date_to: pd.Timestamp | None = None
    cities: list[str] | None = None
    personas: list[str] | None = None
    memberships: list[str] | None = None
    segments: list[str] | None = None
    services: list[str] | None = None
    weather: list[str] | None = None
    holidays_only: bool = False

    def is_active(self) -> bool:
        """Whether anything is actually narrowed."""
        return any([
            self.date_from is not None,
            self.date_to is not None,
            self.cities, self.personas, self.memberships,
            self.segments, self.services, self.weather,
            self.holidays_only,
        ])

    def describe(self) -> str:
        """Human summary, used to caption charts and copilot answers.

        A number without its filter context is not a number a manager
        can act on.
        """

        parts: list[str] = []

        if self.date_from is not None and self.date_to is not None:
            parts.append(
                f"{self.date_from:%d %b} to {self.date_to:%d %b %Y}"
            )

        for label, values in [
            ("city", self.cities), ("persona", self.personas),
            ("membership", self.memberships), ("segment", self.segments),
            ("service", self.services), ("weather", self.weather),
        ]:
            if values:
                joined = ", ".join(values[:3])
                more = f" +{len(values) - 3}" if len(values) > 3 else ""
                parts.append(f"{label}: {joined}{more}")

        if self.holidays_only:
            parts.append("holidays only")

        return " · ".join(parts) if parts else "all data"


def apply_filters(
    frame: pd.DataFrame,
    filters: Filters,
    date_column: str = "Booking_Date",
) -> pd.DataFrame:
    """Narrow a frame to the active filter selection.

    Silently skips any dimension the frame does not carry, so the same
    Filters object works across tables with different shapes.
    """

    if frame.empty:
        return frame

    mask = pd.Series(True, index=frame.index)

    if date_column in frame.columns:
        dates = pd.to_datetime(frame[date_column], errors="coerce")
        if filters.date_from is not None:
            mask &= dates >= filters.date_from
        if filters.date_to is not None:
            mask &= dates <= filters.date_to

    for column, values in [
        ("City", filters.cities),
        ("Persona", filters.personas),
        ("Membership", filters.memberships),
        ("Customer_Segment", filters.segments),
        ("Service_Name", filters.services),
        ("Weather_Condition", filters.weather),
    ]:
        if values and column in frame.columns:
            mask &= frame[column].astype(str).isin(values)

    if filters.holidays_only and "Holiday_Name" in frame.columns:
        mask &= frame["Holiday_Name"].notna()

    return frame[mask]


# ============================================================
# HEADLINE PERFORMANCE
# ============================================================

def headline_metrics(
    bookings: pd.DataFrame,
    activity: pd.DataFrame,
    promotions: pd.DataFrame,
) -> dict[str, float]:
    """The numbers on the Executive Dashboard.

    Answers: *how is the business performing?*

    Revenue counts completed jobs only. Cancelled bookings were
    refunded and No_Provider bookings never transacted, so including
    either would overstate the top line.
    """

    completed = bookings[bookings["Booking_Status"] == "Completed"]

    attempts = len(bookings)
    needed = float(activity["Needed_Service"].sum()) if not activity.empty else 0

    revenue = float(completed["Final_Price"].sum()) if not completed.empty else 0

    promotions_sent = (
        int(promotions["Promotion_Sent"].sum()) if not promotions.empty else 0
    )

    redeemed = (
        int(bookings["Promotion_Redeemed"].sum())
        if "Promotion_Redeemed" in bookings.columns else 0
    )

    return {
        "revenue": revenue,
        "bookings_completed": len(completed),
        "booking_attempts": attempts,
        "customers": int(bookings["Customer_ID"].nunique()) if attempts else 0,
        "service_need_events": needed,
        "conversion_rate": attempts / needed if needed else 0.0,
        "completion_rate": len(completed) / attempts if attempts else 0.0,
        "cancellation_rate": (
            float((bookings["Booking_Status"] == "Cancelled").mean())
            if attempts else 0.0
        ),
        "no_provider_rate": (
            float((bookings["Booking_Status"] == "No_Provider").mean())
            if attempts else 0.0
        ),
        "avg_order_value": (
            float(completed["Final_Price"].mean()) if not completed.empty else 0.0
        ),
        "avg_rating": (
            float(completed["Rating"].mean()) if not completed.empty else 0.0
        ),
        "promotions_sent": promotions_sent,
        "promotion_redemption_rate": (
            redeemed / promotions_sent if promotions_sent else 0.0
        ),
        "discount_given": (
            float((completed["Seasonal_Price"] - completed["Final_Price"]).sum())
            if not completed.empty else 0.0
        ),
    }


def revenue_trend(
    bookings: pd.DataFrame,
    freq: Literal["D", "W", "M"] = "M",
) -> pd.DataFrame:
    """Revenue and bookings over time.

    Answers: *is the business growing, and when did it spike?*
    """

    completed = bookings[bookings["Booking_Status"] == "Completed"].copy()

    if completed.empty:
        return pd.DataFrame(columns=["period", "revenue", "bookings", "aov"])

    completed["period"] = (
        pd.to_datetime(completed["Booking_Date"]).dt.to_period(freq).dt.to_timestamp()
    )

    trend = completed.groupby("period", as_index=False).agg(
        revenue=("Final_Price", "sum"),
        bookings=("Booking_ID", "count"),
    )

    trend["aov"] = trend["revenue"] / trend["bookings"]

    return trend


def booking_funnel(
    activity: pd.DataFrame,
    bookings: pd.DataFrame,
) -> pd.DataFrame:
    """Where customers drop out between intent and delivery.

    Answers: *where are we losing people?*

    The last two stages separate deliberately. A customer who wanted to
    book and found no provider is a supply failure, not a demand one,
    and the fixes are completely different.
    """

    opened = float(activity["Opened_App"].sum()) if not activity.empty else 0
    needed = float(activity["Needed_Service"].sum()) if not activity.empty else 0

    attempted = len(bookings)
    assigned = int((bookings["Booking_Status"] != "No_Provider").sum())
    completed = int((bookings["Booking_Status"] == "Completed").sum())

    stages = [
        ("Opened the app", opened),
        ("Wanted a service", needed),
        ("Decided to book", attempted),
        ("Provider assigned", assigned),
        ("Job completed", completed),
    ]

    frame = pd.DataFrame(stages, columns=["stage", "customers"])

    top = frame["customers"].iloc[0] or 1

    frame["share_of_top"] = frame["customers"] / top
    frame["dropped"] = frame["customers"].shift(1) - frame["customers"]
    frame["drop_rate"] = frame["dropped"] / frame["customers"].shift(1)

    return frame


def revenue_by_dimension(
    bookings: pd.DataFrame,
    dimension: str,
    top_n: int | None = None,
) -> pd.DataFrame:
    """Revenue, volume and order value split by any column.

    Answers: *where does the money come from?*
    """

    completed = bookings[bookings["Booking_Status"] == "Completed"]

    if completed.empty or dimension not in completed.columns:
        return pd.DataFrame(columns=[dimension, "revenue", "bookings", "aov"])

    grouped = completed.groupby(dimension, as_index=False).agg(
        revenue=("Final_Price", "sum"),
        bookings=("Booking_ID", "count"),
        avg_rating=("Rating", "mean"),
    )

    grouped["aov"] = grouped["revenue"] / grouped["bookings"]

    grouped = grouped.sort_values("revenue", ascending=False)

    return grouped.head(top_n) if top_n else grouped


# ============================================================
# PROMOTION PERFORMANCE
# ============================================================

def promotion_uplift(
    promotions: pd.DataFrame,
    bookings: pd.DataFrame,
    by: str = "Persona",
    exploration_only: bool = False,
) -> pd.DataFrame:
    """Conversion with and without a promotion.

    Answers: *what is a promotion actually worth, and to whom?*

    Args:
        exploration_only: Restrict to the randomised subset. Outside it
            the comparison is confounded - promoted customers were
            chosen *because* they looked responsive, so some of the
            apparent uplift is selection rather than treatment.
    """

    frame = promotions.copy()

    if exploration_only and "Targeting_Mode" in frame.columns:
        frame = frame[frame["Targeting_Mode"] == "Exploration"]

    if frame.empty or by not in frame.columns:
        return pd.DataFrame(
            columns=[by, "not_promoted", "promoted", "uplift", "relative_lift"]
        )

    booked_ids = set(bookings["Activity_ID"])

    frame["booked"] = frame["Activity_ID"].isin(booked_ids).astype(int)

    pivot = (
        frame.groupby([by, "Promotion_Sent"], as_index=False)
        .agg(customers=("Activity_ID", "count"), conversion=("booked", "mean"))
    )

    wide = pivot.pivot(index=by, columns="Promotion_Sent", values="conversion")

    counts = pivot.pivot(index=by, columns="Promotion_Sent", values="customers")

    result = pd.DataFrame({
        by: wide.index,
        "not_promoted": wide.get(False, pd.Series(dtype=float)).values,
        "promoted": wide.get(True, pd.Series(dtype=float)).values,
        "n_not_promoted": counts.get(False, pd.Series(dtype=float)).values,
        "n_promoted": counts.get(True, pd.Series(dtype=float)).values,
    })

    result["uplift"] = result["promoted"] - result["not_promoted"]

    result["relative_lift"] = np.where(
        result["not_promoted"] > 0,
        result["promoted"] / result["not_promoted"],
        np.nan,
    )

    return result.sort_values("uplift", ascending=False)


def promotion_roi(bookings: pd.DataFrame) -> pd.DataFrame:
    """Return on each promotion type and discount level.

    Answers: *which promotions pay for themselves?*

    Cost is the discount surrendered. Return is the gross margin on the
    revenue collected. A negative net means that promotion destroyed
    value on the bookings it was attached to.
    """

    completed = bookings[
        (bookings["Booking_Status"] == "Completed")
        & (bookings["Promotion_Offered"] == True)
    ]

    if completed.empty:
        return pd.DataFrame()

    grouped = completed.groupby(
        ["Promotion_Type", "Discount_Percent"], as_index=False
    ).agg(
        bookings=("Booking_ID", "count"),
        revenue=("Final_Price", "sum"),
        full_price_value=("Seasonal_Price", "sum"),
    )

    grouped["discount_cost"] = (
        grouped["full_price_value"] - grouped["revenue"]
    )

    grouped["gross_margin"] = grouped["revenue"] * config.GROSS_MARGIN

    grouped["net_contribution"] = (
        grouped["gross_margin"] - grouped["discount_cost"]
    )

    grouped["roi"] = np.where(
        grouped["discount_cost"] > 0,
        grouped["net_contribution"] / grouped["discount_cost"],
        np.nan,
    )

    return grouped.sort_values("net_contribution", ascending=False)


def promotion_waste(
    promotions: pd.DataFrame,
    bookings: pd.DataFrame,
) -> dict[str, float]:
    """How much promotional spend went to customers who needed no push.

    Answers: *how much of the budget is wasted?*

    Estimated from the randomised group, where baseline conversion is
    observable. Promoted customers who would have converted anyway are
    the waste: the discount changed nothing but still cost money.
    """

    exploration = promotions[
        promotions.get("Targeting_Mode", pd.Series(dtype=str)) == "Exploration"
    ]

    if exploration.empty:
        return {"baseline_conversion": 0.0, "waste_rate": 0.0, "wasted_spend": 0.0}

    booked_ids = set(bookings["Activity_ID"])

    exploration = exploration.copy()
    exploration["booked"] = exploration["Activity_ID"].isin(booked_ids)

    untreated = exploration[~exploration["Promotion_Sent"]]

    baseline = float(untreated["booked"].mean()) if not untreated.empty else 0.0

    promoted_bookings = bookings[bookings["Promotion_Offered"] == True]

    discount_spend = float(
        (promoted_bookings["Seasonal_Price"] - promoted_bookings["Final_Price"]).sum()
    ) if not promoted_bookings.empty else 0.0

    return {
        "baseline_conversion": baseline,
        "waste_rate": baseline,
        "wasted_spend": discount_spend * baseline,
        "total_spend": discount_spend,
    }


# ============================================================
# SUPPLY
# ============================================================

def provider_utilisation(
    bookings: pd.DataFrame,
    providers: pd.DataFrame,
) -> pd.DataFrame:
    """How hard each provider is working against their capacity.

    Answers: *are we short of providers, or carrying idle ones?*
    """

    if providers.empty:
        return pd.DataFrame()

    assigned = bookings[bookings["Provider_ID"].notna()]

    days = (
        pd.to_datetime(bookings["Booking_Date"]).dt.normalize().nunique()
        if not bookings.empty else 1
    ) or 1

    per_provider = (
        assigned.groupby("Provider_ID", as_index=False)
        .agg(bookings=("Booking_ID", "count"))
    )

    merged = providers.merge(per_provider, on="Provider_ID", how="left")

    merged["bookings"] = merged["bookings"].fillna(0)

    merged["capacity_over_period"] = merged["Daily_Capacity"] * days

    merged["utilisation"] = (
        merged["bookings"] / merged["capacity_over_period"]
    ).clip(upper=1.0)

    return merged.sort_values("utilisation", ascending=False)


def supply_gaps(
    bookings: pd.DataFrame,
    providers: pd.DataFrame,
) -> pd.DataFrame:
    """City and service combinations where customers were turned away.

    Answers: *where should we recruit providers?*

    These are lost bookings that have nothing to do with price,
    targeting or the customer. No promotion can fix them.
    """

    turned_away = bookings[bookings["Booking_Status"] == "No_Provider"]

    if turned_away.empty:
        return pd.DataFrame(
            columns=["City", "Service_Name", "lost_bookings",
                     "lost_revenue", "active_providers"]
        )

    gaps = turned_away.groupby(
        ["City", "Service_Name"], as_index=False
    ).agg(
        lost_bookings=("Booking_ID", "count"),
        lost_revenue=("Seasonal_Price", "sum"),
    )

    active = providers[providers["Provider_Status"] == "Active"]

    coverage = active.groupby(
        ["City", "Primary_Service_Name"], as_index=False
    ).agg(active_providers=("Provider_ID", "count"))

    coverage = coverage.rename(columns={"Primary_Service_Name": "Service_Name"})

    gaps = gaps.merge(coverage, on=["City", "Service_Name"], how="left")

    gaps["active_providers"] = gaps["active_providers"].fillna(0).astype(int)

    return gaps.sort_values("lost_revenue", ascending=False)


# ============================================================
# DEMAND CONTEXT
# ============================================================

def weather_impact(bookings: pd.DataFrame) -> pd.DataFrame:
    """Bookings and revenue by weather condition and service.

    Answers: *which services should we staff up when it rains?*
    """

    completed = bookings[bookings["Booking_Status"] == "Completed"]

    if completed.empty:
        return pd.DataFrame()

    by_weather = completed.groupby(
        ["Weather_Condition", "Service_Name"], as_index=False
    ).agg(
        bookings=("Booking_ID", "count"),
        revenue=("Final_Price", "sum"),
    )

    totals = completed.groupby("Weather_Condition", as_index=False).agg(
        weather_days=("Booking_Date", "nunique")
    )

    merged = by_weather.merge(totals, on="Weather_Condition", how="left")

    merged["bookings_per_day"] = merged["bookings"] / merged["weather_days"]

    return merged.sort_values("bookings", ascending=False)


def holiday_impact(bookings: pd.DataFrame) -> pd.DataFrame:
    """Average daily bookings on each holiday versus an ordinary day.

    Answers: *which days deserve extra capacity and budget?*

    Compared per day rather than in total, because a single holiday
    cannot out-total three hundred ordinary days no matter how large
    its spike.
    """

    completed = bookings[bookings["Booking_Status"] == "Completed"].copy()

    if completed.empty:
        return pd.DataFrame()

    completed["day_type"] = completed["Holiday_Name"].fillna("Ordinary day")

    daily = completed.groupby(
        ["Booking_Date", "day_type"], as_index=False
    ).agg(
        bookings=("Booking_ID", "count"),
        revenue=("Final_Price", "sum"),
    )

    summary = daily.groupby("day_type", as_index=False).agg(
        days=("Booking_Date", "nunique"),
        avg_bookings=("bookings", "mean"),
        avg_revenue=("revenue", "mean"),
    )

    ordinary = summary[summary["day_type"] == "Ordinary day"]

    baseline = (
        float(ordinary["avg_bookings"].iloc[0]) if not ordinary.empty else 1.0
    ) or 1.0

    summary["vs_ordinary"] = summary["avg_bookings"] / baseline

    return summary.sort_values("avg_bookings", ascending=False)


def seasonality(bookings: pd.DataFrame) -> pd.DataFrame:
    """Service demand by season.

    Answers: *what should we push, and when?*
    """

    completed = bookings[bookings["Booking_Status"] == "Completed"]

    if completed.empty:
        return pd.DataFrame()

    grouped = completed.groupby(
        ["Season", "Service_Name"], as_index=False
    ).agg(bookings=("Booking_ID", "count"))

    totals = grouped.groupby("Service_Name")["bookings"].transform("sum")

    grouped["share_of_service"] = grouped["bookings"] / totals

    return grouped


def demand_heatmap(bookings: pd.DataFrame) -> pd.DataFrame:
    """Booking volume by day of week and time window.

    Answers: *when is demand concentrated?*
    """

    if bookings.empty or "Booking_Window" not in bookings.columns:
        return pd.DataFrame()

    return bookings.groupby(
        ["Day_Of_Week", "Booking_Window"], as_index=False
    ).agg(bookings=("Booking_ID", "count"))


# ============================================================
# CUSTOMERS
# ============================================================

def segment_summary(state_year_end: pd.DataFrame) -> pd.DataFrame:
    """Value and behaviour of each customer segment.

    Answers: *who is worth investing in?*
    """

    if state_year_end.empty:
        return pd.DataFrame()

    return (
        state_year_end.groupby("Customer_Segment", as_index=False)
        .agg(
            customers=("Customer_ID", "count"),
            avg_spend=("Lifetime_Spend", "mean"),
            total_spend=("Lifetime_Spend", "sum"),
            avg_bookings=("Completed_To_Date", "mean"),
            avg_loyalty=("Loyalty_Score", "mean"),
            avg_promo_response=("Promotion_Response_Rate", "mean"),
        )
        .sort_values("total_spend", ascending=False)
    )


def top_customers(
    state_year_end: pd.DataFrame,
    n: int = 20,
) -> pd.DataFrame:
    """Highest lifetime value customers.

    Answers: *who must we not lose?*
    """

    if state_year_end.empty:
        return pd.DataFrame()

    return state_year_end.nlargest(n, "Lifetime_Spend")[[
        "Customer_ID", "City", "Persona", "Membership", "Customer_Segment",
        "Lifetime_Spend", "Completed_To_Date", "Loyalty_Score",
        "Promotion_Response_Rate", "Days_Since_Last_Booking",
    ]]


def at_risk_customers(
    state_year_end: pd.DataFrame,
    min_spend: float = 10000,
    min_days_quiet: int = 45,
) -> pd.DataFrame:
    """Valuable customers who have gone quiet.

    Answers: *who should we win back first?*

    Sorted by spend rather than by how long they have been away: a
    customer worth fifty thousand who left three months ago matters
    more than one worth two thousand who left six.
    """

    if state_year_end.empty:
        return pd.DataFrame()

    at_risk = state_year_end[
        (state_year_end["Lifetime_Spend"] >= min_spend)
        & (state_year_end["Days_Since_Last_Attempt"] >= min_days_quiet)
    ]

    return at_risk.sort_values("Lifetime_Spend", ascending=False)


# ============================================================
# ALERTS
# ============================================================

@dataclass
class Alert:
    """A condition a manager should act on.

    Attributes:
        severity: ``critical``, ``warning`` or ``info``.
        title: What is wrong, in one line.
        detail: The evidence, with numbers.
        action: What to do about it.
    """

    severity: Literal["critical", "warning", "info"]
    title: str
    detail: str
    action: str


def build_alerts(
    bookings: pd.DataFrame,
    providers: pd.DataFrame,
    promotions: pd.DataFrame,
    state_year_end: pd.DataFrame,
) -> list[Alert]:
    """Conditions worth a manager's attention right now.

    This is what separates a decision platform from a report. KPI tiles
    describe what happened; these say what to do about it.

    Ordered critical first, because a strip of ten warnings is read as
    zero warnings.
    """

    alerts: list[Alert] = []

    if bookings.empty:
        return alerts

    # --- supply failure ---------------------------------------

    no_provider_rate = float(
        (bookings["Booking_Status"] == "No_Provider").mean()
    )

    if no_provider_rate > config.ALERT_NO_PROVIDER_RATE:

        gaps = supply_gaps(bookings, providers)

        lost_revenue = float(gaps["lost_revenue"].sum()) if not gaps.empty else 0
        pairs = len(gaps[gaps["active_providers"] == 0]) if not gaps.empty else 0

        alerts.append(Alert(
            severity="critical",
            title=(
                f"{config.count((bookings['Booking_Status'] == 'No_Provider').sum())} "
                f"bookings lost to zero provider coverage"
            ),
            detail=(
                f"{config.percent(no_provider_rate)} of booking attempts found "
                f"nobody available, worth {config.money(lost_revenue)} in "
                f"foregone bookings. {pairs} city/service combinations have no "
                f"active provider at all."
            ),
            action=(
                "Recruit into the uncovered combinations. No promotion can "
                "recover these - the demand already existed."
            ),
        ))

    # --- promotional waste -------------------------------------

    waste = promotion_waste(promotions, bookings)

    if waste["waste_rate"] > config.ALERT_PROMO_WASTE_RATE:
        alerts.append(Alert(
            severity="warning",
            title=(
                f"{config.money(waste['wasted_spend'])} of discount went to "
                f"customers who would have booked anyway"
            ),
            detail=(
                f"{config.percent(waste['baseline_conversion'])} of promoted "
                f"customers convert without any offer, measured on the "
                f"randomised control group. That share of "
                f"{config.money(waste['total_spend'])} total discount spend "
                f"bought nothing."
            ),
            action=(
                "Shift budget toward personas with genuine uplift. The "
                "Strategy Lab quantifies the trade."
            ),
        ))

    # --- cancellations ------------------------------------------

    cancellation_rate = float(
        (bookings["Booking_Status"] == "Cancelled").mean()
    )

    if cancellation_rate > config.ALERT_CANCELLATION_RATE:

        worst = (
            bookings[bookings["Booking_Status"] == "Cancelled"]
            ["Persona"].value_counts()
        )

        driver = worst.index[0] if not worst.empty else "unknown"

        alerts.append(Alert(
            severity="warning",
            title=f"Cancellation rate at {config.percent(cancellation_rate)}",
            detail=(
                f"Concentrated in the {driver} segment. Cancelled jobs consume "
                f"provider capacity that could have served a completing customer."
            ),
            action=(
                "Consider deposits or confirmation steps for high-cancellation "
                "segments rather than discounts."
            ),
        ))

    # --- dormancy ------------------------------------------------

    if not state_year_end.empty:

        dormant_share = float(
            (state_year_end["Customer_Segment"] == "Dormant").mean()
        )

        if dormant_share > config.ALERT_DORMANT_SHARE:

            at_risk = at_risk_customers(state_year_end)
            at_risk_value = float(at_risk["Lifetime_Spend"].sum()) if not at_risk.empty else 0

            alerts.append(Alert(
                severity="warning",
                title=f"{config.percent(dormant_share)} of customers have gone quiet",
                detail=(
                    f"{len(at_risk)} of them are high value, representing "
                    f"{config.money(at_risk_value)} of historic spend."
                ),
                action=(
                    "Target reactivation at the high-value dormant list in "
                    "Customer Intelligence, not the whole segment."
                ),
            ))

    order = {"critical": 0, "warning": 1, "info": 2}

    return sorted(alerts, key=lambda a: order[a.severity])
