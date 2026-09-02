"""
Customer profile and lifecycle timeline.

The profile answers *who is this person?*. The timeline answers *what
has happened to them, and what did the platform do about it?* - which
is where the pipeline becomes legible: activity, promotion, booking,
provider, payment, state update, recommendation, all for one customer
in order.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

import config


# ============================================================
# PROFILE
# ============================================================

def header(state_row: pd.Series) -> None:
    """Identity and standing, from the customer state table."""

    segment_tone = {
        "VIP": "good", "Loyal": "good", "Active": "neutral",
        "Dormant": "warn", "New": "neutral",
    }.get(str(state_row.get("Customer_Segment")), "neutral")

    st.markdown(
        f"""
        <div class="profile-header">
            <div class="profile-identity">
                <div class="profile-id">{state_row['Customer_ID']}</div>
                <div class="profile-tags">
                    <span class="tag tag-{segment_tone}">
                        {state_row.get('Customer_Segment', '—')}
                    </span>
                    <span class="tag">{state_row.get('Persona', '—')}</span>
                    <span class="tag">{state_row.get('Membership', '—')}</span>
                    <span class="tag">{state_row.get('City', '—')}</span>
                </div>
            </div>
            <div class="profile-loyalty">
                <div class="profile-loyalty-value">
                    {state_row.get('Loyalty_Score', 0):.0f}
                </div>
                <div class="profile-loyalty-label">Loyalty score</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metrics(state_row: pd.Series) -> None:
    """The numbers that describe this customer's history.

    Nulls render as an em dash rather than zero. A customer who has
    never been sent a promotion has no response rate, and showing 0%
    would say they were offered and refused - a different and worse
    thing to believe.
    """

    def value(field: str, formatter, fallback: str = "—") -> str:
        raw = state_row.get(field)
        if raw is None or pd.isna(raw):
            return fallback
        return formatter(raw)

    rows = [
        [
            ("Lifetime spend", value("Lifetime_Spend", config.money)),
            ("Completed bookings", value("Completed_To_Date", lambda v: f"{v:,.0f}")),
            ("Average order", value("Avg_Order_Value", config.money)),
            ("Services tried", value("Distinct_Services_Tried", lambda v: f"{v:.0f}")),
        ],
        [
            ("Promotions received", value("Promotions_Received_To_Date", lambda v: f"{v:,.0f}")),
            ("Redeemed", value("Promotions_Redeemed_To_Date", lambda v: f"{v:,.0f}")),
            ("Response rate", value("Promotion_Response_Rate", config.percent)),
            ("Cancellation rate", value("Cancellation_Rate", config.percent)),
        ],
        [
            ("Last booking", value("Days_Since_Last_Booking", lambda v: f"{v:.0f} days ago")),
            ("Last attempt", value("Days_Since_Last_Attempt", lambda v: f"{v:.0f} days ago")),
            ("Preferred category", value("Preferred_Category", str)),
            ("Average rating given", value("Avg_Rating_Given", lambda v: f"{v:.1f}")),
        ],
    ]

    for row in rows:
        cols = st.columns(len(row))
        for col, (label, text) in zip(cols, row):
            col.markdown(
                f'<div class="mini-metric">'
                f'<div class="mini-label">{label}</div>'
                f'<div class="mini-value">{text}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )


def engagement_gap(state_row: pd.Series) -> None:
    """Flag the difference between attempting and completing.

    A Frequent Canceller books constantly and completes rarely. Judged
    on completed bookings alone they look absent; judged on attempts
    they are clearly present. Confusing the two sends a win-back
    campaign to someone who never left.
    """

    booking_gap = state_row.get("Days_Since_Last_Booking")
    attempt_gap = state_row.get("Days_Since_Last_Attempt")

    if pd.isna(booking_gap) or pd.isna(attempt_gap):
        return

    if booking_gap - attempt_gap < 14:
        return

    st.warning(
        f"**Attempting but not completing.** Last tried to book "
        f"{attempt_gap:.0f} days ago, last actually completed one "
        f"{booking_gap:.0f} days ago. This is a cancellation problem, "
        f"not an absence — a win-back discount would be the wrong "
        f"intervention."
    )


# ============================================================
# TIMELINE
# ============================================================

def timeline(
    customer_id: str,
    bookings: pd.DataFrame,
    promotions: pd.DataFrame,
    recommendations: pd.DataFrame,
    limit: int = 25,
) -> None:
    """Everything that happened to this customer, in order.

    Merges promotions, bookings and recommendations into one sequence,
    so the pipeline reads as a story rather than as separate tables:
    an offer went out, a booking followed or did not, a provider was
    assigned or was not, and the platform now suggests something next.
    """

    events: list[dict] = []

    for _, row in promotions[
        promotions["Customer_ID"] == customer_id
    ].iterrows():

        if row.get("Promotion_Sent"):
            events.append({
                "date": row["Promotion_Date"],
                "kind": "promotion",
                "title": (
                    f"{row['Promotion_Type']} {row['Discount_Percent']:.0f}% "
                    f"offered"
                ),
                "detail": (
                    f"on {row['Service_Name']} · "
                    f"{row.get('Targeting_Mode', 'Targeted')}"
                ),
            })

    for _, row in bookings[
        bookings["Customer_ID"] == customer_id
    ].iterrows():

        status = row["Booking_Status"]

        detail_parts = [row["Service_Name"]]

        if pd.notna(row.get("Provider_Name")):
            detail_parts.append(f"provider {row['Provider_Name']}")

        if status == "Completed":
            detail_parts.append(config.money(row["Final_Price"]))
            if pd.notna(row.get("Rating")):
                detail_parts.append(f"rated {row['Rating']:.1f}")
        elif status == "No_Provider":
            detail_parts.append("nobody available")

        events.append({
            "date": row["Booking_Date"],
            "kind": {
                "Completed": "completed",
                "Cancelled": "cancelled",
                "No_Provider": "failed",
            }.get(status, "booking"),
            "title": {
                "Completed": "Booking completed",
                "Cancelled": "Booking cancelled",
                "No_Provider": "Turned away — no provider",
            }.get(status, status),
            "detail": " · ".join(detail_parts),
        })

    for _, row in recommendations[
        recommendations["Customer_ID"] == customer_id
    ].iterrows():

        offered = row["Offer_Discount"] > 0

        events.append({
            "date": row["Activity_Date"],
            "kind": "recommendation",
            "title": (
                f"Platform recommends {row['Offer_Type']} "
                f"{row['Offer_Discount']:.0f}%" if offered
                else "Platform recommends no promotion"
            ),
            "detail": (
                f"expected profit {config.money(row['Expected_Profit'])} · "
                f"uplift {row['Uplift']:+.1%}"
            ),
        })

    if not events:
        st.info("No recorded activity for this customer.")
        return

    frame = pd.DataFrame(events).sort_values("date", ascending=False)

    st.caption(
        f"{len(frame)} events · showing the most recent {min(limit, len(frame))}"
    )

    for _, event in frame.head(limit).iterrows():

        date = pd.Timestamp(event["date"]).strftime("%d %b %Y")

        st.markdown(
            f"""
            <div class="timeline-event timeline-{event['kind']}">
                <div class="timeline-marker"></div>
                <div class="timeline-body">
                    <div class="timeline-date">{date}</div>
                    <div class="timeline-title">{event['title']}</div>
                    <div class="timeline-detail">{event['detail']}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def state_progression(
    customer_id: str,
    state_daily: pd.DataFrame,
) -> pd.DataFrame:
    """This customer's loyalty and spend, day by day.

    Charting how state accumulates is the clearest demonstration that
    features are point-in-time: the lines only ever move forward, and a
    prediction made in March could not have seen November.
    """

    rows = state_daily[state_daily["Customer_ID"] == customer_id]

    if rows.empty:
        return pd.DataFrame()

    return rows.sort_values("State_Date")[[
        "State_Date", "Loyalty_Score", "Lifetime_Spend",
        "Completed_To_Date", "Customer_Segment",
    ]]


# ============================================================
# SELECTOR
# ============================================================

def selector(
    customers: list[str],
    key: str = "customer_picker",
    label: str = "Customer",
) -> str:
    """Searchable customer picker.

    A dropdown over a thousand IDs is unusable, so this is a
    `selectbox` with typeahead - Streamlit filters as you type.
    """

    return st.selectbox(
        label,
        options=customers,
        key=key,
        help="Type to search. Only customers active in October–December can be scored.",
    )
