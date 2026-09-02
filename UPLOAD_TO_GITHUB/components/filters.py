"""
The filter bar every page shares.

Selections live in `st.session_state`, so moving between pages keeps the
context. A manager who filters to Mumbai on the Executive Dashboard and
clicks through to Promotion Performance should still be looking at
Mumbai - re-selecting the same filters on every page is the fastest way
to make a multi-page tool feel broken.

The filter bar renders in the sidebar. Pages get a `Filters` object back
and pass it to `analytics_service.apply_filters`, which is the only
place filtering actually happens.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

import config
from services import data_loader
from services.analytics_service import Filters


STATE_KEY = "global_filters"


# ============================================================
# STATE
# ============================================================

def _default_filters() -> dict:
    """Opening state: the whole simulated year, nothing narrowed."""

    return {
        "date_from": pd.Timestamp(config.SIMULATION_START),
        "date_to": pd.Timestamp(config.SIMULATION_END),
        "cities": [],
        "personas": [],
        "memberships": [],
        "segments": [],
        "services": [],
        "weather": [],
        "holidays_only": False,
    }


def _ensure_state() -> dict:
    if STATE_KEY not in st.session_state:
        st.session_state[STATE_KEY] = _default_filters()
    return st.session_state[STATE_KEY]


def current_filters() -> Filters:
    """The active selection, without rendering anything.

    For pages that need the filter state but draw their own controls.
    """

    state = _ensure_state()

    return Filters(**state)


# ============================================================
# RENDER
# ============================================================

def render(
    show_date: bool = True,
    show_weather: bool = False,
    show_segments: bool = True,
    key_prefix: str = "",
) -> Filters:
    """Draw the filter bar and return the selection.

    Args:
        show_date: Hide on pages that are inherently point-in-time,
            such as year-end customer segments.
        show_weather: Only shown where weather is a live dimension,
            to keep the sidebar short elsewhere.
        show_segments: Hide where segment is not a meaningful cut.
        key_prefix: Namespace for widget keys when two filter bars
            appear on one page.

    Returns:
        The active `Filters`.
    """

    state = _ensure_state()

    with st.sidebar:

        st.markdown("### Filters")

        if show_date:
            _render_date(state, key_prefix)

        state["cities"] = st.multiselect(
            "City",
            options=config.CITIES,
            default=state["cities"],
            key=f"{key_prefix}filter_cities",
            placeholder="All cities",
        )

        state["personas"] = st.multiselect(
            "Persona",
            options=config.PERSONAS,
            default=state["personas"],
            key=f"{key_prefix}filter_personas",
            placeholder="All personas",
            help=(
                "Behavioural type. Drives baseline conversion and how "
                "strongly a customer responds to promotions."
            ),
        )

        if show_segments:
            state["segments"] = st.multiselect(
                "Segment",
                options=config.CUSTOMER_SEGMENTS,
                default=state["segments"],
                key=f"{key_prefix}filter_segments",
                placeholder="All segments",
                help=(
                    "Where the customer stands today: VIP, Loyal, Active, "
                    "Dormant or New."
                ),
            )

        with st.expander("More filters"):

            state["memberships"] = st.multiselect(
                "Membership",
                options=config.MEMBERSHIPS,
                default=state["memberships"],
                key=f"{key_prefix}filter_memberships",
                placeholder="All tiers",
            )

            state["services"] = st.multiselect(
                "Service",
                options=_service_options(),
                default=state["services"],
                key=f"{key_prefix}filter_services",
                placeholder="All services",
            )

            if show_weather:
                state["weather"] = st.multiselect(
                    "Weather",
                    options=_weather_options(),
                    default=state["weather"],
                    key=f"{key_prefix}filter_weather",
                    placeholder="All conditions",
                )

            state["holidays_only"] = st.checkbox(
                "Holidays only",
                value=state["holidays_only"],
                key=f"{key_prefix}filter_holidays",
                help=(
                    "Restrict to the seven holidays in the simulated year. "
                    "Diwali alone runs at several times an ordinary day."
                ),
            )

        filters = Filters(**state)

        if filters.is_active():
            st.caption(f"Showing: {filters.describe()}")

            if st.button("Clear all", use_container_width=True):
                st.session_state[STATE_KEY] = _default_filters()
                st.rerun()

    return filters


def _render_date(state: dict, key_prefix: str) -> None:
    """Date range, with quick presets.

    The presets matter: typing two dates to see last quarter is enough
    friction that people stop doing it, and a filter nobody uses is a
    filter that may as well not exist.
    """

    preset = st.selectbox(
        "Period",
        options=["Full year", "Q4 (test period)", "Festival season", "Custom"],
        key=f"{key_prefix}filter_preset",
        help=(
            "Q4 is the period the model never trained on, and the only "
            "months with recommendations."
        ),
    )

    if preset == "Full year":
        state["date_from"] = pd.Timestamp(config.SIMULATION_START)
        state["date_to"] = pd.Timestamp(config.SIMULATION_END)

    elif preset == "Q4 (test period)":
        state["date_from"] = pd.Timestamp(config.TEST_PERIOD_START)
        state["date_to"] = pd.Timestamp(config.SIMULATION_END)

    elif preset == "Festival season":
        state["date_from"] = pd.Timestamp("2025-10-01")
        state["date_to"] = pd.Timestamp("2025-11-30")

    else:
        chosen = st.date_input(
            "Date range",
            value=(state["date_from"], state["date_to"]),
            min_value=pd.Timestamp(config.SIMULATION_START),
            max_value=pd.Timestamp(config.SIMULATION_END),
            key=f"{key_prefix}filter_dates",
        )

        if isinstance(chosen, tuple) and len(chosen) == 2:
            state["date_from"] = pd.Timestamp(chosen[0])
            state["date_to"] = pd.Timestamp(chosen[1])


@st.cache_data(show_spinner=False)
def _service_options() -> list[str]:
    try:
        return data_loader.distinct_values("services", "Service_Name")
    except Exception:
        return []


@st.cache_data(show_spinner=False)
def _weather_options() -> list[str]:
    try:
        return data_loader.distinct_values("bookings", "Weather_Condition")
    except Exception:
        return []


# ============================================================
# SCENARIO PICKERS
# ============================================================

def cohort_selector(
    key_prefix: str,
    defaults: dict | None = None,
) -> dict:
    """Targeting controls for the Strategy Lab and recommendation pages.

    Separate from the global filter bar because these describe *who a
    campaign targets*, not *what data is on screen*. Conflating the two
    would mean changing the view silently changed the scenario.
    """

    defaults = defaults or {}

    cols = st.columns(3)

    cities = cols[0].multiselect(
        "Cities",
        options=config.CITIES,
        default=defaults.get("cities", []),
        key=f"{key_prefix}_cities",
        placeholder="All cities",
    )

    personas = cols[1].multiselect(
        "Personas",
        options=config.PERSONAS,
        default=defaults.get("personas", []),
        key=f"{key_prefix}_personas",
        placeholder="All personas",
    )

    segments = cols[2].multiselect(
        "Segments",
        options=config.CUSTOMER_SEGMENTS,
        default=defaults.get("segments", []),
        key=f"{key_prefix}_segments",
        placeholder="All segments",
    )

    return {"cities": cities, "personas": personas, "segments": segments}


def offer_selector(key_prefix: str, default_discount: int = 10) -> dict:
    """Promotion type and discount, with the break-even line visible.

    The caption is not decoration. At a 35% margin a 20% discount
    surrenders more than half the profit on the job, and a user
    dragging a slider deserves to see that while they drag it rather
    than discover it in the results.
    """

    cols = st.columns([2, 3])

    promotion_type = cols[0].selectbox(
        "Promotion type",
        options=config.PROMOTION_TYPES,
        key=f"{key_prefix}_type",
    )

    discount = cols[1].select_slider(
        "Discount",
        options=[0] + config.DISCOUNT_LEVELS,
        value=default_discount,
        format_func=lambda v: "None" if v == 0 else f"{v}%",
        key=f"{key_prefix}_discount",
    )

    margin_left = config.GROSS_MARGIN - discount / 100

    if margin_left <= 0:
        cols[1].error(
            f"A {discount}% discount exceeds the "
            f"{config.percent(config.GROSS_MARGIN)} margin. Every booking "
            f"this wins loses money."
        )
    elif discount >= config.DISCOUNT_CAUTION_THRESHOLD:
        cols[1].warning(
            f"Margin falls to {config.percent(margin_left)}. Only pays "
            f"where uplift is large."
        )
    elif discount > 0:
        cols[1].caption(
            f"Margin after discount: {config.percent(margin_left)} "
            f"of {config.percent(config.GROSS_MARGIN)}"
        )

    return {"promotion_type": promotion_type, "discount": discount}
