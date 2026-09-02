"""
Loading layer for the exported Databricks tables.

Every table is read once and cached for the session. Streamlit re-runs the
entire script on each widget change, so uncached loading would re-read
Parquet on every filter click and make the application unusable.

This module is the ONLY place that touches the filesystem. If the data
later moves to a live Databricks SQL warehouse, this file changes and
nothing else does.

Note on the caching decorators: `services/` otherwise avoids importing
Streamlit, so that business logic stays testable. This module is the
deliberate exception - caching is its entire purpose.
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

import pandas as pd
import streamlit as st

import config


# ============================================================
# LOW LEVEL
# ============================================================

class DataNotFoundError(RuntimeError):
    """Raised when the export is missing, with instructions to fix it."""


def _missing_data_message(name: str) -> str:
    return (
        f"Could not find `{name}` in `{config.DATA_DIR}`.\n\n"
        f"**To fix this:**\n\n"
        f"1. Run `17_Export_For_App` in Databricks\n"
        f"2. Download everything it writes\n"
        f"3. Place the `.parquet` files in `streamlit_app/sample_data/`\n"
        f"4. Place `model.joblib`, `encoders.json` and "
        f"`feature_order.json` in `streamlit_app/sample_data/model/`"
    )


@st.cache_data(show_spinner=False)
def load_table(name: str) -> pd.DataFrame:
    """Read one exported table by its logical name.

    Args:
        name: A key from `config.TABLES`, for example ``"bookings"``.

    Returns:
        The table as a DataFrame. Date-like columns are converted to
        datetime so downstream filtering and resampling behave.

    Raises:
        DataNotFoundError: If the file is absent.
        KeyError: If the name is not a known table.
    """

    if name not in config.TABLES:
        raise KeyError(
            f"Unknown table '{name}'. Known tables: "
            f"{sorted(config.TABLES)}"
        )

    path = config.DATA_DIR / config.TABLES[name]

    if not path.exists():
        raise DataNotFoundError(_missing_data_message(config.TABLES[name]))

    frame = pd.read_parquet(path)

    return _normalise_dates(frame)


def _normalise_dates(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert any column whose name ends in _Date to real datetimes.

    Parquet round-trips dates inconsistently depending on the writer, and
    a column that is sometimes ``object`` and sometimes ``datetime64``
    breaks comparisons in ways that are tedious to debug.

    The `_To_Date` suffix is the trap. `Bookings_To_Date` means "bookings
    so far", not a date - it is a running count, and coercing it turns
    small integers into timestamps in 1970 with no error raised. The
    model then receives nanoseconds where it was trained on counts.
    """

    for column in frame.columns:

        if column.endswith("_To_Date"):
            continue

        if column.endswith("_Date") or column == "State_Date":
            frame[column] = pd.to_datetime(frame[column], errors="coerce")

    return frame


@st.cache_data(show_spinner=False)
def load_manifest() -> dict[str, Any]:
    """Read the export manifest.

    Carries row counts, the export timestamp and the verified encoding
    deviation. The application shows this so a user can always tell how
    old the snapshot is - a dashboard whose age is unknowable is a
    liability.
    """

    if not config.MANIFEST_PATH.exists():
        raise DataNotFoundError(_missing_data_message("MANIFEST.json"))

    with open(config.MANIFEST_PATH) as handle:
        return json.load(handle)


@lru_cache(maxsize=1)
def data_is_available() -> bool:
    """Whether the export is present, without raising.

    Used by `app.py` to show a setup screen instead of a stack trace on
    a fresh clone.
    """

    if not config.MANIFEST_PATH.exists():
        return False

    required = ["recommendations", "bookings", "customers"]

    return all(
        (config.DATA_DIR / config.TABLES[name]).exists()
        for name in required
    )


# ============================================================
# TYPED ACCESSORS
# ------------------------------------------------------------
# Named functions rather than string keys everywhere. An editor can
# autocomplete these, and a typo fails at import rather than at
# render time in front of a user.
# ============================================================

def load_customers() -> pd.DataFrame:
    """One row per customer: persona, city, membership, signup."""
    return load_table("customers")


def load_services() -> pd.DataFrame:
    """Ten services with price, category, duration and demand level."""
    return load_table("services")


def load_providers() -> pd.DataFrame:
    """250 providers with city, primary service, rating and daily capacity."""
    return load_table("providers")


def load_bookings() -> pd.DataFrame:
    """Every booking attempt across the simulated year.

    ``Booking_Status`` separates three outcomes that must not be
    conflated: ``Completed``, ``Cancelled``, and ``No_Provider`` - the
    last being a supply failure rather than a customer decision.
    """
    return load_table("bookings")


def load_promotions() -> pd.DataFrame:
    """Every promotion decision, including those where nothing was sent.

    Rows with ``Promotion_Sent = False`` are the control group and are
    what make uplift measurable. ``Targeting_Mode = "Exploration"``
    marks the randomised subset where the comparison is unconfounded.
    """
    return load_table("promotions")


def load_recommendations() -> pd.DataFrame:
    """The model's chosen offer per customer for the test period.

    Covers October to December only - the months the model never saw
    during training.
    """
    return load_table("recommendations")


def load_capacity() -> pd.DataFrame:
    """Daily capacity and utilisation per city and service.

    Written by notebook 23, exported by notebook 17. ``capacity_status``
    is the column that matters: Idle, Comfortable, Busy, Oversubscribed
    or No cover. It is judged on a BUSY day rather than an average one -
    a cell that averages 27% and peaks at 300% is a cell that turns
    customers away, and the average hides it.
    """
    return load_table("capacity")


def capacity_is_available() -> bool:
    """Whether notebook 23's export is present.

    The campaign builder degrades rather than crashing when it is not -
    it still ranks by profit, it just cannot say "this cell is full".
    """
    return (config.DATA_DIR / config.TABLES["capacity"]).exists()


def load_activity_daily() -> pd.DataFrame:
    """Funnel counts by date, city, persona and membership.

    Aggregated during export. The raw table is 365,000 rows and the
    application only ever needs counts from it.
    """
    return load_table("activity_daily")


def load_customer_state_monthly() -> pd.DataFrame:
    """Customer state sampled on the first of each month, for trends."""
    return load_table("customer_state_monthly")


def load_customer_state_year_end() -> pd.DataFrame:
    """Customer state on 31 December: one row per customer."""
    return load_table("customer_state_year_end")


def load_customer_state_daily() -> pd.DataFrame:
    """Full daily customer state, 365,000 rows.

    Only the Customer Journey timeline needs this. Load it lazily and
    filter to one customer immediately.
    """
    return load_table("customer_state_daily")


def load_training_features() -> pd.DataFrame:
    """The modelling table: features, labels and the split marker.

    This is what the live scoring pages build their input rows from, so
    every feature the model expects is already present and correct.
    """
    return load_table("training_features")


# ============================================================
# CONVENIENCE
# ============================================================

@st.cache_data(show_spinner=False)
def distinct_values(table: str, column: str) -> list[str]:
    """Sorted distinct values of a column, for populating filters.

    Nulls are dropped. Everything is cast to string so a filter never
    mixes types and silently fails to match.
    """

    frame = load_table(table)

    if column not in frame.columns:
        return []

    values = frame[column].dropna().astype(str).unique().tolist()

    return sorted(values)


@st.cache_data(show_spinner=False)
def date_bounds(table: str, column: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Earliest and latest date in a column, for date range pickers."""

    frame = load_table(table)

    series = pd.to_datetime(frame[column], errors="coerce").dropna()

    if series.empty:
        return (
            pd.Timestamp(config.SIMULATION_START),
            pd.Timestamp(config.SIMULATION_END),
        )

    return series.min(), series.max()
