"""
Central configuration for the Urban Company Promotion Intelligence platform.

Everything that could reasonably change - file locations, business rules,
colours, alert thresholds - lives here. Nothing else in the application
should hardcode a constant.

The BUSINESS RULES section mirrors 00_Config in Databricks. Those values
must stay in step: the model was trained under them, and the simulation
maths in the Strategy Lab assumes them. A mismatch here produces an app
that disagrees with the pipeline while looking perfectly plausible.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

# ============================================================
# PATHS
# ============================================================

APP_ROOT: Final[Path] = Path(__file__).parent

DATA_DIR: Final[Path] = APP_ROOT / "sample_data"

MODEL_DIR: Final[Path] = DATA_DIR / "model"

MANIFEST_PATH: Final[Path] = DATA_DIR / "MANIFEST.json"


# Tables written by 17_Export_For_App. The keys are how the rest of the
# application refers to them; the values are the filenames on disk.

TABLES: Final[dict[str, str]] = {
    "capacity": "capacity.parquet",
    "customers": "customers.parquet",
    "services": "services.parquet",
    "providers": "providers.parquet",
    "bookings": "bookings.parquet",
    "promotions": "promotions.parquet",
    "recommendations": "recommendations.parquet",
    "activity_daily": "activity_daily.parquet",
    "customer_state_monthly": "customer_state_monthly.parquet",
    "customer_state_year_end": "customer_state_year_end.parquet",
    "customer_state_daily": "customer_state_daily.parquet",
    "training_features": "training_features.parquet",
}


MODEL_FILE: Final[Path] = MODEL_DIR / "model.joblib"

ENCODERS_FILE: Final[Path] = MODEL_DIR / "encoders.json"

FEATURE_ORDER_FILE: Final[Path] = MODEL_DIR / "feature_order.json"


# ============================================================
# BUSINESS RULES
# ------------------------------------------------------------
# These mirror 00_Config in Databricks. Keep them in step.
# ============================================================

GROSS_MARGIN: Final[float] = 0.35
"""Gross margin on a completed job, before any discount.

This single number is why the platform recommends silence for 71% of
customers. At 35%, a 20% discount surrenders more than half the profit
on the job, so it only pays where uplift is large.
"""

DISCOUNT_LEVELS: Final[list[int]] = [5, 10, 15, 20]

PROMOTION_TYPES: Final[list[str]] = [
    "Direct Discount",
    "Cashback",
    "Coupon",
    "Free Add-on",
]

NO_PROMOTION: Final[str] = "None"
"""How an untreated row encodes Promotion_Type. Must match notebook 10."""


CITIES: Final[list[str]] = [
    "Bangalore", "Delhi", "Mumbai", "Hyderabad", "Pune", "Jaipur",
]

PERSONAS: Final[list[str]] = [
    "Loyal", "Promotion Sensitive", "Seasonal",
    "Premium", "Dormant", "Frequent Canceller",
]

MEMBERSHIPS: Final[list[str]] = ["None", "Silver", "Gold", "Platinum"]

CUSTOMER_SEGMENTS: Final[list[str]] = [
    "VIP", "Loyal", "Active", "Dormant", "New",
]

SEASONS: Final[list[str]] = ["Summer", "Monsoon", "Festival", "Winter"]


SIMULATION_START: Final[str] = "2025-01-01"

SIMULATION_END: Final[str] = "2025-12-31"

TEST_PERIOD_START: Final[str] = "2025-10-01"
"""Recommendations exist only for the model's out-of-time test period."""


# ============================================================
# DERIVED BUSINESS THRESHOLDS
# ============================================================

BREAK_EVEN_DISCOUNT: Final[float] = GROSS_MARGIN * 100
"""Discount percentage at which margin reaches zero.

At 35% margin this is 35%. Above it, every additional booking loses
money no matter how large the uplift. The Strategy Lab draws this line
so the boundary is visible rather than inferred.
"""

DISCOUNT_CAUTION_THRESHOLD: Final[float] = 20.0
"""Above this, a discount rarely pays for itself on a typical cohort.

Derived from the observed policy comparison: blanket 20% discounting
cost 42% of expected profit across the full customer base.
"""


# ============================================================
# ALERT THRESHOLDS
# ------------------------------------------------------------
# Drive the warning strip on the Executive Dashboard. Each is a
# level at which a business person should actually do something.
# ============================================================

ALERT_NO_PROVIDER_RATE: Final[float] = 0.02
"""Share of booking attempts lost to zero provider coverage."""

ALERT_CANCELLATION_RATE: Final[float] = 0.08
"""Share of assigned bookings cancelled by the customer."""

ALERT_DORMANT_SHARE: Final[float] = 0.20
"""Share of the customer base that has gone quiet."""

ALERT_PROMO_WASTE_RATE: Final[float] = 0.30
"""Share of promotions sent to customers who would have booked anyway."""


# ============================================================
# THEME
# ------------------------------------------------------------
# One restrained palette. Semantic colours (good / warn / bad) are
# deliberately separate from the brand accent so status never
# competes with identity.
# ============================================================

COLOR: Final[dict[str, str]] = {
    "primary": "#1B4965",
    "primary_light": "#5FA8D3",
    "accent": "#0F766E",
    "ink": "#0F172A",
    "ink_muted": "#64748B",
    "ink_faint": "#94A3B8",
    "surface": "#FFFFFF",
    "surface_alt": "#F8FAFC",
    "border": "#E2E8F0",
    "good": "#15803D",
    "good_soft": "#DCFCE7",
    "warn": "#B45309",
    "warn_soft": "#FEF3C7",
    "bad": "#B91C1C",
    "bad_soft": "#FEE2E2",
    "neutral": "#475569",
}


CATEGORICAL_PALETTE: Final[list[str]] = [
    "#1B4965", "#5FA8D3", "#0F766E", "#B45309",
    "#7C3AED", "#BE185D", "#0369A1", "#4D7C0F",
]
"""Ordered categorical colours. Eight is enough for six personas,
six cities or five segments without reuse."""


SEQUENTIAL_PALETTE: Final[list[str]] = [
    "#EFF6FF", "#DBEAFE", "#93C5FD", "#5FA8D3", "#1B4965",
]


PERSONA_COLOR: Final[dict[str, str]] = {
    "Promotion Sensitive": "#0F766E",
    "Loyal": "#1B4965",
    "Premium": "#7C3AED",
    "Seasonal": "#5FA8D3",
    "Frequent Canceller": "#B45309",
    "Dormant": "#94A3B8",
}
"""Fixed per persona so a colour means the same thing on every page."""


# ============================================================
# CHART DEFAULTS
# ============================================================

CHART_HEIGHT: Final[int] = 320

CHART_HEIGHT_TALL: Final[int] = 460

CHART_FONT: Final[str] = (
    "system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif"
)


# ============================================================
# APPLICATION
# ============================================================

APP_TITLE: Final[str] = "Urban Company Promotion Intelligence"

APP_ICON: Final[str] = "◆"

CURRENCY: Final[str] = "₹"


PAGES: Final[list[tuple[str, str]]] = [
    ("1_Executive_Dashboard", "Executive Dashboard"),
    ("2_Customer_Intelligence", "Customer Intelligence"),
    ("3_Demand_And_Bookings", "Demand & Bookings"),
    ("4_Promotion_Performance", "Promotion Performance"),
    ("5_AI_Recommendation_Center", "AI Recommendation Center"),
    ("6_Strategy_Lab", "Strategy Lab"),
]


# ============================================================
# FORMATTING
# ============================================================

def money(value: float, decimals: int = 0) -> str:
    """Format a rupee amount, abbreviating anything above a lakh."""

    if value is None:
        return "—"

    if abs(value) >= 1e7:
        return f"{CURRENCY}{value / 1e7:.2f} Cr"

    if abs(value) >= 1e5:
        return f"{CURRENCY}{value / 1e5:.2f} L"

    return f"{CURRENCY}{value:,.{decimals}f}"


def percent(value: float, decimals: int = 1) -> str:
    """Format a 0-1 fraction as a percentage."""

    if value is None:
        return "—"

    return f"{value * 100:.{decimals}f}%"


def count(value: float) -> str:
    """Format a whole number with thousands separators."""

    if value is None:
        return "—"

    return f"{value:,.0f}"
