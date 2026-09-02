"""
Strategy Lab — what happens if we change the strategy?

Build up to three strategies, run them against the real model, and see
which one wins on **incremental profit** rather than revenue.

That ranking choice is deliberate. A strategy can top the revenue table
while destroying value, and ranking on revenue is how organisations
discount their way to a bigger, less profitable business.

Every figure here is either model output or arithmetic on model output.
Where something rests on an assumption — campaign duration scaling
linearly — the page says so rather than presenting it as a finding.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

import config
from components import charts, filters as filter_bar
from components.metric_cards import (
    metric_row, page_header, recommended_action, section,
)
from services import data_loader, simulation_service
from services.model_service import Offer
from services.simulation_service import Scenario

st.set_page_config(page_title="Strategy Lab", page_icon="◆", layout="wide")

stylesheet = config.APP_ROOT / "assets" / "styles.css"
if stylesheet.exists():
    st.markdown(f"<style>{stylesheet.read_text(encoding='utf-8')}</style>",
                unsafe_allow_html=True)


# ============================================================
# LOAD
# ============================================================

try:
    training = data_loader.load_training_features()
    providers = data_loader.load_providers()
except Exception as error:
    st.error(str(error))
    st.stop()


page_header(
    "Strategy Lab",
    "What happens if we change the strategy?",
    f"Scored against the real model on the October–December period. "
    f"Gross margin {config.percent(config.GROSS_MARGIN)}, so break-even "
    f"sits at a {config.BREAK_EVEN_DISCOUNT:.0f}% discount.",
)


# ============================================================
# BUILD STRATEGIES
# ============================================================

def strategy_builder(name: str, defaults: dict) -> Scenario:
    """Controls for one strategy."""

    st.markdown(f"**{name}**")

    offer = filter_bar.offer_selector(
        f"lab_{name}", default_discount=defaults["discount"]
    )

    cohort = filter_bar.cohort_selector(f"lab_{name}", defaults)

    return Scenario(
        name=name,
        offer=Offer(offer["promotion_type"], offer["discount"]),
        cities=cohort["cities"] or None,
        personas=cohort["personas"] or None,
        segments=cohort["segments"] or None,
    )


PRESETS = {
    "Blanket discount": {
        "discount": 20, "cities": [], "personas": [], "segments": [],
    },
    "Target the responsive": {
        "discount": 15, "cities": [],
        "personas": ["Promotion Sensitive", "Seasonal"], "segments": [],
    },
    "Win back the quiet": {
        "discount": 10, "cities": [], "personas": [], "segments": ["Dormant"],
    },
}

section("Build the strategies", "Three at a time, compared side by side")

preset_cols = st.columns(3)

for col, (label, preset) in zip(preset_cols, PRESETS.items()):
    col.caption(f"**{label}** — {preset['discount']}% off"
                + (f", {', '.join(preset['personas'])}" if preset["personas"] else "")
                + (f", {', '.join(preset['segments'])}" if preset["segments"] else ""))

st.caption(
    "The three columns below start from those presets. Change anything."
)

builder_cols = st.columns(3, gap="large")

scenarios: list[Scenario] = []

for col, (label, preset) in zip(builder_cols, PRESETS.items()):
    with col:
        scenarios.append(strategy_builder(label, preset))


# ============================================================
# CONSTRAINTS
# ============================================================

with st.expander("Campaign constraints"):

    cols = st.columns(3)

    budget = cols[0].number_input(
        "Discount budget",
        min_value=0, max_value=5_000_000, value=0, step=50_000,
        help=(
            "0 means unlimited. When set, customers are served in "
            "descending order of profit until it runs out — which is what "
            "a real campaign does, and also the optimal ordering."
        ),
    )

    duration = cols[1].number_input(
        "Campaign days",
        min_value=0, max_value=365, value=0, step=7,
        help=(
            "0 uses the observed period. Any other value scales the result "
            "linearly, which is an assumption rather than a model output."
        ),
    )

    capacity = cols[2].slider(
        "Provider capacity",
        min_value=0.5, max_value=2.0, value=1.0, step=0.1,
        format="%.1f×",
        help="Test supply-constrained scenarios.",
    )

for scenario in scenarios:
    scenario.budget = budget or None
    scenario.campaign_days = duration or None
    scenario.capacity_factor = capacity


# ============================================================
# RUN
# ============================================================

with st.spinner("Scoring each strategy against the model…"):
    results = [
        simulation_service.run_scenario(training, providers, scenario)
        for scenario in scenarios
    ]

for result in results:
    for warning in result.warnings:
        st.warning(f"**{result.scenario.name}** — {warning}")


# ============================================================
# VERDICT
# ============================================================

section("The verdict", "Which strategy wins, and why?")

st.markdown(simulation_service.executive_summary(results))

comparison = simulation_service.compare_scenarios(results)

st.plotly_chart(
    charts.scenario_comparison(comparison),
    use_container_width=True,
    config={"displayModeBar": False},
)


# ============================================================
# SIDE BY SIDE
# ============================================================

section("Side by side", "What each one costs and returns")

result_cols = st.columns(len(results), gap="large")

for col, result in zip(result_cols, results):
    with col:

        tone = "bad" if result.is_value_destroying else "good"

        st.markdown(f"##### {result.scenario.name}")
        st.caption(result.scenario.describe())

        metric_row([
            {
                "label": "Profit vs doing nothing",
                "value": config.money(result.incremental_profit),
                "severity": tone,
                "caption": (
                    "destroys value" if result.is_value_destroying
                    else "creates value"
                ),
            },
        ], columns=1)

        metric_row([
            {
                "label": "Extra bookings",
                "value": f"{result.incremental_bookings:,.0f}",
            },
            {
                "label": "Discount cost",
                "value": config.money(result.discount_cost),
            },
        ], columns=2)

        metric_row([
            {
                "label": "Return per rupee",
                "value": f"{result.roi:.2f}" if result.discount_cost else "—",
                "severity": tone,
            },
            {
                "label": "Reactivated",
                "value": f"{result.customers_reactivated:,}",
                "caption": "dormant customers",
            },
        ], columns=2)

        metric_row([
            {
                "label": "Provider load",
                "value": config.percent(result.provider_utilisation),
                "severity": (
                    "bad" if result.provider_utilisation > 1
                    else "warn" if result.provider_utilisation > 0.85
                    else "neutral"
                ),
                "caption": "of capacity",
            },
        ], columns=1)


# ============================================================
# DISCOUNT SWEEP
# ============================================================

section("Where the discount stops paying", "At what point does it turn negative?")

st.caption(
    "The winning strategy's targeting, swept across every discount level. "
    "The red line is where margin reaches zero."
)

best = max(results, key=lambda r: r.incremental_profit)

sweep_rows = []

with st.spinner("Sweeping discount levels…"):

    for level in [0] + config.DISCOUNT_LEVELS + [25, 30, 35]:

        swept = Scenario(
            name=f"{level}%",
            offer=Offer(best.scenario.offer.promotion_type, level),
            cities=best.scenario.cities,
            personas=best.scenario.personas,
            segments=best.scenario.segments,
            capacity_factor=capacity,
        )

        outcome = simulation_service.run_scenario(training, providers, swept)

        sweep_rows.append({
            "discount": level,
            "incremental_profit": outcome.incremental_profit,
            "bookings": outcome.expected_bookings,
        })

sweep = pd.DataFrame(sweep_rows)

st.plotly_chart(
    charts.profit_curve(sweep),
    use_container_width=True,
    config={"displayModeBar": False},
)

positive = sweep[sweep["incremental_profit"] > 0]

if positive.empty:
    recommended_action(
        f"<b>No discount level pays</b> for this targeting. Every option "
        f"loses money against sending nothing. Narrow the cohort to the "
        f"personas with genuine uplift, or accept that this segment should "
        f"not be discounted at all.",
        severity="bad",
    )
else:
    optimal = positive.loc[positive["incremental_profit"].idxmax()]

    recommended_action(
        f"<b>{optimal['discount']:.0f}% is the sweet spot</b> for this "
        f"targeting, adding {config.money(optimal['incremental_profit'])} "
        f"over doing nothing. Beyond it the extra bookings stop covering the "
        f"margin surrendered — at "
        f"{config.BREAK_EVEN_DISCOUNT:.0f}% there is no margin left to "
        f"surrender at all.",
        severity="good",
    )


# ============================================================
# EXPORT
# ============================================================

st.divider()

export = comparison.copy()

for column in ["Revenue", "Discount cost", "Net profit", "Incremental profit"]:
    export[column] = export[column].round(0)

left, right = st.columns(2)

left.download_button(
    "Download the comparison",
    export.to_csv(index=False).encode(),
    "strategy_comparison.csv",
    "text/csv",
    use_container_width=True,
)

summary_text = (
    f"STRATEGY COMPARISON\n"
    f"{'=' * 60}\n\n"
    f"{simulation_service.executive_summary(results)}\n\n"
    f"{'=' * 60}\n\n"
    f"{export.to_string(index=False)}\n\n"
    f"Assumptions\n"
    f"  Gross margin: {config.percent(config.GROSS_MARGIN)}\n"
    f"  Break-even discount: {config.BREAK_EVEN_DISCOUNT:.0f}%\n"
    f"  Scored on the October-December period, out of model training.\n"
    f"  Booking probability is model output; all financials are\n"
    f"  arithmetic on it.\n"
)

right.download_button(
    "Download the executive summary",
    summary_text.encode(),
    "executive_summary.txt",
    "text/plain",
    use_container_width=True,
)
