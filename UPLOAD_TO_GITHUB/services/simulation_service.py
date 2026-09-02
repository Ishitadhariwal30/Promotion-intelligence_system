"""
The Strategy Lab engine: what happens if we do X instead of Y.

An executive changes a discount, a target segment or a budget, and this
module answers what it would cost and what it would return - across a
whole cohort, using the real model rather than a rule of thumb.

WHAT IS MODELLED AND WHAT IS NOT
--------------------------------

Honesty about provenance matters more here than anywhere else in the
platform, because these numbers get quoted in meetings.

    Booking conversion    the model
    Expected bookings     the model, times cohort size
    Revenue               the model, times price, less discount
    Promotion cost        arithmetic
    Profit and ROI        arithmetic
    Provider utilisation  predicted bookings against real capacity
    Customers reactivated the model, on the dormant subset

There is deliberately no "retention rate". No churn model exists in the
pipeline, so any retention percentage would be invented. Reactivation -
dormant customers whose predicted probability crosses a threshold - is
computable from real data and answers the same question honestly.

Campaign duration scales linearly, which is a genuine assumption rather
than a finding. It is surfaced in the UI as such.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import config
from services import model_service
from services.model_service import Offer


REACTIVATION_THRESHOLD = 0.35
"""Predicted booking probability at which a dormant customer counts as
reactivated. Set at roughly the platform-wide conversion rate: a dormant
customer brought up to an average customer's likelihood has, in any
meaningful sense, come back."""


# ============================================================
# SCENARIO
# ============================================================

@dataclass
class Scenario:
    """One promotional strategy to evaluate.

    Attributes:
        name: Label shown in comparisons, e.g. "Strategy A".
        offer: The promotion to send to everyone in the cohort.
        cities: Restrict the cohort. None means all.
        personas: Restrict the cohort. None means all.
        segments: Restrict the cohort. None means all.
        memberships: Restrict the cohort. None means all.
        services: Restrict the cohort. None means all.
        budget: Maximum discount spend. When set, the cohort is served
            in descending order of expected profit until it runs out -
            which is what a real campaign does.
        campaign_days: Scales the result linearly. An assumption, not a
            model output.
        capacity_factor: Multiplies available provider capacity, for
            testing supply-constrained scenarios.
    """

    name: str
    offer: Offer

    cities: list[str] | None = None
    personas: list[str] | None = None
    segments: list[str] | None = None
    memberships: list[str] | None = None
    services: list[str] | None = None

    budget: float | None = None
    campaign_days: int | None = None
    capacity_factor: float = 1.0

    def describe(self) -> str:
        """One-line summary for charts and executive summaries."""

        parts = [self.offer.label]

        for label, values in [
            ("in", self.cities), ("to", self.personas),
            ("segment", self.segments), ("tier", self.memberships),
            ("for", self.services),
        ]:
            if values:
                parts.append(f"{label} {', '.join(values[:2])}")

        if self.budget:
            parts.append(f"capped at {config.money(self.budget)}")

        return " · ".join(parts)


@dataclass
class ScenarioResult:
    """What a scenario is projected to produce."""

    scenario: Scenario

    customers_targeted: int
    customers_in_cohort: int

    conversion_rate: float
    baseline_conversion: float
    expected_bookings: float
    baseline_bookings: float

    revenue: float
    discount_cost: float
    gross_profit: float
    net_profit: float
    baseline_profit: float

    avg_order_value: float
    provider_utilisation: float
    customers_reactivated: int

    budget_exhausted: bool = False
    warnings: list[str] = field(default_factory=list)

    @property
    def incremental_bookings(self) -> float:
        """Bookings this strategy wins over sending nothing."""
        return self.expected_bookings - self.baseline_bookings

    @property
    def incremental_profit(self) -> float:
        """Profit gained or destroyed against sending nothing.

        The number that matters. A strategy can raise bookings and
        revenue while sitting well below this line.
        """
        return self.net_profit - self.baseline_profit

    @property
    def roi(self) -> float:
        """Incremental profit per rupee of discount spent.

        Above 0 means the campaign paid for itself. Below 0 means the
        discount cost more than the extra bookings returned.
        """
        if self.discount_cost <= 0:
            return 0.0
        return self.incremental_profit / self.discount_cost

    @property
    def is_value_destroying(self) -> bool:
        return self.incremental_profit < 0


# ============================================================
# COHORT
# ============================================================

def build_cohort(
    training_features: pd.DataFrame,
    scenario: Scenario,
) -> pd.DataFrame:
    """Rows matching the scenario's targeting.

    Restricted to the model's test period, because those are the only
    months it did not train on.
    """

    frame = training_features[training_features["Data_Split"] == "test"]

    for column, values in [
        ("City", scenario.cities),
        ("Persona", scenario.personas),
        ("Customer_Segment", scenario.segments),
        ("Membership", scenario.memberships),
        ("Service_Name", scenario.services),
    ]:
        if values and column in frame.columns:
            frame = frame[frame[column].astype(str).isin(values)]

    return frame.reset_index(drop=True)


# ============================================================
# RUN
# ============================================================

def run_scenario(
    training_features: pd.DataFrame,
    providers: pd.DataFrame,
    scenario: Scenario,
) -> ScenarioResult:
    """Project the business outcome of one strategy.

    Scores the cohort twice - once under the scenario's offer, once
    under no promotion - because every figure that matters is a
    *difference*, not a level. Revenue going up means nothing if the
    discount that bought it cost more.
    """

    cohort = build_cohort(training_features, scenario)

    warnings: list[str] = []

    if cohort.empty:
        return _empty_result(scenario, ["No customers match this targeting."])

    price = cohort["Seasonal_Price"].astype(float)

    treated_probability = model_service.score_offer(cohort, scenario.offer)

    baseline_offer = Offer(config.NO_PROMOTION, 0)
    baseline_probability = model_service.score_offer(cohort, baseline_offer)

    discount_fraction = scenario.offer.discount_percent / 100.0

    per_customer = pd.DataFrame({
        "price": price,
        "treated_probability": treated_probability,
        "baseline_probability": baseline_probability,
        "revenue": treated_probability * price * (1 - discount_fraction),
        "discount_cost": treated_probability * price * discount_fraction,
        "baseline_profit": baseline_probability * price * config.GROSS_MARGIN,
    })

    per_customer["net_profit"] = (
        per_customer["treated_probability"]
        * per_customer["price"]
        * scenario.offer.margin_after_discount()
    )

    per_customer["incremental_profit"] = (
        per_customer["net_profit"] - per_customer["baseline_profit"]
    )

    # --- budget -------------------------------------------------
    #
    # A real campaign does not treat everyone. Serve the cohort in
    # descending order of incremental profit until the money runs out -
    # which is also the optimal ordering.

    budget_exhausted = False

    if scenario.budget is not None and scenario.budget > 0:

        ordered = per_customer.sort_values(
            "incremental_profit", ascending=False
        )

        cumulative = ordered["discount_cost"].cumsum()

        within = cumulative <= scenario.budget

        if not within.all():
            budget_exhausted = True
            warnings.append(
                f"Budget of {config.money(scenario.budget)} covers "
                f"{int(within.sum()):,} of {len(ordered):,} customers. The "
                f"rest receive nothing."
            )

        targeted = ordered[within]
        untargeted = ordered[~within]

    else:
        targeted = per_customer
        untargeted = per_customer.iloc[0:0]

    # Untargeted customers still book at their baseline rate.

    revenue = float(
        targeted["revenue"].sum()
        + (untargeted["baseline_probability"] * untargeted["price"]).sum()
    )

    discount_cost = float(targeted["discount_cost"].sum())

    net_profit = float(
        targeted["net_profit"].sum() + untargeted["baseline_profit"].sum()
    )

    baseline_profit = float(per_customer["baseline_profit"].sum())

    expected_bookings = float(
        targeted["treated_probability"].sum()
        + untargeted["baseline_probability"].sum()
    )

    baseline_bookings = float(per_customer["baseline_probability"].sum())

    # --- reactivation ------------------------------------------

    reactivated = 0

    if "Customer_Segment" in cohort.columns:
        dormant = cohort["Customer_Segment"].astype(str) == "Dormant"
        if dormant.any():
            reactivated = int(
                (
                    (treated_probability >= REACTIVATION_THRESHOLD)
                    & (baseline_probability < REACTIVATION_THRESHOLD)
                    & dormant.values
                ).sum()
            )

    # --- supply -------------------------------------------------

    utilisation = _projected_utilisation(
        cohort, providers, expected_bookings, scenario.capacity_factor
    )

    if utilisation > 1.0:
        warnings.append(
            f"Projected demand exceeds provider capacity by "
            f"{config.percent(utilisation - 1)}. Some of these bookings "
            f"cannot be fulfilled."
        )

    # --- duration -----------------------------------------------

    scale = 1.0

    if scenario.campaign_days:
        period_days = _cohort_days(cohort)
        scale = scenario.campaign_days / period_days if period_days else 1.0
        warnings.append(
            f"Scaled linearly to {scenario.campaign_days} days from an "
            f"observed {period_days} days. Linear scaling is an assumption, "
            f"not a model output."
        )

    # --- economics guardrail ------------------------------------

    if scenario.offer.margin_after_discount() <= 0:
        warnings.append(
            f"A {scenario.offer.discount_percent}% discount exceeds the "
            f"{config.percent(config.GROSS_MARGIN)} gross margin. Every "
            f"booking this wins loses money."
        )

    return ScenarioResult(
        scenario=scenario,
        customers_targeted=len(targeted),
        customers_in_cohort=len(cohort),
        conversion_rate=float(np.mean(treated_probability)),
        baseline_conversion=float(np.mean(baseline_probability)),
        expected_bookings=expected_bookings * scale,
        baseline_bookings=baseline_bookings * scale,
        revenue=revenue * scale,
        discount_cost=discount_cost * scale,
        gross_profit=revenue * config.GROSS_MARGIN * scale,
        net_profit=net_profit * scale,
        baseline_profit=baseline_profit * scale,
        avg_order_value=(
            revenue / expected_bookings if expected_bookings else 0.0
        ),
        provider_utilisation=utilisation,
        customers_reactivated=int(reactivated * scale),
        budget_exhausted=budget_exhausted,
        warnings=warnings,
    )


def _empty_result(scenario: Scenario, warnings: list[str]) -> ScenarioResult:
    return ScenarioResult(
        scenario=scenario,
        customers_targeted=0, customers_in_cohort=0,
        conversion_rate=0.0, baseline_conversion=0.0,
        expected_bookings=0.0, baseline_bookings=0.0,
        revenue=0.0, discount_cost=0.0, gross_profit=0.0,
        net_profit=0.0, baseline_profit=0.0,
        avg_order_value=0.0, provider_utilisation=0.0,
        customers_reactivated=0, warnings=warnings,
    )


def _cohort_days(cohort: pd.DataFrame) -> int:
    if "Activity_Date" not in cohort.columns:
        return 1
    dates = pd.to_datetime(cohort["Activity_Date"], errors="coerce")
    return max(int(dates.dt.normalize().nunique()), 1)


def _projected_utilisation(
    cohort: pd.DataFrame,
    providers: pd.DataFrame,
    expected_bookings: float,
    capacity_factor: float,
) -> float:
    """Predicted bookings against real provider capacity.

    Restricted to the cities and services the cohort actually covers -
    national capacity is irrelevant when a campaign targets one city.
    """

    if providers.empty:
        return 0.0

    active = providers[providers["Provider_Status"] == "Active"]

    if "City" in cohort.columns:
        cities = set(cohort["City"].astype(str))
        active = active[active["City"].astype(str).isin(cities)]

    if "Service_Name" in cohort.columns:
        services = set(cohort["Service_Name"].astype(str))
        active = active[
            active["Primary_Service_Name"].astype(str).isin(services)
        ]

    if active.empty:
        return float("inf")

    days = _cohort_days(cohort)

    capacity = float(active["Daily_Capacity"].sum()) * days * capacity_factor

    return expected_bookings / capacity if capacity else float("inf")


# ============================================================
# COMPARE
# ============================================================

def compare_scenarios(results: list[ScenarioResult]) -> pd.DataFrame:
    """Side-by-side comparison, ranked by incremental profit.

    Ranked on incremental profit rather than revenue deliberately. A
    strategy can top the revenue table while destroying value, and
    ranking on revenue is how organisations end up discounting their
    way to a bigger, less profitable business.
    """

    if not results:
        return pd.DataFrame()

    rows = [{
        "Strategy": r.scenario.name,
        "Offer": r.scenario.offer.label,
        "Targeting": r.scenario.describe(),
        "Customers": r.customers_targeted,
        "Conversion": r.conversion_rate,
        "Expected bookings": r.expected_bookings,
        "Incremental bookings": r.incremental_bookings,
        "Revenue": r.revenue,
        "Discount cost": r.discount_cost,
        "Net profit": r.net_profit,
        "Incremental profit": r.incremental_profit,
        "ROI": r.roi,
        "Reactivated": r.customers_reactivated,
        "Provider load": r.provider_utilisation,
    } for r in results]

    frame = pd.DataFrame(rows)

    return frame.sort_values("Incremental profit", ascending=False)


def executive_summary(results: list[ScenarioResult]) -> str:
    """Plain-language verdict on which strategy wins, and why.

    Written for someone who will read one paragraph and make a call.
    """

    if not results:
        return "No strategies to compare."

    ranked = sorted(results, key=lambda r: r.incremental_profit, reverse=True)

    best = ranked[0]

    lines: list[str] = []

    if best.is_value_destroying:
        lines.append(
            f"**None of these strategies pays for itself.** The strongest, "
            f"{best.scenario.name}, still destroys "
            f"{config.money(abs(best.incremental_profit))} against simply "
            f"sending nothing. At a "
            f"{config.percent(config.GROSS_MARGIN)} margin the discounts "
            f"cost more than the bookings they win."
        )
        lines.append(
            "**Recommendation: do not run any of these.** Test smaller "
            "discounts, or narrower targeting on the personas with genuine "
            "uplift."
        )
        return "\n\n".join(lines)

    lines.append(
        f"**{best.scenario.name} wins** — {best.scenario.describe()}.\n\n"
        f"It adds {config.money(best.incremental_profit)} of profit over "
        f"doing nothing, from {best.incremental_bookings:,.0f} extra "
        f"bookings, at a discount cost of "
        f"{config.money(best.discount_cost)}. Every rupee discounted "
        f"returns {best.roi:.2f}."
    )

    if len(ranked) > 1:
        second = ranked[1]
        gap = best.incremental_profit - second.incremental_profit

        if gap < abs(best.incremental_profit) * 0.05:
            lines.append(
                f"**It is close.** {second.scenario.name} is only "
                f"{config.money(gap)} behind, which is inside the margin of "
                f"error on these predictions. Either is defensible; pick on "
                f"operational grounds."
            )
        else:
            lines.append(
                f"It beats {second.scenario.name} by "
                f"{config.money(gap)}, a clear enough gap to act on."
            )

    destroying = [r for r in ranked if r.is_value_destroying]

    if destroying:
        names = ", ".join(r.scenario.name for r in destroying)
        lines.append(
            f"**Avoid {names}** — {'it destroys' if len(destroying) == 1 else 'they destroy'} "
            f"value against sending nothing at all."
        )

    if best.provider_utilisation > 0.85:
        lines.append(
            f"**Check supply before launching.** This would run providers at "
            f"{config.percent(best.provider_utilisation)} of capacity. "
            f"Bookings that cannot be fulfilled cost goodwill as well as "
            f"revenue."
        )

    if best.customers_reactivated > 0:
        lines.append(
            f"It also brings back {best.customers_reactivated:,} dormant "
            f"customers, whose future value is not counted in the figure "
            f"above."
        )

    return "\n\n".join(lines)
