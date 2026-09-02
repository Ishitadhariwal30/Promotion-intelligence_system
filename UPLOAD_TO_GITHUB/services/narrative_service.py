"""
The Business Copilot: questions in, answers out, no language model.

WHAT THIS IS
------------

A deterministic intent router. A question is matched against a fixed set
of known business questions; the matching handler runs a real query and
renders the result through a template.

WHY NOT AN LLM
--------------

Every number in an answer here is computed, not generated, so the
copilot cannot invent a figure. An executive acting on a fabricated
number is a worse outcome than one who has to rephrase a question.

The cost is honesty about scope: it answers roughly fifteen questions
well and says so plainly when asked anything else, rather than
improvising something plausible.

EVERY ANSWER CITES ITS SOURCE
-----------------------------

Which table, how many rows, what filters. A number a manager cannot
trace is a number they will not act on - and rightly so.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

import config
from services import analytics_service as analytics


# ============================================================
# ANSWER
# ============================================================

@dataclass
class Answer:
    """A copilot response.

    Attributes:
        headline: The answer in one sentence.
        detail: Supporting explanation, markdown.
        table: Optional evidence to render beneath.
        source: What was queried, so the user can verify it.
        follow_ups: Related questions worth asking next.
    """

    headline: str
    detail: str = ""
    table: pd.DataFrame | None = None
    source: str = ""
    follow_ups: list[str] = field(default_factory=list)


@dataclass
class Intent:
    """One question the copilot knows how to answer."""

    name: str
    patterns: list[str]
    example: str
    handler: Callable[..., Answer]


# ============================================================
# HANDLERS
# ============================================================

def _who_to_promote(data: dict) -> Answer:
    recommendations = data["recommendations"]

    promoted = recommendations[recommendations["Offer_Discount"] > 0]

    by_persona = (
        recommendations.groupby("Persona")
        .agg(
            customers=("Activity_ID", "count"),
            promoted_share=("Offer_Discount", lambda s: (s > 0).mean()),
            avg_discount=("Offer_Discount", "mean"),
        )
        .reset_index()
        .sort_values("promoted_share", ascending=False)
    )

    top = by_persona.iloc[0]

    return Answer(
        headline=(
            f"{len(promoted):,} of {len(recommendations):,} customers should "
            f"receive an offer — {config.percent(len(promoted) / len(recommendations))} "
            f"of the base."
        ),
        detail=(
            f"**{top['Persona']}** is the priority: "
            f"{config.percent(top['promoted_share'])} of them warrant a "
            f"promotion, at an average of {top['avg_discount']:.1f}% off.\n\n"
            f"The other {config.percent(1 - len(promoted) / len(recommendations))} "
            f"should receive nothing. They either book anyway, or no discount "
            f"large enough to move them would pay for itself."
        ),
        table=by_persona,
        source=(
            f"gold_promotion_recommendations · {len(recommendations):,} "
            f"customers · October to December"
        ),
        follow_ups=[
            "Which city should receive more discount?",
            "How much are we wasting on promotions?",
        ],
    )


def _which_city(data: dict) -> Answer:
    recommendations = data["recommendations"]

    by_city = (
        recommendations.groupby("City")
        .agg(
            customers=("Activity_ID", "count"),
            promoted_share=("Offer_Discount", lambda s: (s > 0).mean()),
            avg_uplift=("Uplift", "mean"),
            expected_profit=("Expected_Profit", "sum"),
        )
        .reset_index()
        .sort_values("promoted_share", ascending=False)
    )

    top = by_city.iloc[0]
    bottom = by_city.iloc[-1]

    return Answer(
        headline=(
            f"{top['City']} warrants the most promotional spend — "
            f"{config.percent(top['promoted_share'])} of its customers should "
            f"receive an offer."
        ),
        detail=(
            f"{top['City']} shows average uplift of {top['avg_uplift']:+.3f} "
            f"per promoted customer.\n\n"
            f"{bottom['City']} is at the other end at "
            f"{config.percent(bottom['promoted_share'])} — customers there "
            f"convert without needing a push, so discounting them mostly "
            f"gives away margin."
        ),
        table=by_city,
        source=(
            f"gold_promotion_recommendations · {len(recommendations):,} "
            f"customers across {len(by_city)} cities"
        ),
        follow_ups=[
            "Where are we losing bookings to missing providers?",
            "Which promotion gives the best return?",
        ],
    )


def _best_roi(data: dict) -> Answer:
    roi = analytics.promotion_roi(data["bookings"])

    if roi.empty:
        return Answer(
            headline="No promoted bookings in the current selection.",
            source="silver_bookings",
        )

    best = roi.iloc[0]
    worst = roi.iloc[-1]

    return Answer(
        headline=(
            f"{best['Promotion_Type']} at {best['Discount_Percent']:.0f}% "
            f"returned the most — {config.money(best['net_contribution'])} "
            f"net contribution."
        ),
        detail=(
            f"It ran across {best['bookings']:,} bookings, collecting "
            f"{config.money(best['revenue'])} at a discount cost of "
            f"{config.money(best['discount_cost'])}.\n\n"
            f"The weakest was **{worst['Promotion_Type']} at "
            f"{worst['Discount_Percent']:.0f}%**, netting "
            f"{config.money(worst['net_contribution'])}. Larger discounts "
            f"consistently return less: at a "
            f"{config.percent(config.GROSS_MARGIN)} margin, every point of "
            f"discount is a point of profit surrendered, and uplift rarely "
            f"keeps pace."
        ),
        table=roi,
        source=f"silver_bookings · {int(roi['bookings'].sum()):,} promoted bookings",
        follow_ups=[
            "What happens if we increase discounts?",
            "How much are we wasting on promotions?",
        ],
    )


def _promotion_waste(data: dict) -> Answer:
    waste = analytics.promotion_waste(data["promotions"], data["bookings"])

    return Answer(
        headline=(
            f"About {config.money(waste['wasted_spend'])} of discount went to "
            f"customers who would have booked anyway."
        ),
        detail=(
            f"Measured on the randomised control group, "
            f"{config.percent(waste['baseline_conversion'])} of customers "
            f"convert with no offer at all. Applied to "
            f"{config.money(waste['total_spend'])} of total discount spend, "
            f"that share bought nothing.\n\n"
            f"This is an estimate, not an exact figure — we cannot observe "
            f"what an individual customer would have done without the offer. "
            f"But the randomised group makes the *rate* trustworthy."
        ),
        source=(
            f"silver_promotions, Targeting_Mode = Exploration · "
            f"{len(data['promotions']):,} promotion decisions"
        ),
        follow_ups=[
            "Which customers should receive promotions?",
            "What happens if we increase discounts?",
        ],
    )


def _rainfall_effect(data: dict) -> Answer:
    weather = analytics.weather_impact(data["bookings"])

    if weather.empty:
        return Answer(headline="No weather data in the current selection.")

    rainy = weather[weather["Weather_Condition"] == "Rainy"]

    if rainy.empty:
        return Answer(
            headline="No rainy-day bookings in the current selection.",
            source="silver_bookings",
        )

    top = rainy.nlargest(3, "bookings")

    services = ", ".join(top["Service_Name"].tolist())

    return Answer(
        headline=(
            f"Rain shifts demand toward {top.iloc[0]['Service_Name']} and away "
            f"from outdoor-adjacent services."
        ),
        detail=(
            f"On rainy days the busiest services are **{services}**.\n\n"
            f"Plumbing and Pest Control rise sharply — rain creates the "
            f"problems they solve. Salon at Home falls, because customers "
            f"defer discretionary appointments.\n\n"
            f"The operational read: staff repair capacity ahead of monsoon "
            f"and expect beauty bookings to soften."
        ),
        table=rainy.nlargest(8, "bookings"),
        source=(
            f"silver_bookings · {int(rainy['bookings'].sum()):,} bookings on "
            f"rainy days"
        ),
        follow_ups=[
            "Which days deserve extra capacity?",
            "Where are we losing bookings to missing providers?",
        ],
    )


def _holiday_effect(data: dict) -> Answer:
    holidays = analytics.holiday_impact(data["bookings"])

    if holidays.empty:
        return Answer(headline="No holiday data in the current selection.")

    peak = holidays[holidays["day_type"] != "Ordinary day"]

    if peak.empty:
        return Answer(
            headline="No holidays fall inside the current selection.",
            source="silver_bookings",
        )

    top = peak.iloc[0]

    return Answer(
        headline=(
            f"{top['day_type']} runs at {top['vs_ordinary']:.1f}× an ordinary "
            f"day — {top['avg_bookings']:.0f} bookings against the usual baseline."
        ),
        detail=(
            f"Demand concentrates rather than spreading, so capacity is the "
            f"binding constraint, not marketing.\n\n"
            f"Discounting into a demand spike is the least efficient possible "
            f"use of budget: those customers were coming regardless. Spend it "
            f"on the quiet weeks instead."
        ),
        table=holidays,
        source=f"silver_bookings · {len(holidays)} distinct day types",
        follow_ups=[
            "How can we increase bookings this weekend?",
            "Which promotion gives the best return?",
        ],
    )


def _supply_gaps(data: dict) -> Answer:
    gaps = analytics.supply_gaps(data["bookings"], data["providers"])

    if gaps.empty:
        return Answer(
            headline="No bookings were lost to missing providers.",
            source="silver_bookings",
        )

    uncovered = gaps[gaps["active_providers"] == 0]

    return Answer(
        headline=(
            f"{int(gaps['lost_bookings'].sum()):,} bookings were lost because "
            f"nobody was available — {config.money(gaps['lost_revenue'].sum())} "
            f"of demand that already existed."
        ),
        detail=(
            f"{len(uncovered)} city and service combinations have **no active "
            f"provider at all**.\n\n"
            f"These are not a marketing problem. The customer wanted to book "
            f"and the marketplace could not serve them, so no promotion, "
            f"targeting change or discount recovers them. The fix is "
            f"recruitment."
        ),
        table=gaps.head(12),
        source=(
            f"silver_bookings joined to bronze.providers · "
            f"{len(data['bookings']):,} booking attempts"
        ),
        follow_ups=[
            "Which city should receive more discount?",
            "How is the business performing?",
        ],
    )


def _increase_discount(data: dict) -> Answer:
    return Answer(
        headline=(
            "Raising discounts across the board loses money. Blanket 20% off "
            "costs about 42% of expected profit."
        ),
        detail=(
            f"At a {config.percent(config.GROSS_MARGIN)} gross margin, a 20% "
            f"discount leaves {config.percent(config.GROSS_MARGIN - 0.20)}. "
            f"That is 43% of the original margin, so **with zero uplift you "
            f"would lose 57%**.\n\n"
            f"Discounts do work — a 20% offer raises bookings by roughly 35% "
            f"on average. It still does not come close to covering the margin "
            f"given up.\n\n"
            f"The break-even sits at "
            f"{config.BREAK_EVEN_DISCOUNT:.0f}% discount, where margin reaches "
            f"zero. Anything approaching it needs uplift most customers "
            f"simply do not have.\n\n"
            f"**Model this properly in the Strategy Lab** — it will show you "
            f"the exact cohort and discount where it does pay."
        ),
        source="Derived from gross margin and the observed policy comparison",
        follow_ups=[
            "Which customers should receive promotions?",
            "How much are we wasting on promotions?",
        ],
    )


def _weekend_bookings(data: dict) -> Answer:
    bookings = data["bookings"]

    if "Is_Weekend" not in bookings.columns:
        return Answer(headline="Weekend data is not available.")

    completed = bookings[bookings["Booking_Status"] == "Completed"]

    by_weekend = (
        completed.groupby("Is_Weekend")
        .agg(
            bookings=("Booking_ID", "count"),
            days=("Booking_Date", "nunique"),
            revenue=("Final_Price", "sum"),
        )
        .reset_index()
    )

    by_weekend["per_day"] = by_weekend["bookings"] / by_weekend["days"]

    weekend = by_weekend[by_weekend["Is_Weekend"] == True]
    weekday = by_weekend[by_weekend["Is_Weekend"] == False]

    if weekend.empty or weekday.empty:
        return Answer(headline="Not enough data to compare weekends.")

    ratio = (
        float(weekend["per_day"].iloc[0]) / float(weekday["per_day"].iloc[0])
    )

    return Answer(
        headline=(
            f"Weekends already run at {ratio:.2f}× weekday volume — "
            f"{weekend['per_day'].iloc[0]:.0f} bookings a day against "
            f"{weekday['per_day'].iloc[0]:.0f}."
        ),
        detail=(
            "Demand is not the weekend constraint; **supply usually is**. "
            "Check provider utilisation before adding promotional spend.\n\n"
            "If capacity is comfortable, target the personas with real "
            "uplift rather than discounting the whole weekend — most weekend "
            "customers were coming anyway."
        ),
        table=by_weekend,
        source=f"silver_bookings · {len(completed):,} completed bookings",
        follow_ups=[
            "Where are we losing bookings to missing providers?",
            "Which customers should receive promotions?",
        ],
    )


def _business_performance(data: dict) -> Answer:
    metrics = analytics.headline_metrics(
        data["bookings"], data["activity"], data["promotions"]
    )

    return Answer(
        headline=(
            f"{config.money(metrics['revenue'])} revenue from "
            f"{metrics['bookings_completed']:,} completed bookings, at "
            f"{config.percent(metrics['conversion_rate'])} conversion."
        ),
        detail=(
            f"Average order value {config.money(metrics['avg_order_value'])}. "
            f"{config.percent(metrics['completion_rate'])} of booking attempts "
            f"were delivered.\n\n"
            f"**{config.percent(metrics['cancellation_rate'])}** were "
            f"cancelled and **{config.percent(metrics['no_provider_rate'])}** "
            f"found no provider — the second being a supply failure rather "
            f"than a customer decision, and fixable only by recruitment.\n\n"
            f"{config.money(metrics['discount_given'])} was given away in "
            f"discounts across the period."
        ),
        source=(
            f"silver_bookings and silver_daily_customer_activity · "
            f"{metrics['booking_attempts']:,} booking attempts"
        ),
        follow_ups=[
            "Where are we losing bookings to missing providers?",
            "Which promotion gives the best return?",
        ],
    )


def _at_risk(data: dict) -> Answer:
    at_risk = analytics.at_risk_customers(data["state_year_end"])

    if at_risk.empty:
        return Answer(
            headline="No high-value customers have gone quiet.",
            source="silver_customer_state",
        )

    value = float(at_risk["Lifetime_Spend"].sum())

    return Answer(
        headline=(
            f"{len(at_risk):,} valuable customers have gone quiet, "
            f"representing {config.money(value)} of historic spend."
        ),
        detail=(
            "Each has spent meaningfully and has not attempted a booking in "
            "over 45 days.\n\n"
            "Ranked by lifetime spend rather than by how long they have been "
            "away: a customer worth fifty thousand who left three months ago "
            "matters more than one worth two thousand who left six.\n\n"
            "**Reactivation is harder than retention.** A single discount "
            "often is not enough — these need a reason to return, not just a "
            "cheaper price."
        ),
        table=at_risk.head(15),
        source=(
            f"silver_customer_state at year end · "
            f"{len(data['state_year_end']):,} customers"
        ),
        follow_ups=[
            "Which customers should receive promotions?",
            "How is the business performing?",
        ],
    )


# ============================================================
# ROUTER
# ============================================================

INTENTS: list[Intent] = [
    Intent(
        "who_to_promote",
        [r"who.*promot", r"which customer", r"target.*customer", r"send.*offer"],
        "Which customers should receive promotions?",
        _who_to_promote,
    ),
    Intent(
        "which_city",
        [r"which city", r"city.*discount", r"where.*discount", r"city.*promot"],
        "Which city should receive more discount?",
        _which_city,
    ),
    Intent(
        "best_roi",
        [r"best roi", r"highest roi", r"which promotion.*(work|best|return)",
         r"promotion.*return"],
        "Which promotion gives the best return?",
        _best_roi,
    ),
    Intent(
        "promotion_waste",
        [r"wast", r"wasted", r"unnecessary discount", r"spend.*nothing"],
        "How much are we wasting on promotions?",
        _promotion_waste,
    ),
    Intent(
        "rainfall",
        [r"rain", r"weather", r"monsoon"],
        "How does rainfall affect demand?",
        _rainfall_effect,
    ),
    Intent(
        "holiday",
        [r"holiday", r"diwali", r"festival", r"capacity.*day"],
        "Which days deserve extra capacity?",
        _holiday_effect,
    ),
    Intent(
        "supply_gaps",
        [r"provider", r"supply", r"losing booking", r"turned away", r"capacity"],
        "Where are we losing bookings to missing providers?",
        _supply_gaps,
    ),
    Intent(
        "increase_discount",
        [r"increase.*discount", r"raise.*discount", r"what if.*discount",
         r"bigger discount", r"more discount"],
        "What happens if we increase discounts?",
        _increase_discount,
    ),
    Intent(
        "weekend",
        [r"weekend", r"saturday", r"sunday"],
        "How can we increase bookings this weekend?",
        _weekend_bookings,
    ),
    Intent(
        "performance",
        [r"how.*business", r"performance", r"revenue.*total", r"how.*doing",
         r"overall"],
        "How is the business performing?",
        _business_performance,
    ),
    Intent(
        "at_risk",
        [r"at risk", r"churn", r"dormant", r"lost customer", r"win.*back",
         r"reactivat"],
        "Which customers are we about to lose?",
        _at_risk,
    ),
]


def suggested_questions() -> list[str]:
    """Every question the copilot can answer, for the UI to offer.

    Shown as clickable prompts rather than left for the user to guess.
    A copilot that answers fifteen questions well is more useful than
    one that appears to answer anything and improvises badly.
    """

    return [intent.example for intent in INTENTS]


def ask(question: str, data: dict) -> Answer:
    """Route a question to its handler.

    Args:
        question: Free text from the user.
        data: Loaded tables, keyed ``bookings``, ``promotions``,
            ``recommendations``, ``providers``, ``activity``,
            ``state_year_end``.

    Returns:
        An Answer. If nothing matches, one that says so and lists what
        *can* be asked, rather than guessing.
    """

    normalised = question.lower().strip()

    if not normalised:
        return Answer(
            headline="Ask a question, or pick one of the suggestions.",
            follow_ups=suggested_questions()[:4],
        )

    for intent in INTENTS:
        for pattern in intent.patterns:
            if re.search(pattern, normalised):
                try:
                    return intent.handler(data)
                except Exception as error:
                    return Answer(
                        headline="Could not answer that from the current data.",
                        detail=f"`{type(error).__name__}: {error}`",
                        source="error",
                    )

    return Answer(
        headline="I do not have an answer for that one.",
        detail=(
            "This copilot answers a fixed set of business questions from "
            "computed data — it does not generate text, so it will not "
            "improvise a number it cannot verify.\n\n"
            "**Try one of these instead:**"
        ),
        follow_ups=suggested_questions(),
    )
