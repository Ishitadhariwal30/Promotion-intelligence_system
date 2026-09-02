"""Build a promotion campaign under a budget.

This is the decision layer. Every other service in this application
describes what happened; this one decides what to do.

WHAT IT DOES
    Takes the scored offers from notebook 13, applies the constraints a
    real marketing team works under - a budget, and providers who are
    already full - and returns a named list of customers to promote.

WHY IT LIVES HERE AND NOT IN THE NOTEBOOK
    Notebook 13 runs the same logic once, at one budget. The whole point
    of the campaign screen is that the budget is a lever the user moves,
    so the ranking has to run in the application. The notebook keeps the
    columns needed to re-rank (`Profit_Per_Rupee`, `Best_Offer_*`) rather
    than overwriting them, precisely so this is possible.

WHAT BREAKS IF THE RANKING IS SKIPPED
    Promoting everyone whose best offer beats "send nothing" reaches 51%
    of customers. Ranked by return, the top decile repays 2.13x per rupee
    and the bottom decile 0.29x - so an unranked campaign spends four
    times more on the customers who repay least.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

import config


# ============================================================
# CONSTANTS
# ============================================================

DEFAULT_BUDGET_SHARE: float = 0.03
"""Promotion budget as a share of the revenue we would earn sending
nothing.

Stated as a share rather than a rupee figure so it survives the dataset
changing size. 3% is the default because it keeps roughly three quarters
of the available profit for a little over half the spend - but it is a
business decision, and the trade-off curve is on screen so it can be
changed with the evidence in view.
"""

BLOCKED_CAPACITY_STATUSES: tuple[str, ...] = ("Busy", "Oversubscribed")
"""Cell states where a promotion is withheld.

Busy is 85-100% full and Oversubscribed is over 100%. Discounting into
either buys almost nothing - those customers were coming anyway - and
risks generating a booking there is nobody to serve. Notebook 23 counted
612 bookings already lost for want of a provider; advertising into those
cells makes that number larger, not smaller.
"""

REASON_PROMOTED = "Promoted"
REASON_WOULD_BOOK = "Would book anyway"
REASON_CELL_FULL = "Cell already full"
REASON_NO_BUDGET = "Budget ran out"


# ============================================================
# RESULT
# ============================================================

@dataclass(frozen=True)
class Campaign:
    """One campaign, and everything a screen needs to explain it."""

    selected: pd.DataFrame
    """The customers to promote, best return first."""

    scored: pd.DataFrame
    """Every customer considered, with a `Decision_Reason` on each."""

    budget: float
    """Rupees available."""

    spend: float
    """Rupees committed. Always at or under `budget`."""

    expected_profit: float
    """Incremental profit over sending nothing, in rupees."""

    baseline_revenue: float
    """Revenue if we promoted nobody. The denominator for the budget."""

    @property
    def promoted_count(self) -> int:
        return len(self.selected)

    @property
    def considered_count(self) -> int:
        return len(self.scored)

    @property
    def return_per_rupee(self) -> float:
        """Profit gained for each rupee of discount committed."""
        return self.expected_profit / self.spend if self.spend else 0.0

    @property
    def reason_counts(self) -> pd.Series:
        """How many customers landed on each outcome, and why."""
        return self.scored["Decision_Reason"].value_counts()


# ============================================================
# ECONOMICS
# ============================================================

def add_economics(
    recommendations: pd.DataFrame,
    margin: float = config.GROSS_MARGIN,
) -> pd.DataFrame:
    """Attach the four numbers the ranking needs.

    Notebook 13 writes these already, but recomputing them here means the
    screen still works against an older export - and means the definition
    lives somewhere a reader can check it.

    The one that is easy to get wrong is `Promotion_Cost`. A discount is
    only paid if the customer actually books, so the expected spend is
    the face value multiplied by the booking probability. Charging full
    face value to everyone we email overstates the budget badly, and
    would make the cap far more conservative than intended.
    """

    frame = recommendations.copy()

    # What the model wanted, before any constraint was applied. Falls
    # back to the post-constraint columns when reading an older export.
    for wanted, fallback in [
        ("Best_Offer_Type", "Offer_Type"),
        ("Best_Offer_Discount", "Offer_Discount"),
        ("Best_Offer_Probability", "Booking_Probability"),
        ("Best_Offer_Profit", "Expected_Profit"),
    ]:
        if wanted not in frame.columns:
            frame[wanted] = frame[fallback]

    frame["Baseline_Profit"] = (
        frame["Baseline_Probability"] * frame["Seasonal_Price"] * margin
    )

    frame["Incremental_Profit"] = (
        frame["Best_Offer_Profit"] - frame["Baseline_Profit"]
    )

    frame["Promotion_Cost"] = (
        frame["Best_Offer_Probability"]
        * frame["Seasonal_Price"]
        * frame["Best_Offer_Discount"]
        / 100.0
    )

    # Dividing by a zero cost would give infinity and sort to the top of
    # the queue - a "free" promotion that is really just no promotion.
    frame["Profit_Per_Rupee"] = (
        frame["Incremental_Profit"]
        / frame["Promotion_Cost"].where(frame["Promotion_Cost"] > 0)
    )

    return frame


def attach_capacity(
    recommendations: pd.DataFrame,
    capacity: pd.DataFrame | None,
) -> pd.DataFrame:
    """Join each customer to how strained their cell was that day.

    Returns the frame with a `capacity_status` column. When the capacity
    export is missing, every customer is marked ``Unknown`` and no
    promotion is withheld - the screen says so rather than pretending the
    check ran.
    """

    frame = recommendations.copy()

    # Notebook 13 already writes a capacity_status. Keeping it would make
    # the merge produce capacity_status_x and capacity_status_y, and the
    # column this function promises to return would not exist. We drop it
    # and re-derive, so the screen always reflects the capacity file on
    # disk rather than whatever the notebook saw when it last ran.
    frame = frame.drop(columns=["capacity_status"], errors="ignore")

    if capacity is None or capacity.empty:
        frame["capacity_status"] = "Unknown"
        return frame

    cells = (
        capacity[["Date", "City", "Service_Name", "capacity_status"]]
        .drop_duplicates()
        .copy()
    )

    # Both sides must be the same kind of date. A datetime meeting a date
    # matches nothing, silently, and every customer would look
    # unconstrained.
    cells["_join_date"] = pd.to_datetime(cells["Date"]).dt.normalize()
    frame["_join_date"] = pd.to_datetime(frame["Activity_Date"]).dt.normalize()

    frame = frame.merge(
        cells.drop(columns=["Date"]),
        on=["_join_date", "City", "Service_Name"],
        how="left",
    )

    frame["capacity_status"] = frame["capacity_status"].fillna("Unknown")

    return frame.drop(columns=["_join_date"])


# ============================================================
# THE DECISION
# ============================================================

def build_campaign(
    recommendations: pd.DataFrame,
    capacity: pd.DataFrame | None = None,
    budget_share: float = DEFAULT_BUDGET_SHARE,
    respect_capacity: bool = True,
    margin: float = config.GROSS_MARGIN,
) -> Campaign:
    """Decide who to promote, under a budget and capacity constraints.

    Args:
        recommendations: Scored offers from notebook 13, already filtered
            to whatever date and city the user selected.
        capacity: Daily utilisation per cell, or None if not exported.
        budget_share: Budget as a share of baseline revenue.
        respect_capacity: Turn the capacity gate off to show what it is
            worth. Useful in a demo; not something to ship as a default.
        margin: Gross margin on a completed job.

    Returns:
        A `Campaign` carrying the selection, the rejects with reasons,
        and the totals.
    """

    scored = add_economics(recommendations, margin=margin)
    scored = attach_capacity(scored, capacity)

    baseline_revenue = float(
        (scored["Baseline_Probability"] * scored["Seasonal_Price"]).sum()
    )
    budget = baseline_revenue * budget_share

    # ---- who is even a candidate ----

    wants_promotion = scored["Best_Offer_Type"] != "None"

    cell_is_full = (
        scored["capacity_status"].isin(BLOCKED_CAPACITY_STATUSES)
        if respect_capacity
        else pd.Series(False, index=scored.index)
    )

    eligible = wants_promotion & ~cell_is_full & scored["Profit_Per_Rupee"].notna()

    # ---- rank, then spend down the list ----
    #
    # Sorting by return per rupee and taking until the money runs out is
    # the greedy solution to a knapsack problem. It is not provably
    # optimal, but the gap is small enough that the extra complexity
    # would be harder to explain than it is worth.

    queue = scored[eligible].sort_values("Profit_Per_Rupee", ascending=False)

    running_cost = queue["Promotion_Cost"].cumsum()
    affordable = running_cost <= budget

    scored["Budget_Rank"] = pd.Series(
        range(1, len(queue) + 1), index=queue.index
    )

    selected = queue[affordable].copy()
    selected_ids = set(selected.index)

    # ---- one plain reason per customer ----

    scored["Decision_Reason"] = REASON_WOULD_BOOK
    scored.loc[wants_promotion, "Decision_Reason"] = REASON_NO_BUDGET
    scored.loc[cell_is_full & wants_promotion, "Decision_Reason"] = REASON_CELL_FULL
    scored.loc[list(selected_ids), "Decision_Reason"] = REASON_PROMOTED

    return Campaign(
        selected=selected,
        scored=scored,
        budget=budget,
        spend=float(selected["Promotion_Cost"].sum()),
        expected_profit=float(selected["Incremental_Profit"].sum()),
        baseline_revenue=baseline_revenue,
    )


def budget_curve(
    recommendations: pd.DataFrame,
    capacity: pd.DataFrame | None = None,
    shares: tuple[float, ...] = (0.01, 0.02, 0.03, 0.05, 0.075, 0.10),
    respect_capacity: bool = True,
) -> pd.DataFrame:
    """What each budget level buys.

    The point of showing this rather than a single number: the curve is
    steeply diminishing. Doubling the spend does not double the return,
    and the person who owns the budget should see where it flattens
    before picking a figure.
    """

    scored = add_economics(recommendations)
    scored = attach_capacity(scored, capacity)

    baseline_revenue = float(
        (scored["Baseline_Probability"] * scored["Seasonal_Price"]).sum()
    )

    eligible = scored["Best_Offer_Type"] != "None"
    if respect_capacity:
        eligible &= ~scored["capacity_status"].isin(BLOCKED_CAPACITY_STATUSES)
    eligible &= scored["Profit_Per_Rupee"].notna()

    queue = scored[eligible].sort_values("Profit_Per_Rupee", ascending=False)
    running_cost = queue["Promotion_Cost"].cumsum()
    total_profit = float(queue["Incremental_Profit"].sum())

    rows = []

    for share in shares:
        affordable = queue[running_cost <= baseline_revenue * share]
        profit = float(affordable["Incremental_Profit"].sum())
        rows.append({
            "budget_share": share,
            "budget": baseline_revenue * share,
            "promotions": len(affordable),
            "share_of_customers": len(affordable) / max(len(scored), 1),
            "spend": float(affordable["Promotion_Cost"].sum()),
            "profit": profit,
            "share_of_profit": profit / total_profit if total_profit else 0.0,
        })

    return pd.DataFrame(rows)
