"""Campaign Builder - the decision screen.

WHAT THIS IS
    Every other page in this application describes what happened. This
    one decides what to do: you set a budget, it returns a named list of
    customers to promote, tells you what it rejected and why, and lets
    you export the result.

WHY IT IS PAGE ZERO
    The product is a promotion recommendation system. A viewer should
    meet the decision first and the evidence second, not the other way
    round.

INPUT   recommendations.parquet, capacity.parquet (optional)
OUTPUT  an on-screen campaign and a downloadable CSV
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from services import campaign_service as cs
from services import data_loader

st.set_page_config(page_title="Campaign Builder", page_icon="🎯",
                   layout="wide")

st.title("Campaign Builder")
st.caption(
    "Set the constraints. The system decides who to promote, and shows "
    "you what it turned down."
)

# ============================================================
# 1. LOAD
# ============================================================

try:
    recommendations = data_loader.load_recommendations()
except data_loader.DataNotFoundError as error:
    st.error(str(error))
    st.stop()

capacity = None
capacity_available = data_loader.capacity_is_available()

if capacity_available:
    capacity = data_loader.load_capacity()

# ============================================================
# 2. CONTROLS
# ============================================================

left, middle, right = st.columns([2, 2, 3])

with left:
    cities = ["All"] + sorted(recommendations["City"].dropna().unique())
    chosen_city = st.selectbox("City", cities)

with middle:
    services = ["All"] + sorted(
        recommendations["Service_Name"].dropna().unique()
    )
    chosen_service = st.selectbox("Service", services)

with right:
    # The slider runs in percentage POINTS, not in a 0-1 fraction.
    # Streamlit's `format` is a printf template applied to the raw
    # value - it does not scale - so a fraction here would render 10%
    # as "0.1%". Everything downstream wants the fraction, so convert
    # once, immediately.

    budget_percent = st.slider(
        "Promotion budget, as a share of baseline revenue",
        min_value=0.5,
        max_value=10.0,
        value=cs.DEFAULT_BUDGET_SHARE * 100,
        step=0.5,
        format="%.1f%%",
        help=(
            "Baseline revenue is what we would earn promoting nobody. "
            "Expressing the budget as a share of it keeps the number "
            "meaningful when the dataset changes size."
        ),
    )

    budget_share = budget_percent / 100.0

respect_capacity = st.checkbox(
    "Withhold promotions where providers are already full",
    value=True,
    disabled=not capacity_available,
    help=(
        "Discounting into a full cell buys almost nothing - those "
        "customers were coming anyway - and risks a booking nobody can "
        "serve. Untick to see what the constraint is costing you."
    ),
)

if not capacity_available:
    st.info(
        "**Capacity data not found.** The campaign will still rank by "
        "return, but it cannot withhold promotions where providers are "
        "full. Run notebook 23, then notebook 17, and place "
        "`capacity.parquet` in `sample_data/`."
    )

# ============================================================
# 3. FILTER, THEN DECIDE
# ============================================================

selection = recommendations

if chosen_city != "All":
    selection = selection[selection["City"] == chosen_city]

if chosen_service != "All":
    selection = selection[selection["Service_Name"] == chosen_service]

if selection.empty:
    st.warning("No scored customers match that combination.")
    st.stop()

campaign = cs.build_campaign(
    selection,
    capacity=capacity,
    budget_share=budget_share,
    respect_capacity=respect_capacity and capacity_available,
)

# ============================================================
# 4. THE ANSWER
# ============================================================

st.divider()

a, b, c, d = st.columns(4)

a.metric(
    "Customers to promote",
    f"{campaign.promoted_count:,}",
    f"{campaign.promoted_count / campaign.considered_count:.1%} of "
    f"{campaign.considered_count:,} scored",
)
b.metric("Discount spend", f"₹{campaign.spend:,.0f}",
         f"budget ₹{campaign.budget:,.0f}")
c.metric("Expected profit added", f"₹{campaign.expected_profit:,.0f}")
d.metric("Return per rupee spent", f"{campaign.return_per_rupee:.2f}×")

# ============================================================
# 5. WHAT WAS TURNED DOWN, AND WHY
# ============================================================

st.subheader("Who did not get a promotion, and why")

st.caption(
    "Three different reasons, three different meanings. Only one of "
    "them is a budget problem."
)

reasons = campaign.reason_counts

EXPLANATIONS = {
    cs.REASON_WOULD_BOOK: (
        "The model expects them to book without a discount. Promoting "
        "them would pay for a booking we were getting anyway."
    ),
    cs.REASON_CELL_FULL: (
        "Their city and service is already at or near capacity. A "
        "discount here generates demand we cannot serve."
    ),
    cs.REASON_NO_BUDGET: (
        "A worthwhile promotion that ranked below the cut-off. Raise "
        "the budget and these are the customers it buys."
    ),
}

for reason, explanation in EXPLANATIONS.items():

    count = int(reasons.get(reason, 0))

    if not count:
        continue

    with st.container(border=True):
        st.markdown(f"**{count:,} — {reason}**")
        st.caption(explanation)

# ============================================================
# 6. WHAT A DIFFERENT BUDGET WOULD BUY
# ============================================================

st.subheader("What a different budget would buy")

st.caption(
    "The curve flattens. Doubling the spend does not double the return, "
    "because promotions are funded best-first."
)

curve = cs.budget_curve(
    selection,
    capacity=capacity,
    respect_capacity=respect_capacity and capacity_available,
)

display_curve = pd.DataFrame({
    "Budget": curve["budget_share"].map("{:.1%}".format),
    "Promotions": curve["promotions"].map("{:,}".format),
    "Spend": curve["spend"].map("₹{:,.0f}".format),
    "Profit added": curve["profit"].map("₹{:,.0f}".format),
    "Share of available profit": curve["share_of_profit"],
})

st.dataframe(
    display_curve,
    hide_index=True,
    use_container_width=True,
    column_config={
        "Share of available profit": st.column_config.ProgressColumn(
            "Share of available profit", min_value=0.0, max_value=1.0,
            format="%.0f%%",
        )
    },
)

# ============================================================
# 7. THE CAMPAIGN ITSELF
# ============================================================

st.subheader("The campaign")

st.caption(
    "Best return first. Rank 1 is the promotion that repays most per "
    "rupee of discount."
)

campaign_table = campaign.selected.copy()
campaign_table.insert(0, "Rank", range(1, len(campaign_table) + 1))

COLUMNS = {
    "Rank": "Rank",
    "Customer_ID": "Customer",
    "City": "City",
    "Service_Name": "Service",
    "Customer_Segment": "Segment",
    "Best_Offer_Type": "Offer",
    "Best_Offer_Discount": "Discount %",
    "Best_Offer_Uplift": "Uplift",
    "Promotion_Cost": "Cost",
    "Incremental_Profit": "Profit added",
    "Profit_Per_Rupee": "Return",
}

present = [c for c in COLUMNS if c in campaign_table.columns]

st.dataframe(
    campaign_table[present].rename(columns=COLUMNS).head(500),
    hide_index=True,
    use_container_width=True,
    column_config={
        "Cost": st.column_config.NumberColumn("Cost", format="₹%.0f"),
        "Profit added": st.column_config.NumberColumn(
            "Profit added", format="₹%.0f"),
        "Return": st.column_config.NumberColumn("Return", format="%.2f×"),
        "Uplift": st.column_config.NumberColumn("Uplift", format="%.3f"),
    },
)

if len(campaign_table) > 500:
    st.caption(
        f"Showing the top 500 of {len(campaign_table):,}. The export "
        f"below contains all of them."
    )

# ============================================================
# 8. TAKE IT SOMEWHERE
# ============================================================

export = campaign_table[present].rename(columns=COLUMNS)

st.download_button(
    "Export campaign list (CSV)",
    data=export.to_csv(index=False).encode("utf-8"),
    file_name=(
        f"campaign_{chosen_city.lower().replace(' ', '_')}"
        f"_{budget_share:.3f}.csv"
    ),
    mime="text/csv",
    type="primary",
)

st.caption(
    f"Margin assumed at {config.GROSS_MARGIN:.0%}. Discount cost is "
    f"weighted by booking probability, because a discount is only paid "
    f"if the customer actually books."
)
