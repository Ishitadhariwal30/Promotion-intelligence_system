"""
Urban Company Promotion Intelligence Platform.

Entry point. Streamlit discovers `pages/` automatically, so this file's
jobs are narrow: configure the app, load styling, check the data is
present, and give the platform a front door that explains what it does.

    streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

import config
from services import data_loader


# ============================================================
# SETUP
# ============================================================

st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon=config.APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_styles() -> None:
    """Inject the stylesheet.

    Widget colours come from `.streamlit/config.toml`; this covers the
    custom components Streamlit knows nothing about.
    """

    stylesheet = config.APP_ROOT / "assets" / "styles.css"

    if stylesheet.exists():
        st.markdown(
            f"<style>{stylesheet.read_text(encoding='utf-8')}</style>",
            unsafe_allow_html=True,
        )


load_styles()


# ============================================================
# SETUP SCREEN
# ============================================================

def setup_screen() -> None:
    """Shown when the exported data is missing.

    A fresh clone has no `sample_data/`, and a stack trace is a poor
    first impression. This explains the one thing that needs doing.
    """

    st.markdown(
        """
        <div class="page-header">
            <div class="page-question">Setup required</div>
            <h1 class="page-title">Connect the pipeline data</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.warning(
        "The platform runs on a snapshot exported from Databricks, and "
        "that snapshot is not here yet."
    )

    st.markdown(
        f"""
### Three steps

**1. Run `17_Export_For_App` in Databricks.**

It reads every table and the registered model, rebuilds the encodings,
and verifies they reproduce Databricks scoring before writing anything.
It modifies nothing.

**2. Download what it produces.**

Eleven Parquet files, three model artifacts and a manifest.

**3. Place them here.**

```
streamlit_app/sample_data/
├── MANIFEST.json
├── bookings.parquet
├── customers.parquet
├── ... (nine more)
└── model/
    ├── model.joblib
    ├── encoders.json
    └── feature_order.json
```

Expecting them in `{config.DATA_DIR}`.
        """
    )

    st.info(
        "**Why a snapshot rather than a live connection?** The Strategy "
        "Lab re-filters on every widget change. Network round-trips to a "
        "SQL warehouse would make it feel broken, and the simulated year "
        "is not changing anyway."
    )


# ============================================================
# LANDING
# ============================================================

def landing() -> None:
    """Front door: what this is, and where to go first."""

    st.markdown(
        f"""
        <div class="page-header">
            <div class="page-question">AI Decision Support Platform</div>
            <h1 class="page-title">Promotion Intelligence</h1>
            <div class="page-caption">
                Who should receive a promotion, which one, when, and how
                much — answered per customer, and priced against the
                margin it costs.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        manifest = data_loader.load_manifest()
        recommendations = data_loader.load_recommendations()
        bookings = data_loader.load_bookings()
    except Exception as error:
        st.error(f"Could not load the snapshot: {error}")
        return

    # --- the finding -------------------------------------------

    promoted = (recommendations["Offer_Discount"] > 0).mean()

    st.markdown("### The headline")

    cols = st.columns(3)

    cols[0].metric(
        "Customers who should get nothing",
        config.percent(1 - promoted),
        help=(
            "They book anyway, or no discount large enough to move them "
            "would pay for itself."
        ),
    )

    cols[1].metric(
        "Cost of blanket 20% discounting",
        "−42%",
        delta="of expected profit",
        delta_color="inverse",
        help=(
            "At a 35% margin, 20% off leaves 15%. Even with a 35% lift in "
            "bookings it does not pay for itself."
        ),
    )

    cols[2].metric(
        "Customers scored",
        f"{len(recommendations):,}",
        delta="17 offers each",
        delta_color="off",
    )

    st.markdown(
        """
        <div class="inline-insight">
            <div class="inline-insight-text">
                Promotions work — a 20% discount raises bookings by roughly
                35%. It still loses money, because at a 35% gross margin
                that discount surrenders more than half the profit on the
                job. The platform exists to find the customers where it
                does pay, and there are fewer of them than intuition
                suggests.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- navigation ---------------------------------------------

    st.markdown("### Where to start")

    pages = [
        ("Executive Dashboard", "How is the business performing, and what needs attention?"),
        ("Customer Intelligence", "Who are our customers, and what has happened to them?"),
        ("Demand & Bookings", "How is demand behaving, and can we serve it?"),
        ("Promotion Performance", "Which promotions work, and what are they costing?"),
        ("AI Recommendation Center", "What should we offer this specific customer?"),
        ("Strategy Lab", "What happens if we change the strategy?"),
    ]

    left, right = st.columns(2, gap="large")

    for i, (name, question) in enumerate(pages):
        target = left if i % 2 == 0 else right
        with target:
            st.markdown(
                f"""
                <div class="metric-card" style="margin-bottom:0.6rem">
                    <div class="metric-label">{question}</div>
                    <div style="font-size:1.05rem;font-weight:650;
                                color:var(--ink);margin-top:0.3rem">
                        {name}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.caption("Pages are in the sidebar.")

    # --- provenance ----------------------------------------------

    from components.metric_cards import data_quality_panel

    data_quality_panel(manifest)

    st.markdown(
        """
        <div class="action-panel">
            <div class="action-label">How to read this platform</div>
            <div class="action-text">
                Every number comes from the Databricks pipeline or the
                model trained on it — nothing is generated. Where a figure
                rests on an assumption, the page says so. Where the model
                is known to be weak, the page says that too.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# ROUTE
# ============================================================

if data_loader.data_is_available():
    landing()
else:
    setup_screen()
