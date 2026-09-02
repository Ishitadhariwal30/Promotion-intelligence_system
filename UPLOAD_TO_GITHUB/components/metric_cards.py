"""
KPI tiles and the alert strip.

Two components with different jobs and a deliberate order on the page:
alerts sit **above** metrics, because a manager needs to know what is
wrong before they read what is happening. A dashboard that leads with
nine healthy-looking numbers buries the one thing worth acting on.
"""

from __future__ import annotations

from typing import Literal

import streamlit as st

import config
from services.analytics_service import Alert


Severity = Literal["good", "warn", "bad", "neutral"]


# ============================================================
# METRIC TILE
# ============================================================

def metric_card(
    label: str,
    value: str,
    caption: str | None = None,
    severity: Severity = "neutral",
    help_text: str | None = None,
) -> None:
    """One KPI tile.

    Args:
        label: What the number is.
        value: The formatted number.
        caption: Context beneath it - a comparison, a share, a period.
        severity: Colours the value. Use sparingly; if every tile is
            coloured, none of them stand out.
        help_text: Tooltip explaining how the figure is calculated.
            Worth filling in for anything derived, because an
            unexplained number invites a question rather than a
            decision.
    """

    colour = {
        "good": config.COLOR["good"],
        "warn": config.COLOR["warn"],
        "bad": config.COLOR["bad"],
        "neutral": config.COLOR["ink"],
    }[severity]

    tooltip = f' title="{help_text}"' if help_text else ""

    st.markdown(
        f"""
        <div class="metric-card"{tooltip}>
            <div class="metric-label">{label}</div>
            <div class="metric-value" style="color:{colour}">{value}</div>
            <div class="metric-caption">{caption or "&nbsp;"}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_row(
    metrics: list[dict],
    columns: int | None = None,
) -> None:
    """A row of KPI tiles.

    Args:
        metrics: Dicts matching `metric_card`'s arguments.
        columns: Tiles per row. Defaults to one per metric, which is
            right up to about five; beyond that pass a number so they
            wrap instead of shrinking to unreadable.
    """

    if not metrics:
        return

    per_row = columns or len(metrics)

    for start in range(0, len(metrics), per_row):

        chunk = metrics[start:start + per_row]

        cols = st.columns(len(chunk), gap="small")

        for col, metric in zip(cols, chunk):
            with col:
                metric_card(**metric)


# ============================================================
# ALERTS
# ============================================================

def alert_strip(alerts: list[Alert], max_shown: int = 3) -> None:
    """Conditions worth acting on, above everything else.

    Capped deliberately. A strip of ten warnings is read as zero
    warnings - the eye stops counting and the page becomes noise. The
    remainder collapse into an expander.

    Args:
        alerts: Already sorted critical-first by `build_alerts`.
        max_shown: How many stay open.
    """

    if not alerts:
        st.markdown(
            f"""
            <div class="alert alert-good">
                <div class="alert-icon">✓</div>
                <div class="alert-body">
                    <div class="alert-title">Nothing needs attention</div>
                    <div class="alert-detail">
                        No supply gaps, promotional waste or cancellation
                        spikes above their thresholds in this selection.
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    for alert in alerts[:max_shown]:
        _render_alert(alert)

    remaining = alerts[max_shown:]

    if remaining:
        with st.expander(f"{len(remaining)} more"):
            for alert in remaining:
                _render_alert(alert)


def _render_alert(alert: Alert) -> None:
    """One alert: what is wrong, the evidence, and what to do."""

    icon = {"critical": "!", "warning": "!", "info": "i"}[alert.severity]

    st.markdown(
        f"""
        <div class="alert alert-{alert.severity}">
            <div class="alert-icon">{icon}</div>
            <div class="alert-body">
                <div class="alert-title">{alert.title}</div>
                <div class="alert-detail">{alert.detail}</div>
                <div class="alert-action">→ {alert.action}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# PAGE FURNITURE
# ============================================================

def page_header(
    title: str,
    question: str,
    caption: str | None = None,
) -> None:
    """Standard header for every page.

    The `question` line states what the page is for. Every page in this
    platform exists to answer one business question, and naming it
    keeps both the user and the next developer honest about scope.
    """

    st.markdown(
        f"""
        <div class="page-header">
            <div class="page-question">{question}</div>
            <h1 class="page-title">{title}</h1>
            {f'<div class="page-caption">{caption}</div>' if caption else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )


def recommended_action(text: str, severity: Severity = "neutral") -> None:
    """The one-line verdict that closes every page.

    Without it a page stops at "here are the charts" and leaves the
    reader to work out the implication - which is the difference
    between a report and a decision tool.
    """

    colour = {
        "good": config.COLOR["good"],
        "warn": config.COLOR["warn"],
        "bad": config.COLOR["bad"],
        "neutral": config.COLOR["accent"],
    }[severity]

    st.markdown(
        f"""
        <div class="action-panel" style="border-left-color:{colour}">
            <div class="action-label">What to do</div>
            <div class="action-text">{text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section(title: str, question: str | None = None) -> None:
    """Section heading, optionally with the question it answers."""

    st.markdown(
        f"""
        <div class="section-head">
            <h2 class="section-title">{title}</h2>
            {f'<div class="section-question">{question}</div>' if question else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )


def data_quality_panel(manifest: dict, verification: tuple | None = None) -> None:
    """Snapshot provenance: how old, how big, and does it reconcile.

    Answers *can I trust these numbers?* - which is a real business
    question, unlike the generic exploratory charts this platform
    deliberately leaves out.
    """

    exported = manifest.get("exported_at", "unknown")[:16].replace("T", " ")

    tables = manifest.get("tables", {})

    total_rows = sum(t.get("rows", 0) for t in tables.values())

    with st.expander("Data quality and freshness"):

        cols = st.columns(4)

        cols[0].metric("Snapshot taken", exported)
        cols[1].metric("Tables", len(tables))
        cols[2].metric("Total rows", f"{total_rows:,}")
        cols[3].metric(
            "Period",
            f"{manifest.get('simulation_start', '')[:7]} to "
            f"{manifest.get('simulation_end', '')[:7]}"
        )

        deviation = manifest.get("encoding_verified_deviation")

        if deviation is not None:
            st.markdown(
                f"**Model encoding verified at export** — re-scoring "
                f"reproduced Databricks predictions to within "
                f"`{deviation:.2e}`. The app scores identically to the "
                f"pipeline."
            )

        if verification is not None:
            passed, largest, message = verification
            if passed:
                st.success(f"Live check passed. {message}")
            else:
                st.error(f"Live check FAILED. {message}")

        st.caption(
            "This is a point-in-time snapshot exported from Databricks, "
            "not a live connection. Re-run `17_Export_For_App` to refresh."
        )
