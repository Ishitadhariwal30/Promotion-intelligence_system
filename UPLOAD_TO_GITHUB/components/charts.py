"""
Every chart in the platform, built here so they all look the same.

Two rules govern this module.

**One theme.** A page that mixes chart styles reads as a collection of
widgets rather than a product. `_apply_theme` runs on every figure and
nothing bypasses it.

**Every chart answers a question.** Each builder's docstring names the
question. If a chart cannot be given one, it does not belong in the
platform - which is the design rule this project set for itself.

Colour is used semantically, never decoratively. Persona colours are
fixed in `config` so a colour means the same thing on every page, and
good/warning/bad are reserved for status so they never compete with
identity.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

import config


# ============================================================
# THEME
# ============================================================

def _apply_theme(
    fig: go.Figure,
    height: int | None = None,
    show_legend: bool = True,
) -> go.Figure:
    """Apply the platform's visual identity to any figure.

    Deliberately sparse: no gridlines on the category axis, a faint one
    on the value axis, no chart border, and a horizontal legend above
    the plot where it does not steal width from the data.
    """

    fig.update_layout(
        height=height or config.CHART_HEIGHT,
        margin=dict(l=8, r=8, t=32, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            family=config.CHART_FONT,
            size=12,
            color=config.COLOR["ink_muted"],
        ),
        hoverlabel=dict(
            bgcolor=config.COLOR["surface"],
            font_size=12,
            font_family=config.CHART_FONT,
            bordercolor=config.COLOR["border"],
        ),
        showlegend=show_legend,
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="left", x=0,
            title_text="",
        ),
    )

    fig.update_xaxes(
        showgrid=False,
        showline=True,
        linecolor=config.COLOR["border"],
        tickfont=dict(size=11),
    )

    fig.update_yaxes(
        showgrid=True,
        gridcolor=config.COLOR["border"],
        gridwidth=1,
        zeroline=False,
        showline=False,
        tickfont=dict(size=11),
    )

    return fig


def _empty(message: str = "No data for this selection") -> go.Figure:
    """Placeholder when filters exclude everything.

    An empty chart with an explanation beats a blank rectangle, which
    users read as a broken page rather than an empty result.
    """

    fig = go.Figure()

    fig.add_annotation(
        text=message,
        showarrow=False,
        font=dict(size=13, color=config.COLOR["ink_faint"]),
        xref="paper", yref="paper", x=0.5, y=0.5,
    )

    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)

    return _apply_theme(fig, show_legend=False)


def _persona_colors(values: pd.Series) -> list[str]:
    """Fixed colour per persona, falling back to the palette."""

    return [
        config.PERSONA_COLOR.get(
            str(v), config.CATEGORICAL_PALETTE[i % len(config.CATEGORICAL_PALETTE)]
        )
        for i, v in enumerate(values)
    ]


# ============================================================
# TREND
# ============================================================

def revenue_trend(frame: pd.DataFrame) -> go.Figure:
    """*Is the business growing, and when did it spike?*

    Bars for volume, a line for revenue on a second axis. Volume and
    money move differently - a month can gain bookings while losing
    revenue if the mix shifts cheap - and one chart showing both makes
    that visible.
    """

    if frame.empty:
        return _empty()

    fig = go.Figure()

    fig.add_bar(
        x=frame["period"],
        y=frame["bookings"],
        name="Bookings",
        marker_color=config.COLOR["primary_light"],
        hovertemplate="%{x|%b %Y}<br>%{y:,.0f} bookings<extra></extra>",
    )

    fig.add_scatter(
        x=frame["period"],
        y=frame["revenue"],
        name="Revenue",
        yaxis="y2",
        mode="lines+markers",
        line=dict(color=config.COLOR["primary"], width=2.5),
        marker=dict(size=6),
        hovertemplate="%{x|%b %Y}<br>₹%{y:,.0f}<extra></extra>",
    )

    fig.update_layout(
        yaxis=dict(title="Bookings"),
        yaxis2=dict(
            title="Revenue", overlaying="y", side="right",
            showgrid=False, tickformat=",.0f",
        ),
    )

    return _apply_theme(fig)


def daily_trend(
    frame: pd.DataFrame,
    x: str,
    y: str,
    label: str,
    highlight: pd.Series | None = None,
) -> go.Figure:
    """*What happened day to day, and which days were unusual?*

    Args:
        highlight: Optional boolean mask marking days worth flagging,
            such as holidays. Those points are drawn larger and in the
            accent colour rather than annotated, which keeps a
            365-point chart readable.
    """

    if frame.empty:
        return _empty()

    fig = go.Figure()

    fig.add_scatter(
        x=frame[x], y=frame[y],
        mode="lines",
        line=dict(color=config.COLOR["primary"], width=1.6),
        name=label,
        hovertemplate="%{x|%d %b}<br>%{y:,.0f}<extra></extra>",
    )

    if highlight is not None and highlight.any():
        marked = frame[highlight]
        fig.add_scatter(
            x=marked[x], y=marked[y],
            mode="markers",
            marker=dict(size=9, color=config.COLOR["accent"]),
            name="Holiday",
            hovertemplate="%{x|%d %b}<br>%{y:,.0f}<extra></extra>",
        )

    return _apply_theme(fig)


# ============================================================
# COMPARISON
# ============================================================

def ranked_bar(
    frame: pd.DataFrame,
    category: str,
    value: str,
    title: str = "",
    value_format: str = ",.0f",
    color_by_persona: bool = False,
    top_n: int | None = None,
    horizontal: bool = True,
) -> go.Figure:
    """*Which categories contribute most?*

    Horizontal by default: category names are words, and words read
    better along the y-axis than rotated under the x.
    """

    if frame.empty or category not in frame.columns:
        return _empty()

    data = frame.nlargest(top_n, value) if top_n else frame.copy()

    data = data.sort_values(value, ascending=horizontal)

    colors = (
        _persona_colors(data[category])
        if color_by_persona else config.COLOR["primary"]
    )

    fig = go.Figure()

    if horizontal:
        fig.add_bar(
            y=data[category], x=data[value],
            orientation="h",
            marker_color=colors,
            text=data[value].map(lambda v: f"{v:{value_format}}"),
            textposition="auto",
            hovertemplate=f"%{{y}}<br>%{{x:{value_format}}}<extra></extra>",
        )
    else:
        fig.add_bar(
            x=data[category], y=data[value],
            marker_color=colors,
            text=data[value].map(lambda v: f"{v:{value_format}}"),
            textposition="outside",
            hovertemplate=f"%{{x}}<br>%{{y:{value_format}}}<extra></extra>",
        )

    fig.update_layout(title=title)

    return _apply_theme(fig, show_legend=False)


def grouped_bar(
    frame: pd.DataFrame,
    category: str,
    value: str,
    group: str,
    title: str = "",
    value_format: str = ".1%",
) -> go.Figure:
    """*How does one dimension differ across another?*

    The workhorse for uplift: category on the x-axis, one bar per
    treatment group, and the gap between them is the answer.
    """

    if frame.empty:
        return _empty()

    fig = go.Figure()

    for i, (name, group_data) in enumerate(frame.groupby(group)):
        fig.add_bar(
            name=str(name),
            x=group_data[category],
            y=group_data[value],
            marker_color=config.CATEGORICAL_PALETTE[
                i % len(config.CATEGORICAL_PALETTE)
            ],
            hovertemplate=(
                f"%{{x}}<br>{name}: %{{y:{value_format}}}<extra></extra>"
            ),
        )

    fig.update_layout(barmode="group", title=title)

    return _apply_theme(fig)


def uplift_chart(frame: pd.DataFrame, by: str = "Persona") -> go.Figure:
    """*What is a promotion worth, and to whom?*

    The most important chart in the platform. Two bars per persona -
    conversion without an offer and with one - and the gap between them
    is what the promotion buys.

    A flat set of gaps would mean targeting cannot beat a blanket
    policy, and the whole product would have no reason to exist.
    """

    if frame.empty:
        return _empty("No promotion data for this selection")

    data = frame.sort_values("uplift", ascending=False)

    fig = go.Figure()

    fig.add_bar(
        name="No promotion",
        x=data[by], y=data["not_promoted"],
        marker_color=config.COLOR["ink_faint"],
        hovertemplate="%{x}<br>No offer: %{y:.1%}<extra></extra>",
    )

    fig.add_bar(
        name="Promoted",
        x=data[by], y=data["promoted"],
        marker_color=config.COLOR["accent"],
        hovertemplate="%{x}<br>Promoted: %{y:.1%}<extra></extra>",
    )

    for _, row in data.iterrows():
        fig.add_annotation(
            x=row[by],
            y=max(row["promoted"], row["not_promoted"]),
            text=f"+{row['uplift']:.1%}",
            showarrow=False,
            yshift=14,
            font=dict(
                size=11, color=config.COLOR["accent"],
                family=config.CHART_FONT,
            ),
        )

    fig.update_layout(barmode="group", yaxis=dict(tickformat=".0%"))

    return _apply_theme(fig)


# ============================================================
# FUNNEL
# ============================================================

def funnel(frame: pd.DataFrame) -> go.Figure:
    """*Where are we losing people?*

    Drop-off is labelled between stages rather than left to be inferred
    from bar widths, because the number is the point.
    """

    if frame.empty:
        return _empty()

    fig = go.Figure(go.Funnel(
        y=frame["stage"],
        x=frame["customers"],
        textposition="inside",
        textinfo="value+percent initial",
        marker=dict(
            color=[
                config.SEQUENTIAL_PALETTE[
                    min(i, len(config.SEQUENTIAL_PALETTE) - 1)
                ][::1]
                for i in range(len(frame))
            ][::-1],
            line=dict(width=0),
        ),
        connector=dict(line=dict(color=config.COLOR["border"], width=1)),
        hovertemplate="%{y}<br>%{x:,.0f}<extra></extra>",
    ))

    return _apply_theme(fig, height=config.CHART_HEIGHT_TALL, show_legend=False)


# ============================================================
# DISTRIBUTION
# ============================================================

def donut(
    frame: pd.DataFrame,
    category: str,
    value: str,
    title: str = "",
    color_by_persona: bool = False,
) -> go.Figure:
    """*How is the whole split up?*

    A donut rather than a pie: the centre carries the total, which is
    the number people look for first.
    """

    if frame.empty:
        return _empty()

    colors = (
        _persona_colors(frame[category])
        if color_by_persona
        else config.CATEGORICAL_PALETTE[:len(frame)]
    )

    total = frame[value].sum()

    fig = go.Figure(go.Pie(
        labels=frame[category],
        values=frame[value],
        hole=0.62,
        marker=dict(colors=colors, line=dict(color="white", width=2)),
        textinfo="percent",
        hovertemplate="%{label}<br>%{value:,.0f} (%{percent})<extra></extra>",
    ))

    fig.add_annotation(
        text=f"<b>{total:,.0f}</b><br><span style='font-size:11px'>total</span>",
        showarrow=False,
        font=dict(size=20, color=config.COLOR["ink"]),
    )

    fig.update_layout(title=title)

    return _apply_theme(fig, show_legend=True)


def heatmap(
    frame: pd.DataFrame,
    x: str,
    y: str,
    value: str,
    title: str = "",
) -> go.Figure:
    """*When is demand concentrated?*"""

    if frame.empty:
        return _empty()

    pivot = frame.pivot_table(
        index=y, columns=x, values=value, aggfunc="sum"
    ).fillna(0)

    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=pivot.columns,
        y=pivot.index,
        colorscale=[[0, config.SEQUENTIAL_PALETTE[0]],
                    [1, config.SEQUENTIAL_PALETTE[-1]]],
        hovertemplate="%{y} · %{x}<br>%{z:,.0f}<extra></extra>",
        colorbar=dict(thickness=10, outlinewidth=0),
    ))

    fig.update_layout(title=title)

    return _apply_theme(fig, show_legend=False)


# ============================================================
# DECISION
# ============================================================

def offer_comparison(frame: pd.DataFrame, top_n: int = 6) -> go.Figure:
    """*Why did this offer win?*

    Expected profit per candidate offer, winner highlighted. Showing
    the alternatives is what turns a recommendation into an argument -
    the user can see what was considered and rejected.
    """

    if frame.empty:
        return _empty()

    data = frame.nlargest(top_n, "expected_profit").sort_values(
        "expected_profit"
    )

    colors = [
        config.COLOR["good"] if i == len(data) - 1
        else config.COLOR["ink_faint"]
        for i in range(len(data))
    ]

    fig = go.Figure()

    fig.add_bar(
        y=data["offer"],
        x=data["expected_profit"],
        orientation="h",
        marker_color=colors,
        text=data["expected_profit"].map(lambda v: f"₹{v:,.0f}"),
        textposition="auto",
        hovertemplate="%{y}<br>Expected profit ₹%{x:,.0f}<extra></extra>",
    )

    fig.update_layout(xaxis_title="Expected profit")

    return _apply_theme(fig, show_legend=False)


def discount_response_curve(frame: pd.DataFrame) -> go.Figure:
    """*How does response change as the discount grows?*

    One line per persona. Different slopes are the entire basis for
    targeting - parallel lines would mean everyone responds the same
    and one blanket discount would be optimal.

    The break-even discount is drawn as a hard boundary, because
    everything beyond it loses money regardless of how well it converts.
    """

    if frame.empty:
        return _empty()

    fig = go.Figure()

    for persona, group in frame.groupby("Persona"):

        group = group.sort_values("Discount_Percent")

        fig.add_scatter(
            x=group["Discount_Percent"],
            y=group["conversion"],
            mode="lines+markers",
            name=str(persona),
            line=dict(
                color=config.PERSONA_COLOR.get(
                    str(persona), config.COLOR["primary"]
                ),
                width=2.2,
            ),
            marker=dict(size=7),
            hovertemplate=(
                f"{persona}<br>%{{x}}% off → %{{y:.1%}}<extra></extra>"
            ),
        )

    fig.add_vline(
        x=config.DISCOUNT_CAUTION_THRESHOLD,
        line=dict(color=config.COLOR["warn"], width=1.5, dash="dot"),
        annotation_text="rarely pays beyond here",
        annotation_position="top",
        annotation_font=dict(size=10, color=config.COLOR["warn"]),
    )

    fig.update_layout(
        xaxis_title="Discount %",
        yaxis=dict(title="Booking rate", tickformat=".0%"),
    )

    return _apply_theme(fig)


def profit_curve(frame: pd.DataFrame) -> go.Figure:
    """*At what discount does this stop paying?*

    The Strategy Lab's central chart. Incremental profit against
    discount, with a zero line and the break-even boundary marked.
    Above zero the campaign pays; below it, the discount costs more
    than the bookings it wins.
    """

    if frame.empty:
        return _empty()

    positive = frame["incremental_profit"] >= 0

    fig = go.Figure()

    fig.add_bar(
        x=frame["discount"],
        y=frame["incremental_profit"],
        marker_color=[
            config.COLOR["good"] if p else config.COLOR["bad"]
            for p in positive
        ],
        hovertemplate=(
            "%{x}% off<br>Incremental profit ₹%{y:,.0f}<extra></extra>"
        ),
    )

    fig.add_hline(
        y=0,
        line=dict(color=config.COLOR["ink_muted"], width=1.5),
    )

    fig.add_vline(
        x=config.BREAK_EVEN_DISCOUNT,
        line=dict(color=config.COLOR["bad"], width=1.5, dash="dash"),
        annotation_text=f"margin hits zero at {config.BREAK_EVEN_DISCOUNT:.0f}%",
        annotation_position="top left",
        annotation_font=dict(size=10, color=config.COLOR["bad"]),
    )

    fig.update_layout(
        xaxis_title="Discount %",
        yaxis_title="Profit vs sending nothing",
    )

    return _apply_theme(fig, show_legend=False)


def scenario_comparison(frame: pd.DataFrame) -> go.Figure:
    """*Which strategy wins, and by how much?*

    Ranked on incremental profit, not revenue. A strategy can top the
    revenue table while destroying value, and ranking on revenue is how
    organisations discount their way to a bigger, less profitable
    business.
    """

    if frame.empty:
        return _empty()

    data = frame.sort_values("Incremental profit")

    colors = [
        config.COLOR["good"] if v >= 0 else config.COLOR["bad"]
        for v in data["Incremental profit"]
    ]

    fig = go.Figure()

    fig.add_bar(
        y=data["Strategy"],
        x=data["Incremental profit"],
        orientation="h",
        marker_color=colors,
        text=data["Incremental profit"].map(lambda v: f"₹{v:,.0f}"),
        textposition="auto",
        customdata=data[["Offer", "Customers"]],
        hovertemplate=(
            "%{y}<br>%{customdata[0]}<br>"
            "%{customdata[1]:,.0f} customers<br>"
            "Incremental profit ₹%{x:,.0f}<extra></extra>"
        ),
    )

    fig.add_vline(x=0, line=dict(color=config.COLOR["ink_muted"], width=1.5))

    fig.update_layout(xaxis_title="Profit vs sending nothing")

    return _apply_theme(fig, show_legend=False)


# ============================================================
# SUPPLY
# ============================================================

def utilisation_distribution(frame: pd.DataFrame) -> go.Figure:
    """*Are we short of providers, or carrying idle ones?*

    A histogram rather than an average, because the average hides the
    shape: a marketplace where half the providers are saturated and
    half are idle averages to "fine" and is nothing of the sort.
    """

    if frame.empty:
        return _empty()

    fig = go.Figure()

    fig.add_histogram(
        x=frame["utilisation"],
        nbinsx=20,
        marker_color=config.COLOR["primary_light"],
        hovertemplate="%{x:.0%} utilised<br>%{y} providers<extra></extra>",
    )

    fig.add_vline(
        x=float(frame["utilisation"].mean()),
        line=dict(color=config.COLOR["primary"], width=2),
        annotation_text=f"average {frame['utilisation'].mean():.0%}",
        annotation_position="top",
        annotation_font=dict(size=10),
    )

    fig.update_layout(
        xaxis=dict(title="Capacity used", tickformat=".0%"),
        yaxis_title="Providers",
    )

    return _apply_theme(fig, show_legend=False)
