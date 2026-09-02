"""
The Business Copilot panel.

Renders questions and answers. All the thinking happens in
`narrative_service`; this file only decides how it looks.

One deliberate choice: the suggested questions are always visible, not
hidden behind a "what can I ask?" link. This copilot answers a fixed set
of questions well rather than any question badly, and showing that set
up front sets the right expectation. A blank box implies it can answer
anything, and the first unanswerable question then reads as a failure
rather than a documented boundary.
"""

from __future__ import annotations

import streamlit as st

import config
from services import narrative_service
from services.narrative_service import Answer


HISTORY_KEY = "copilot_history"


def _history() -> list[tuple[str, Answer]]:
    if HISTORY_KEY not in st.session_state:
        st.session_state[HISTORY_KEY] = []
    return st.session_state[HISTORY_KEY]


# ============================================================
# PANEL
# ============================================================

def render(data: dict, compact: bool = False) -> None:
    """The full copilot panel.

    Args:
        data: Loaded tables, keyed as `narrative_service.ask` expects.
        compact: Sidebar mode - suggestions only, no history.
    """

    st.markdown("### Business Copilot")

    st.caption(
        "Answers a fixed set of business questions from computed data. "
        "It has no language model, so it cannot invent a number — and "
        "will say so rather than guess."
    )

    question = st.text_input(
        "Ask a question",
        placeholder="Which customers should receive promotions?",
        key="copilot_input",
        label_visibility="collapsed",
    )

    _suggestions(compact)

    pending = st.session_state.pop("copilot_pending", None)

    asked = pending or question

    if asked:
        answer = narrative_service.ask(asked, data)
        _render_answer(asked, answer)

        history = _history()
        if not history or history[-1][0] != asked:
            history.append((asked, answer))

    if not compact:
        _history_panel()


def _suggestions(compact: bool) -> None:
    """Clickable prompts.

    Every question the copilot knows, offered rather than guessed at.
    """

    questions = narrative_service.suggested_questions()

    shown = questions[:4] if compact else questions

    st.markdown('<div class="copilot-suggestions">', unsafe_allow_html=True)

    per_row = 2 if compact else 3

    for start in range(0, len(shown), per_row):

        cols = st.columns(min(per_row, len(shown) - start))

        for col, question in zip(cols, shown[start:start + per_row]):
            if col.button(question, key=f"suggest_{question}", use_container_width=True):
                st.session_state["copilot_pending"] = question
                st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


def _render_answer(question: str, answer: Answer) -> None:
    """One answer: headline, detail, evidence, source, next questions."""

    st.markdown(
        f"""
        <div class="copilot-answer">
            <div class="copilot-question">{question}</div>
            <div class="copilot-headline">{answer.headline}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if answer.detail:
        st.markdown(answer.detail)

    if answer.table is not None and not answer.table.empty:
        st.dataframe(
            answer.table,
            hide_index=True,
            use_container_width=True,
        )

    if answer.source:
        st.markdown(
            f'<div class="copilot-source">Source: {answer.source}</div>',
            unsafe_allow_html=True,
        )

    if answer.follow_ups:
        st.markdown("**Related questions**")

        cols = st.columns(min(len(answer.follow_ups), 3))

        for col, follow_up in zip(cols, answer.follow_ups[:3]):
            if col.button(follow_up, key=f"follow_{follow_up}", use_container_width=True):
                st.session_state["copilot_pending"] = follow_up
                st.rerun()


def _history_panel() -> None:
    """Earlier questions from this session."""

    history = _history()

    if len(history) < 2:
        return

    with st.expander(f"Earlier questions ({len(history) - 1})"):

        for question, answer in reversed(history[:-1]):
            st.markdown(f"**{question}**")
            st.caption(answer.headline)
            st.divider()

        if st.button("Clear history"):
            st.session_state[HISTORY_KEY] = []
            st.rerun()


# ============================================================
# INLINE
# ============================================================

def inline_insight(text: str, source: str | None = None) -> None:
    """A single generated observation, for embedding inside a page.

    Used where a chart benefits from a sentence naming what it shows.
    Always carries its source for the same reason the copilot does - a
    number a manager cannot trace is a number they will not act on.
    """

    st.markdown(
        f"""
        <div class="inline-insight">
            <div class="inline-insight-text">{text}</div>
            {f'<div class="inline-insight-source">{source}</div>' if source else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )
