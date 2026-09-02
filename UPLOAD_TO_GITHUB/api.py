"""
REST API - Urban Company Promotion Intelligence Platform

PURPOSE
    A second front door onto the same scoring logic the Streamlit app uses.
    Programs call this; people call app.py. Neither is a client of the other.

RUN
    uvicorn api:app --reload --port 8000

DOCS
    http://localhost:8000/docs   - FastAPI generates this from the type hints
                                   below. It is a real, clickable test console.

WHY IT SHARES services/ RATHER THAN CALLING THE STREAMLIT APP
    The scoring rules - encoding, the 17 offers, expected profit - live in
    services/. Both entry points import them, so there is exactly one
    implementation. If the API re-implemented scoring, the two would drift
    apart and only a customer would notice.

A NOTE ON THE STREAMLIT IMPORT
    services/ uses @st.cache_data for caching. Those decorators work fine
    outside a Streamlit runtime - they fall back to an in-memory cache and
    log "No runtime found", which is noise, not an error. Streamlit is
    already a dependency, so nothing extra is installed for this.
"""

from __future__ import annotations

from dataclasses import asdict

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from services import data_loader, recommendation_service

app = FastAPI(
    title="Urban Company Promotion Intelligence",
    description=(
        "Scores all 17 candidate offers for a customer and returns the one "
        "with the highest expected profit - which is frequently 'send nothing'."
    ),
    version="1.0.0",
)


# ============================================================
# Response shapes
# ============================================================
#
# Declared as models rather than returning raw dicts so that FastAPI can
# document them, and so a change to the internal Recommendation dataclass
# cannot silently change the API contract. The contract is here, in one
# place, deliberately separate from the internals.


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    data_available: bool


class RecommendationResponse(BaseModel):
    customer_id: str
    activity_id: str

    offer: dict = Field(description="The recommended offer - type and discount")
    is_promotion: bool = Field(description="False means the answer is 'send nothing'")

    booking_probability: float = Field(description="P(book) under the recommended offer")
    baseline_probability: float = Field(description="P(book) with no promotion")
    uplift: float = Field(description="Percentage points gained over sending nothing")

    revenue_if_booked: float
    expected_profit: float
    baseline_expected_profit: float
    profit_gain: float = Field(description="Expected profit over sending nothing")
    edge_over_next_best: float

    reasons: list[str]
    risks: list[str]


# ============================================================
# Endpoints
# ============================================================


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    """Liveness check.

    Deliberately reports whether the MODEL and DATA loaded, not merely that
    the process is up. A container that answers "I am running" while unable
    to score is exactly the failure an orchestrator needs to catch.
    """
    from services import model_service

    try:
        model_loaded = model_service.load_model() is not None
    except Exception:
        model_loaded = False

    return HealthResponse(
        status="ok" if model_loaded else "degraded",
        model_loaded=model_loaded,
        data_available=data_loader.data_is_available(),
    )


@app.get("/customers", tags=["data"])
def list_customers(
    limit: int = Query(50, ge=1, le=500, description="How many IDs to return"),
) -> dict:
    """Customer IDs that can be scored.

    Only customers with an event in the out-of-time TEST period are
    scoreable. Recommendations for a training-period customer would be
    scored on data the model already learned from, which tells you nothing.
    """
    features = data_loader.load_training_features()
    customers = recommendation_service.available_customers(features)

    return {"total": len(customers), "customers": customers[:limit]}


@app.get(
    "/recommend/{customer_id}",
    response_model=RecommendationResponse,
    tags=["scoring"],
)
def recommend(customer_id: str) -> RecommendationResponse:
    """Score all 17 offers for one customer and return the best.

    "Best" is highest EXPECTED PROFIT, not highest booking probability. A
    larger discount always raises P(book), so optimising probability alone
    would recommend the deepest discount to everyone and give away more
    margin than it earns back.
    """
    features = data_loader.load_training_features()

    try:
        row = recommendation_service.customer_row(features, customer_id)
    except ValueError as exc:
        # Raised when the customer has no event in the test period. A 404 is
        # right: the customer may exist, but there is nothing here to score.
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    result = recommendation_service.recommend(row)

    return RecommendationResponse(
        customer_id=result.customer_id,
        activity_id=result.activity_id,
        # asdict() rather than naming fields, so adding a field to Offer does
        # not silently drop it from the response.
        offer=asdict(result.offer),
        is_promotion=result.is_promotion,
        booking_probability=round(float(result.booking_probability), 4),
        baseline_probability=round(float(result.baseline_probability), 4),
        uplift=round(float(result.uplift), 4),
        revenue_if_booked=round(float(result.revenue_if_booked), 2),
        expected_profit=round(float(result.expected_profit), 2),
        baseline_expected_profit=round(float(result.baseline_expected_profit), 2),
        profit_gain=round(float(result.profit_gain), 2),
        edge_over_next_best=round(float(result.edge_over_next_best), 2),
        reasons=result.reasons,
        risks=result.risks,
    )


@app.get("/recommend/{customer_id}/alternatives", tags=["scoring"])
def alternatives(customer_id: str) -> dict:
    """The runner-up offers, not just the winner.

    All 17 candidates are scored; `Recommendation.alternatives` keeps the
    closest few rather than every one, which is what the UI displays.
    `offers_considered` reports how many are returned, not how many were
    evaluated.

    This is the endpoint that makes a recommendation defensible. Anyone can
    assert "send 10% off"; showing what the near-misses were worth is what
    makes it an argument rather than an opinion.
    """
    features = data_loader.load_training_features()

    try:
        row = recommendation_service.customer_row(features, customer_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    result = recommendation_service.recommend(row)

    frame: pd.DataFrame = result.alternatives

    return {
        "customer_id": customer_id,
        "offers_considered": len(frame),
        "alternatives": frame.to_dict(orient="records"),
    }
