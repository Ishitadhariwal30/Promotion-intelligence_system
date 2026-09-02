# Urban Company Promotion Intelligence Platform

An AI decision support platform that answers four questions per customer:
**who** should receive a promotion, **what** promotion, **when**, and
**how much** discount — priced against the margin it costs.

Built on a Databricks pipeline of 18 notebooks. This application consumes
its output; it does not compute it.

---

## The finding

```
Never promote          ₹2,206,531       baseline
Always 20% off         ₹1,273,707       −42.3%
Model recommendation   ₹2,277,810       +3.2%
```

At a 35% gross margin, a 20% discount leaves 15% — **43% of the original
margin, so with zero uplift you lose 57%.** The observed loss is 42.3%,
meaning promotions recovered about 15 points.

Working backwards: a 20% discount raises bookings by roughly **35%**. A
real, substantial effect. It still does not come close to paying for
itself.

The platform exists to find the customers where it does. **There are
fewer of them than intuition suggests — it recommends sending nothing to
71% of customers.**

---

## Running it

```bash
cd streamlit_app
pip install -r requirements.txt
streamlit run app.py
```

The app opens at `http://localhost:8501`.

### First, get the data

The platform runs on a snapshot exported from Databricks.

1. Run **`17_Export_For_App`** in Databricks
2. Download what it writes
3. Place it as below

```
streamlit_app/sample_data/
├── MANIFEST.json
├── bookings.parquet
├── customers.parquet
├── promotions.parquet
├── providers.parquet
├── recommendations.parquet
├── services.parquet
├── activity_daily.parquet
├── customer_state_daily.parquet
├── customer_state_monthly.parquet
├── customer_state_year_end.parquet
├── training_features.parquet
└── model/
    ├── model.joblib
    ├── encoders.json
    └── feature_order.json
```

Without it, the app shows a setup screen rather than a stack trace.

---

## The pages

| Page | Question it answers |
|---|---|
| **Executive Dashboard** | How is the business performing, and what needs attention? |
| **Customer Intelligence** | Who are our customers, and what has happened to them? |
| **Demand & Bookings** | How is demand behaving, and can we serve it? |
| **Promotion Performance** | Which promotions work, and what are they costing? |
| **AI Recommendation Center** | What should we offer this specific customer? |
| **Strategy Lab** | What happens if we change the strategy? |

Filters persist across pages. Filtering to Mumbai on one page keeps you
in Mumbai on the next.

---

## How it works

The model answers exactly one question: **how likely is a booking?**

A recommendation comes from asking it seventeen times — once per
candidate offer — and comparing:

```
expected profit = P(book | offer) × price × (margin − discount)
```

Ranked on **profit**, not probability. Ranking on probability would
recommend the largest discount to everybody, because a bigger discount
always raises the chance of a booking. That is not a decision, it is an
identity.

This is also why **"send nothing" is always a candidate.** Choosing not
to promote is a real decision and is usually the right one.

---

## Architecture

```
app.py                 entry, routing, setup screen
config.py              every setting and business constant

services/              business logic — no Streamlit imports
  data_loader.py         cached Parquet reading
  model_service.py       model, encoding, offer scoring
  analytics_service.py   every number behind every chart
  recommendation_service.py  single-customer decisions and reasoning
  simulation_service.py  Strategy Lab scenarios
  narrative_service.py   copilot intent routing

components/            presentation — no computation
  charts.py              Plotly builders, one theme
  metric_cards.py        KPI tiles, alerts, action panels
  filters.py             the shared filter bar
  recommendation_card.py the offer panel
  customer_profile.py    profile and lifecycle timeline
  copilot.py             question box and answers

pages/                 six screens, each wiring the two together
```

**The rule:** `services/` never imports Streamlit, `components/` never
computes. Pages connect them and do nothing else. That is what makes the
logic testable and lets the Strategy Lab reuse the Executive Dashboard's
maths.

---

## Design decisions worth knowing

**The filters select real customers, never synthetic ones.** A made-up
customer would need values for all 57 features, and any combination we
invented might be one the model has never seen — it would answer
confidently regardless.

**Uplift is measured on the randomised group by default.** Elsewhere the
comparison is confounded: promoted customers were chosen *because* they
looked responsive, so some apparent uplift is selection rather than
treatment. The toggle on Promotion Performance makes the difference
visible.

**There is no confidence score.** A single probability has no meaningful
confidence interval. It is replaced by **edge over next best offer** —
how far ahead the winner finished — which is real and tells you when a
choice is close.

**There is no retention metric.** No churn model exists in the pipeline,
so any retention percentage would be invented. **Customers reactivated**
— dormant customers whose predicted probability crosses a threshold —
answers the same question from real data.

**The copilot has no language model.** It routes questions to a fixed set
of handlers that run real queries. It cannot invent a number, and says so
plainly when asked something outside its range. Every answer cites its
source.

---

## Known limitations

**Premium customers are over-promoted.** Their true response rate is the
lowest of any persona; the model ranks them second for promotion. Partly
defensible — Premium buy the most expensive services, so a small discount
can pay on job value alone — but the ordering is wrong. Flagged in the
risks panel wherever it appears.

**The +3.2% cannot be independently verified.** It is the model's own
estimate of its own policy. Proving it would need a live test.

**Campaign duration scales linearly.** An assumption, not a model output.
The Strategy Lab says so when you use it.

**This is a snapshot, not a live connection.** The Strategy Lab
re-filters on every widget change; network round-trips to a SQL warehouse
would make it feel broken. Re-run `17_Export_For_App` to refresh.

---

## Refreshing the data

```
Databricks:  run 17_Export_For_App
             → verifies encodings reproduce Databricks scoring
             → writes Parquet and model artifacts

Local:       replace sample_data/
             → restart Streamlit
```

The export **verifies before it writes.** The model reads encoded
integers, and if a category mapped to a different number the app would
score confidently and wrongly with nothing to notice. Notebook 17
re-scores 500 customers and refuses to export if they do not match.

The same check is available in-app under *AI Recommendation Center → Is
the model scoring correctly?*
