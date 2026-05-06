# Causally-Aware ML Incident Command Center

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)
![Tests](https://img.shields.io/badge/tests-133%20passing-brightgreen)
![License](https://img.shields.io/badge/license-MIT-blue)
![WebSocket](https://img.shields.io/badge/transport-WebSocket-purple)
![Prometheus](https://img.shields.io/badge/metrics-Prometheus-E6522C?logo=prometheus&logoColor=white)

A production-style ML observability platform that goes beyond "model accuracy dropped" to answer **why**, **how fast**, and **how much it costs** — in real time.

Built with FastAPI, streaming pub-sub, and a browser-based Incident Command Center dashboard. Designed around the same operational primitives Google and Stripe use to run large-scale ML systems in production.

---

## What Makes This Different

Most ML monitoring tools alert you when a metric crosses a threshold. This platform answers the operator's actual questions:

| Question | How it's answered |
|---|---|
| Is the model drifting? | PSI + Z-score drift detector with per-feature ranking |
| Why is it drifting? | Causal attribution engine with temporal scoring and field overlap |
| How fast is the budget burning? | Google SRE-style error budget burn rate (slow-burn + fast-burn) |
| What will it cost if I wait? | Financial loss model based on excess fraud above baseline |
| Should I roll back? | Deployment posture advisor wired to live SLO state |
| What happened and in what order? | Causal timeline with narrative generation |

---

## Architecture

```
StreamProducer  ──pub/sub──►  DriftDetector
     │                              │
     │                     FeatureDriftScore (PSI, Z)
     │                              │
     ▼                              ▼
BatchHistory              SchemaRegistry ──► CausalEngine
                                │                  │
                          SLOEngine           Attribution
                                │
                          BurnRateAlert
                                │
                     ┌──────────┴──────────────────┐
                     │                             │
               RiskForecaster              DecisionSimulator
                     │                             │
               ImpactEngine               CanaryController
                     │                             │
                     └──────────┬──────────────────┘
                                │
                         DependencyGraph
                                │
                           FastAPI ──── WebSocket ──── Dashboard
```

### Component breakdown

**`StreamProducer`** — Generates batches from a real Kaggle credit card fraud dataset (`real` mode) or a synthetic simulator (`synthetic` mode). Publishes to an in-process pub-sub broker. Supports configurable drift injection after N batches.

**`DriftDetector`** — Maintains a rolling window of feature means. Computes PSI (Population Stability Index) against the training baseline and per-feature Z-scores. PSI measures _distribution shape shift_; Z-score measures _mean displacement_. They are intentionally independent signals — the dashboard explains discrepancies rather than hiding them.

**`CausalEngine`** — Ingests `UpstreamEvent` objects (deployments, pipeline anomalies, schema changes). On each drift detection it scores candidates using temporal proximity (exponential decay with 10-minute half-life) and feature field overlap. Returns ranked hypotheses with confidence scores. The platform auto-injects `pipeline_anomaly` events at drift onset and every 150 batches of sustained drift so attribution works throughout an incident — not just in the first five minutes.

**`SLOEngine`** — Tracks three SLOs: fraud rate ceiling, confidence floor, and drift-free rate. Computes error budget burn rate following the Google SRE model: burn > 1× means depleting budget faster than allowed; burn > 14.4× (fast-burn) triggers immediate page. Budget exhaustion triggers a floor on the risk score.

**`RiskForecaster`** — Produces a 30-minute forward projection using OLS linear regression over the last 20 batches. Projects fraud rate and model confidence with a ±30% total-delta cap (not per-step cap — the per-step approach clamped projections to zero at fast batch rates). Loss model: `excess_fraud_rate × 0.35 leak_factor × $85/txn × 600 txns/hr`. The 35% leak factor reflects that a fraud detector blocks detected fraud; only excess fraud above the healthy baseline converts to real loss.

**`ImpactEngine`** — Translates technical signals into business impact labels. Enforces an invariant: if `requires_escalation=True`, the business impact label must be at least "material". PSI ≥ 0.50 or Z ≥ 3.0 trigger KPI breach independently of the raw fraud rate.

**`CanaryController`** — When an active canary is deployed, evaluates promotion/hold/rollback based on SLO burn rate and health score. When no canary is active, emits a production posture recommendation (promote/hold/rollback) based on current drift severity and budget state.

**`DependencyGraph`** — Tracks health of the model node and its upstream dependencies (feature pipelines, data stores, schema contracts). Health degrades on drift detection and recovers on healthy batches.

---

## Design Decisions

### Why PSI _and_ Z-score?

They measure different things. PSI compares the shape of the rolling distribution against training baseline — it catches variance shifts, bimodal splits, and long-tail changes. Z-score measures how many standard deviations the recent batch mean is from the baseline mean — it catches sharp point shifts. You can have PSI=1.0 (the distribution has completely changed shape) with Z=1.3 (the means haven't moved much). The dashboard explains this rather than masking it.

### Why calibrated confidence instead of raw prediction score?

Raw logistic regression output near 0.12 doesn't mean "12% confidence." It means the model scored the transaction near the decision boundary. After calibration: `confidence = 0.5 + 0.5 × (distance_from_threshold / max_possible_distance)`. This maps cleanly to [0.5, 1.0] where 0.5 = model at threshold (maximally uncertain), 1.0 = far from threshold (maximally certain). The SLO target of 0.75 now has an interpretable operational meaning.

### Why excess-above-baseline for the loss model?

A fraud detector blocks what it catches. The loss only comes from fraud that slips through because the model degraded. Using total fraud rate × transaction value overstates loss by 10-50×. The correct model is `max(0, current_rate - healthy_baseline_rate) × leak_factor × txn_value × volume`.

### Why a 1-hour causal lookback window?

The original 5-minute window caused attribution to fail on every incident that lasted more than 5 minutes — which is all of them. Real ML incidents are investigated over hours, not seconds. A deployment at 09:00 that causes drift at 09:20 is 20 minutes of lag, not 5. The 10-minute temporal half-life means old events decay in confidence but remain attributable.

---

## Running Locally

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python -m uvicorn src.api:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000` for the dashboard.

**Real mode** (Kaggle credit card dataset):

```bash
# Download creditcard.csv from https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
# Place it at data/creditcard.csv
MODEL__DATA_MODE=real python -m uvicorn src.api:app --port 8000
```

**Docker:**

```bash
docker compose up --build
# With Prometheus:
docker compose --profile monitoring up --build
```

---

## API Endpoints

| Endpoint | Purpose |
|---|---|
| `GET /` | Incident Command Center dashboard |
| `GET /dashboard/bootstrap` | Full dashboard payload snapshot (HTTP fallback) |
| `WS /ws/live` | Live WebSocket stream (2s push interval) |
| `GET /health` | Application health + uptime |
| `GET /livez` / `GET /readyz` | Kubernetes liveness / readiness probes |
| `GET /metrics/prometheus` | Prometheus scrape endpoint |
| `POST /causal/events` | Inject upstream event (deployment, config change) |
| `GET /causal/events` | List recent upstream events |
| `GET /risk/forecast` | Current risk forecast |
| `GET /slo/budgets` | SLO budget snapshot |
| `GET /state/snapshots` | Persisted incident snapshots (if enabled) |

---

## Configuration

All settings are environment-variable driven (nested with `__` separator). Key knobs:

```bash
MODEL__DATA_MODE=real          # "real" or "synthetic"
MODEL__DRIFT_AFTER_BATCHES=10  # when synthetic drift injection starts
DETECTOR__PSI_THRESHOLD=0.20   # feature-level PSI drift trigger
DETECTOR__Z_SCORE_THRESHOLD=2.0
MONITORING__ALERT_FRAUD_RATE=0.15
STATE_STORE__ENABLED=true      # SQLite incident snapshot persistence
LOGGING__JSON_FORMAT=true      # structured JSON logs for log pipelines
```

---

## Testing

```bash
python -m pytest -q           # 133 tests — unit + integration
python -m pytest tests/test_integration.py -v  # intelligence pipeline chain tests
```

The test suite covers:
- Drift detection math (PSI computation, Z-score, severity thresholds)
- Risk forecaster OLS slope and projection capping
- Causal attribution scoring (temporal decay, field overlap, confidence)
- SLO engine burn rate and budget exhaustion
- Impact engine business label invariants
- Calibrated confidence formula
- Rolling confusion matrix (precision, recall, F1)
- Canary controller posture logic
- Integration: full bootstrap payload completeness, canary decisions, alert feature ranking, timeline array contract, dependency graph wiring

---

## Known Limitations

**No action execution.** The dashboard recommends rollback or retrain but cannot execute either. Closing the loop would require integration with a model registry API (MLflow, Vertex AI), a deployment system, or a Slack/PagerDuty runbook trigger.

**In-process state.** All runtime state lives in memory in the single FastAPI process. Horizontal scaling would require moving state to Redis, a time-series DB (InfluxDB, TimescaleDB), and a shared event bus (Kafka, Pub/Sub).

**Synthetic drift model.** The simulator injects global distribution shift across all features uniformly. Real model drift is usually sparse — 1-3 features shift while others remain stable. The alert ranking by PSI score partially addresses this in the UI.

**Single-model scope.** The platform monitors one model (`fraud_detection_v1`). Extending to a model fleet would require namespacing all state by model ID and a registry-level aggregation view.

---

## Acknowledgements

- Google SRE Book Ch. 5 — error budget burn rate model
- Stripe's ML risk quantification framework — financial loss decomposition
- Kaggle ULB Credit Card Fraud dataset — real-mode replay data
