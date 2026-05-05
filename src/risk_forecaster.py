"""
src/risk_forecaster.py
----------------------
Forward-looking risk quantification for ML model reliability.

Industry context:
    SRE teams at Google and Stripe don't just react to current alerts — they
    project forward: "At the current burn rate, we'll exhaust our error budget
    in 47 minutes. At $4,200/hr fraud exposure, that's $3,290 in unmitigated
    losses." This forecaster implements that model.

    Inspired by:
        - Google SRE Book Ch.5 (Error Budget burn-rate extrapolation)
        - Stripe's ML risk quantification framework
        - Netflix's failure impact projection

Algorithm:
    KPI trajectory   — linear regression over the last N drift result values
    SLO breach ETA   — from ErrorBudgetState.exhaustion_eta_batches × batch_interval
    Loss per hour    — fraud_rate × avg_transaction_value × batch_volume × batches_per_hour
    Risk score       — weighted composite of budget_remaining, drift_severity, trajectory slope

Uncertainty:
    When fewer than MIN_OBSERVATIONS data points are available, confidence is
    explicitly lowered and the assumptions list documents the gap. The engine
    returns a valid forecast object either way — callers should check
    forecast_confidence before acting on projections.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from src.drift_detector import DriftResult
from src.slo_engine import ErrorBudgetState, SLOEngine

logger = logging.getLogger(__name__)

__all__ = [
    "RiskForecast",
    "RiskForecaster",
    "get_risk_forecaster",
]

# Business constants — override via environment / config in production
_AVG_TRANSACTION_USD: float = 85.0        # average transaction value
_BATCH_VOLUME: float = 50.0               # transactions per batch (matches StreamProducer batch_size)
_BATCHES_PER_HOUR: float = 12.0           # 1 batch per 5 minutes
_MANUAL_REVIEW_COST_PER_HOUR: float = 45.0  # ops analyst hourly cost

# Loss model parameters
# A fraud detector blocks detected fraud — the loss is only the EXCESS fraud
# that slips through because model degradation raises the false-negative rate.
_BASELINE_FRAUD_RATE: float = 0.05        # expected fraud rate when model is healthy
_DEGRADATION_LEAK_FACTOR: float = 0.35    # fraction of excess fraud that converts to real $ loss


@dataclass(slots=True)
class RiskForecast:
    """
    Forward-looking risk projection for the current model health state.

    All monetary values are in USD. ETA values are in minutes.
    risk_score is 0-100 where 100 is maximum risk.
    forecast_confidence is 0.0-1.0; below 0.5 means insufficient data.
    """
    # Financial projections
    loss_per_hour_usd: float
    accumulated_loss_usd: float       # estimated since drift onset
    manual_review_cost_per_hour: float

    # SLO projections
    slo_breach_eta_minutes: float | None   # None when budget is healthy
    budget_remaining_pct: float
    worst_burn_rate: float

    # KPI trajectory (linear regression slopes)
    projected_fraud_rate: float       # predicted rate at end of projection window
    projected_confidence: float       # predicted avg_confidence
    fraud_rate_slope: float           # positive = worsening
    confidence_slope: float           # negative = worsening

    # Risk composite
    risk_score: float                 # 0-100
    risk_level: str                   # "low" | "moderate" | "high" | "critical"

    # Metadata
    forecast_confidence: float        # 0-1, low when data is sparse
    projection_window_minutes: float
    observations_used: int
    assumptions: list[str]
    generated_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "loss_per_hour_usd": round(self.loss_per_hour_usd, 2),
            "accumulated_loss_usd": round(self.accumulated_loss_usd, 2),
            "manual_review_cost_per_hour": round(self.manual_review_cost_per_hour, 2),
            "slo_breach_eta_minutes": (
                round(self.slo_breach_eta_minutes, 1)
                if self.slo_breach_eta_minutes is not None else None
            ),
            "budget_remaining_pct": round(self.budget_remaining_pct, 1),
            "worst_burn_rate": round(self.worst_burn_rate, 2),
            "projected_fraud_rate": round(self.projected_fraud_rate, 4),
            "projected_confidence": round(self.projected_confidence, 4),
            "fraud_rate_slope": round(self.fraud_rate_slope, 6),
            "confidence_slope": round(self.confidence_slope, 6),
            "risk_score": round(self.risk_score, 1),
            "risk_level": self.risk_level,
            "forecast_confidence": round(self.forecast_confidence, 3),
            "projection_window_minutes": self.projection_window_minutes,
            "observations_used": self.observations_used,
            "assumptions": self.assumptions,
            "generated_at": self.generated_at,
        }


# ────────────────────────────────────────────────────────────────
# FORECASTER
# ────────────────────────────────────────────────────────────────

class RiskForecaster:
    """
    Quantify forward-looking risk from recent drift and SLO state.

    Call forecast() after each batch to get the latest risk projection.
    The forecaster is stateless between calls — it reads from the provided
    drift_results list and slo_engine snapshot each time.

    Thread-safety: stateless per call — safe for concurrent use.
    """

    MIN_OBSERVATIONS: int = 3         # minimum batches for confident regression
    MAX_OBSERVATIONS: int = 20        # cap to prevent stale data dominating
    PROJECTION_WINDOW_MINUTES: float = 30.0

    def forecast(
        self,
        drift_results: list[DriftResult],
        slo_engine: SLOEngine,
        drift_onset_batches: int = 0,
        batch_interval_seconds: float = 120.0,
    ) -> RiskForecast:
        """
        Generate a risk forecast from recent drift observations and SLO state.

        Args:
            drift_results:          recent drift results, most recent last
            slo_engine:             live SLOEngine instance for budget state
            drift_onset_batches:    how many batches since drift was first detected
                                    (used for accumulated loss calculation)
            batch_interval_seconds: seconds between batches (from config)

        Returns:
            RiskForecast with financial projections, SLO ETAs, and risk score.
        """
        recent = drift_results[-self.MAX_OBSERVATIONS:] if drift_results else []
        n = len(recent)
        assumptions: list[str] = []

        # ── KPI TRAJECTORY ───────────────────────────────────
        fraud_rates = [r.fraud_rate for r in recent]
        confidences = [r.avg_confidence for r in recent]

        if n >= self.MIN_OBSERVATIONS:
            fraud_slope = _linear_slope(fraud_rates)
            conf_slope = _linear_slope(confidences)
            proj_steps = (self.PROJECTION_WINDOW_MINUTES * 60.0) / max(batch_interval_seconds, 1.0)
            # Cap the total projected *delta* at ±30% — not the per-step slope.
            # Capping the per-step slope causes projected_fraud_rate to clamp to 0
            # when proj_steps is large (e.g. 900 at 2s/batch): slope × 900 ≫ current value.
            delta_fraud = max(-0.30, min(0.30, fraud_slope * proj_steps))
            delta_conf = max(-0.30, min(0.30, conf_slope * proj_steps))
            proj_fraud = max(0.0, min(1.0, (fraud_rates[-1] if fraud_rates else 0.10) + delta_fraud))
            # Use the signed slope — negative means deteriorating, positive means recovering.
            # abs() was a bug: it projected confidence as always declining even when improving.
            proj_conf = max(0.0, min(1.0, (confidences[-1] if confidences else 0.80) + delta_conf))
            trajectory_confidence = 1.0
        else:
            # Insufficient data — use last known value, flag uncertainty
            fraud_slope = 0.0
            conf_slope = 0.0
            proj_fraud = fraud_rates[-1] if fraud_rates else 0.10
            proj_conf = confidences[-1] if confidences else 0.80
            trajectory_confidence = 0.4
            assumptions.append(
                f"Only {n} observations available (need ≥{self.MIN_OBSERVATIONS}); "
                "trajectory projection is a flat extrapolation from latest value."
            )

        # ── SLO STATE ─────────────────────────────────────────
        budgets = slo_engine.get_budgets_snapshot()
        worst_burn = 0.0
        min_budget_remaining = 100.0
        slo_breach_eta_minutes: float | None = None

        for name, bdata in budgets.items():
            burn = bdata.get("burn_rate", 0.0)
            remaining = bdata.get("budget_remaining_pct", 100.0)
            eta_batches = bdata.get("exhaustion_eta_batches")

            worst_burn = max(worst_burn, burn)
            min_budget_remaining = min(min_budget_remaining, remaining)

            if remaining <= 0.0:
                # Budget already exhausted — eta of 0 would be misleading ("0 min away").
                # Signal "already breached" to callers by leaving eta_minutes as None;
                # callers check budget_remaining_pct == 0 separately.
                pass
            elif eta_batches is not None and eta_batches > 0:
                eta_minutes = (eta_batches * batch_interval_seconds) / 60.0
                if slo_breach_eta_minutes is None or eta_minutes < slo_breach_eta_minutes:
                    slo_breach_eta_minutes = eta_minutes

        if not budgets:
            min_budget_remaining = 100.0
            assumptions.append("No SLO budgets configured; SLO risk assumed minimal.")

        # ── FINANCIAL PROJECTIONS ─────────────────────────────
        # A fraud detector blocks what it catches — losses only come from the
        # EXCESS fraud that slips through due to model degradation above baseline.
        current_fraud_rate = fraud_rates[-1] if fraud_rates else _BASELINE_FRAUD_RATE
        txns_per_hour = _BATCH_VOLUME * _BATCHES_PER_HOUR
        excess_rate = max(0.0, current_fraud_rate - _BASELINE_FRAUD_RATE)
        loss_per_hour = (
            excess_rate
            * _DEGRADATION_LEAK_FACTOR
            * _AVG_TRANSACTION_USD
            * txns_per_hour
        )
        # Manual review cost rises with fraud rate (more transactions flagged)
        flagging_multiplier = max(1.0, current_fraud_rate / _BASELINE_FRAUD_RATE)
        manual_cost_per_hour = _MANUAL_REVIEW_COST_PER_HOUR * flagging_multiplier

        drift_hours = (drift_onset_batches * batch_interval_seconds) / 3600.0
        accumulated_loss = loss_per_hour * drift_hours

        assumptions.append(
            f"Loss model: excess fraud above {_BASELINE_FRAUD_RATE*100:.0f}% baseline × "
            f"{_DEGRADATION_LEAK_FACTOR*100:.0f}% leak factor × "
            f"${_AVG_TRANSACTION_USD:.0f}/txn × {int(txns_per_hour):,} txns/hr."
        )

        # ── RISK SCORE ────────────────────────────────────────
        risk_score = _compute_risk_score(
            fraud_rate=current_fraud_rate,
            avg_confidence=confidences[-1] if confidences else 0.80,
            budget_remaining_pct=min_budget_remaining,
            worst_burn_rate=worst_burn,
            fraud_slope=fraud_slope,
        )
        risk_level = _risk_level(risk_score)

        # ── FORECAST CONFIDENCE ───────────────────────────────
        # Degrade confidence proportionally to missing data
        obs_confidence = min(1.0, n / self.MIN_OBSERVATIONS) * trajectory_confidence
        forecast_confidence = round(obs_confidence, 3)

        if forecast_confidence < 0.5:
            assumptions.append(
                "Forecast confidence is LOW — projections are directionally indicative "
                "only. Do not use for automated decision making."
            )

        logger.debug(
            "Risk forecast | risk_score=%.1f | risk_level=%s | "
            "loss_per_hr=$%.0f | slo_eta=%.1f min | confidence=%.2f",
            risk_score, risk_level, loss_per_hour,
            slo_breach_eta_minutes if slo_breach_eta_minutes is not None else -1,
            forecast_confidence,
        )

        return RiskForecast(
            loss_per_hour_usd=round(loss_per_hour, 2),
            accumulated_loss_usd=round(accumulated_loss, 2),
            manual_review_cost_per_hour=round(manual_cost_per_hour, 2),
            slo_breach_eta_minutes=slo_breach_eta_minutes,
            budget_remaining_pct=round(min_budget_remaining, 1),
            worst_burn_rate=round(worst_burn, 3),
            projected_fraud_rate=round(proj_fraud, 4),
            projected_confidence=round(proj_conf, 4),
            fraud_rate_slope=round(fraud_slope, 6),
            confidence_slope=round(conf_slope, 6),
            risk_score=round(risk_score, 1),
            risk_level=risk_level,
            forecast_confidence=forecast_confidence,
            projection_window_minutes=self.PROJECTION_WINDOW_MINUTES,
            observations_used=n,
            assumptions=assumptions,
            generated_at=datetime.now(timezone.utc).isoformat(),
        )


# ────────────────────────────────────────────────────────────────
# MATH HELPERS
# ────────────────────────────────────────────────────────────────

def _linear_slope(values: list[float]) -> float:
    """Ordinary least squares slope for a sequence of y values (x = 0,1,2,...)."""
    n = len(values)
    if n < 2:
        return 0.0
    x_mean = (n - 1) / 2.0
    y_mean = sum(values) / n
    numerator = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
    denominator = sum((i - x_mean) ** 2 for i in range(n))
    return numerator / denominator if denominator else 0.0


def _compute_risk_score(
    fraud_rate: float,
    avg_confidence: float,
    budget_remaining_pct: float,
    worst_burn_rate: float,
    fraud_slope: float,
) -> float:
    """
    Weighted composite risk score, 0-100.

    Components:
        fraud_rate component  (40%) — normalized against a 50% ceiling
        confidence gap        (20%) — how far below 90% confidence
        budget depletion      (25%) — inverse of remaining budget
        burn acceleration     (15%) — burn_rate / 14.4 (fast_burn threshold)
    """
    fraud_component = min(1.0, fraud_rate / 0.50) * 40.0
    confidence_gap = max(0.0, 0.90 - avg_confidence) / 0.90
    confidence_component = confidence_gap * 20.0
    budget_component = max(0.0, (100.0 - budget_remaining_pct) / 100.0) * 25.0
    burn_component = min(1.0, worst_burn_rate / 14.4) * 15.0

    # Slope penalty: worsening trajectory adds up to 10 bonus points
    slope_penalty = min(10.0, max(0.0, fraud_slope * 1000.0))

    raw = fraud_component + confidence_component + budget_component + burn_component + slope_penalty

    # Floor: an exhausted budget is always at least "high" risk (≥ 75)
    if budget_remaining_pct <= 0.0:
        raw = max(raw, 75.0)

    return min(100.0, raw)


def _risk_level(score: float) -> str:
    if score >= 75:
        return "critical"
    if score >= 50:
        return "high"
    if score >= 25:
        return "moderate"
    return "low"


# ────────────────────────────────────────────────────────────────
# SINGLETON
# ────────────────────────────────────────────────────────────────

_forecaster: RiskForecaster | None = None


def get_risk_forecaster() -> RiskForecaster:
    global _forecaster
    if _forecaster is None:
        _forecaster = RiskForecaster()
    return _forecaster
