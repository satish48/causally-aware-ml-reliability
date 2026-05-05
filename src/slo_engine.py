"""
src/slo_engine.py
-----------------
ML SLO tracking with Google SRE-style error-budget burn-rate alerting.

Industry context:
    Google's SRE book Chapter 5 defines the error-budget methodology for services.
    This module applies the same framework to ML model quality. Every model
    deserves an SLO, not just the infrastructure running it.

    Core idea: instead of alerting on a single threshold crossing, we track
    how fast the error budget is being *consumed*. A 100× burn rate for 5 minutes
    is more alarming than a 1.5× burn rate sustained for a week.

Burn-rate alert thresholds (Google-standard, adapted for batch cadence):
    Fast burn  burn_rate > 14.4   → page immediately
    Slow burn  burn_rate >  1.0   → ticket, review within 24h
    Exhausted  remaining ≤   0    → escalate, freeze deploys

SLOs defined for this fraud-detection platform:
    fraud_rate_slo      fraud_rate ≤ 0.30   (upper-bound KPI SLO)
    confidence_slo      avg_confidence ≥ 0.75
    drift_free_slo      drift_detected = False for ≥ 90% of batches
"""

from __future__ import annotations

import logging
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

from src.drift_detector import DriftResult

logger = logging.getLogger(__name__)

__all__ = [
    "SLOTarget",
    "SLOObservation",
    "ErrorBudgetState",
    "BurnRateAlert",
    "BurnAlertType",
    "SLOEngine",
    "get_slo_engine",
    "DEFAULT_SLO_TARGETS",
]


BurnAlertType = Literal["fast_burn", "slow_burn", "budget_exhausted"]
SLODirection = Literal["above", "below"]


# ────────────────────────────────────────────────────────────────
# DOMAIN TYPES
# ────────────────────────────────────────────────────────────────

@dataclass(slots=True)
class SLOTarget:
    name: str
    metric_key: str          # key into the metric_map built from DriftResult
    target: float
    direction: SLODirection  # "above": value >= target is good; "below": value <= target is good
    window_batches: int = 30
    description: str = ""


@dataclass(slots=True)
class SLOObservation:
    batch_id: int
    value: float
    is_good: bool


@dataclass
class ErrorBudgetState:
    slo_name: str
    target: float
    direction: SLODirection
    window_batches: int
    observations: deque[SLOObservation] = field(default_factory=lambda: deque(maxlen=30))

    def __post_init__(self) -> None:
        # Ensure deque respects the configured window.
        object.__setattr__(
            self,
            "observations",
            deque(self.observations, maxlen=self.window_batches),
        )

    # ── COMPUTED PROPERTIES ───────────────────────────────────

    @property
    def total_window(self) -> int:
        return len(self.observations)

    @property
    def bad_count(self) -> int:
        return sum(1 for o in self.observations if not o.is_good)

    @property
    def error_rate(self) -> float:
        return self.bad_count / self.total_window if self.total_window > 0 else 0.0

    @property
    def allowed_error_rate(self) -> float:
        """Fraction of batches that are allowed to be bad under the SLO."""
        return 1.0 - self.target if self.direction == "above" else self.target

    @property
    def burn_rate(self) -> float:
        """
        Burn rate = current_error_rate / allowed_error_rate.

        Interpretation:
            < 1.0  — consuming budget slower than allowed (healthy)
            = 1.0  — exactly on pace (neutral)
            > 1.0  — consuming faster than allowed (degraded)
            > 14.4 — fast-burn: budget will exhaust in < 1/14.4 of the window
        """
        allowed = self.allowed_error_rate
        if allowed <= 0.0:
            return float("inf") if self.error_rate > 0.0 else 0.0
        return self.error_rate / allowed

    @property
    def budget_consumed_pct(self) -> float:
        allowed_errors = self.allowed_error_rate * self.window_batches
        if allowed_errors <= 0.0:
            return 100.0 if self.bad_count > 0 else 0.0
        return min(100.0, (self.bad_count / allowed_errors) * 100.0)

    @property
    def budget_remaining_pct(self) -> float:
        return max(0.0, 100.0 - self.budget_consumed_pct)

    @property
    def exhaustion_eta_batches(self) -> int | None:
        """Estimated batches until budget exhaustion at current burn rate. None if stable."""
        br = self.burn_rate
        if br <= 1.0:
            return None
        allowed_total = self.allowed_error_rate * self.window_batches
        remaining_errors = max(0.0, allowed_total - self.bad_count)
        if remaining_errors <= 0.0:
            return 0
        per_batch = self.error_rate
        if per_batch <= 0.0:
            return None
        return max(0, int(remaining_errors / per_batch))


@dataclass(slots=True)
class BurnRateAlert:
    alert_id: str
    slo_name: str
    alert_type: BurnAlertType
    burn_rate: float
    budget_remaining_pct: float
    exhaustion_eta_batches: int | None
    batch_id: int
    triggered_at: str
    message: str


# ────────────────────────────────────────────────────────────────
# DEFAULT SLO CONFIGURATION
# ────────────────────────────────────────────────────────────────

DEFAULT_SLO_TARGETS: list[SLOTarget] = [
    SLOTarget(
        name="fraud_rate_slo",
        metric_key="fraud_rate",
        target=0.30,
        direction="below",
        window_batches=30,
        description="Fraud rate must stay ≤ 30% (upper-bound KPI SLO). "
                    "Breach indicates the model is flagging too little — financial exposure rising.",
    ),
    SLOTarget(
        name="avg_confidence_slo",
        metric_key="avg_confidence",
        target=0.75,
        direction="above",
        window_batches=30,
        description="Average model confidence must stay ≥ 75%. "
                    "Drops indicate covariate shift or distribution mismatch.",
    ),
    SLOTarget(
        name="drift_free_slo",
        metric_key="drift_detected",
        target=0.10,   # at most 10% of batches may have drift_detected=True
        direction="below",
        window_batches=30,
        description="No more than 10% of batches should have drift detected. "
                    "Sustained drift means retraining or rollback is overdue.",
    ),
]


# ────────────────────────────────────────────────────────────────
# ENGINE
# ────────────────────────────────────────────────────────────────

class SLOEngine:
    """
    Track ML model SLOs with error-budget burn-rate alerting.

    Usage:
        slo_alerts = slo_engine.record_batch(
            batch_id=batch_id,
            batch_summary=latest_batch,
            drift_result=drift_result,
        )
        for alert in slo_alerts:
            logger.critical(alert.message)
    """

    FAST_BURN_THRESHOLD: float = 14.4
    SLOW_BURN_THRESHOLD: float = 1.0
    MIN_OBSERVATIONS_BEFORE_ALERT: int = 5
    MAX_ALERTS: int = 200

    def __init__(
        self,
        slo_targets: list[SLOTarget] | None = None,
    ) -> None:
        targets = slo_targets or DEFAULT_SLO_TARGETS
        self._targets: dict[str, SLOTarget] = {t.name: t for t in targets}
        self._budgets: dict[str, ErrorBudgetState] = {
            name: ErrorBudgetState(
                slo_name=name,
                target=t.target,
                direction=t.direction,
                window_batches=t.window_batches,
            )
            for name, t in self._targets.items()
        }
        self._alerts: list[BurnRateAlert] = []
        self._batches_evaluated: int = 0

        logger.info(
            "SLOEngine initialized | slos=%s | fast_burn_threshold=%.1fx",
            list(self._targets.keys()),
            self.FAST_BURN_THRESHOLD,
        )

    # ── PUBLIC API ────────────────────────────────────────────

    def record_batch(
        self,
        *,
        batch_id: int,
        batch_summary: dict[str, Any],
        drift_result: DriftResult,
    ) -> list[BurnRateAlert]:
        """
        Record one batch observation against all SLO targets.

        Builds a unified metric map from the drift result, then evaluates each
        SLO target. Returns any new burn-rate alerts emitted this batch.
        """
        self._batches_evaluated += 1

        metric_map: dict[str, float] = {
            "fraud_rate": drift_result.fraud_rate,
            "avg_confidence": drift_result.avg_confidence,
            "drift_detected": float(drift_result.drift_detected),
            "overall_psi": drift_result.overall_psi,
            "max_z_score": drift_result.max_z_score,
        }
        metric_map.update(
            {k: float(v) for k, v in batch_summary.items() if isinstance(v, (int, float))}
        )

        new_alerts: list[BurnRateAlert] = []

        for slo_name, budget in self._budgets.items():
            target = self._targets[slo_name]
            value = metric_map.get(target.metric_key)
            if value is None:
                continue

            if target.direction == "above":
                is_good = value >= target.target
            else:
                is_good = value <= target.target

            budget.observations.append(
                SLOObservation(batch_id=batch_id, value=value, is_good=is_good)
            )

            if budget.total_window >= self.MIN_OBSERVATIONS_BEFORE_ALERT:
                new_alerts.extend(self._check_burn_rate(budget, batch_id))

        if new_alerts:
            self._alerts.extend(new_alerts)
            if len(self._alerts) > self.MAX_ALERTS:
                self._alerts = self._alerts[-self.MAX_ALERTS:]

        return new_alerts

    def get_budgets_snapshot(self) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        for name, budget in self._budgets.items():
            target = self._targets[name]
            br = budget.burn_rate
            result[name] = {
                "slo_name": name,
                "target": target.target,
                "direction": target.direction,
                "description": target.description,
                "window_batches": budget.window_batches,
                "observations_count": budget.total_window,
                "bad_count": budget.bad_count,
                "error_rate": round(budget.error_rate, 4),
                "allowed_error_rate": round(budget.allowed_error_rate, 4),
                "burn_rate": round(br, 3),
                "budget_consumed_pct": round(budget.budget_consumed_pct, 1),
                "budget_remaining_pct": round(budget.budget_remaining_pct, 1),
                "exhaustion_eta_batches": budget.exhaustion_eta_batches,
                "health": (
                    "healthy" if br <= self.SLOW_BURN_THRESHOLD
                    else "degraded" if br <= self.FAST_BURN_THRESHOLD
                    else "critical"
                ),
            }
        return result

    def get_alerts_snapshot(self, limit: int = 30) -> list[dict[str, Any]]:
        return [
            {
                "alert_id": a.alert_id,
                "slo_name": a.slo_name,
                "alert_type": a.alert_type,
                "burn_rate": a.burn_rate,
                "budget_remaining_pct": a.budget_remaining_pct,
                "exhaustion_eta_batches": a.exhaustion_eta_batches,
                "batch_id": a.batch_id,
                "triggered_at": a.triggered_at,
                "message": a.message,
            }
            for a in self._alerts[-limit:]
        ]

    def get_alert_objects(self, limit: int | None = None) -> list[BurnRateAlert]:
        items = self._alerts if limit is None else self._alerts[-limit:]
        return list(items)

    def get_stats(self) -> dict[str, Any]:
        return {
            "slos_tracked": len(self._targets),
            "batches_evaluated": self._batches_evaluated,
            "total_burn_alerts": len(self._alerts),
            "fast_burn_alerts": sum(1 for a in self._alerts if a.alert_type == "fast_burn"),
            "slow_burn_alerts": sum(1 for a in self._alerts if a.alert_type == "slow_burn"),
            "exhausted_alerts": sum(1 for a in self._alerts if a.alert_type == "budget_exhausted"),
            "slo_health": {
                name: (
                    "healthy" if b.burn_rate <= self.SLOW_BURN_THRESHOLD
                    else "degraded" if b.burn_rate <= self.FAST_BURN_THRESHOLD
                    else "critical"
                )
                for name, b in self._budgets.items()
                if b.total_window >= self.MIN_OBSERVATIONS_BEFORE_ALERT
            },
        }

    # ── INTERNAL ──────────────────────────────────────────────

    def _check_burn_rate(
        self,
        budget: ErrorBudgetState,
        batch_id: int,
    ) -> list[BurnRateAlert]:
        br = budget.burn_rate
        now = datetime.now(timezone.utc).isoformat()
        alerts: list[BurnRateAlert] = []

        if budget.budget_remaining_pct <= 0.0:
            alerts.append(BurnRateAlert(
                alert_id=f"slo-{uuid.uuid4().hex[:8]}",
                slo_name=budget.slo_name,
                alert_type="budget_exhausted",
                burn_rate=round(br, 2),
                budget_remaining_pct=0.0,
                exhaustion_eta_batches=0,
                batch_id=batch_id,
                triggered_at=now,
                message=(
                    f"[BUDGET EXHAUSTED] {budget.slo_name}: error budget fully consumed. "
                    "Escalate immediately. Freeze deploys until resolved."
                ),
            ))
            logger.critical(
                "SLO BUDGET EXHAUSTED | slo=%s | burn_rate=%.1fx | batch_id=%d",
                budget.slo_name, br, batch_id,
            )
        elif br > self.FAST_BURN_THRESHOLD:
            alerts.append(BurnRateAlert(
                alert_id=f"slo-{uuid.uuid4().hex[:8]}",
                slo_name=budget.slo_name,
                alert_type="fast_burn",
                burn_rate=round(br, 2),
                budget_remaining_pct=round(budget.budget_remaining_pct, 1),
                exhaustion_eta_batches=budget.exhaustion_eta_batches,
                batch_id=batch_id,
                triggered_at=now,
                message=(
                    f"[FAST BURN] {budget.slo_name}: burn rate {br:.1f}x "
                    f"(threshold {self.FAST_BURN_THRESHOLD}x). "
                    f"Budget {budget.budget_remaining_pct:.0f}% remaining. "
                    "Page on-call immediately."
                ),
            ))
            logger.critical(
                "SLO FAST BURN | slo=%s | burn_rate=%.1fx | budget=%.0f%% | batch_id=%d",
                budget.slo_name, br, budget.budget_remaining_pct, batch_id,
            )
        elif br > self.SLOW_BURN_THRESHOLD:
            alerts.append(BurnRateAlert(
                alert_id=f"slo-{uuid.uuid4().hex[:8]}",
                slo_name=budget.slo_name,
                alert_type="slow_burn",
                burn_rate=round(br, 2),
                budget_remaining_pct=round(budget.budget_remaining_pct, 1),
                exhaustion_eta_batches=budget.exhaustion_eta_batches,
                batch_id=batch_id,
                triggered_at=now,
                message=(
                    f"[SLOW BURN] {budget.slo_name}: burn rate {br:.1f}x. "
                    f"Budget {budget.budget_remaining_pct:.0f}% remaining. "
                    f"ETA exhaustion: {budget.exhaustion_eta_batches} batches. "
                    "Create ticket and review within 24h."
                ),
            ))
            logger.warning(
                "SLO SLOW BURN | slo=%s | burn_rate=%.1fx | eta=%s batches | batch_id=%d",
                budget.slo_name, br, budget.exhaustion_eta_batches, batch_id,
            )

        return alerts


# ────────────────────────────────────────────────────────────────
# SINGLETON
# ────────────────────────────────────────────────────────────────

_engine: SLOEngine | None = None


def get_slo_engine() -> SLOEngine:
    global _engine
    if _engine is None:
        _engine = SLOEngine()
    return _engine
