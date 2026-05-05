"""
tests/test_slo_engine.py
-------------------------
Unit tests for SLOEngine: error-budget tracking, burn-rate computation,
and fast/slow/exhausted alert emission.
"""

import pytest

from src.drift_detector import DriftResult, FeatureDriftScore
from src.slo_engine import (
    DEFAULT_SLO_TARGETS,
    BurnAlertType,
    ErrorBudgetState,
    SLOEngine,
    SLOTarget,
)


# ────────────────────────────────────────────────────────────────
# HELPERS
# ────────────────────────────────────────────────────────────────

def make_engine(*targets: SLOTarget) -> SLOEngine:
    return SLOEngine(slo_targets=list(targets) if targets else None)


def make_drift_result(
    batch_id: int = 1,
    fraud_rate: float = 0.10,
    avg_confidence: float = 0.85,
    drift_detected: bool = False,
    overall_psi: float = 0.05,
    max_z_score: float = 0.5,
) -> DriftResult:
    from datetime import datetime, timezone
    return DriftResult(
        batch_id=batch_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        overall_psi=overall_psi,
        max_z_score=max_z_score,
        drift_detected=drift_detected,
        severity="stable",
        feature_scores=[],
        fraud_rate=fraud_rate,
        avg_confidence=avg_confidence,
        processing_ms=1.0,
        warmed_up=True,
    )


def make_batch_summary(batch_id: int = 1, **kwargs: float) -> dict:
    return {"batch_id": batch_id, **kwargs}


def make_slo(
    name: str = "test_slo",
    metric_key: str = "fraud_rate",
    target: float = 0.30,
    direction: str = "below",
    window_batches: int = 10,
) -> SLOTarget:
    return SLOTarget(
        name=name,
        metric_key=metric_key,
        target=target,
        direction=direction,
        window_batches=window_batches,
    )


# ────────────────────────────────────────────────────────────────
# ERROR BUDGET STATE (unit)
# ────────────────────────────────────────────────────────────────

class TestErrorBudgetState:
    def make_budget(self, target=0.90, direction="above", window=10):
        from collections import deque
        b = ErrorBudgetState(
            slo_name="test",
            target=target,
            direction=direction,
            window_batches=window,
        )
        return b

    def test_all_good_burn_rate_zero(self):
        b = self.make_budget(target=0.90, direction="above")
        from src.slo_engine import SLOObservation
        for i in range(10):
            b.observations.append(SLOObservation(batch_id=i, value=0.95, is_good=True))
        assert b.burn_rate == 0.0
        assert b.budget_remaining_pct == 100.0

    def test_all_bad_burn_rate_very_high(self):
        # target=0.90, direction="above" → allowed_error_rate=0.10
        # All 10 batches bad → error_rate=1.0, burn_rate=1.0/0.10=10.0
        b = self.make_budget(target=0.90, direction="above")
        from src.slo_engine import SLOObservation
        for i in range(10):
            b.observations.append(SLOObservation(batch_id=i, value=0.50, is_good=False))
        assert b.burn_rate > 1.0
        assert b.budget_remaining_pct == 0.0

    def test_burn_rate_exactly_one_when_on_pace(self):
        """If error rate == allowed_error_rate, burn_rate must be 1.0."""
        b = self.make_budget(target=0.90, direction="above", window=10)
        from src.slo_engine import SLOObservation
        # allowed_error_rate = 1 - 0.90 = 0.10 → 1 bad in 10 = 1.0×
        for i in range(9):
            b.observations.append(SLOObservation(batch_id=i, value=0.95, is_good=True))
        b.observations.append(SLOObservation(batch_id=9, value=0.50, is_good=False))
        assert abs(b.burn_rate - 1.0) < 1e-6

    def test_exhaustion_eta_none_when_healthy(self):
        b = self.make_budget(target=0.90, direction="above")
        from src.slo_engine import SLOObservation
        for i in range(10):
            b.observations.append(SLOObservation(batch_id=i, value=0.95, is_good=True))
        assert b.exhaustion_eta_batches is None

    def test_exhaustion_eta_positive_when_burning(self):
        b = self.make_budget(target=0.90, direction="above", window=20)
        from src.slo_engine import SLOObservation
        # Insert 5 bad out of 10 → burn_rate high
        for i in range(5):
            b.observations.append(SLOObservation(batch_id=i, value=0.50, is_good=False))
        for i in range(5, 10):
            b.observations.append(SLOObservation(batch_id=i, value=0.95, is_good=True))
        assert b.exhaustion_eta_batches is not None


# ────────────────────────────────────────────────────────────────
# SLO ENGINE: HEALTHY BATCHES
# ────────────────────────────────────────────────────────────────

class TestSLOEngineHealthy:
    def test_healthy_batches_produce_no_alerts(self):
        eng = make_engine(make_slo(target=0.30, direction="below"))
        for i in range(10):
            alerts = eng.record_batch(
                batch_id=i,
                batch_summary=make_batch_summary(i),
                drift_result=make_drift_result(batch_id=i, fraud_rate=0.05),
            )
            assert alerts == [], f"Unexpected alert at batch {i}: {alerts}"

    def test_budgets_show_healthy_when_no_bad_batches(self):
        eng = make_engine(make_slo())
        for i in range(10):
            eng.record_batch(
                batch_id=i,
                batch_summary=make_batch_summary(i),
                drift_result=make_drift_result(batch_id=i, fraud_rate=0.05),
            )
        budgets = eng.get_budgets_snapshot()
        for name, b in budgets.items():
            assert b["health"] == "healthy", f"SLO {name} unexpectedly not healthy"


# ────────────────────────────────────────────────────────────────
# SLO ENGINE: SLOW BURN
# ────────────────────────────────────────────────────────────────

class TestSlowBurn:
    def test_slow_burn_alert_emitted(self):
        slo = make_slo(name="fraud_slo", metric_key="fraud_rate", target=0.10, direction="below", window_batches=10)
        eng = make_engine(slo)

        all_alerts = []
        for i in range(20):
            # fraud_rate=0.15 > target=0.10 → bad batch → burn_rate > 1
            alerts = eng.record_batch(
                batch_id=i,
                batch_summary=make_batch_summary(i),
                drift_result=make_drift_result(batch_id=i, fraud_rate=0.15),
            )
            all_alerts.extend(alerts)

        burn_alerts = [a for a in all_alerts if a.alert_type in ("slow_burn", "fast_burn", "budget_exhausted")]
        assert len(burn_alerts) > 0, "Expected at least one burn alert for sustained bad batches"


# ────────────────────────────────────────────────────────────────
# SLO ENGINE: FAST BURN
# ────────────────────────────────────────────────────────────────

class TestFastBurn:
    def test_fast_burn_alert_type(self):
        """A very tight SLO (target=0.99) with bad batches should trigger fast burn."""
        slo = make_slo(
            name="tight_slo",
            metric_key="fraud_rate",
            target=0.01,   # only 1% error allowed → any bad batch is fast burn
            direction="below",
            window_batches=10,
        )
        eng = make_engine(slo)
        eng.MIN_OBSERVATIONS_BEFORE_ALERT = 1

        all_alerts = []
        for i in range(10):
            # fraud_rate=0.50 >> target=0.01 → massively bad
            alerts = eng.record_batch(
                batch_id=i,
                batch_summary=make_batch_summary(i),
                drift_result=make_drift_result(batch_id=i, fraud_rate=0.50),
            )
            all_alerts.extend(alerts)

        fast_burns = [a for a in all_alerts if a.alert_type == "fast_burn"]
        exhausted = [a for a in all_alerts if a.alert_type == "budget_exhausted"]
        assert len(fast_burns) > 0 or len(exhausted) > 0


# ────────────────────────────────────────────────────────────────
# SLO ENGINE: DIRECTION "above"
# ────────────────────────────────────────────────────────────────

class TestDirectionAbove:
    def test_above_direction_good_when_value_exceeds_target(self):
        slo = make_slo(
            name="conf_slo",
            metric_key="avg_confidence",
            target=0.75,
            direction="above",
            window_batches=10,
        )
        eng = make_engine(slo)
        for i in range(10):
            alerts = eng.record_batch(
                batch_id=i,
                batch_summary=make_batch_summary(i),
                drift_result=make_drift_result(batch_id=i, avg_confidence=0.90),
            )
            assert alerts == []

    def test_above_direction_bad_when_value_below_target(self):
        slo = make_slo(
            name="conf_slo",
            metric_key="avg_confidence",
            target=0.75,
            direction="above",
            window_batches=10,
        )
        eng = make_engine(slo)
        eng.MIN_OBSERVATIONS_BEFORE_ALERT = 1
        all_alerts = []
        for i in range(15):
            alerts = eng.record_batch(
                batch_id=i,
                batch_summary=make_batch_summary(i),
                drift_result=make_drift_result(batch_id=i, avg_confidence=0.30),
            )
            all_alerts.extend(alerts)
        assert len(all_alerts) > 0


# ────────────────────────────────────────────────────────────────
# DEFAULT SLO TARGETS
# ────────────────────────────────────────────────────────────────

class TestDefaultSLOs:
    def test_default_slos_present(self):
        eng = make_engine()
        budgets = eng.get_budgets_snapshot()
        assert "fraud_rate_slo" in budgets
        assert "avg_confidence_slo" in budgets
        assert "drift_free_slo" in budgets

    def test_stats_reflect_evaluated_batches(self):
        eng = make_engine()
        for i in range(5):
            eng.record_batch(
                batch_id=i,
                batch_summary=make_batch_summary(i),
                drift_result=make_drift_result(batch_id=i),
            )
        stats = eng.get_stats()
        assert stats["batches_evaluated"] == 5
