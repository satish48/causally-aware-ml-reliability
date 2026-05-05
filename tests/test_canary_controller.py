"""
tests/test_canary_controller.py
--------------------------------
Unit tests for CanaryController: promote/hold/rollback decisions,
stage progression, and SLO-gated traffic management.
"""

from datetime import datetime, timezone

import pytest

from src.canary_controller import CanaryController, CanaryStage
from src.drift_detector import DriftResult, FeatureDriftScore
from src.model_registry import ModelRegistry
from src.slo_engine import BurnRateAlert, SLOEngine, SLOTarget


# ────────────────────────────────────────────────────────────────
# HELPERS
# ────────────────────────────────────────────────────────────────

def make_slo_target(
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


def make_setup(seed_canary: bool = True):
    """Return (controller, model_registry, slo_engine) with optional canary seeded."""
    registry = ModelRegistry()
    slo = SLOEngine(slo_targets=[make_slo_target()])
    controller = CanaryController(model_registry=registry, slo_engine=slo)
    if seed_canary:
        controller.deploy_canary("fraud_detection_v1", notes="test canary")
    return controller, registry, slo


def make_drift_result(
    batch_id: int = 1,
    drift_detected: bool = False,
    severity: str = "stable",
) -> DriftResult:
    return DriftResult(
        batch_id=batch_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        overall_psi=0.05,
        max_z_score=0.5,
        drift_detected=drift_detected,
        severity=severity,
        feature_scores=[],
        fraud_rate=0.05,
        avg_confidence=0.90,
        processing_ms=1.0,
        warmed_up=True,
    )


def make_fast_burn_alert(slo_name: str = "test_slo", batch_id: int = 1) -> BurnRateAlert:
    return BurnRateAlert(
        alert_id="slo-test",
        slo_name=slo_name,
        alert_type="fast_burn",
        burn_rate=20.0,
        budget_remaining_pct=10.0,
        exhaustion_eta_batches=2,
        batch_id=batch_id,
        triggered_at=datetime.now(timezone.utc).isoformat(),
        message="Fast burn test",
    )


def _pump_healthy(controller: CanaryController, slo: SLOEngine, n: int, start_batch: int = 1):
    """Simulate n healthy batches through the controller."""
    evals = []
    for i in range(n):
        batch_id = start_batch + i
        dr = make_drift_result(batch_id=batch_id)
        slo_alerts = slo.record_batch(
            batch_id=batch_id,
            batch_summary={"batch_id": batch_id},
            drift_result=dr,
        )
        ev = controller.evaluate(
            model_name="fraud_detection_v1",
            batch_id=batch_id,
            drift_result=dr,
            burn_rate_alerts=slo_alerts,
        )
        evals.append(ev)
    return evals


# ────────────────────────────────────────────────────────────────
# CANARY STAGE ENUM
# ────────────────────────────────────────────────────────────────

class TestCanaryStage:
    def test_next_stage_from_canary(self):
        assert CanaryStage.next_stage(CanaryStage.CANARY) == CanaryStage.EXPANDING

    def test_next_stage_from_majority(self):
        assert CanaryStage.next_stage(CanaryStage.MAJORITY) == CanaryStage.STABLE

    def test_next_stage_from_stable_is_none(self):
        assert CanaryStage.next_stage(CanaryStage.STABLE) is None

    def test_from_weight(self):
        assert CanaryStage.from_weight(0.10) == CanaryStage.CANARY
        assert CanaryStage.from_weight(0.30) == CanaryStage.EXPANDING
        assert CanaryStage.from_weight(1.00) == CanaryStage.STABLE


# ────────────────────────────────────────────────────────────────
# NO CANARY
# ────────────────────────────────────────────────────────────────

class TestNoCanary:
    def test_no_canary_returns_production_posture_decision(self):
        # When no canary is active, the controller evaluates production model health
        # and returns a meaningful posture (promote/hold/rollback) — never "no_canary".
        registry = ModelRegistry()
        slo = SLOEngine(slo_targets=[make_slo_target()])
        controller = CanaryController(model_registry=registry, slo_engine=slo)

        ev = controller.evaluate(
            model_name="fraud_detection_v1",
            batch_id=1,
            drift_result=make_drift_result(),
            burn_rate_alerts=[],
        )
        assert ev.decision in ("promote", "hold", "rollback"), (
            f"Expected a posture decision, got {ev.decision!r}"
        )
        assert ev.rationale, "Posture decision must include a rationale"

    def test_deploy_canary_creates_canary_version(self):
        controller, registry, _ = make_setup(seed_canary=False)
        canary = controller.deploy_canary("fraud_detection_v1")
        assert canary.status == "canary"
        assert abs(canary.canary_weight - 0.10) < 1e-6


# ────────────────────────────────────────────────────────────────
# HOLD DECISION
# ────────────────────────────────────────────────────────────────

class TestHold:
    def test_hold_before_promotion_threshold(self):
        controller, registry, slo = make_setup()
        # Feed fewer healthy batches than CONSECUTIVE_HEALTHY_FOR_PROMOTE
        evals = _pump_healthy(controller, slo, n=3)
        holds = [e for e in evals if e.decision == "hold"]
        assert len(holds) > 0

    def test_hold_decision_does_not_change_weight(self):
        controller, registry, slo = make_setup()
        canary = registry.get_canary_version("fraud_detection_v1")
        assert canary is not None
        initial_weight = canary.canary_weight
        _pump_healthy(controller, slo, n=2)
        assert abs(canary.canary_weight - initial_weight) < 1e-6


# ────────────────────────────────────────────────────────────────
# PROMOTE DECISION
# ────────────────────────────────────────────────────────────────

class TestPromote:
    def test_promote_after_consecutive_healthy_batches(self):
        controller, registry, slo = make_setup()
        n = controller.CONSECUTIVE_HEALTHY_FOR_PROMOTE + 2
        evals = _pump_healthy(controller, slo, n=n)
        promotes = [e for e in evals if e.decision == "promote"]
        assert len(promotes) >= 1

    def test_promote_advances_traffic_weight(self):
        controller, registry, slo = make_setup()
        canary = registry.get_canary_version("fraud_detection_v1")
        assert canary is not None
        initial_weight = canary.canary_weight

        n = controller.CONSECUTIVE_HEALTHY_FOR_PROMOTE + 2
        _pump_healthy(controller, slo, n=n)

        assert canary.canary_weight > initial_weight

    def test_promote_resets_consecutive_counter(self):
        """After a promote, the consecutive-healthy counter resets so the next
        stage also requires N healthy batches."""
        controller, registry, slo = make_setup()
        canary = registry.get_canary_version("fraud_detection_v1")
        assert canary is not None

        n = controller.CONSECUTIVE_HEALTHY_FOR_PROMOTE + 2
        evals = _pump_healthy(controller, slo, n=n)

        promote_evals = [e for e in evals if e.decision == "promote"]
        if promote_evals:
            # The batch right after a promote should have consecutive_healthy reset
            promote_idx = evals.index(promote_evals[0])
            if promote_idx + 1 < len(evals):
                assert evals[promote_idx + 1].consecutive_healthy <= 1


# ────────────────────────────────────────────────────────────────
# ROLLBACK DECISION
# ────────────────────────────────────────────────────────────────

class TestRollback:
    def test_fast_burn_triggers_rollback(self):
        controller, registry, slo = make_setup()
        dr = make_drift_result(batch_id=10)
        fast_burn = make_fast_burn_alert(batch_id=10)

        ev = controller.evaluate(
            model_name="fraud_detection_v1",
            batch_id=10,
            drift_result=dr,
            burn_rate_alerts=[fast_burn],
        )
        assert ev.decision == "rollback"
        assert ev.auto_executed is True

    def test_rollback_transitions_canary_to_rolled_back(self):
        controller, registry, slo = make_setup()
        canary = registry.get_canary_version("fraud_detection_v1")
        assert canary is not None
        canary_id = canary.version_id

        controller.evaluate(
            model_name="fraud_detection_v1",
            batch_id=5,
            drift_result=make_drift_result(batch_id=5),
            burn_rate_alerts=[make_fast_burn_alert(batch_id=5)],
        )

        rolled = next(
            (v for v in registry.get_all_versions().get("fraud_detection_v1", [])
             if v["version_id"] == canary_id),
            None,
        )
        assert rolled is not None
        assert rolled["status"] == "rolled_back"

    def test_rollback_restores_stable_to_full_weight(self):
        controller, registry, slo = make_setup()
        stable = registry.get_stable_version("fraud_detection_v1")
        assert stable is not None
        stable_id = stable.version_id

        controller.evaluate(
            model_name="fraud_detection_v1",
            batch_id=5,
            drift_result=make_drift_result(batch_id=5),
            burn_rate_alerts=[make_fast_burn_alert(batch_id=5)],
        )

        stable_after = next(
            (v for v in registry.get_all_versions().get("fraud_detection_v1", [])
             if v["version_id"] == stable_id),
            None,
        )
        # Stable should have weight restored
        assert stable_after is not None
        assert abs(stable_after["canary_weight"] - 1.0) < 1e-6


# ────────────────────────────────────────────────────────────────
# STATS & SNAPSHOTS
# ────────────────────────────────────────────────────────────────

class TestStatsAndSnapshots:
    def test_stats_count_decisions(self):
        controller, registry, slo = make_setup()
        _pump_healthy(controller, slo, n=3)
        stats = controller.get_stats()
        assert stats["total_evaluations"] == 3
        assert stats["holds"] + stats["promotes"] + stats["rollbacks"] == 3

    def test_evaluations_snapshot_bounded(self):
        controller, registry, slo = make_setup()
        _pump_healthy(controller, slo, n=10)
        snap = controller.get_evaluations_snapshot(limit=3)
        assert len(snap) <= 3

    def test_latest_evaluation_is_last(self):
        controller, registry, slo = make_setup()
        evals = _pump_healthy(controller, slo, n=4)
        latest = controller.get_latest_evaluation()
        assert latest is not None
        assert latest.batch_id == evals[-1].batch_id
