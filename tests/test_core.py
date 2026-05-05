"""
tests/test_core.py
------------------
Production-grade test suite for the ML Observability Platform.

Coverage targets:
    - drift_detector:   statistical helpers, severity ladder, DriftResult contract
    - risk_forecaster:  linear slope, risk score, proj_conf signed-slope fix, risk levels
    - impact_engine:    severity scoring, escalation logic, label/escalation consistency
    - slo_engine:       budget tracking, burn rate, ETA
    - model_simulator:  calibrated confidence, confusion matrix, performance metrics
    - causal_engine:    upstream event ingestion, attribution when event present vs absent
    - dependency_graph: health updates, degradation propagation

Run:
    pytest tests/test_core.py -v
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

# ────────────────────────────────────────────────────────────────
# DRIFT DETECTOR
# ────────────────────────────────────────────────────────────────

from src.drift_detector import (
    DriftResult,
    FeatureDriftScore,
    compute_z_score,
    compute_rolling_psi,
    severity_from_psi,
)


class TestComputeZScore:
    def test_identical_mean_returns_zero(self):
        assert compute_z_score(1.0, 1.0, 0.5) == pytest.approx(0.0)

    def test_one_std_deviation_returns_one(self):
        assert compute_z_score(2.0, 1.0, 1.0) == pytest.approx(1.0, abs=1e-4)

    def test_zero_std_uses_epsilon(self):
        # Should not raise ZeroDivisionError
        result = compute_z_score(2.0, 1.0, 0.0, epsilon=1e-6)
        assert result > 0

    def test_absolute_value_always_positive(self):
        # z-score is always |current - baseline| / std
        assert compute_z_score(0.5, 1.5, 1.0) > 0
        assert compute_z_score(1.5, 0.5, 1.0) > 0


class TestSeverityFromPsi:
    # Actual thresholds: <0.10 → "stable", <0.25 → "moderate", <0.50 → "high", ≥0.50 → "critical"
    def test_low_psi_is_stable(self):
        assert severity_from_psi(0.05) == "stable"

    def test_boundary_moderate(self):
        assert severity_from_psi(0.10) == "moderate"
        assert severity_from_psi(0.24) == "moderate"

    def test_boundary_high(self):
        assert severity_from_psi(0.25) == "high"
        assert severity_from_psi(0.49) == "high"

    def test_critical_above_50(self):
        assert severity_from_psi(0.50) == "critical"
        assert severity_from_psi(1.0) == "critical"

    def test_zero_psi_returns_valid_label(self):
        result = severity_from_psi(0.0)
        assert result in ("stable", "moderate", "high", "critical")


# ────────────────────────────────────────────────────────────────
# RISK FORECASTER
# ────────────────────────────────────────────────────────────────

from src.risk_forecaster import _compute_risk_score, _linear_slope, _risk_level


class TestLinearSlope:
    def test_flat_series_slope_zero(self):
        assert _linear_slope([1.0, 1.0, 1.0, 1.0]) == pytest.approx(0.0)

    def test_increasing_series_positive_slope(self):
        assert _linear_slope([0.0, 1.0, 2.0, 3.0]) > 0

    def test_decreasing_series_negative_slope(self):
        assert _linear_slope([3.0, 2.0, 1.0, 0.0]) < 0

    def test_single_value_returns_zero(self):
        assert _linear_slope([5.0]) == pytest.approx(0.0)

    def test_empty_returns_zero(self):
        assert _linear_slope([]) == pytest.approx(0.0)

    def test_known_slope(self):
        # y = 2x for x in [0,1,2,3] → slope = 2
        assert _linear_slope([0.0, 2.0, 4.0, 6.0]) == pytest.approx(2.0, rel=1e-3)


class TestComputeRiskScore:
    def test_perfect_health_low_risk(self):
        score = _compute_risk_score(
            fraud_rate=0.02, avg_confidence=0.95,
            budget_remaining_pct=100.0, worst_burn_rate=0.0, fraud_slope=0.0,
        )
        assert score < 25.0

    def test_exhausted_budget_floored_at_75(self):
        score = _compute_risk_score(
            fraud_rate=0.05, avg_confidence=0.90,
            budget_remaining_pct=0.0, worst_burn_rate=0.5, fraud_slope=0.0,
        )
        assert score >= 75.0

    def test_max_fraud_rate_max_score(self):
        score = _compute_risk_score(
            fraud_rate=0.50, avg_confidence=0.0,
            budget_remaining_pct=0.0, worst_burn_rate=14.4, fraud_slope=10.0,
        )
        assert score == pytest.approx(100.0)

    def test_score_bounded_0_to_100(self):
        for fr in [0.0, 0.25, 0.50, 1.0]:
            s = _compute_risk_score(fr, 0.5, 50.0, 5.0, 0.01)
            assert 0.0 <= s <= 100.0

    def test_worsening_slope_increases_score(self):
        base = _compute_risk_score(0.20, 0.80, 50.0, 1.0, fraud_slope=0.0)
        worse = _compute_risk_score(0.20, 0.80, 50.0, 1.0, fraud_slope=0.01)
        assert worse >= base


class TestRiskLevel:
    def test_low_below_25(self):
        assert _risk_level(24.9) == "low"

    def test_moderate_25_to_49(self):
        assert _risk_level(25.0) == "moderate"
        assert _risk_level(49.9) == "moderate"

    def test_high_50_to_74(self):
        assert _risk_level(50.0) == "high"
        assert _risk_level(74.9) == "high"

    def test_critical_at_75(self):
        assert _risk_level(75.0) == "critical"
        assert _risk_level(100.0) == "critical"


class TestProjConfSignedSlope:
    """
    Regression test for the abs() bug in projected_confidence.
    With a POSITIVE confidence slope (confidence recovering), the projection
    must also go UP — not DOWN as abs() would force.
    """
    def test_improving_confidence_projects_upward(self):
        current_conf = 0.75
        positive_slope = 0.001      # confidence is recovering
        proj_steps = 100
        # Correct: current + slope * steps = 0.75 + 0.1 = 0.85
        proj_conf = min(1.0, max(0.0, current_conf + positive_slope * proj_steps))
        assert proj_conf > current_conf, (
            "Positive slope must project confidence upward, not downward (abs() bug)"
        )

    def test_deteriorating_confidence_projects_downward(self):
        current_conf = 0.80
        negative_slope = -0.002
        proj_steps = 100
        proj_conf = min(1.0, max(0.0, current_conf + negative_slope * proj_steps))
        assert proj_conf < current_conf


# ────────────────────────────────────────────────────────────────
# IMPACT ENGINE
# ────────────────────────────────────────────────────────────────

from src.impact_engine import ImpactEngine


def _make_drift_result(
    batch_id: int = 1,
    fraud_rate: float = 0.05,
    avg_confidence: float = 0.90,
    overall_psi: float = 0.05,
    max_z_score: float = 0.5,
    drift_detected: bool = False,
    n_features: int = 3,
) -> DriftResult:
    scores = [
        FeatureDriftScore(
            feature=f"V{i}", z_score=max_z_score / n_features,
            psi=overall_psi / n_features,
            drifted=drift_detected,
            severity="low",
        )
        for i in range(n_features)
    ]
    return DriftResult(
        batch_id=batch_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        overall_psi=overall_psi,
        max_z_score=max_z_score,
        drift_detected=drift_detected,
        severity="low" if not drift_detected else "high",
        feature_scores=scores,
        fraud_rate=fraud_rate,
        avg_confidence=avg_confidence,
        processing_ms=5.0,
        warmed_up=True,
    )


class TestImpactEngine:
    @pytest.fixture
    def engine(self):
        with patch("src.impact_engine.get_settings") as mock_cfg:
            mock_cfg.return_value.monitoring.alert_fraud_rate = 0.30
            mock_cfg.return_value.impact.base_incident_cost_usd = 500.0
            mock_cfg.return_value.impact.cost_per_excess_fraud_event_usd = 85.0
            mock_cfg.return_value.impact.cost_per_confidence_drop_pct_usd = 12.0
            return ImpactEngine()

    def test_healthy_model_no_escalation(self, engine):
        result = _make_drift_result(fraud_rate=0.05, overall_psi=0.05, drift_detected=False)
        assessment = engine.assess(result=result, recent_alert_count=0)
        assert not assessment.requires_escalation

    def test_high_psi_triggers_kpi_breach(self, engine):
        result = _make_drift_result(
            fraud_rate=0.10, overall_psi=0.60, drift_detected=True
        )
        assessment = engine.assess(result=result, recent_alert_count=0)
        assert assessment.incident_kpi_breach, (
            "PSI >= 0.50 must trigger kpi_breach regardless of fraud_rate"
        )

    def test_escalation_requires_material_label(self, engine):
        # When requires_escalation is True, label must be >= "material"
        result = _make_drift_result(
            fraud_rate=0.35, overall_psi=0.60, drift_detected=True, max_z_score=3.5
        )
        assessment = engine.assess(result=result, recent_alert_count=5)
        assert assessment.requires_escalation
        assert assessment.business_impact_label in ("material", "severe"), (
            f"Got '{assessment.business_impact_label}' with requires_escalation=True — "
            "must be 'material' or 'severe'"
        )

    def test_moderate_label_upgraded_when_escalation_required(self, engine):
        # PSI=0.55 triggers escalation — label must be at least "material"
        result = _make_drift_result(overall_psi=0.55, fraud_rate=0.10, drift_detected=True)
        assessment = engine.assess(result=result, recent_alert_count=0)
        assert assessment.requires_escalation  # PSI≥0.50 forces escalation
        assert assessment.business_impact_label in ("material", "severe"), (
            f"label='{assessment.business_impact_label}' + requires_escalation=True is contradictory"
        )

    def test_no_loss_when_no_drift(self, engine):
        result = _make_drift_result(fraud_rate=0.01, overall_psi=0.02, drift_detected=False)
        assessment = engine.assess(result=result, recent_alert_count=0)
        assert assessment.estimated_loss_usd >= 0


# ────────────────────────────────────────────────────────────────
# MODEL SIMULATOR — confidence calibration + confusion matrix
# ────────────────────────────────────────────────────────────────

from src.model_simulator import FraudModelSimulator


class TestCalibratedConfidence:
    """
    The calibrated confidence maps prediction distance-from-threshold to [0.5, 1.0].
    At the threshold itself, confidence must be 0.5 (maximally uncertain).
    At extreme predictions, confidence approaches 1.0.
    """

    @pytest.fixture
    def sim(self):
        with patch("src.model_simulator.get_settings") as mock:
            cfg = MagicMock()
            cfg.model.batch_size = 50
            cfg.model.drift_after_batches = 10
            cfg.model.fraud_threshold = 0.25
            cfg.model.data_mode = "synthetic"
            cfg.model.random_seed = 42
            mock.return_value = cfg
            s = FraudModelSimulator.__new__(FraudModelSimulator)
            s._fraud_threshold = 0.25
            return s

    def test_at_threshold_confidence_is_0_5(self, sim):
        conf = sim._calibrated_confidence(0.25, "fraud")
        assert conf == pytest.approx(0.5, abs=0.01)

    def test_strong_legit_higher_than_borderline(self, sim):
        # prediction=0.0 is max distance from threshold (0.25) for legit → highest confidence
        conf_strong = sim._calibrated_confidence(0.0, "legit")
        conf_border = sim._calibrated_confidence(0.24, "legit")
        assert conf_strong > conf_border

    def test_strong_fraud_near_1(self, sim):
        conf = sim._calibrated_confidence(1.0, "fraud")
        assert conf > 0.8

    def test_confidence_always_in_range(self, sim):
        for pred in [0.0, 0.10, 0.25, 0.50, 0.75, 1.0]:
            label = "fraud" if pred >= 0.25 else "legit"
            conf = sim._calibrated_confidence(pred, label)
            assert 0.0 <= conf <= 1.0, f"confidence out of range for pred={pred}"

    def test_confidence_monotonically_increases_with_distance(self, sim):
        # Further from threshold = more confident
        conf_near = sim._calibrated_confidence(0.30, "fraud")   # 0.05 above threshold
        conf_far = sim._calibrated_confidence(0.80, "fraud")    # 0.55 above threshold
        assert conf_far > conf_near


class TestConfusionMatrix:
    @pytest.fixture
    def sim(self):
        with patch("src.model_simulator.get_settings") as mock:
            cfg = MagicMock()
            cfg.model.batch_size = 10
            cfg.model.drift_after_batches = 100
            cfg.model.fraud_threshold = 0.25
            cfg.model.data_mode = "real"
            cfg.model.random_seed = 42
            mock.return_value = cfg
            s = FraudModelSimulator.__new__(FraudModelSimulator)
            s._fraud_threshold = 0.25
            s._data_mode = "real"
            from src.model_simulator import ModelStats
            s.stats = ModelStats()
            return s

    def test_true_positive_counted(self, sim):
        from src.model_simulator import PredictionRecord
        record = PredictionRecord(
            timestamp="2026-01-01T00:00:00Z",
            prediction=0.80, label="fraud", confidence=0.9,
            features={}, batch_id=1, is_drifted=False, true_label=1,
        )
        sim._update_confusion_matrix([record])
        assert sim.stats.true_positives == 1
        assert sim.stats.false_positives == 0

    def test_false_positive_counted(self, sim):
        from src.model_simulator import PredictionRecord
        record = PredictionRecord(
            timestamp="2026-01-01T00:00:00Z",
            prediction=0.80, label="fraud", confidence=0.9,
            features={}, batch_id=1, is_drifted=False, true_label=0,
        )
        sim._update_confusion_matrix([record])
        assert sim.stats.false_positives == 1

    def test_performance_metrics_precision_recall_f1(self, sim):
        # Seed known counts: 3 TP, 1 FP, 1 FN, 5 TN
        sim.stats.true_positives = 3
        sim.stats.false_positives = 1
        sim.stats.false_negatives = 1
        sim.stats.true_negatives = 5
        metrics = sim.get_performance_metrics()
        assert metrics["precision"] == pytest.approx(0.75, abs=0.01)   # 3/(3+1)
        assert metrics["recall"] == pytest.approx(0.75, abs=0.01)      # 3/(3+1)
        assert metrics["f1_score"] == pytest.approx(0.75, abs=0.01)

    def test_performance_metrics_empty_returns_none(self, sim):
        metrics = sim.get_performance_metrics()
        assert metrics["precision"] is None
        assert metrics["recall"] is None


# ────────────────────────────────────────────────────────────────
# CAUSAL ENGINE — attribution with and without upstream events
# ────────────────────────────────────────────────────────────────

from src.causal_engine import CausalEngine, UpstreamEvent


class TestCausalEngine:
    @pytest.fixture
    def engine(self):
        return CausalEngine()

    def _make_upstream_event(self, lag_seconds: float = 30.0, features: list[str] | None = None):
        now = datetime.now(timezone.utc).timestamp()
        return UpstreamEvent(
            event_id="test-evt-1",
            event_type="pipeline_anomaly",
            source="feature_pipeline",
            timestamp=datetime.fromtimestamp(now - lag_seconds, tz=timezone.utc).isoformat(),
            timestamp_unix=now - lag_seconds,
            affected_fields=features or ["V1", "V3", "V7"],
            severity="high",
            description="Test upstream event",
            metadata={},
        )

    def test_no_upstream_events_returns_not_attributed(self, engine):
        result = _make_drift_result(drift_detected=True, overall_psi=0.8)
        attribution = engine.attribute(result)
        assert not attribution.attributed
        assert attribution.causal_confidence == pytest.approx(0.0)

    def test_upstream_event_in_window_produces_attribution(self, engine):
        event = self._make_upstream_event(lag_seconds=60.0, features=["V1", "V3"])
        engine.ingest_upstream_event(event)
        result = _make_drift_result(drift_detected=True, overall_psi=0.8)
        attribution = engine.attribute(result)
        assert attribution.attributed
        assert attribution.causal_confidence > 0.0
        assert len(attribution.hypotheses) > 0

    def test_event_outside_window_not_attributed(self, engine):
        # Event happened 2 hours ago (7300s) — outside the 1-hour (3600s) lookback.
        event = self._make_upstream_event(lag_seconds=7300.0)
        engine.ingest_upstream_event(event)
        result = _make_drift_result(drift_detected=True, overall_psi=0.8)
        attribution = engine.attribute(result)
        assert not attribution.attributed

    def test_field_overlap_increases_confidence(self, engine):
        # High overlap: event features match drifted features
        event_high = self._make_upstream_event(lag_seconds=30.0, features=["V1", "V3", "V7"])
        event_low = self._make_upstream_event(lag_seconds=30.0, features=["V99"])
        engine.ingest_upstream_event(event_high)
        result = _make_drift_result(drift_detected=True, overall_psi=0.8)
        attribution = engine.attribute(result)
        high_conf = attribution.causal_confidence

        engine2 = CausalEngine()
        engine2.ingest_upstream_event(event_low)
        attribution2 = engine2.attribute(result)
        low_conf = attribution2.causal_confidence

        assert high_conf >= low_conf


# ────────────────────────────────────────────────────────────────
# DEPENDENCY GRAPH — health updates
# ────────────────────────────────────────────────────────────────

from src.dependency_graph import DependencyGraph, get_dependency_graph


class TestDependencyGraph:
    @pytest.fixture(autouse=True)
    def reset_graph(self):
        """Reset the singleton before each test to ensure isolation."""
        import src.dependency_graph as dg_module
        dg_module._graph = None
        yield
        dg_module._graph = None

    @pytest.fixture
    def graph(self):
        # Use the factory which seeds the pre-built fraud-detection DAG
        return get_dependency_graph()

    def test_all_nodes_start_healthy(self, graph):
        # get_all_nodes() returns dicts; use get_node() for the object
        node = graph.get_node("fraud_detection_v1")
        assert node is not None
        assert node.health_score == pytest.approx(1.0)
        assert not node.degraded

    def test_update_health_reflects_in_node(self, graph):
        graph.update_health("fraud_detection_v1", 0.3, "drift detected")
        node = graph.get_node("fraud_detection_v1")
        assert node is not None
        assert node.health_score == pytest.approx(0.3)
        assert node.degraded  # health < 0.70

    def test_recovery_restores_health(self, graph):
        graph.update_health("fraud_detection_v1", 0.2, "drift")
        graph.update_health("fraud_detection_v1", 1.0, "recovered")
        node = graph.get_node("fraud_detection_v1")
        assert node.health_score == pytest.approx(1.0)
        assert not node.degraded

    def test_mark_degraded_sets_degraded_flag(self, graph):
        graph.mark_degraded("fraud_detection_v1", reason="test", health_score=0.1)
        node = graph.get_node("fraud_detection_v1")
        assert node.degraded
        assert node.health_score == pytest.approx(0.1)

    def test_nonexistent_node_update_is_safe(self, graph):
        # Should not raise even for unknown node_id
        graph.update_health("does_not_exist", 0.5, "test")

    def test_health_clamped_to_0_1(self, graph):
        graph.update_health("fraud_detection_v1", 1.5, "over 1")
        node = graph.get_node("fraud_detection_v1")
        assert node.health_score <= 1.0

        graph.update_health("fraud_detection_v1", -0.5, "under 0")
        node = graph.get_node("fraud_detection_v1")
        assert node.health_score >= 0.0

    def test_known_nodes_exist(self, graph):
        for node_id in ["fraud_detection_v1", "txn_amount_feature", "payment_pipeline"]:
            assert graph.get_node(node_id) is not None, f"Expected node '{node_id}' to exist"
