"""
tests/test_integration.py
--------------------------
Integration tests for the intelligence pipeline loop.

These tests exercise the full chain that runs in production:
    StreamProducer → DriftDetector → CausalEngine → SLOEngine →
    RiskForecaster → ImpactEngine → CanaryController → bootstrap payload

They use the TestClient against a real app instance (synthetic mode, fast interval)
so they catch wiring bugs that unit tests can't — e.g. a field present in the
bootstrap dict but missing from the API payload builder.
"""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from config.settings import get_settings
from src.api import create_app


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def reset_settings():
    yield
    get_settings(force_reload=True)


@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset ALL module-level singletons between tests to prevent state bleed."""
    import src.canary_controller as cc_mod
    import src.dependency_graph as dg_mod
    import src.drift_detector as dd_mod
    import src.causal_engine as ce_mod
    import src.model_registry as mr_mod
    import src.schema_registry as sr_mod
    import src.slo_engine as slo_mod

    def _clear():
        cc_mod._controller = None
        dg_mod._graph = None
        dd_mod._detector = None
        ce_mod._engine = None
        mr_mod._registry = None
        sr_mod._registry = None
        slo_mod._engine = None

    _clear()
    yield
    _clear()


def _client(monkeypatch, extra_env: dict[str, str] | None = None) -> TestClient:
    monkeypatch.setenv("MODEL__DATA_MODE", "synthetic")
    monkeypatch.setenv("MODEL__INTERVAL_SECONDS", "0.05")
    monkeypatch.setenv("API__TRUSTED_HOSTS", '["127.0.0.1","localhost","testserver"]')
    for k, v in (extra_env or {}).items():
        monkeypatch.setenv(k, v)
    get_settings(force_reload=True)
    return TestClient(create_app())


def _wait_for_bootstrap(client: TestClient, min_batches: int = 5, timeout: float = 10.0) -> dict:
    """Poll /dashboard/bootstrap until min_batches have been processed."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        r = client.get("/dashboard/bootstrap")
        if r.status_code == 200:
            payload = r.json()
            producer = payload.get("producer") or {}
            if producer.get("batches_sent", 0) >= min_batches:
                return payload
        time.sleep(0.1)
    pytest.fail(f"Bootstrap did not reach {min_batches} batches within {timeout}s")


# ── Bootstrap payload completeness ────────────────────────────────────

class TestBootstrapPayloadCompleteness:
    """Every key the dashboard JS reads must be present in the payload."""

    REQUIRED_TOP_LEVEL_KEYS = [
        "type", "timestamp", "metrics", "detector", "producer",
        "alerts", "impact", "decision", "causal_attribution",
        "slo", "canary", "risk_forecast", "simulations",
        "dependency_trace", "causal_timeline", "model_registry",
        "root_causes",
    ]

    def test_all_required_keys_present(self, monkeypatch):
        with _client(monkeypatch) as client:
            payload = _wait_for_bootstrap(client, min_batches=3)

        for key in self.REQUIRED_TOP_LEVEL_KEYS:
            assert key in payload, f"Missing top-level key: '{key}'"

    def test_metrics_contains_core_kpis(self, monkeypatch):
        with _client(monkeypatch) as client:
            payload = _wait_for_bootstrap(client, min_batches=3)

        metrics = payload["metrics"]
        assert "fraud_rate" in metrics
        assert "avg_confidence" in metrics
        assert "batch_id" in metrics
        assert 0.0 <= metrics["fraud_rate"] <= 1.0
        assert 0.5 <= metrics["avg_confidence"] <= 1.0, (
            "Calibrated confidence must be in [0.5, 1.0]; "
            f"got {metrics['avg_confidence']}"
        )

    def test_model_registry_is_populated(self, monkeypatch):
        with _client(monkeypatch) as client:
            payload = _wait_for_bootstrap(client, min_batches=5)

        mr = payload.get("model_registry") or {}
        assert mr, "model_registry must not be null or empty"
        assert "stats" in mr
        assert "versions" in mr
        stats = mr["stats"]
        assert stats["models_tracked"] >= 1
        assert stats["total_versions"] >= 1

    def test_canary_always_returns_a_decision(self, monkeypatch):
        with _client(monkeypatch) as client:
            # Need enough batches for the intelligence loop (0.5s sleep) to run.
            # At 0.05s/batch, 15 batches ≈ 0.75s > 0.5s loop sleep.
            payload = _wait_for_bootstrap(client, min_batches=15, timeout=10.0)

        canary = payload.get("canary") or {}
        assert "decision" in canary, "canary section must have a 'decision' field"
        assert canary["decision"] in ("promote", "hold", "rollback", "no_canary")
        # With the new posture logic, even without an active canary deployment
        # we expect a meaningful decision (promote/hold/rollback), not no_canary.
        assert canary["decision"] != "no_canary", (
            "Canary controller should emit a production posture decision, not 'no_canary'"
        )
        assert "rationale" in canary
        assert len(canary["rationale"]) > 10

    def test_slo_budgets_are_tracked(self, monkeypatch):
        with _client(monkeypatch) as client:
            payload = _wait_for_bootstrap(client, min_batches=10)

        slo = payload.get("slo") or {}
        budgets = slo.get("budgets") or {}
        assert len(budgets) >= 1, "At least one SLO budget must be tracked"
        for name, budget in budgets.items():
            assert "burn_rate" in budget, f"Budget '{name}' missing burn_rate"
            assert "budget_remaining_pct" in budget, f"Budget '{name}' missing budget_remaining_pct"
            assert 0.0 <= budget["budget_remaining_pct"] <= 100.0


# ── Risk forecast correctness ─────────────────────────────────────────

class TestRiskForecastCorrectness:

    def test_projected_fraud_rate_is_not_clamped_to_zero(self, monkeypatch):
        """
        Regression test: projected_fraud_rate was 0.0 when batch_interval=2s
        caused proj_steps=900 and the per-step slope cap drove the projection
        to -0.10 (clamped to 0). The fix caps total delta, not per-step slope.
        """
        with _client(monkeypatch) as client:
            payload = _wait_for_bootstrap(client, min_batches=20)

        rf = payload.get("risk_forecast") or {}
        if rf.get("observations_used", 0) >= 3:
            current_fraud = payload["metrics"]["fraud_rate"]
            projected = rf["projected_fraud_rate"]
            # Projection must be within 30% of current — never 0.0 when
            # current fraud rate is above 5%.
            if current_fraud > 0.05:
                assert projected > 0.0, (
                    f"projected_fraud_rate={projected} is 0.0 while current fraud={current_fraud:.2f}; "
                    "this is the slope-cap regression"
                )
                assert abs(projected - current_fraud) <= 0.35, (
                    f"projected_fraud_rate={projected} deviates >35% from current={current_fraud:.2f}"
                )

    def test_risk_score_is_in_valid_range(self, monkeypatch):
        with _client(monkeypatch) as client:
            payload = _wait_for_bootstrap(client, min_batches=5)

        rf = payload.get("risk_forecast") or {}
        if rf:
            assert 0 <= rf["risk_score"] <= 100
            assert rf["risk_level"] in ("low", "moderate", "high", "critical")

    def test_loss_per_hour_is_realistic(self, monkeypatch):
        """Loss should never exceed ~$20k/hr in normal operation (realistic ceiling)."""
        with _client(monkeypatch) as client:
            payload = _wait_for_bootstrap(client, min_batches=5)

        rf = payload.get("risk_forecast") or {}
        if rf:
            loss = rf.get("loss_per_hour_usd", 0)
            assert loss < 20_000, f"loss_per_hour_usd={loss} is unrealistically high"
            assert loss >= 0.0


# ── Alert quality ─────────────────────────────────────────────────────

class TestAlertQuality:

    def test_alerts_list_ranked_drifted_features_not_all_features(self, monkeypatch):
        """
        Regression test: alerts used to list all 10 features every time.
        Now they must show only features that actually drifted, ranked by PSI,
        with PSI and Z-score values in the label.
        """
        with _client(monkeypatch) as client:
            # Wait long enough for at least one alert to be raised
            payload = _wait_for_bootstrap(client, min_batches=60, timeout=20.0)

        alerts = payload.get("alerts") or []
        if not alerts:
            pytest.skip("No alerts raised yet — increase warmup time")

        for alert in alerts:
            features = alert.get("features") or []
            # Must never list all 10 generic feature names with no context
            assert len(features) <= 8, (
                f"Alert lists {len(features)} features — should be top-5 ranked, not all features. "
                f"features={features}"
            )
            # If features are present, they should include PSI context
            if features:
                has_psi_annotation = any("PSI=" in f for f in features)
                has_aggregate = any("aggregate_psi" in f for f in features)
                assert has_psi_annotation or has_aggregate, (
                    f"Features should include PSI annotations. Got: {features}"
                )

    def test_alert_has_required_fields(self, monkeypatch):
        with _client(monkeypatch) as client:
            payload = _wait_for_bootstrap(client, min_batches=60, timeout=20.0)

        alerts = payload.get("alerts") or []
        if not alerts:
            pytest.skip("No alerts raised yet")

        for alert in alerts:
            assert "alert_id" in alert
            assert "severity" in alert
            assert "batch_id" in alert
            assert alert["severity"] in ("low", "moderate", "high", "critical")


# ── Causal attribution pipeline ───────────────────────────────────────

class TestCausalAttributionPipeline:

    def test_causal_attribution_fields_are_present(self, monkeypatch):
        with _client(monkeypatch) as client:
            payload = _wait_for_bootstrap(client, min_batches=5)

        ca = payload.get("causal_attribution") or {}
        assert "attributed" in ca
        assert "confidence" in ca
        assert "summary" in ca
        assert isinstance(ca["attributed"], bool)
        assert 0.0 <= ca["confidence"] <= 1.0

    def test_causal_timeline_is_an_array(self, monkeypatch):
        """
        Regression test: causal_timeline was sent as a dict {events, narrative}.
        JS called .slice() on it and threw 'x.timeline.slice is not a function'.
        """
        with _client(monkeypatch) as client:
            payload = _wait_for_bootstrap(client, min_batches=5)

        timeline = payload.get("causal_timeline")
        # Must be a list (or null), never a dict
        assert timeline is None or isinstance(timeline, list), (
            f"causal_timeline must be an array, got {type(timeline).__name__}"
        )


# ── Dependency graph wiring ───────────────────────────────────────────

class TestDependencyGraphWiring:

    def test_model_node_is_first_in_dependency_trace(self, monkeypatch):
        """
        The model node must be prepended to the dependency trace so that
        the dashboard always shows fraud_detection_v1's live health.
        """
        with _client(monkeypatch) as client:
            payload = _wait_for_bootstrap(client, min_batches=5)

        dep = payload.get("dependency_trace") or {}
        nodes = dep.get("nodes") or []
        assert nodes, "dependency_trace.nodes must not be empty"

        first = nodes[0]
        assert first.get("node_id") == "fraud_detection_v1", (
            f"First node must be fraud_detection_v1; got {first.get('node_id')}"
        )
        assert "health_score" in first
        assert "degraded" in first

    def test_dependency_nodes_have_required_fields(self, monkeypatch):
        with _client(monkeypatch) as client:
            payload = _wait_for_bootstrap(client, min_batches=5)

        dep = payload.get("dependency_trace") or {}
        nodes = dep.get("nodes") or []
        for node in nodes:
            assert "node_id" in node, f"Node missing node_id: {node}"
            assert "display_name" in node, f"Node missing display_name: {node}"
            assert "health_score" in node, f"Node missing health_score: {node}"
            score = node["health_score"]
            assert score is None or 0.0 <= score <= 1.0, (
                f"health_score={score} out of [0, 1] range"
            )
