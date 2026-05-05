import time

import pytest
from fastapi.testclient import TestClient

from config.settings import get_settings
from src.api import create_app
from src.drift_detector import DriftDetector


@pytest.fixture(autouse=True)
def reload_settings_after_test():
    yield
    get_settings(force_reload=True)


def _build_test_client(monkeypatch, *, state_store_path: str | None = None) -> TestClient:
    monkeypatch.setenv("MODEL__DATA_MODE", "synthetic")
    monkeypatch.setenv("MODEL__INTERVAL_SECONDS", "0.05")
    monkeypatch.setenv(
        "API__TRUSTED_HOSTS",
        '["127.0.0.1","localhost","testserver"]',
    )
    if state_store_path is not None:
        monkeypatch.setenv("STATE_STORE__ENABLED", "true")
        monkeypatch.setenv("STATE_STORE__SQLITE_PATH", state_store_path)
    get_settings(force_reload=True)
    return TestClient(create_app())


class TestDashboardDelivery:
    def test_root_serves_dashboard_html(self, monkeypatch):
        with _build_test_client(monkeypatch) as client:
            response = client.get("/")
        assert response.status_code == 200
        assert "Incident Command Center" in response.text
        assert response.headers["content-type"].startswith("text/html")
        assert response.headers["cache-control"] == "no-store"

    def test_dashboard_assets_are_served_with_cache_headers(self, monkeypatch):
        with _build_test_client(monkeypatch) as client:
            response = client.get("/assets/css/styles.css")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/css")
        assert "max-age=3600" in response.headers["cache-control"]


class TestOperationalEndpoints:
    def test_liveness_and_readiness_endpoints(self, monkeypatch):
        with _build_test_client(monkeypatch) as client:
            live = client.get("/livez")
            ready = client.get("/readyz")

        assert live.status_code == 200
        assert live.json()["status"] == "ok"

        payload = ready.json()
        assert ready.status_code == 200
        assert payload["status"] == "ok"
        assert payload["checks"]["producer_initialized"] is True
        assert payload["checks"]["detector_initialized"] is True
        assert payload["checks"]["dashboard_assets_ready"] is True

    def test_prometheus_metrics_endpoint_exposes_runtime_metrics(self, monkeypatch):
        with _build_test_client(monkeypatch) as client:
            client.get("/livez")
            response = client.get("/metrics/prometheus")

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/plain")
        assert "ml_platform_http_requests_total" in response.text
        assert 'path="/livez"' in response.text
        assert "ml_platform_build_info" in response.text

    def test_can_ingest_and_read_back_upstream_events(self, monkeypatch):
        with _build_test_client(monkeypatch) as client:
            create_response = client.post(
                "/causal/events",
                json={
                    "event_type": "deployment",
                    "source": "feature-store",
                    "affected_fields": ["transaction_amount"],
                    "severity": "high",
                    "description": "Deployed new feature-store transform",
                },
            )
            events_response = client.get("/causal/events?limit=5")

        assert create_response.status_code == 201
        assert "event_id" in create_response.json()

        events = events_response.json()
        assert events_response.status_code == 200
        assert any(event["source"] == "feature-store" for event in events)

    def test_state_snapshots_endpoint_returns_persisted_batches(self, monkeypatch, tmp_path):
        db_path = tmp_path / "platform.sqlite3"
        with _build_test_client(monkeypatch, state_store_path=str(db_path)) as client:
            snapshots = []
            for _ in range(30):
                response = client.get("/state/snapshots?limit=5")
                payload = response.json()
                snapshots = payload["snapshots"]
                if snapshots:
                    break
                time.sleep(0.05)

        assert payload["enabled"] is True
        assert payload["stats"]["enabled"] is True
        assert snapshots, "expected at least one persisted incident snapshot"
        assert snapshots[0]["snapshot_kind"] == "incident_state"


class TestConfigDrivenDetector:
    def test_detector_uses_runtime_detector_config(self, monkeypatch):
        monkeypatch.setenv("MODEL__DATA_MODE", "synthetic")
        monkeypatch.setenv("DETECTOR__WARMUP_BATCHES", "7")
        monkeypatch.setenv("DETECTOR__Z_SCORE_THRESHOLD", "3.5")
        monkeypatch.setenv("DETECTOR__PSI_THRESHOLD", "0.35")
        monkeypatch.setenv("DETECTOR__WINDOW_SIZE", "12")
        monkeypatch.setenv("DETECTOR__ALERT_COOLDOWN_BATCHES", "9")
        monkeypatch.setenv("DETECTOR__MAX_RESULTS", "77")
        monkeypatch.setenv("DETECTOR__MAX_ALERTS", "55")
        monkeypatch.setenv("MONITORING__DRIFT_THRESHOLD", "0.25")

        get_settings(force_reload=True)
        detector = DriftDetector()

        assert detector._warmup_batches == 7
        assert detector._z_score_threshold == 3.5
        assert detector._psi_threshold == 0.35
        assert detector._aggregate_psi_threshold == 0.35
        assert detector._window_size == 12
        assert detector._alert_cooldown_batches == 9
        assert detector._max_results == 77
        assert detector._max_alerts == 55
