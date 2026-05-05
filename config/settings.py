"""
config/settings.py
-------------------
Production-grade configuration management for the
Real-Time ML Observability Platform.

Design principles:
    - Only AppSettings uses BaseSettings (env vars + .env loading)
    - Sub-configs use BaseModel (pure validation)
    - Cross-field validation happens at config boundaries
    - All values validated at startup
    - Singleton factory supports force_reload for tests

Environment variable examples:
    MODEL__BATCH_SIZE=64
    MODEL__DATA_MODE=real
    API__PORT=9000
    MONITORING__DRIFT_THRESHOLD=0.2
    DATA__PATH=data/creditcard.csv
    DETECTOR__WINDOW_SIZE=30
"""

from __future__ import annotations

import logging
from functools import lru_cache

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────
# SUB-CONFIGS
# ────────────────────────────────────────────────────────────────

class ModelConfig(BaseModel):
    """
    Controls batch generation and simulator mode.
    """

    batch_size: int = Field(
        default=50,
        gt=0,
        description="Predictions generated per batch.",
    )
    interval_seconds: float = Field(
        default=2.0,
        gt=0.0,
        description="Seconds between generated batches.",
    )
    drift_after_batches: int = Field(
        default=10,
        ge=0,
        description="Number of healthy batches before drift is injected.",
    )
    random_seed: int = Field(
        default=42,
        ge=0,
        description="Seed for deterministic simulation behavior.",
    )
    fraud_threshold: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="Prediction threshold above which a record is labeled as fraud.",
    )
    data_mode: str = Field(
        default="synthetic",
        description="Simulation source mode: synthetic or real.",
    )

    @field_validator("data_mode")
    @classmethod
    def validate_data_mode(cls, value: str) -> str:
        allowed = {"synthetic", "real"}
        normalized = value.lower().strip()
        if normalized not in allowed:
            raise ValueError(f"data_mode must be one of {allowed}, got '{value}'")
        return normalized

    @model_validator(mode="after")
    def validate_drift_timing(self) -> "ModelConfig":
        if self.drift_after_batches < 3:
            raise ValueError(
                "drift_after_batches must be >= 3 to collect a meaningful baseline"
            )
        return self


class APIConfig(BaseModel):
    """
    Controls FastAPI service behavior.
    """

    host: str = Field(
        default="0.0.0.0",
        description="Host address for FastAPI server.",
    )
    port: int = Field(
        default=8000,
        ge=1024,
        le=65535,
        description="Port for FastAPI server.",
    )
    log_level: str = Field(
        default="info",
        description="Uvicorn log level.",
    )
    cors_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:3000", "http://127.0.0.1:3000"],
        description="Allowed CORS origins for frontend clients.",
    )
    trusted_hosts: list[str] = Field(
        default_factory=lambda: ["127.0.0.1", "localhost", "testserver"],
        description="Allowed Host headers for inbound requests.",
    )
    dashboard_enabled: bool = Field(
        default=True,
        description="Whether the API should serve the Incident Command Center UI.",
    )
    websocket_push_interval_seconds: float = Field(
        default=2.0,
        gt=0.0,
        description="Seconds between live dashboard websocket updates.",
    )
    static_asset_cache_seconds: int = Field(
        default=3600,
        ge=0,
        description="Cache lifetime for dashboard static assets.",
    )

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, value: str) -> str:
        allowed = {"debug", "info", "warning", "error", "critical"}
        normalized = value.lower().strip()
        if normalized not in allowed:
            raise ValueError(f"log_level must be one of {allowed}, got '{value}'")
        return normalized

    @field_validator("trusted_hosts")
    @classmethod
    def validate_trusted_hosts(cls, value: list[str]) -> list[str]:
        cleaned = [host.strip() for host in value if host.strip()]
        if not cleaned:
            raise ValueError("trusted_hosts must contain at least one host")
        return cleaned


class LoggingConfig(BaseModel):
    """
    Controls application log formatting and service identity.
    """

    json_format: bool = Field(
        default=True,
        description="Emit logs as structured JSON.",
    )
    service_name: str = Field(
        default="ml-incident-command-center",
        description="Logical service name included in structured logs.",
    )


class StateStoreConfig(BaseModel):
    """
    Controls local durable snapshot persistence.
    """

    enabled: bool = Field(
        default=False,
        description="Persist incident snapshots to a local SQLite database.",
    )
    sqlite_path: str = Field(
        default="state/platform.sqlite3",
        description="Filesystem path to the SQLite state store.",
    )
    max_snapshots: int = Field(
        default=2000,
        gt=0,
        description="Maximum number of incident snapshots retained.",
    )


class MonitoringConfig(BaseModel):
    """
    Controls queueing and high-level monitoring behavior.
    """

    drift_threshold: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="PSI threshold above which drift contributes to confirmation.",
    )
    alert_fraud_rate: float = Field(
        default=0.30,
        ge=0.0,
        le=1.0,
        description="Fraud-rate threshold above which an alert can fire.",
    )
    max_batch_log_size: int = Field(
        default=100,
        gt=0,
        description="Maximum batch summaries retained in memory.",
    )
    queue_maxsize: int = Field(
        default=1000,
        gt=0,
        description="Maximum subscriber queue size before rotation occurs.",
    )

    @model_validator(mode="after")
    def validate_thresholds(self) -> "MonitoringConfig":
        if self.drift_threshold >= self.alert_fraud_rate:
            raise ValueError(
                f"drift_threshold ({self.drift_threshold}) must be lower than "
                f"alert_fraud_rate ({self.alert_fraud_rate})."
            )
        return self


class DetectorConfig(BaseModel):
    """
    Technical policy for drift detection.
    """

    warmup_batches: int = Field(
        default=5,
        ge=1,
        description="Number of batches to observe before drift alerts are allowed.",
    )
    z_score_threshold: float = Field(
        default=2.0,
        gt=0.0,
        description="Z-score threshold for sudden feature shift detection.",
    )
    psi_threshold: float = Field(
        default=0.20,
        gt=0.0,
        le=1.0,
        description="PSI threshold for gradual feature drift detection.",
    )
    window_size: int = Field(
        default=30,
        ge=5,
        description="Rolling window size used for PSI calculations.",
    )
    alert_cooldown_batches: int = Field(
        default=5,
        ge=0,
        description="Minimum batches between emitted alerts.",
    )
    max_results: int = Field(
        default=200,
        gt=0,
        description="Maximum drift analysis results stored in memory.",
    )
    max_alerts: int = Field(
        default=100,
        gt=0,
        description="Maximum alerts stored in memory.",
    )


class DataConfig(BaseModel):
    """
    Controls real CSV replay mode.
    """

    path: str = Field(
        default="data/creditcard.csv",
        description="Path to real CSV dataset used in real mode.",
    )
    label_column: str | None = Field(
        default="Class",
        description="Preferred label column name when present.",
    )
    feature_columns: list[str] = Field(
        default_factory=list,
        description="Optional explicit feature columns for real mode.",
    )
    max_features: int = Field(
        default=10,
        ge=1,
        description="Maximum number of auto-inferred numeric features.",
    )


class ImpactConfig(BaseModel):
    """
    Controls business-impact estimation.
    """

    base_incident_cost_usd: float = Field(
        default=250.0,
        ge=0.0,
        description="Base cost assigned to any material incident.",
    )
    cost_per_excess_fraud_event_usd: float = Field(
        default=120.0,
        ge=0.0,
        description="Estimated business cost per excess fraud event.",
    )
    cost_per_confidence_drop_pct_usd: float = Field(
        default=25.0,
        ge=0.0,
        description="Estimated cost per percentage-point confidence degradation.",
    )


class DecisionConfig(BaseModel):
    """
    Controls recommendation policy for the decision engine.
    """

    retraining_severity_threshold: int = Field(
        default=75,
        ge=0,
        le=100,
        description="Severity score threshold above which retraining is recommended.",
    )
    rollback_severity_threshold: int = Field(
        default=90,
        ge=0,
        le=100,
        description="Severity score threshold above which rollback is recommended.",
    )
    repeated_alert_count_for_rollback: int = Field(
        default=2,
        ge=1,
        description="Number of recent alerts required before rollback is considered.",
    )


class DistributionConfig(BaseModel):
    """
    Represents one feature distribution.
    """

    mean: float = Field(..., description="Distribution mean.")
    std: float = Field(..., gt=0.0, description="Distribution standard deviation.")


class BaselineConfig(BaseModel):
    """
    Synthetic baseline distributions representing healthy traffic.
    """

    transaction_amount: DistributionConfig = DistributionConfig(mean=85.0, std=40.0)
    hour_of_day: DistributionConfig = DistributionConfig(mean=13.0, std=4.0)
    transactions_last_hour: DistributionConfig = DistributionConfig(mean=3.0, std=1.5)
    account_age_days: DistributionConfig = DistributionConfig(mean=400.0, std=120.0)
    distance_from_home: DistributionConfig = DistributionConfig(mean=15.0, std=10.0)

    def as_dict(self) -> dict[str, dict[str, float]]:
        dumped = self.model_dump()
        return {
            feature: {"mean": values["mean"], "std": values["std"]}
            for feature, values in dumped.items()
        }


class DriftedConfig(BaseModel):
    """
    Synthetic drifted distributions representing degraded traffic.
    """

    transaction_amount: DistributionConfig = DistributionConfig(mean=160.0, std=80.0)
    hour_of_day: DistributionConfig = DistributionConfig(mean=3.0, std=4.0)
    transactions_last_hour: DistributionConfig = DistributionConfig(mean=5.5, std=2.0)
    account_age_days: DistributionConfig = DistributionConfig(mean=150.0, std=60.0)
    distance_from_home: DistributionConfig = DistributionConfig(mean=35.0, std=20.0)

    def as_dict(self) -> dict[str, dict[str, float]]:
        dumped = self.model_dump()
        return {
            feature: {"mean": values["mean"], "std": values["std"]}
            for feature, values in dumped.items()
        }


# ────────────────────────────────────────────────────────────────
# TOP-LEVEL SETTINGS
# ────────────────────────────────────────────────────────────────

class AppSettings(BaseSettings):
    """
    Master application settings object.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore",
    )

    model: ModelConfig = Field(default_factory=ModelConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    detector: DetectorConfig = Field(default_factory=DetectorConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    state_store: StateStoreConfig = Field(default_factory=StateStoreConfig)
    impact: ImpactConfig = Field(default_factory=ImpactConfig)
    decision: DecisionConfig = Field(default_factory=DecisionConfig)
    baseline: BaselineConfig = Field(default_factory=BaselineConfig)
    drifted: DriftedConfig = Field(default_factory=DriftedConfig)
    api_key: str = Field(
        default="",
        description=(
            "Shared secret for write endpoints (POST /causal/events, POST /canary/deploy). "
            "Pass as X-API-Key header. Empty string disables auth (development default)."
        ),
    )

    @model_validator(mode="after")
    def validate_cross_config(self) -> "AppSettings":
        if self.model.batch_size > self.monitoring.queue_maxsize:
            logger.warning(
                "batch_size (%d) exceeds queue_maxsize (%d). This is allowed, but may increase queue pressure.",
                self.model.batch_size,
                self.monitoring.queue_maxsize,
            )

        if self.detector.psi_threshold < self.monitoring.drift_threshold:
            logger.warning(
                "detector.psi_threshold (%.2f) is lower than monitoring.drift_threshold (%.2f). "
                "This is allowed, but may create policy confusion.",
                self.detector.psi_threshold,
                self.monitoring.drift_threshold,
            )

        if self.data.max_features < 3:
            logger.warning(
                "data.max_features=%d is quite low and may weaken detector signal.",
                self.data.max_features,
            )

        return self


# ────────────────────────────────────────────────────────────────
# SINGLETON FACTORY
# ────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _cached_settings() -> AppSettings:
    cfg = AppSettings()
    logger.info(
        "AppSettings validated | batch_size=%d | drift_after=%d | data_mode=%s | "
        "api_port=%d | psi_threshold=%.2f | queue_maxsize=%d",
        cfg.model.batch_size,
        cfg.model.drift_after_batches,
        cfg.model.data_mode,
        cfg.api.port,
        cfg.detector.psi_threshold,
        cfg.monitoring.queue_maxsize,
    )
    return cfg


def get_settings(force_reload: bool = False) -> AppSettings:
    """
    Return validated singleton settings.

    Args:
        force_reload: Clear cache and rebuild settings. Useful in tests only.
    """
    if force_reload:
        _cached_settings.cache_clear()
        logger.warning("Settings cache cleared — reloading from environment")

    return _cached_settings()
