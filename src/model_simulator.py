"""
src/model_simulator.py
-----------------------
Production-grade fraud / drift simulator with unified support for:

    - synthetic mode: config-driven distributions
    - real mode: CSV-backed replay with realistic drift injection

Public API:
    FraudModelSimulator
    PredictionRecord
    ModelStats
    generate_batch()
    get_stats_snapshot()
    get_baseline_stats()

Design goals:
    - Keep downstream contract identical across modes
    - Simulate realistic degradation, not cartoonish failure
    - Preserve enough signal for drift detector + dashboard
    - Remain easy to extend for new datasets later
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from config.settings import get_settings

logger = logging.getLogger(__name__)

__all__ = [
    "FraudModelSimulator",
    "PredictionRecord",
    "ModelStats",
]

DEFAULT_REAL_DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "data",
    "creditcard.csv",
)

MAX_INFERRED_FEATURES = 10


# ────────────────────────────────────────────────────────────────
# DATA MODELS
# ────────────────────────────────────────────────────────────────

@dataclass(slots=True)
class PredictionRecord:
    timestamp: str
    prediction: float
    label: str
    confidence: float
    features: dict[str, float]
    batch_id: int
    is_drifted: bool
    true_label: int | None = None


@dataclass(slots=True)
class ModelStats:
    total_predictions: int = 0
    fraud_rate: float = 0.0
    avg_confidence: float = 0.0
    drift_injected: bool = False
    batches_processed: int = 0
    healthy_batches: int = 0
    drifted_batches: int = 0
    # Running confusion-matrix counts (only populated in real mode where true_label exists)
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    true_negatives: int = 0


# ────────────────────────────────────────────────────────────────
# SIMULATOR
# ────────────────────────────────────────────────────────────────

class FraudModelSimulator:
    """
    Unified simulator supporting synthetic and real CSV modes.

    Synthetic mode:
        - samples from configured baseline distributions
        - injects drifted distributions after drift_after_batches

    Real mode:
        - loads CSV dataset
        - infers numeric feature columns
        - builds healthy baseline from majority class when binary label exists
        - simulates gradual fraud-rate increase + feature shift + confidence compression
    """

    def __init__(self) -> None:
        cfg = get_settings()

        self._cfg = cfg
        self._batch_size = cfg.model.batch_size
        self._drift_after = cfg.model.drift_after_batches
        self._fraud_threshold = cfg.model.fraud_threshold
        self._data_mode = cfg.model.data_mode

        self.stats = ModelStats()
        np.random.seed(cfg.model.random_seed)

        # Synthetic mode state
        self._baseline_dist: dict[str, dict[str, float]] = {}
        self._drifted_dist: dict[str, dict[str, float]] = {}

        # Real mode state
        self._data_path: str = self._resolve_data_path()
        self._label_col: str | None = self._resolve_configured_label_column()
        self._feature_cols: list[str] = []
        self._healthy_df: pd.DataFrame | None = None
        self._drifted_df: pd.DataFrame | None = None
        self._baseline_stats: dict[str, dict[str, float]] = {}

        # Selective drift: only these features drift significantly during an incident.
        # Real-world drift is sparse — 2-4 upstream features shift while others remain
        # stable. This is chosen at init so the same features drift consistently.
        # Populated after mode init (needs feature list to be resolved first).
        self._primary_drift_features: list[str] = []
        self._secondary_drift_features: list[str] = []

        if self._data_mode == "real":
            self._init_real_mode()
        else:
            self._init_synthetic_mode()

        self._init_drift_feature_selection()

        logger.info(
            "FraudModelSimulator initialized | mode=%s | batch_size=%d | drift_after=%d | features=%s",
            self._data_mode,
            self._batch_size,
            self._drift_after,
            self._feature_cols if self._data_mode == "real" else list(self._baseline_dist.keys()),
        )

    # ── PUBLIC API ─────────────────────────────────────────────

    def generate_batch(self, batch_size: int | None = None) -> list[PredictionRecord]:
        size = batch_size or self._batch_size
        self.stats.batches_processed += 1
        drifted = self.stats.batches_processed > self._drift_after

        if drifted and not self.stats.drift_injected:
            logger.warning(
                "DRIFT INJECTED at batch %d | mode=%s",
                self.stats.batches_processed,
                self._data_mode,
            )
            self.stats.drift_injected = True

        if self._data_mode == "real":
            batch = self._generate_real_batch(size=size, drifted=drifted)
        else:
            batch = self._generate_synthetic_batch(size=size, drifted=drifted)

        fraud_count = sum(1 for record in batch if record.label == "fraud")
        self.stats.total_predictions += size
        self.stats.fraud_rate = round(fraud_count / size, 4)
        self.stats.avg_confidence = round(
            sum(record.confidence for record in batch) / size,
            4,
        )

        # Update running confusion matrix from true_label (real mode only)
        self._update_confusion_matrix(batch)

        if drifted:
            self.stats.drifted_batches += 1
        else:
            self.stats.healthy_batches += 1

        logger.debug(
            "Batch %03d | size=%d | fraud=%.1f%% | conf=%.3f | drifted=%s | mode=%s",
            self.stats.batches_processed,
            size,
            self.stats.fraud_rate * 100,
            self.stats.avg_confidence,
            drifted,
            self._data_mode,
        )
        return batch

    def get_stats_snapshot(self) -> dict[str, Any]:
        return {
            "total_predictions": self.stats.total_predictions,
            "fraud_rate": self.stats.fraud_rate,
            "avg_confidence": self.stats.avg_confidence,
            "drift_injected": self.stats.drift_injected,
            "batches_processed": self.stats.batches_processed,
            "healthy_batches": self.stats.healthy_batches,
            "drifted_batches": self.stats.drifted_batches,
            "mode": self._data_mode,
        }

    def get_performance_metrics(self) -> dict[str, float | None]:
        """
        Return live precision, recall, F1, and AUC-ROC proxy from the running
        confusion matrix.  Only meaningful in real mode where true_label is set;
        returns None for all metrics in synthetic mode.
        """
        tp = self.stats.true_positives
        fp = self.stats.false_positives
        fn = self.stats.false_negatives
        tn = self.stats.true_negatives
        total_labeled = tp + fp + fn + tn

        if total_labeled == 0 or self._data_mode != "real":
            return {"precision": None, "recall": None, "f1_score": None, "auc_roc_proxy": None}

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        # AUC-ROC proxy: balanced accuracy = (TPR + TNR) / 2
        tpr = recall
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        auc_proxy = (tpr + tnr) / 2.0

        return {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "auc_roc_proxy": round(auc_proxy, 4),
        }

    def get_baseline_stats(self) -> dict[str, dict[str, float]]:
        return self._baseline_stats if self._data_mode == "real" else self._baseline_dist

    def get_batch_as_dataframe(self, batch: list[PredictionRecord]) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "timestamp": record.timestamp,
                    "prediction": record.prediction,
                    "label": record.label,
                    "batch_id": record.batch_id,
                    "true_label": record.true_label,
                    **record.features,
                }
                for record in batch
            ]
        )

    def _init_drift_feature_selection(self) -> None:
        """
        Pick which features drift during an incident — once, at startup.

        Real drift is sparse: 2-4 upstream features shift significantly while
        the rest stay near baseline. Seeding from the random state keeps the
        selection deterministic across runs for a given random_seed.
        """
        all_features = (
            self._feature_cols
            if self._data_mode == "real"
            else list(self._baseline_dist.keys())
        )
        if not all_features:
            return

        rng = np.random.default_rng(self._cfg.model.random_seed + 42)
        n_primary = min(3, len(all_features))
        n_secondary = min(2, len(all_features) - n_primary)

        shuffled = rng.permutation(all_features).tolist()
        self._primary_drift_features = shuffled[:n_primary]
        self._secondary_drift_features = shuffled[n_primary: n_primary + n_secondary]

        logger.info(
            "Drift feature selection | primary=%s | secondary=%s | stable=%d features",
            self._primary_drift_features,
            self._secondary_drift_features,
            len(all_features) - n_primary - n_secondary,
        )

    # ── SYNTHETIC MODE ─────────────────────────────────────────

    def _init_synthetic_mode(self) -> None:
        self._baseline_dist = self._cfg.baseline.as_dict()
        self._drifted_dist = self._cfg.drifted.as_dict()

    def _generate_synthetic_batch(self, size: int, drifted: bool) -> list[PredictionRecord]:
        batch: list[PredictionRecord] = []

        for _ in range(size):
            features = self._sample_synthetic_features(drifted)
            prediction = self._score_synthetic(features)
            label = "fraud" if prediction >= self._fraud_threshold else "legit"
            confidence = self._calibrated_confidence(prediction, label)

            batch.append(
                PredictionRecord(
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    prediction=round(prediction, 4),
                    label=label,
                    confidence=round(confidence, 4),
                    features=features,
                    batch_id=self.stats.batches_processed,
                    is_drifted=drifted,
                    true_label=None,
                )
            )

        return batch

    def _sample_synthetic_features(self, drifted: bool) -> dict[str, float]:
        features: dict[str, float] = {}

        for feature, base_params in self._baseline_dist.items():
            if not drifted:
                params = base_params
            elif feature in self._primary_drift_features:
                # Primary features use the full drifted distribution
                params = self._drifted_dist.get(feature, base_params)
            elif feature in self._secondary_drift_features:
                # Secondary features get a mild blend: 80% baseline + 20% drift direction
                drift_params = self._drifted_dist.get(feature, base_params)
                blended_mean = base_params["mean"] + 0.2 * (drift_params["mean"] - base_params["mean"])
                params = {"mean": blended_mean, "std": base_params["std"]}
            else:
                # Stable features sample from baseline — no drift
                params = base_params

            value = float(np.random.normal(params["mean"], params["std"]))
            if feature == "hour_of_day":
                value = float(np.clip(round(value), 0, 23))
            else:
                value = float(max(0.0, value))
            features[feature] = round(value, 4)

        return features

    def _score_synthetic(self, features: dict[str, float]) -> float:
        score = 0.02

        amount = features.get("transaction_amount", 0.0)
        if amount > 120:
            score += 0.03
        if amount > 200:
            score += 0.03

        hour = features.get("hour_of_day", 12.0)
        if not (7 <= hour <= 21):
            score += 0.04

        txn_rate = features.get("transactions_last_hour", 0.0)
        if txn_rate > 3:
            score += 0.03
        if txn_rate > 6:
            score += 0.03

        account_age = features.get("account_age_days", 999.0)
        if account_age < 200:
            score += 0.02
        if account_age < 80:
            score += 0.02

        distance = features.get("distance_from_home", 0.0)
        if distance > 20:
            score += 0.02
        if distance > 50:
            score += 0.02

        score += float(np.random.normal(0.0, 0.01))
        return float(np.clip(score, 0.0, 1.0))

    # ── REAL MODE ──────────────────────────────────────────────

    def _init_real_mode(self) -> None:
        dataframe = self._load_real_data()

        self._feature_cols = self._infer_feature_columns(dataframe)
        self._label_col = self._resolve_label_column(dataframe)

        logger.info(
            "Real mode schema | label_col=%s | feature_cols=%s",
            self._label_col,
            self._feature_cols,
        )

        if self._label_col and self._is_binary_label(dataframe[self._label_col]):
            self._healthy_df, self._drifted_df = self._split_binary_classes(dataframe)
        else:
            self._healthy_df = dataframe.reset_index(drop=True)
            self._drifted_df = dataframe.reset_index(drop=True)
            logger.warning(
                "No usable binary label found — falling back to feature-distribution shift only"
            )

        assert self._healthy_df is not None
        self._baseline_stats = {
            column: {
                "mean": float(self._healthy_df[column].mean()),
                "std": float(self._healthy_df[column].std() or 1e-6),
            }
            for column in self._feature_cols
        }

        logger.info(
            "Real baseline built | healthy_rows=%d | drift_rows=%d | features=%d",
            len(self._healthy_df),
            len(self._drifted_df) if self._drifted_df is not None else 0,
            len(self._feature_cols),
        )

    def _generate_real_batch(self, size: int, drifted: bool) -> list[PredictionRecord]:
        rows = self._sample_real_rows(size=size, drifted=drifted)
        batch: list[PredictionRecord] = []

        for _, row in rows.iterrows():
            features = {
                column: round(float(row[column]), 4)
                for column in self._feature_cols
            }

            true_label: int | None = None
            if self._label_col and self._label_col in row.index:
                try:
                    true_label = int(row[self._label_col])
                except Exception:
                    true_label = None

            if drifted:
                features = self._inject_feature_shift(features)

            prediction = self._score_real(features, true_label=true_label, drifted=drifted)
            label = "fraud" if prediction >= self._fraud_threshold else "legit"
            confidence = self._calibrated_confidence(prediction, label)

            batch.append(
                PredictionRecord(
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    prediction=round(prediction, 4),
                    label=label,
                    confidence=round(confidence, 4),
                    features=features,
                    batch_id=self.stats.batches_processed,
                    is_drifted=drifted,
                    true_label=true_label,
                )
            )

        return batch

    def _load_real_data(self) -> pd.DataFrame:
        if not os.path.exists(self._data_path):
            logger.error("Dataset not found at %s", self._data_path)
            raise FileNotFoundError(f"Dataset not found at {self._data_path}")

        logger.info("Loading dataset from %s", self._data_path)
        dataframe = pd.read_csv(self._data_path)
        dataframe.columns = [column.strip() for column in dataframe.columns]

        logger.info(
            "Dataset loaded | rows=%d | cols=%d",
            len(dataframe),
            len(dataframe.columns),
        )
        return dataframe

    def _resolve_data_path(self) -> str:
        possible = getattr(self._cfg, "data", None)
        if possible is not None and hasattr(possible, "path") and possible.path:
            return str(possible.path)
        return DEFAULT_REAL_DATA_PATH

    def _resolve_configured_label_column(self) -> str | None:
        possible = getattr(self._cfg, "data", None)
        if possible is not None and hasattr(possible, "label_column"):
            return possible.label_column
        return "Class"

    def _resolve_label_column(self, dataframe: pd.DataFrame) -> str | None:
        if self._label_col and self._label_col in dataframe.columns:
            return self._label_col

        candidates = ["Class", "class", "label", "target", "fraud", "y"]
        for candidate in candidates:
            if candidate in dataframe.columns:
                return candidate
        return None

    def _infer_feature_columns(self, dataframe: pd.DataFrame) -> list[str]:
        """
        Infer numeric feature columns from a generic tabular CSV.
        """
        possible = getattr(self._cfg, "data", None)
        configured_columns: list[str] = []
        max_features = MAX_INFERRED_FEATURES

        if possible is not None:
            if hasattr(possible, "feature_columns") and possible.feature_columns:
                configured_columns = list(possible.feature_columns)
            if hasattr(possible, "max_features") and possible.max_features:
                max_features = int(possible.max_features)

        if configured_columns:
            missing = [column for column in configured_columns if column not in dataframe.columns]
            if missing:
                raise ValueError(f"Configured feature columns not found: {missing}")
            return configured_columns

        exclude_always = {
            "time",
            "timestamp",
            "id",
            "index",
            "row_id",
            "record_id",
            "sequence",
            "step",
        }

        label_col = self._resolve_label_column(dataframe)
        numeric_columns = dataframe.select_dtypes(include=[np.number]).columns.tolist()

        feature_columns = [
            column
            for column in numeric_columns
            if column != label_col
            and dataframe[column].nunique(dropna=True) > 1
            and column.lower() not in exclude_always
            and not column.lower().endswith("_id")
        ][:max_features]

        if not feature_columns:
            raise ValueError("No usable numeric feature columns found in dataset")

        logger.info(
            "Auto-inferred feature columns | count=%d | max_features=%d | features=%s",
            len(feature_columns),
            max_features,
            feature_columns,
        )
        return feature_columns

    @staticmethod
    def _is_binary_label(series: pd.Series) -> bool:
        unique_values = set(series.dropna().unique().tolist())
        return unique_values.issubset({0, 1}) and len(unique_values) >= 1

    def _split_binary_classes(
        self,
        dataframe: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        assert self._label_col is not None

        label_counts = dataframe[self._label_col].value_counts()
        majority_label = int(label_counts.idxmax())
        minority_label = int(label_counts.idxmin())

        healthy = dataframe[dataframe[self._label_col] == majority_label].reset_index(drop=True)
        drifted = dataframe[dataframe[self._label_col] == minority_label].reset_index(drop=True)

        logger.info(
            "Binary split | healthy_label=%d (%d rows) | drift_label=%d (%d rows)",
            majority_label,
            len(healthy),
            minority_label,
            len(drifted),
        )
        return healthy, drifted

    def _sample_real_rows(self, size: int, drifted: bool) -> pd.DataFrame:
        """
        Sampling strategy:
            healthy -> ~2% fraud
            drifted -> gradually rises toward ~22%
        """
        assert self._healthy_df is not None
        assert self._drifted_df is not None

        if self._healthy_df is self._drifted_df:
            return self._healthy_df.sample(
                n=size,
                replace=len(self._healthy_df) < size,
            ).reset_index(drop=True)

        if not drifted:
            fraud_fraction = 0.02
        else:
            batches_into_drift = max(0, self.stats.batches_processed - self._drift_after)
            target = min(0.22, 0.02 + batches_into_drift * 0.01)
            noise = np.random.normal(0, 0.01)
            fraud_fraction = float(np.clip(target + noise, 0.02, 0.25))

        n_fraud = max(1, int(size * fraud_fraction))
        n_legit = max(0, size - n_fraud)

        legit_rows = self._healthy_df.sample(
            n=n_legit,
            replace=len(self._healthy_df) < n_legit,
        )
        fraud_rows = self._drifted_df.sample(
            n=n_fraud,
            replace=len(self._drifted_df) < n_fraud,
        )

        return pd.concat([legit_rows, fraud_rows], axis=0).sample(frac=1.0).reset_index(drop=True)

    def _update_confusion_matrix(self, batch: list[PredictionRecord]) -> None:
        """Update running TP/FP/FN/TN counts from a batch (real mode only)."""
        for record in batch:
            if record.true_label is not None:
                predicted_fraud = record.label == "fraud"
                actual_fraud = record.true_label == 1
                if predicted_fraud and actual_fraud:
                    self.stats.true_positives += 1
                elif predicted_fraud and not actual_fraud:
                    self.stats.false_positives += 1
                elif not predicted_fraud and actual_fraud:
                    self.stats.false_negatives += 1
                else:
                    self.stats.true_negatives += 1

    def _calibrated_confidence(self, prediction: float, label: str) -> float:
        """
        Map a raw fraud-probability score to a semantically meaningful confidence.

        The raw score (e.g. 0.20) used directly as "confidence" is misleading:
        a prediction barely above the threshold (0.15) gets confidence 0.20, which
        implies the model is only 20% sure — but it's actually AT the boundary of
        its decision and is as uncertain as it can be.

        Instead we normalise the distance from the decision boundary so that:
            - prediction exactly AT the threshold  → confidence 0.50 (maximally uncertain)
            - prediction = 0.0 (strong legit)      → confidence 1.00
            - prediction = 1.0 (strong fraud)      → confidence 1.00

        Formula: conf = 0.5 + 0.5 * |prediction - threshold| / max(threshold, 1-threshold)
        """
        t = self._fraud_threshold
        max_dist = max(t, 1.0 - t)
        dist = abs(prediction - t)
        return round(min(1.0, 0.5 + 0.5 * (dist / max_dist)), 4)

    def _inject_feature_shift(self, features: dict[str, float]) -> dict[str, float]:
        """
        Apply selective feature drift.

        Primary features:   amplified shift (2× baseline std above mean) — high PSI
        Secondary features: mild perturbation (0.4× std) — moderate PSI
        Stable features:    snapped back to baseline mean + small noise — PSI ~0
        """
        shifted = dict(features)
        for feature, value in shifted.items():
            baseline = self._baseline_stats.get(feature, {})
            b_mean = baseline.get("mean", value)
            b_std = baseline.get("std", 1.0)

            if feature in self._primary_drift_features:
                # Strong shift: push 2 std deviations above baseline mean
                shifted[feature] = round(b_mean + 2.0 * b_std + float(np.random.normal(0, 0.1 * b_std)), 4)
            elif feature in self._secondary_drift_features:
                # Mild shift: 0.4 std nudge for a moderate but non-alarming PSI
                shifted[feature] = round(value + 0.4 * b_std + float(np.random.normal(0, 0.05 * b_std)), 4)
            else:
                # Stable: resample near baseline to cancel the class-distribution shift
                shifted[feature] = round(float(np.random.normal(b_mean, b_std * 0.5)), 4)

        return shifted

    def _score_real(
        self,
        features: dict[str, float],
        *,
        true_label: int | None,
        drifted: bool,
    ) -> float:
        """
        Simulate real-model predictions with drift-induced calibration loss.
        """
        if true_label is not None:
            if true_label == 1:
                if drifted:
                    base = np.random.uniform(0.18, 0.34)
                else:
                    base = np.random.uniform(0.24, 0.40)
            else:
                if drifted:
                    base = np.random.uniform(0.03, 0.14)
                else:
                    base = np.random.uniform(0.01, 0.10)

            score = base + float(np.random.normal(0, 0.01))
            return float(np.clip(score, 0.0, 1.0))

        z_scores = [
            abs(
                (float(value) - self._baseline_stats[column]["mean"])
                / (self._baseline_stats[column]["std"] + 1e-6)
            )
            for column, value in features.items()
            if column in self._baseline_stats
        ]

        if not z_scores:
            return float(np.random.uniform(0.02, 0.10))

        avg_z = float(np.mean(z_scores))
        score = (1.0 / (1.0 + np.exp(-(avg_z - 3.0)))) * 0.20

        if drifted:
            score += 0.02

        score += float(np.random.normal(0, 0.01))
        return float(np.clip(score, 0.0, 1.0))


# ────────────────────────────────────────────────────────────────
# ENTRYPOINT TEST
# ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    simulator = FraudModelSimulator()

    print(f"\nMode     : {simulator._data_mode}")
    print(
        f"Features : {simulator._feature_cols if simulator._data_mode == 'real' else list(simulator._baseline_dist.keys())}"
    )
    print(f"\nRunning 14 batches (drift after batch {simulator._drift_after})...\n")

    for index in range(14):
        simulator.generate_batch(batch_size=50)
        print(
            f"  Batch {index + 1:02d} | "
            f"fraud={simulator.stats.fraud_rate:.1%} | "
            f"conf={simulator.stats.avg_confidence:.3f} | "
            f"drift={simulator.stats.drift_injected}"
        )

    print("\nFinal stats:")
    for key, value in simulator.get_stats_snapshot().items():
        print(f"  {key:<22} {value}")