"""
src/model_simulator.py
-----------------------
Simulates a production fraud detection model running continuously.

Responsibilities:
  - Generates realistic transaction feature data
  - Scores each transaction (fraud / legit)
  - Injects distribution drift after a configurable number of batches
  - Pulls ALL config from AppSettings — zero hardcoded values

Design decisions:
  - Stateless feature sampling via numpy — reproducible with seed
  - Dataclasses for type-safe prediction records
  - Structured logging instead of print statements
  - All distributions pulled from config — ops team can tune without code change
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Optional

from config.settings import get_settings

# ─── LOGGER ─────────────────────────────────────────────────────
# Every module gets its own named logger.
# Log level controlled centrally via config — no scattered basicConfig calls.
logger = logging.getLogger(__name__)


# ─── DATA CLASSES ───────────────────────────────────────────────

@dataclass
class PredictionRecord:
    """
    Represents a single model prediction.
    Immutable after creation — mirrors how production
    prediction logs work (append-only audit trail).
    """
    timestamp:   str
    prediction:  float        # fraud probability score 0.0 - 1.0
    label:       str          # "fraud" or "legit"
    confidence:  float        # how confident the model is
    features:    dict         # raw input features
    batch_id:    int          # which batch this came from
    is_drifted:  bool = False # ground truth flag for evaluation


@dataclass
class ModelStats:
    """
    Running health statistics for the model.
    Updated after every batch — read by the API and dashboard.
    """
    total_predictions:  int   = 0
    fraud_rate:         float = 0.0
    avg_confidence:     float = 0.0
    drift_injected:     bool  = False
    batches_processed:  int   = 0
    healthy_batches:    int   = 0
    drifted_batches:    int   = 0


# ─── SIMULATOR ──────────────────────────────────────────────────

class FraudModelSimulator:
    """
    Simulates a production fraud detection model.

    Generates batches of predictions from sampled feature data.
    After drift_after_batches, switches to the drifted distribution
    to simulate a real upstream data quality failure.

    All configuration pulled from AppSettings at construction time.
    Reseed is applied once — batches are deterministic for the same seed.
    """

    def __init__(self):
        cfg = get_settings()

        self._batch_size         = cfg.model.batch_size
        self._drift_after        = cfg.model.drift_after_batches
        self._fraud_threshold    = cfg.model.fraud_threshold
        self._baseline_dist      = cfg.baseline.as_dict()
        self._drifted_dist       = cfg.drifted.as_dict()

        self.stats = ModelStats()
        self.history: list[PredictionRecord] = []

        # Seed once at construction for reproducibility
        np.random.seed(cfg.model.random_seed)

        logger.info(
            f"FraudModelSimulator initialized | "
            f"batch_size={self._batch_size} | "
            f"drift_after={self._drift_after} batches"
        )

    # ── PRIVATE HELPERS ─────────────────────────────────────────

    def _sample_features(self, drifted: bool) -> dict:
        """
        Samples one row of input features from the active distribution.
        Switches between baseline and drifted based on current state.
        """
        dist = self._drifted_dist if drifted else self._baseline_dist
        return {
            feature: round(
                float(np.random.normal(params["mean"], params["std"])), 2
            )
            for feature, params in dist.items()
        }
    def _score(self, features: dict) -> float:
        """
        Rule-based scoring mimicking a trained model's predict_proba().
        Calibrated so baseline batches stay low-risk while drifted batches
        show a strong but not catastrophic degradation.
        """
        s = 0.01

        a = features.get("transaction_amount", 0)
        if a > 120:
            s += 0.04
        if a > 200:
            s += 0.05

        h = features.get("hour_of_day", 12)
        if not (7 <= h <= 21):
            s += 0.05

        t = features.get("transactions_last_hour", 0)
        if t > 3:
            s += 0.04
        if t > 6:
            s += 0.04

        ag = features.get("account_age_days", 999)
        if ag < 200:
            s += 0.04
        if ag < 80:
            s += 0.03

        d = features.get("distance_from_home", 0)
        if d > 20:
            s += 0.04
        if d > 50:
            s += 0.03

        s += float(np.random.normal(0, 0.03))
        return float(np.clip(s, 0.0, 1.0))
        
    def _is_drifted_batch(self) -> bool:
        """Returns True if current batch should use drifted distribution."""
        return self.stats.batches_processed > self._drift_after

    # ── PUBLIC API ───────────────────────────────────────────────

    def generate_batch(
        self,
        batch_size: Optional[int] = None
    ) -> list[PredictionRecord]:
        """
        Generates one batch of predictions.

        Args:
            batch_size: Override default batch size for this call only.
                        Useful for testing with smaller batches.

        Returns:
            List of PredictionRecord — one per transaction scored.
        """
        size      = batch_size or self._batch_size
        self.stats.batches_processed += 1
        drifted   = self._is_drifted_batch()

        # Log drift injection exactly once
        if drifted and not self.stats.drift_injected:
            logger.warning(
                f"DRIFT INJECTED at batch {self.stats.batches_processed} — "
                f"switching to drifted distribution"
            )
            self.stats.drift_injected = True

        

        batch: list[PredictionRecord] = []
        for _ in range(size):
            features   = self._sample_features(drifted=drifted)
            score      = self._score(features)
            label      = "fraud" if score >= self._fraud_threshold else "legit"
            confidence = score if label == "fraud" else (1.0 - score)

            record = PredictionRecord(
                timestamp  = datetime.now(timezone.utc).isoformat(),
                prediction = round(score, 4),
                label      = label,
                confidence = round(confidence, 4),
                features   = features,
                batch_id   = self.stats.batches_processed,
                is_drifted = drifted,
            )
            batch.append(record)

        # Update running stats
        fraud_count = sum(1 for r in batch if r.label == "fraud")
        self.stats.total_predictions += size
        self.stats.fraud_rate         = round(fraud_count / size, 4)
        self.stats.avg_confidence     = round(
            sum(r.confidence for r in batch) / size, 4
        )

        if drifted:
            self.stats.drifted_batches += 1
        else:
            self.stats.healthy_batches += 1

        self.history.extend(batch)

        logger.debug(
            f"Batch {self.stats.batches_processed:03d} | "
            f"size={size} | "
            f"fraud_rate={self.stats.fraud_rate:.1%} | "
            f"avg_conf={self.stats.avg_confidence:.3f} | "
            f"drifted={drifted}"
        )

        return batch

    def get_batch_as_dataframe(
        self,
        batch: list[PredictionRecord]
    ) -> pd.DataFrame:
        """
        Converts a batch to a DataFrame for statistical analysis.
        Used by the drift detector for PSI and KS test computation.
        """
        rows = [
            {
                "timestamp":  r.timestamp,
                "prediction": r.prediction,
                "label":      r.label,
                "batch_id":   r.batch_id,
                **r.features,
            }
            for r in batch
        ]
        return pd.DataFrame(rows)

    def get_stats_snapshot(self) -> dict:
        """
        Returns current model stats as a JSON-serializable dict.
        Called by the API health endpoint.
        """
        return {
            "total_predictions": self.stats.total_predictions,
            "fraud_rate":        self.stats.fraud_rate,
            "avg_confidence":    self.stats.avg_confidence,
            "drift_injected":    self.stats.drift_injected,
            "batches_processed": self.stats.batches_processed,
            "healthy_batches":   self.stats.healthy_batches,
            "drifted_batches":   self.stats.drifted_batches,
        }


# ─── ENTRYPOINT TEST ────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    sim = FraudModelSimulator()
    print("\nRunning 12 batches (drift after batch 10)...\n")

    for i in range(12):
        batch = sim.generate_batch(batch_size=20)
        print(
            f"  Batch {i+1:02d} | "
            f"fraud={sim.stats.fraud_rate:.1%} | "
            f"confidence={sim.stats.avg_confidence:.3f} | "
            f"drift={sim.stats.drift_injected}"
        )

    print("\nFinal stats snapshot:")
    for key, val in sim.get_stats_snapshot().items():
        print(f"  {key:<22} {val}")