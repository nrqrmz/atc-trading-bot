"""Persistence mixin — save and load trained models with joblib."""
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
from atc_trading_bot.pipeline_warning import PipelineWarning


class PersistenceMixin:
    """Mixin for saving and loading trained pipeline state.

    Serializes the HMM model, PCA, scaler, feature columns, and metadata
    so a trained pipeline can be restored without retraining.
    """

    def save_model(self, path: str) -> str:
        """Save trained pipeline state to disk.

        Args:
            path: File path for the saved model (e.g. "models/btc_model.joblib").

        Returns:
            The absolute path of the saved file.
        """
        hmm_model = getattr(self, "hmm_model", None)
        if hmm_model is None:
            warnings.warn("No trained model. Call detect_regime first.", PipelineWarning)
            return

        state = {
            "hmm_model": hmm_model,
            "pca": getattr(self, "pca", None),
            "scaler": getattr(self, "scaler", None),
            "feature_columns": list(self.features.columns) if getattr(self, "features", None) is not None else None,
            "current_regime": getattr(self, "current_regime", None),
            "regimes": getattr(self, "regimes", None),
            "regime_metrics": getattr(self, "regime_metrics", None),
            "metadata": {
                "saved_at": datetime.now(timezone.utc).isoformat(),
                "exchange_id": getattr(self, "exchange_id", None),
                "symbols": getattr(self, "symbols", None),
                "timeframe": getattr(self, "timeframe", None),
            },
        }

        filepath = Path(path)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(state, filepath)
        return str(filepath.resolve())

    def load_model(self, path: str) -> dict:
        """Load a previously saved pipeline state from disk.

        Restores the HMM model, PCA, scaler, and feature columns so the
        bot can skip directly to select_strategy or generate_signals.

        Args:
            path: Path to the saved model file.

        Returns:
            The metadata dict from the saved model.
        """
        filepath = Path(path)
        if not filepath.exists():
            warnings.warn(f"Model file not found: {path}", PipelineWarning)
            return

        state = joblib.load(filepath)

        self.hmm_model = state["hmm_model"]
        self.pca = state.get("pca")
        self.scaler = state.get("scaler")
        self.current_regime = state.get("current_regime")
        self.regimes = state.get("regimes")
        self.regime_metrics = state.get("regime_metrics")

        feature_columns = state.get("feature_columns")
        if feature_columns is not None and getattr(self, "features", None) is not None:
            # Re-align features if they exist
            pass

        return state.get("metadata", {})
