import warnings

import pandas as pd
from atc_trading_bot.config import DEFAULT_N_COMPONENTS, DEFAULT_N_REGIMES
from atc_trading_bot.pipeline_warning import PipelineWarning


class ScannerMixin:
    """Mixin for quick multi-symbol regime scanning.

    Iterates over configured symbols, runs a lightweight pipeline
    (fetch_data -> compute_features -> detect_regime) for each, and
    returns a summary DataFrame showing the current regime, confidence,
    last price, and 24-hour percentage change.
    """

    def _require_symbols(self) -> bool:
        """Check that at least one symbol is configured."""
        symbols = getattr(self, "symbols", None)
        if not symbols:
            warnings.warn(
                "No symbols configured. Pass symbols to the constructor.",
                PipelineWarning,
            )
            return False
        return True

    def scan_regimes(
        self,
        n_components: int = DEFAULT_N_COMPONENTS,
        n_regimes: int = DEFAULT_N_REGIMES,
    ) -> pd.DataFrame | None:
        """Scan all configured symbols and return a regime summary.

        Runs a lightweight pipeline (fetch_data, compute_features,
        detect_regime) for each symbol in self.symbols. Errors on
        individual symbols are caught and skipped so a single failure
        does not abort the entire scan.

        Args:
            n_components: Number of PCA components for feature reduction.
                Default: 10.
            n_regimes: Number of HMM regimes. Default: 3.

        Returns:
            DataFrame with columns: symbol, regime, confidence,
            last_price, pct_change_24h. Returns None if no symbols
            are configured.
        """
        if not self._require_symbols():
            return None

        results = []

        for symbol in self.symbols:
            try:
                # Suppress verbose output during scanning
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")

                    fetch_data = getattr(self, "fetch_data", None)
                    compute_features = getattr(self, "compute_features", None)
                    detect_regime = getattr(self, "detect_regime", None)

                    if not all([fetch_data, compute_features, detect_regime]):
                        continue

                    fetch_data(symbol)
                    compute_features(n_components=n_components)
                    detect_regime(n_regimes=n_regimes)

                # Extract results from the pipeline
                regime = getattr(self, "current_regime", None)
                confidence = self._scan_confidence()
                last_price = self._scan_last_price()
                pct_change = self._scan_pct_change_24h()

                results.append({
                    "symbol": symbol,
                    "regime": regime,
                    "confidence": round(confidence, 4),
                    "last_price": round(last_price, 2),
                    "pct_change_24h": round(pct_change, 4),
                })

            except Exception:
                # Skip symbols that fail — don't abort the whole scan
                continue

        df = pd.DataFrame(
            results,
            columns=["symbol", "regime", "confidence", "last_price", "pct_change_24h"],
        )
        return df

    def _scan_confidence(self) -> float:
        """Compute regime confidence from HMM posteriors for the scan.

        Returns the posterior probability of the current regime for the
        last observation. Falls back to 0.0 if the model is unavailable.
        """
        hmm_model = getattr(self, "hmm_model", None)
        features_pca = getattr(self, "features_pca", None)

        if hmm_model is None or features_pca is None:
            return 0.0

        try:
            posteriors = hmm_model.predict_proba(features_pca)
            last_posteriors = posteriors[-1]
            predicted_state = hmm_model.predict(features_pca[-1:])[-1]
            return float(last_posteriors[predicted_state])
        except Exception:
            return 0.0

    def _scan_last_price(self) -> float:
        """Return the last closing price from the current data."""
        df = getattr(self, "df", None)
        if df is None or df.empty:
            return 0.0
        return float(df["Close"].iloc[-1])

    def _scan_pct_change_24h(self) -> float:
        """Compute the 24-hour percentage change from the current data.

        Uses the last two rows of the DataFrame. If fewer than two rows
        are available, returns 0.0.
        """
        df = getattr(self, "df", None)
        if df is None or len(df) < 2:
            return 0.0
        prev_close = float(df["Close"].iloc[-2])
        last_close = float(df["Close"].iloc[-1])
        if prev_close == 0:
            return 0.0
        return (last_close - prev_close) / prev_close
