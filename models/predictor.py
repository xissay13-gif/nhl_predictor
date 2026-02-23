"""
Main ML Prediction Model.

Ensemble of XGBoost and LightGBM for:
  1. Win probability (classification)
  2. Goal totals (regression)
  3. Spread/puck line (classification)

Includes training, evaluation, and prediction pipelines.
"""

import os
import logging
import json
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, log_loss, brier_score_loss, mean_absolute_error,
)

try:
    import xgboost as xgb
except ImportError:
    xgb = None

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

import joblib

from config import cfg

logger = logging.getLogger("nhl_predictor.models.predictor")


# ── Feature columns to exclude from model input ─────────────────────
NON_FEATURE_COLS = {"home_team", "away_team", "date", "season_phase",
                    "market_favorite", "streak_type", "home_streak_type",
                    "away_streak_type"}


def _prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Select numeric features, fill NAs, return clean DataFrame + feature names."""
    numeric_cols = []
    for col in df.columns:
        if col in NON_FEATURE_COLS:
            continue
        if df[col].dtype in ("float64", "float32", "int64", "int32", "bool"):
            numeric_cols.append(col)

    X = df[numeric_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)

    # Convert booleans to int
    for col in X.columns:
        if X[col].dtype == "bool":
            X[col] = X[col].astype(int)

    return X, numeric_cols


class NHLPredictor:
    """
    Ensemble prediction model for NHL games.

    Three sub-models:
      - win_model: P(home_win)  — binary classification
      - total_model: predicted total goals — regression
      - spread_model: P(home covers -1.5) — binary classification
    """

    def __init__(self):
        self.win_model = None
        self.total_model = None
        self.spread_model = None
        self.scaler = StandardScaler()
        self.feature_names: list[str] = []
        self.is_trained = False

        # Feature importance
        self.feature_importance: Optional[pd.DataFrame] = None

    def train(
        self,
        features_df: pd.DataFrame,
        results_df: pd.DataFrame,
        calibrate: bool = True,
    ):
        """
        Train all sub-models.

        features_df: output of FeatureEngineer.build_features() for each historical game.
        results_df: must have columns:
            - home_win (bool): did home team win?
            - total_goals (int): total goals in the game
            - home_covered (bool): did home cover -1.5 spread?
        """
        X, self.feature_names = _prepare_features(features_df)
        X_scaled = self.scaler.fit_transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=self.feature_names)

        logger.info("Training on %d games with %d features", len(X), len(self.feature_names))

        # ── Win model ────────────────────────────────────────────────
        y_win = results_df["home_win"].astype(int).values
        self.win_model = self._build_classifier("win")
        self.win_model.fit(X_scaled_df, y_win)

        if calibrate:
            self.win_model = CalibratedClassifierCV(
                self.win_model, cv=3, method="isotonic"
            )
            self.win_model.fit(X_scaled_df, y_win)

        # ── Total model ─────────────────────────────────────────────
        y_total = results_df["total_goals"].values
        self.total_model = self._build_regressor("total")
        self.total_model.fit(X_scaled_df, y_total)

        # ── Spread model ─────────────────────────────────────────────
        y_spread = results_df["home_covered"].astype(int).values
        self.spread_model = self._build_classifier("spread")
        self.spread_model.fit(X_scaled_df, y_spread)

        if calibrate:
            self.spread_model = CalibratedClassifierCV(
                self.spread_model, cv=3, method="isotonic"
            )
            self.spread_model.fit(X_scaled_df, y_spread)

        self.is_trained = True

        # Feature importance (from the base estimator)
        self._compute_feature_importance()

        logger.info("Training complete")

    def predict(self, features: dict) -> dict:
        """
        Predict outcomes for a single game.

        features: output of FeatureEngineer.build_features()
        Returns dict with all predictions.
        """
        if not self.is_trained:
            logger.warning("Model not trained — returning defaults")
            return self._default_prediction()

        X = pd.DataFrame([features])
        X, _ = _prepare_features(X)

        # Ensure columns match training
        for col in self.feature_names:
            if col not in X.columns:
                X[col] = 0
        X = X[self.feature_names]

        X_scaled = self.scaler.transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=self.feature_names)

        # Win probability
        win_proba = self.win_model.predict_proba(X_scaled_df)[0]
        home_win_prob = win_proba[1] if len(win_proba) > 1 else win_proba[0]

        # Total goals
        total_pred = self.total_model.predict(X_scaled_df)[0]

        # Spread probability
        spread_proba = self.spread_model.predict_proba(X_scaled_df)[0]
        home_cover_prob = spread_proba[1] if len(spread_proba) > 1 else spread_proba[0]

        return {
            "ml_home_win_prob": float(home_win_prob),
            "ml_away_win_prob": float(1 - home_win_prob),
            "ml_total_pred": float(total_pred),
            "ml_home_cover_prob": float(home_cover_prob),
            "ml_away_cover_prob": float(1 - home_cover_prob),
        }

    def predict_batch(self, features_list: list[dict]) -> pd.DataFrame:
        """Predict for multiple games at once."""
        results = []
        for f in features_list:
            pred = self.predict(f)
            pred["home_team"] = f.get("home_team", "")
            pred["away_team"] = f.get("away_team", "")
            results.append(pred)
        return pd.DataFrame(results)

    def evaluate(
        self,
        features_df: pd.DataFrame,
        results_df: pd.DataFrame,
    ) -> dict:
        """Cross-validated evaluation of model performance."""
        X, _ = _prepare_features(features_df)
        for col in self.feature_names:
            if col not in X.columns:
                X[col] = 0
        X = X[self.feature_names]
        X_scaled = self.scaler.transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=self.feature_names)

        y_win = results_df["home_win"].astype(int).values
        y_total = results_df["total_goals"].values

        # Win model evaluation
        win_proba = self.win_model.predict_proba(X_scaled_df)[:, 1]
        win_pred = (win_proba > 0.5).astype(int)

        metrics = {
            "win_accuracy": accuracy_score(y_win, win_pred),
            "win_log_loss": log_loss(y_win, win_proba),
            "win_brier": brier_score_loss(y_win, win_proba),
            "total_mae": mean_absolute_error(y_total, self.total_model.predict(X_scaled_df)),
        }

        logger.info("Evaluation: accuracy=%.3f, log_loss=%.3f, brier=%.3f, total_MAE=%.2f",
                     metrics["win_accuracy"], metrics["win_log_loss"],
                     metrics["win_brier"], metrics["total_mae"])

        return metrics

    def cross_validate(
        self,
        features_df: pd.DataFrame,
        results_df: pd.DataFrame,
        n_splits: int = 5,
    ) -> dict:
        """Time-series cross-validation."""
        X, feature_names = _prepare_features(features_df)
        X_scaled = self.scaler.fit_transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=feature_names)

        y_win = results_df["home_win"].astype(int).values

        tscv = TimeSeriesSplit(n_splits=n_splits)
        model = self._build_classifier("cv")

        accuracies = []
        log_losses_list = []

        for train_idx, val_idx in tscv.split(X_scaled_df):
            X_train = X_scaled_df.iloc[train_idx]
            X_val = X_scaled_df.iloc[val_idx]
            y_train = y_win[train_idx]
            y_val = y_win[val_idx]

            model.fit(X_train, y_train)
            proba = model.predict_proba(X_val)[:, 1]
            pred = (proba > 0.5).astype(int)

            accuracies.append(accuracy_score(y_val, pred))
            log_losses_list.append(log_loss(y_val, proba))

        return {
            "cv_accuracy_mean": np.mean(accuracies),
            "cv_accuracy_std": np.std(accuracies),
            "cv_log_loss_mean": np.mean(log_losses_list),
            "cv_log_loss_std": np.std(log_losses_list),
        }

    def save(self, path: Optional[str] = None):
        """Save trained model to disk."""
        path = path or cfg.models_dir
        os.makedirs(path, exist_ok=True)

        joblib.dump(self.win_model, os.path.join(path, "win_model.pkl"))
        joblib.dump(self.total_model, os.path.join(path, "total_model.pkl"))
        joblib.dump(self.spread_model, os.path.join(path, "spread_model.pkl"))
        joblib.dump(self.scaler, os.path.join(path, "scaler.pkl"))

        with open(os.path.join(path, "feature_names.json"), "w") as f:
            json.dump(self.feature_names, f)

        logger.info("Model saved to %s", path)

    def load(self, path: Optional[str] = None):
        """Load trained model from disk."""
        path = path or cfg.models_dir

        self.win_model = joblib.load(os.path.join(path, "win_model.pkl"))
        self.total_model = joblib.load(os.path.join(path, "total_model.pkl"))
        self.spread_model = joblib.load(os.path.join(path, "spread_model.pkl"))
        self.scaler = joblib.load(os.path.join(path, "scaler.pkl"))

        with open(os.path.join(path, "feature_names.json"), "r") as f:
            self.feature_names = json.load(f)

        self.is_trained = True
        logger.info("Model loaded from %s (%d features)", path, len(self.feature_names))

    def _build_classifier(self, name: str):
        """Build the best available classifier."""
        if xgb is not None:
            return xgb.XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=3,
                reg_alpha=0.1,
                reg_lambda=1.0,
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=42,
                verbosity=0,
            )
        elif lgb is not None:
            return lgb.LGBMClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=3,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                verbose=-1,
            )
        else:
            from sklearn.ensemble import GradientBoostingClassifier
            logger.warning("Neither XGBoost nor LightGBM available — using sklearn GBM")
            return GradientBoostingClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.05,
                subsample=0.8, random_state=42,
            )

    def _build_regressor(self, name: str):
        """Build the best available regressor."""
        if xgb is not None:
            return xgb.XGBRegressor(
                n_estimators=250,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=5,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                verbosity=0,
            )
        elif lgb is not None:
            return lgb.LGBMRegressor(
                n_estimators=250,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbose=-1,
            )
        else:
            from sklearn.ensemble import GradientBoostingRegressor
            return GradientBoostingRegressor(
                n_estimators=200, max_depth=5, learning_rate=0.05,
                subsample=0.8, random_state=42,
            )

    def _compute_feature_importance(self):
        """Extract feature importance from the win model."""
        try:
            estimator = self.win_model
            # If calibrated, get the base estimator
            if hasattr(estimator, "estimator"):
                estimator = estimator.estimator
            if hasattr(estimator, "calibrated_classifiers_"):
                estimator = estimator.calibrated_classifiers_[0].estimator

            if hasattr(estimator, "feature_importances_"):
                imp = estimator.feature_importances_
                self.feature_importance = pd.DataFrame({
                    "feature": self.feature_names,
                    "importance": imp,
                }).sort_values("importance", ascending=False)
        except Exception as exc:
            logger.warning("Could not extract feature importance: %s", exc)

    def _default_prediction(self) -> dict:
        return {
            "ml_home_win_prob": 0.50,  # neutral when untrained — Elo/Poisson add home edge
            "ml_away_win_prob": 0.50,
            "ml_total_pred": 6.0,
            "ml_home_cover_prob": 0.47,
            "ml_away_cover_prob": 0.53,
        }
