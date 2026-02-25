"""
Main ML Prediction Model.

Ensemble of XGBoost and LightGBM for:
  1. Win probability (classification)
  2. Goal totals (regression)
  3. Spread/puck line (classification)

Includes SHAP feature selection, stacking meta-model, and Optuna param loading.
"""

import os
import logging
import json
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
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

try:
    import shap
except ImportError:
    shap = None

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

    Plus:
      - SHAP-based feature selection (top_n features)
      - Stacking meta-model (learned blend of ML + Poisson + Elo)
      - Optuna tuned params (loaded from best_params.json if available)
    """

    def __init__(self):
        self.win_model = None
        self.total_model = None
        self.spread_model = None
        self.scaler = StandardScaler()
        self.feature_names: list[str] = []
        self.selected_features: list[str] | None = None
        self.is_trained = False

        # Feature importance
        self.feature_importance: Optional[pd.DataFrame] = None
        self.shap_values_summary: Optional[pd.DataFrame] = None

        # Meta-model for stacking
        self.meta_model: Optional[LogisticRegression] = None

        # Tuned params
        self._tuned_params = self._load_tuned_params()

    # ── Training ──────────────────────────────────────────────────────

    def train(
        self,
        features_df: pd.DataFrame,
        results_df: pd.DataFrame,
        calibrate: bool = True,
        feature_selection: bool = True,
        top_n_features: int = 25,
    ):
        """
        Train all sub-models.

        features_df: output of FeatureEngineer.build_features() for each historical game.
        results_df: must have columns:
            - home_win (bool): did home team win?
            - total_goals (int): total goals in the game
            - home_covered (bool): did home cover -1.5 spread?
        """
        X_full, all_feature_names = _prepare_features(features_df)

        y_win = results_df["home_win"].astype(int).values
        y_total = results_df["total_goals"].values
        y_spread = results_df["home_covered"].astype(int).values

        # ── SHAP Feature Selection ────────────────────────────────────
        if feature_selection and shap is not None and len(all_feature_names) > top_n_features:
            logger.info("Running SHAP feature selection (from %d to %d)...",
                        len(all_feature_names), top_n_features)
            selected = self._shap_select(X_full, y_win, top_n=top_n_features)
            if selected:
                self.selected_features = selected
                self.feature_names = selected
                X = X_full[selected]
                logger.info("Selected %d features via SHAP", len(selected))
            else:
                self.feature_names = all_feature_names
                self.selected_features = None
                X = X_full
        else:
            self.feature_names = all_feature_names
            self.selected_features = None
            X = X_full
            if feature_selection and shap is None:
                logger.warning("shap not installed — skipping feature selection")

        X_scaled = self.scaler.fit_transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=self.feature_names)

        logger.info("Training on %d games with %d features", len(X), len(self.feature_names))

        # ── Early stopping split (last 15% as eval set) ──────────────
        split_idx = int(len(X_scaled_df) * 0.85)
        X_train_es = X_scaled_df.iloc[:split_idx]
        X_eval_es = X_scaled_df.iloc[split_idx:]
        y_win_tr, y_win_ev = y_win[:split_idx], y_win[split_idx:]
        y_total_tr, y_total_ev = y_total[:split_idx], y_total[split_idx:]
        y_spread_tr, y_spread_ev = y_spread[:split_idx], y_spread[split_idx:]

        # ── Win model ────────────────────────────────────────────────
        self.win_model = self._build_classifier("win")
        self.win_model.fit(X_train_es, y_win_tr,
                           eval_set=[(X_eval_es, y_win_ev)], verbose=False)
        logger.info("Win model stopped at %d trees",
                     getattr(self.win_model, "best_iteration", -1))

        if calibrate:
            self.win_model = CalibratedClassifierCV(
                self.win_model, cv=3, method="isotonic"
            )
            self.win_model.fit(X_scaled_df, y_win)

        # ── Total model ─────────────────────────────────────────────
        self.total_model = self._build_regressor("total")
        self.total_model.fit(X_train_es, y_total_tr,
                             eval_set=[(X_eval_es, y_total_ev)], verbose=False)
        logger.info("Total model stopped at %d trees",
                     getattr(self.total_model, "best_iteration", -1))

        # ── Spread model ─────────────────────────────────────────────
        self.spread_model = self._build_classifier("spread")
        self.spread_model.fit(X_train_es, y_spread_tr,
                              eval_set=[(X_eval_es, y_spread_ev)], verbose=False)
        logger.info("Spread model stopped at %d trees",
                     getattr(self.spread_model, "best_iteration", -1))

        if calibrate:
            self.spread_model = CalibratedClassifierCV(
                self.spread_model, cv=3, method="isotonic"
            )
            self.spread_model.fit(X_scaled_df, y_spread)

        self.is_trained = True

        # Feature importance (from the base estimator)
        self._compute_feature_importance()

        # ── Train stacking meta-model ────────────────────────────────
        self._train_meta_model(X_full, all_feature_names, y_win, features_df)

        logger.info("Training complete")

    # ── Stacking Meta-model ───────────────────────────────────────────

    def _train_meta_model(
        self,
        X_full: pd.DataFrame,
        all_feature_names: list[str],
        y_win: np.ndarray,
        features_df: pd.DataFrame,
    ):
        """
        Train a logistic regression meta-model on OOF predictions
        from the ML model, Poisson xG, and Elo.
        """
        logger.info("Training stacking meta-model...")

        feature_cols = self.feature_names
        X_sel = X_full[feature_cols] if self.selected_features else X_full

        tscv = TimeSeriesSplit(n_splits=5)
        meta_features = np.full((len(y_win), 3), np.nan)

        for train_idx, val_idx in tscv.split(X_sel):
            X_train = X_sel.iloc[train_idx]
            X_val = X_sel.iloc[val_idx]
            y_train = y_win[train_idx]

            scaler_tmp = StandardScaler()
            X_tr_sc = pd.DataFrame(
                scaler_tmp.fit_transform(X_train), columns=feature_cols
            )
            X_vl_sc = pd.DataFrame(
                scaler_tmp.transform(X_val), columns=feature_cols
            )

            clf = self._build_classifier("meta_oof")
            clf.fit(X_tr_sc, y_train,
                    eval_set=[(X_vl_sc, y_win[val_idx])], verbose=False)
            ml_probs = clf.predict_proba(X_vl_sc)[:, 1]

            meta_features[val_idx, 0] = ml_probs

        # Poisson & Elo from features_df
        # Use xG-based estimate as Poisson proxy
        for col_name, idx in [("elo_home_win_prob", 2)]:
            if col_name in features_df.columns:
                meta_features[:, idx] = features_df[col_name].fillna(0.5).values
            else:
                meta_features[:, idx] = 0.5

        # Poisson proxy: use xG features if available
        xgf_col = None
        for c in ("home_xgf_pg", "home_xg_diff_pg", "diff_xgf_pg"):
            if c in features_df.columns:
                xgf_col = c
                break
        if xgf_col:
            xg_vals = features_df[xgf_col].fillna(0).values
            # Rough sigmoid to convert xG diff to probability
            meta_features[:, 1] = 1.0 / (1.0 + np.exp(-xg_vals))
        else:
            meta_features[:, 1] = 0.5

        # Only use rows where we have OOF predictions (not in first fold's training set)
        valid_mask = ~np.isnan(meta_features[:, 0])
        meta_X = meta_features[valid_mask]
        meta_y = y_win[valid_mask]

        if len(meta_y) < 50:
            logger.warning("Not enough data for meta-model (%d samples), skipping", len(meta_y))
            self.meta_model = None
            return

        self.meta_model = LogisticRegression(C=1.0, max_iter=1000)
        self.meta_model.fit(meta_X, meta_y)

        meta_probs = self.meta_model.predict_proba(meta_X)[:, 1]
        meta_acc = accuracy_score(meta_y, (meta_probs > 0.5).astype(int))
        meta_ll = log_loss(meta_y, meta_probs)

        coefs = self.meta_model.coef_[0]
        logger.info("Meta-model weights: ML=%.3f, Poisson=%.3f, Elo=%.3f",
                     coefs[0], coefs[1], coefs[2])
        logger.info("Meta-model accuracy=%.3f, log_loss=%.3f", meta_acc, meta_ll)

    def blend_predictions(
        self,
        ml_prob: float,
        poisson_prob: float,
        elo_prob: float,
    ) -> float:
        """
        Blend ML, Poisson, and Elo probabilities using the meta-model.
        Falls back to 60/40 ML/Poisson if meta-model not available.
        """
        if self.meta_model is not None:
            X = np.array([[ml_prob, poisson_prob, elo_prob]])
            return float(self.meta_model.predict_proba(X)[0, 1])
        # Fallback: fixed weights
        return 0.60 * ml_prob + 0.40 * poisson_prob

    # ── SHAP Feature Selection ────────────────────────────────────────

    def _shap_select(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        top_n: int = 40,
    ) -> list[str]:
        """Select top N features by mean |SHAP value|."""
        try:
            # Train a quick model for SHAP analysis
            model = self._build_classifier("shap_select")
            model.fit(X, y)

            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(X)

            # For binary classification, shap_values might return a list
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1]  # class 1 (home win)

            mean_abs_shap = np.abs(shap_vals).mean(axis=0)
            importance_df = pd.DataFrame({
                "feature": X.columns.tolist(),
                "mean_abs_shap": mean_abs_shap,
            }).sort_values("mean_abs_shap", ascending=False)

            self.shap_values_summary = importance_df

            selected = importance_df.head(top_n)["feature"].tolist()
            return selected

        except Exception as exc:
            logger.warning("SHAP feature selection failed: %s", exc)
            return []

    def get_shap_report(self, top_n: int = 50) -> str:
        """Return formatted SHAP feature importance report."""
        if self.shap_values_summary is None:
            return "No SHAP data available. Run training with feature_selection=True first."

        lines = []
        lines.append(f"\n{'='*60}")
        lines.append("  SHAP FEATURE IMPORTANCE")
        lines.append(f"{'='*60}\n")
        lines.append(f"  {'Rank':<5} {'Feature':<40} {'|SHAP|':>10}")
        lines.append(f"  {'─'*55}")

        df = self.shap_values_summary.head(top_n)
        for i, (_, row) in enumerate(df.iterrows(), 1):
            marker = " *" if self.selected_features and row["feature"] in self.selected_features else ""
            lines.append(f"  {i:<5} {row['feature']:<40} {row['mean_abs_shap']:>10.4f}{marker}")

        if self.selected_features:
            lines.append(f"\n  * = selected ({len(self.selected_features)} features)")
        lines.append(f"\n{'='*60}\n")

        return "\n".join(lines)

    # ── Prediction ────────────────────────────────────────────────────

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

    # ── Evaluation ────────────────────────────────────────────────────

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

        # Use selected features if available
        if self.selected_features:
            use_features = [f for f in self.selected_features if f in X.columns]
            if use_features:
                X = X[use_features]
                feature_names = use_features

        X_scaled = self.scaler.fit_transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=feature_names)

        y_win = results_df["home_win"].astype(int).values

        tscv = TimeSeriesSplit(n_splits=n_splits)

        accuracies = []
        log_losses_list = []

        for train_idx, val_idx in tscv.split(X_scaled_df):
            X_train = X_scaled_df.iloc[train_idx]
            X_val = X_scaled_df.iloc[val_idx]
            y_train = y_win[train_idx]
            y_val = y_win[val_idx]

            model = self._build_classifier("cv")
            model.fit(X_train, y_train,
                       eval_set=[(X_val, y_val)], verbose=False)
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

    # ── Save / Load ───────────────────────────────────────────────────

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

        # Save selected features
        if self.selected_features:
            with open(os.path.join(path, "selected_features.json"), "w") as f:
                json.dump(self.selected_features, f)

        # Save SHAP summary
        if self.shap_values_summary is not None:
            self.shap_values_summary.to_csv(
                os.path.join(path, "shap_summary.csv"), index=False
            )

        # Save meta-model
        if self.meta_model is not None:
            joblib.dump(self.meta_model, os.path.join(path, "meta_model.pkl"))

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

        # Load selected features
        sel_path = os.path.join(path, "selected_features.json")
        if os.path.exists(sel_path):
            with open(sel_path, "r") as f:
                self.selected_features = json.load(f)

        # Load SHAP summary
        shap_path = os.path.join(path, "shap_summary.csv")
        if os.path.exists(shap_path):
            self.shap_values_summary = pd.read_csv(shap_path)

        # Load meta-model
        meta_path = os.path.join(path, "meta_model.pkl")
        if os.path.exists(meta_path):
            self.meta_model = joblib.load(meta_path)
            logger.info("Meta-model loaded (stacking enabled)")

        self.is_trained = True
        logger.info("Model loaded from %s (%d features%s)", path, len(self.feature_names),
                     f", {len(self.selected_features)} selected" if self.selected_features else "")

    # ── Model builders ────────────────────────────────────────────────

    def _load_tuned_params(self) -> dict | None:
        """Load Optuna-tuned params if available."""
        from models.tuner import HyperparamTuner
        return HyperparamTuner.load_params()

    def _build_classifier(self, name: str):
        """Build the best available classifier, using tuned params if available."""
        params = {}
        if self._tuned_params and "classifier" in self._tuned_params:
            params = self._tuned_params["classifier"]
            logger.debug("Using tuned classifier params for '%s'", name)

        if xgb is not None:
            defaults = dict(
                n_estimators=1000, max_depth=4, learning_rate=0.03,
                subsample=0.7, colsample_bytree=0.7, min_child_weight=5,
                reg_alpha=0.3, reg_lambda=2.0, gamma=0.5,
                early_stopping_rounds=30,
            )
            defaults.update(params)
            return xgb.XGBClassifier(
                **defaults,
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=42,
                verbosity=0,
            )
        elif lgb is not None:
            defaults = dict(
                n_estimators=1000, max_depth=4, learning_rate=0.03,
                subsample=0.7, colsample_bytree=0.7, min_child_weight=5,
                reg_alpha=0.3, reg_lambda=2.0,
            )
            defaults.update(params)
            return lgb.LGBMClassifier(
                **defaults, random_state=42, verbose=-1,
                callbacks=[lgb.early_stopping(30, verbose=False)],
            )
        else:
            from sklearn.ensemble import GradientBoostingClassifier
            logger.warning("Neither XGBoost nor LightGBM available — using sklearn GBM")
            return GradientBoostingClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.03,
                subsample=0.7, random_state=42,
            )

    def _build_regressor(self, name: str):
        """Build the best available regressor, using tuned params if available."""
        params = {}
        if self._tuned_params and "regressor" in self._tuned_params:
            params = self._tuned_params["regressor"]
            logger.debug("Using tuned regressor params for '%s'", name)

        if xgb is not None:
            defaults = dict(
                n_estimators=1000, max_depth=4, learning_rate=0.03,
                subsample=0.7, colsample_bytree=0.7, min_child_weight=5,
                reg_alpha=0.3, reg_lambda=2.0, gamma=0.5,
                early_stopping_rounds=20,
            )
            defaults.update(params)
            return xgb.XGBRegressor(
                **defaults, random_state=42, verbosity=0,
            )
        elif lgb is not None:
            defaults = dict(
                n_estimators=1000, max_depth=4, learning_rate=0.03,
                subsample=0.7, colsample_bytree=0.7, min_child_weight=5,
                reg_alpha=0.3, reg_lambda=2.0,
            )
            defaults.update(params)
            return lgb.LGBMRegressor(
                **defaults, random_state=42, verbose=-1,
                callbacks=[lgb.early_stopping(20, verbose=False)],
            )
        else:
            from sklearn.ensemble import GradientBoostingRegressor
            return GradientBoostingRegressor(
                n_estimators=200, max_depth=4, learning_rate=0.03,
                subsample=0.7, random_state=42,
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
