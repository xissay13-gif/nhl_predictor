"""
Main ML Prediction Model.

Ensemble of XGBoost, LightGBM, and CatBoost for:
  1. Win probability (classification)
  2. Goal totals (regression)
  3. Spread/puck line (classification)

Includes SHAP feature selection, stacking meta-model, sample weighting,
top_n_features optimization, and Optuna param loading.
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
    import catboost as cb
except ImportError:
    cb = None

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
      - Automatic top_n_features optimization via grid search
      - Stacking meta-model with context features (ML + Poisson + Elo + context)
      - Exponential decay sample weighting (recent games matter more)
      - CatBoost support alongside XGBoost / LightGBM
      - Optuna tuned params (loaded from best_params.json if available)
    """

    # Available model engines
    ENGINES = ("xgboost", "lightgbm", "catboost")

    def __init__(self, engine: str = "auto"):
        """
        Args:
            engine: "xgboost", "lightgbm", "catboost", or "auto" (best available).
        """
        self.win_model = None
        self.total_model = None
        self.spread_model = None
        self.scaler = StandardScaler()
        self.feature_names: list[str] = []
        self.selected_features: list[str] | None = None
        self.is_trained = False
        self.engine = self._resolve_engine(engine)

        # Feature importance
        self.feature_importance: Optional[pd.DataFrame] = None
        self.shap_values_summary: Optional[pd.DataFrame] = None

        # Meta-model for stacking (enhanced with context features)
        self.meta_model: Optional[LogisticRegression] = None
        self.meta_feature_names: list[str] = []  # track meta-model input columns

        # Tuned params
        self._tuned_params = self._load_tuned_params()

    @staticmethod
    def _resolve_engine(engine: str) -> str:
        """Resolve 'auto' to the best available engine."""
        if engine != "auto":
            return engine
        if xgb is not None:
            return "xgboost"
        if cb is not None:
            return "catboost"
        if lgb is not None:
            return "lightgbm"
        return "sklearn"

    # ── Sample weighting ──────────────────────────────────────────────

    @staticmethod
    def _compute_sample_weights(
        n_samples: int,
        half_life: int = 200,
    ) -> np.ndarray:
        """
        Exponential decay weights: recent games get higher weight.

        Games are assumed chronologically ordered (oldest first).
        half_life: number of games after which weight halves
                   (200 ≈ ~3 months of NHL games across all teams).
        """
        positions = np.arange(n_samples)
        # Decay from oldest (0) to newest (n-1)
        # w(i) = 2^((i - n + 1) / half_life)   → newest = 1.0, oldest decays
        weights = np.power(2.0, (positions - n_samples + 1) / half_life)
        return weights

    # ── Training ──────────────────────────────────────────────────────

    def train(
        self,
        features_df: pd.DataFrame,
        results_df: pd.DataFrame,
        calibrate: bool = True,
        feature_selection: bool = True,
        top_n_features: int = 25,
        auto_select_top_n: bool = True,
        sample_weighting: bool = True,
        weight_half_life: int = 200,
    ):
        """
        Train all sub-models.

        features_df: output of FeatureEngineer.build_features() for each historical game.
        results_df: must have columns:
            - home_win (bool): did home team win?
            - total_goals (int): total goals in the game
            - home_covered (bool): did home cover -1.5 spread?

        New parameters:
            auto_select_top_n: if True, grid-search for optimal top_n_features value
            sample_weighting: if True, apply exponential decay sample weights
            weight_half_life: half-life for sample weight decay (in games)
        """
        X_full, all_feature_names = _prepare_features(features_df)

        y_win = results_df["home_win"].astype(int).values
        y_total = results_df["total_goals"].values
        y_spread = results_df["home_covered"].astype(int).values

        # ── Sample weights (exponential decay by recency) ─────────────
        sample_weights = None
        if sample_weighting:
            sample_weights = self._compute_sample_weights(len(y_win), half_life=weight_half_life)
            logger.info("Sample weighting enabled (half_life=%d): oldest=%.3f, newest=%.3f",
                        weight_half_life, sample_weights[0], sample_weights[-1])

        # ── SHAP Feature Selection ────────────────────────────────────
        if feature_selection and shap is not None and len(all_feature_names) > top_n_features:
            # Auto-optimize top_n if requested
            if auto_select_top_n:
                top_n_features = self._optimize_top_n(
                    X_full, y_win, all_feature_names, sample_weights
                )

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

        logger.info("Training on %d games with %d features (engine=%s)",
                     len(X), len(self.feature_names), self.engine)

        # ── Early stopping split (last 15% as eval set) ──────────────
        split_idx = int(len(X_scaled_df) * 0.85)
        X_train_es = X_scaled_df.iloc[:split_idx]
        X_eval_es = X_scaled_df.iloc[split_idx:]
        y_win_tr, y_win_ev = y_win[:split_idx], y_win[split_idx:]
        y_total_tr, y_total_ev = y_total[:split_idx], y_total[split_idx:]
        y_spread_tr, y_spread_ev = y_spread[:split_idx], y_spread[split_idx:]
        sw_train = sample_weights[:split_idx] if sample_weights is not None else None

        # ── Win model ────────────────────────────────────────────────
        self.win_model = self._build_classifier("win")
        fit_kwargs = self._fit_kwargs(
            self.win_model, X_eval_es, y_win_ev, sw_train
        )
        self.win_model.fit(X_train_es, y_win_tr, **fit_kwargs)
        logger.info("Win model stopped at %d trees",
                     self._get_best_iteration(self.win_model))

        if calibrate:
            self.win_model = self._calibrate_model(
                self.win_model, X_scaled_df, y_win, sample_weights
            )

        # ── Total model ─────────────────────────────────────────────
        self.total_model = self._build_regressor("total")
        fit_kwargs = self._fit_kwargs(
            self.total_model, X_eval_es, y_total_ev, sw_train
        )
        self.total_model.fit(X_train_es, y_total_tr, **fit_kwargs)
        logger.info("Total model stopped at %d trees",
                     self._get_best_iteration(self.total_model))

        # ── Spread model ─────────────────────────────────────────────
        self.spread_model = self._build_classifier("spread")
        fit_kwargs = self._fit_kwargs(
            self.spread_model, X_eval_es, y_spread_ev, sw_train
        )
        self.spread_model.fit(X_train_es, y_spread_tr, **fit_kwargs)
        logger.info("Spread model stopped at %d trees",
                     self._get_best_iteration(self.spread_model))

        if calibrate:
            self.spread_model = self._calibrate_model(
                self.spread_model, X_scaled_df, y_spread, sample_weights
            )

        self.is_trained = True

        # Feature importance (from the base estimator)
        self._compute_feature_importance()

        # ── Train stacking meta-model (enhanced) ─────────────────────
        self._train_meta_model(X_full, all_feature_names, y_win, features_df,
                               sample_weights=sample_weights)

        logger.info("Training complete")

    # ── Top N feature optimization ─────────────────────────────────────

    def _optimize_top_n(
        self,
        X_full: pd.DataFrame,
        y_win: np.ndarray,
        all_feature_names: list[str],
        sample_weights: np.ndarray | None = None,
        candidates: list[int] | None = None,
    ) -> int:
        """
        Grid search over top_n_features values to find the one that
        minimizes CV log loss.
        """
        if candidates is None:
            candidates = [15, 20, 25, 30, 35, 40]

        # Filter candidates to valid range
        max_feats = len(all_feature_names)
        candidates = [c for c in candidates if c < max_feats]
        if not candidates:
            return min(25, max_feats)

        logger.info("Optimizing top_n_features from candidates %s...", candidates)

        # Train a quick SHAP model to rank features once
        shap_ranking = self._shap_select(X_full, y_win, top_n=max(candidates))
        if not shap_ranking:
            logger.warning("SHAP ranking failed — using default top_n=25")
            return 25

        best_n = candidates[0]
        best_score = float("inf")

        tscv = TimeSeriesSplit(n_splits=3)  # fewer folds for speed

        for n in candidates:
            features_subset = shap_ranking[:n]
            X_sub = X_full[features_subset]

            scores = []
            for train_idx, val_idx in tscv.split(X_sub):
                X_tr = X_sub.iloc[train_idx]
                X_vl = X_sub.iloc[val_idx]
                y_tr = y_win[train_idx]
                y_vl = y_win[val_idx]

                scaler_tmp = StandardScaler()
                X_tr_s = pd.DataFrame(scaler_tmp.fit_transform(X_tr), columns=features_subset)
                X_vl_s = pd.DataFrame(scaler_tmp.transform(X_vl), columns=features_subset)

                model = self._build_classifier("topn_search")
                sw = sample_weights[train_idx] if sample_weights is not None else None
                fit_kw = self._fit_kwargs(model, X_vl_s, y_vl, sw)
                model.fit(X_tr_s, y_tr, **fit_kw)
                proba = model.predict_proba(X_vl_s)[:, 1]
                scores.append(log_loss(y_vl, proba))

            mean_ll = np.mean(scores)
            logger.info("  top_n=%d → CV log_loss=%.4f", n, mean_ll)
            if mean_ll < best_score:
                best_score = mean_ll
                best_n = n

        logger.info("Optimal top_n_features = %d (log_loss=%.4f)", best_n, best_score)
        return best_n

    # ── Calibration helper ─────────────────────────────────────────────

    def _calibrate_model(self, model, X_full, y, sample_weights=None):
        """Calibrate a classifier using isotonic regression."""
        best_iter = self._get_best_iteration(model)
        if best_iter > 0:
            model.set_params(n_estimators=best_iter)
        # Disable early stopping for refit
        self._disable_early_stopping(model)
        cal = CalibratedClassifierCV(model, cv=3, method="isotonic")
        cal.fit(X_full, y, sample_weight=sample_weights)
        return cal

    @staticmethod
    def _get_best_iteration(model) -> int:
        """Get best iteration from any boosting model."""
        if hasattr(model, "best_iteration"):
            return getattr(model, "best_iteration", -1) or -1
        if hasattr(model, "best_iteration_"):
            return getattr(model, "best_iteration_", -1) or -1
        return -1

    @staticmethod
    def _disable_early_stopping(model):
        """Disable early stopping for refit."""
        if hasattr(model, "early_stopping_rounds"):
            try:
                model.set_params(early_stopping_rounds=None)
            except Exception:
                pass

    def _fit_kwargs(self, model, X_eval, y_eval, sample_weights=None) -> dict:
        """Build fit() kwargs appropriate for the model engine."""
        kwargs: dict = {"verbose": False}

        # CatBoost uses different API
        if cb is not None and isinstance(model, (cb.CatBoostClassifier, cb.CatBoostRegressor)):
            kwargs = {
                "eval_set": (X_eval, y_eval),
                "verbose": False,
            }
            if sample_weights is not None:
                kwargs["sample_weight"] = sample_weights
            return kwargs

        # XGBoost / LightGBM
        kwargs["eval_set"] = [(X_eval, y_eval)]
        if sample_weights is not None:
            kwargs["sample_weight"] = sample_weights
        return kwargs

    # ── Stacking Meta-model ───────────────────────────────────────────

    # ── Context columns for enhanced meta-model ─────────────────────

    META_CONTEXT_COLS = [
        "rest_diff",
        "diff_momentum",
        "market_home_true_prob",
        "diff_win_pct",
        "diff_goal_diff_pg",
    ]

    def _train_meta_model(
        self,
        X_full: pd.DataFrame,
        all_feature_names: list[str],
        y_win: np.ndarray,
        features_df: pd.DataFrame,
        sample_weights: np.ndarray | None = None,
    ):
        """
        Train an enhanced logistic regression meta-model on OOF predictions
        from the ML model, Poisson xG, Elo, PLUS context features
        (rest_diff, momentum_diff, market_implied, win_pct_diff, goal_diff).
        """
        logger.info("Training enhanced stacking meta-model...")

        feature_cols = self.feature_names
        X_sel = X_full[feature_cols] if self.selected_features else X_full

        # Determine which context cols are available
        available_context = [c for c in self.META_CONTEXT_COLS if c in features_df.columns]

        # Total meta features: 3 (ml, poisson, elo) + N context
        n_meta = 3 + len(available_context)
        self.meta_feature_names = ["ml_prob", "poisson_prob", "elo_prob"] + available_context

        tscv = TimeSeriesSplit(n_splits=5)
        meta_features = np.full((len(y_win), n_meta), np.nan)

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
            sw = sample_weights[train_idx] if sample_weights is not None else None
            fit_kw = self._fit_kwargs(clf, X_vl_sc, y_win[val_idx], sw)
            clf.fit(X_tr_sc, y_train, **fit_kw)
            ml_probs = clf.predict_proba(X_vl_sc)[:, 1]

            meta_features[val_idx, 0] = ml_probs

        # Elo probabilities
        if "elo_home_win_prob" in features_df.columns:
            meta_features[:, 2] = features_df["elo_home_win_prob"].fillna(0.5).values
        else:
            meta_features[:, 2] = 0.5

        # Poisson proxy: use xG features if available
        xgf_col = None
        for c in ("home_xgf_pg", "home_xg_diff_pg", "diff_xgf_pg"):
            if c in features_df.columns:
                xgf_col = c
                break
        if xgf_col:
            xg_vals = features_df[xgf_col].fillna(0).values
            meta_features[:, 1] = 1.0 / (1.0 + np.exp(-xg_vals))
        else:
            meta_features[:, 1] = 0.5

        # Context features
        for i, col in enumerate(available_context):
            meta_features[:, 3 + i] = features_df[col].fillna(0).values

        # Only use rows where we have OOF predictions
        valid_mask = ~np.isnan(meta_features[:, 0])
        meta_X = meta_features[valid_mask]
        meta_y = y_win[valid_mask]
        meta_sw = sample_weights[valid_mask] if sample_weights is not None else None

        if len(meta_y) < 50:
            logger.warning("Not enough data for meta-model (%d samples), skipping", len(meta_y))
            self.meta_model = None
            return

        # Replace any remaining NaNs in context cols
        meta_X = np.nan_to_num(meta_X, nan=0.0)

        self.meta_model = LogisticRegression(C=1.0, max_iter=1000)
        self.meta_model.fit(meta_X, meta_y, sample_weight=meta_sw)

        meta_probs = self.meta_model.predict_proba(meta_X)[:, 1]
        meta_acc = accuracy_score(meta_y, (meta_probs > 0.5).astype(int))
        meta_ll = log_loss(meta_y, meta_probs)

        coefs = self.meta_model.coef_[0]
        coef_str = ", ".join(f"{name}={c:.3f}" for name, c in zip(self.meta_feature_names, coefs))
        logger.info("Meta-model weights: %s", coef_str)
        logger.info("Meta-model accuracy=%.3f, log_loss=%.3f", meta_acc, meta_ll)

    def blend_predictions(
        self,
        ml_prob: float,
        poisson_prob: float,
        elo_prob: float,
        context: dict | None = None,
    ) -> float:
        """
        Blend ML, Poisson, and Elo probabilities using the enhanced meta-model.
        Falls back to 60/40 ML/Poisson if meta-model not available.

        context: optional dict with keys like rest_diff, diff_momentum, etc.
        """
        if self.meta_model is not None:
            base = [ml_prob, poisson_prob, elo_prob]
            # Add context features in the same order as training
            if context and len(self.meta_feature_names) > 3:
                for col in self.meta_feature_names[3:]:
                    base.append(context.get(col, 0.0))
            elif len(self.meta_feature_names) > 3:
                base.extend([0.0] * (len(self.meta_feature_names) - 3))
            X = np.array([base])
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
            split_idx = int(len(X) * 0.85)
            X_train, X_eval = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_eval = y[:split_idx], y[split_idx:]
            model.fit(X_train, y_train,
                      eval_set=[(X_eval, y_eval)], verbose=False)

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
        sample_weighting: bool = True,
    ) -> dict:
        """Time-series cross-validation with sample weighting."""
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
        all_weights = self._compute_sample_weights(len(y_win)) if sample_weighting else None

        tscv = TimeSeriesSplit(n_splits=n_splits)

        accuracies = []
        log_losses_list = []

        for train_idx, val_idx in tscv.split(X_scaled_df):
            X_train = X_scaled_df.iloc[train_idx]
            X_val = X_scaled_df.iloc[val_idx]
            y_train = y_win[train_idx]
            y_val = y_win[val_idx]

            model = self._build_classifier("cv")
            sw = all_weights[train_idx] if all_weights is not None else None
            fit_kw = self._fit_kwargs(model, X_val, y_val, sw)
            model.fit(X_train, y_train, **fit_kw)
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

    def calibration_analysis(
        self,
        features_df: pd.DataFrame,
        results_df: pd.DataFrame,
        n_splits: int = 5,
        n_bins: int = 10,
    ) -> dict:
        """
        Out-of-fold calibration analysis using TimeSeriesSplit.

        Returns dict with:
          - oof_predictions: array of OOF predicted probabilities
          - oof_actuals: array of actual outcomes
          - calibration_bins: list of {bin_center, predicted_mean, actual_mean, count}
          - ece: Expected Calibration Error
          - mce: Maximum Calibration Error
          - overconfidence: avg(predicted - actual) for bins where predicted > 0.5
          - underconfidence: avg(actual - predicted) for bins where predicted < 0.5
        """
        X, feature_names = _prepare_features(features_df)

        if self.selected_features:
            use_features = [f for f in self.selected_features if f in X.columns]
            if use_features:
                X = X[use_features]
                feature_names = use_features

        y_win = results_df["home_win"].astype(int).values

        tscv = TimeSeriesSplit(n_splits=n_splits)

        oof_probs = np.full(len(y_win), np.nan)

        for train_idx, val_idx in tscv.split(X):
            X_train_raw = X.iloc[train_idx]
            X_val_raw = X.iloc[val_idx]
            y_train = y_win[train_idx]

            scaler_tmp = StandardScaler()
            X_tr = pd.DataFrame(
                scaler_tmp.fit_transform(X_train_raw), columns=feature_names
            )
            X_vl = pd.DataFrame(
                scaler_tmp.transform(X_val_raw), columns=feature_names
            )

            model = self._build_classifier("calibration_oof")
            model.fit(X_tr, y_train,
                      eval_set=[(X_vl, y_win[val_idx])], verbose=False)
            oof_probs[val_idx] = model.predict_proba(X_vl)[:, 1]

        # Filter to rows with OOF predictions
        valid = ~np.isnan(oof_probs)
        probs = oof_probs[valid]
        actuals = y_win[valid]

        # Calibration bins
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bins_data = []
        for i in range(n_bins):
            lo, hi = bin_edges[i], bin_edges[i + 1]
            mask = (probs >= lo) & (probs < hi) if i < n_bins - 1 else (probs >= lo) & (probs <= hi)
            count = mask.sum()
            if count > 0:
                pred_mean = probs[mask].mean()
                actual_mean = actuals[mask].mean()
            else:
                pred_mean = (lo + hi) / 2
                actual_mean = np.nan
            bins_data.append({
                "bin_low": lo,
                "bin_high": hi,
                "bin_center": (lo + hi) / 2,
                "predicted_mean": pred_mean,
                "actual_mean": actual_mean,
                "count": int(count),
            })

        # ECE and MCE
        total = len(probs)
        ece = 0.0
        mce = 0.0
        for b in bins_data:
            if b["count"] > 0 and not np.isnan(b["actual_mean"]):
                gap = abs(b["predicted_mean"] - b["actual_mean"])
                ece += (b["count"] / total) * gap
                mce = max(mce, gap)

        # Overconfidence / underconfidence analysis
        over_gaps = []
        under_gaps = []
        for b in bins_data:
            if b["count"] < 5 or np.isnan(b["actual_mean"]):
                continue
            if b["bin_center"] > 0.5:
                over_gaps.append(b["predicted_mean"] - b["actual_mean"])
            elif b["bin_center"] < 0.5:
                under_gaps.append(b["actual_mean"] - b["predicted_mean"])

        return {
            "oof_predictions": probs,
            "oof_actuals": actuals,
            "calibration_bins": bins_data,
            "ece": float(ece),
            "mce": float(mce),
            "overconfidence": float(np.mean(over_gaps)) if over_gaps else 0.0,
            "underconfidence": float(np.mean(under_gaps)) if under_gaps else 0.0,
            "total_oof_samples": int(len(probs)),
            "oof_accuracy": float(accuracy_score(actuals, (probs > 0.5).astype(int))),
            "oof_log_loss": float(log_loss(actuals, probs)),
            "oof_brier": float(brier_score_loss(actuals, probs)),
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

        # Save meta-model and its feature names
        if self.meta_model is not None:
            joblib.dump(self.meta_model, os.path.join(path, "meta_model.pkl"))
            with open(os.path.join(path, "meta_feature_names.json"), "w") as f:
                json.dump(self.meta_feature_names, f)

        # Save engine info
        with open(os.path.join(path, "model_config.json"), "w") as f:
            json.dump({"engine": self.engine}, f)

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

        # Load meta-model and its feature names
        meta_path = os.path.join(path, "meta_model.pkl")
        if os.path.exists(meta_path):
            self.meta_model = joblib.load(meta_path)
            meta_fn_path = os.path.join(path, "meta_feature_names.json")
            if os.path.exists(meta_fn_path):
                with open(meta_fn_path, "r") as f:
                    self.meta_feature_names = json.load(f)
            else:
                self.meta_feature_names = ["ml_prob", "poisson_prob", "elo_prob"]
            logger.info("Meta-model loaded (stacking enabled, %d features)", len(self.meta_feature_names))

        # Load engine info
        config_path = os.path.join(path, "model_config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                self.engine = json.load(f).get("engine", self.engine)

        self.is_trained = True
        logger.info("Model loaded from %s (%d features%s, engine=%s)", path, len(self.feature_names),
                     f", {len(self.selected_features)} selected" if self.selected_features else "",
                     self.engine)

    # ── Model builders ────────────────────────────────────────────────

    def _load_tuned_params(self) -> dict | None:
        """Load Optuna-tuned params if available."""
        from models.tuner import HyperparamTuner
        return HyperparamTuner.load_params()

    def _build_classifier(self, name: str):
        """Build a classifier using the selected engine, with tuned params if available."""
        params = {}
        if self._tuned_params and "classifier" in self._tuned_params:
            params = self._tuned_params["classifier"]
            logger.debug("Using tuned classifier params for '%s'", name)

        if self.engine == "catboost" and cb is not None:
            defaults = dict(
                iterations=1000, depth=4, learning_rate=0.03,
                subsample=0.7, rsm=0.7,  # rsm = colsample_bytree equivalent
                l2_leaf_reg=2.0,
                early_stopping_rounds=30,
                random_seed=42,
                verbose=0,
                loss_function="Logloss",
                eval_metric="Logloss",
            )
            # Map XGB-style params to CatBoost if tuned params came from XGB
            if "max_depth" in params:
                defaults["depth"] = params.pop("max_depth")
            if "colsample_bytree" in params:
                defaults["rsm"] = params.pop("colsample_bytree")
            if "reg_lambda" in params:
                defaults["l2_leaf_reg"] = params.pop("reg_lambda")
            # Remove XGB-only params
            for k in ("min_child_weight", "reg_alpha", "gamma",
                       "n_estimators", "use_label_encoder", "eval_metric"):
                params.pop(k, None)
            defaults.update(params)
            return cb.CatBoostClassifier(**defaults)

        if self.engine == "xgboost" and xgb is not None:
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

        if self.engine == "lightgbm" and lgb is not None:
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

        # Fallback: try any available engine
        if xgb is not None:
            return xgb.XGBClassifier(
                n_estimators=1000, max_depth=4, learning_rate=0.03,
                subsample=0.7, colsample_bytree=0.7, min_child_weight=5,
                reg_alpha=0.3, reg_lambda=2.0, gamma=0.5,
                early_stopping_rounds=30, use_label_encoder=False,
                eval_metric="logloss", random_state=42, verbosity=0,
            )
        if cb is not None:
            return cb.CatBoostClassifier(
                iterations=1000, depth=4, learning_rate=0.03,
                early_stopping_rounds=30, random_seed=42, verbose=0,
            )
        if lgb is not None:
            return lgb.LGBMClassifier(
                n_estimators=1000, max_depth=4, learning_rate=0.03,
                random_state=42, verbose=-1,
                callbacks=[lgb.early_stopping(30, verbose=False)],
            )

        from sklearn.ensemble import GradientBoostingClassifier
        logger.warning("No boosting library available — using sklearn GBM")
        return GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.03,
            subsample=0.7, random_state=42,
        )

    def _build_regressor(self, name: str):
        """Build a regressor using the selected engine, with tuned params if available."""
        params = {}
        if self._tuned_params and "regressor" in self._tuned_params:
            params = self._tuned_params["regressor"]
            logger.debug("Using tuned regressor params for '%s'", name)

        if self.engine == "catboost" and cb is not None:
            defaults = dict(
                iterations=1000, depth=4, learning_rate=0.03,
                subsample=0.7, rsm=0.7,
                l2_leaf_reg=2.0,
                early_stopping_rounds=20,
                random_seed=42,
                verbose=0,
                loss_function="RMSE",
            )
            if "max_depth" in params:
                defaults["depth"] = params.pop("max_depth")
            if "colsample_bytree" in params:
                defaults["rsm"] = params.pop("colsample_bytree")
            if "reg_lambda" in params:
                defaults["l2_leaf_reg"] = params.pop("reg_lambda")
            for k in ("min_child_weight", "reg_alpha", "gamma",
                       "n_estimators"):
                params.pop(k, None)
            defaults.update(params)
            return cb.CatBoostRegressor(**defaults)

        if self.engine == "xgboost" and xgb is not None:
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

        if self.engine == "lightgbm" and lgb is not None:
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

        # Fallback
        if xgb is not None:
            return xgb.XGBRegressor(
                n_estimators=1000, max_depth=4, learning_rate=0.03,
                early_stopping_rounds=20, random_state=42, verbosity=0,
            )
        if cb is not None:
            return cb.CatBoostRegressor(
                iterations=1000, depth=4, learning_rate=0.03,
                early_stopping_rounds=20, random_seed=42, verbose=0,
            )
        if lgb is not None:
            return lgb.LGBMRegressor(
                n_estimators=1000, max_depth=4, learning_rate=0.03,
                random_state=42, verbose=-1,
                callbacks=[lgb.early_stopping(20, verbose=False)],
            )

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
