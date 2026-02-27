"""
Hyperparameter Tuning with Optuna.

Optimizes XGBoost/LightGBM/CatBoost parameters via Bayesian search,
using TimeSeriesSplit cross-validation to prevent look-ahead bias.
Supports sample weighting for recency bias.
"""

import os
import json
import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss, mean_absolute_error

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    optuna = None

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

from config import cfg

logger = logging.getLogger("nhl_predictor.models.tuner")

PARAMS_FILE = os.path.join(cfg.models_dir, "best_params.json")


class HyperparamTuner:
    """Optuna-based hyperparameter optimization for NHL prediction models."""

    def __init__(self, n_splits: int = 5, engine: str = "auto"):
        if optuna is None:
            raise ImportError("optuna is required for tuning: pip install optuna")
        self.n_splits = n_splits
        self.engine = self._resolve_engine(engine)

    @staticmethod
    def _resolve_engine(engine: str) -> str:
        if engine != "auto":
            return engine
        if xgb is not None:
            return "xgboost"
        if cb is not None:
            return "catboost"
        if lgb is not None:
            return "lightgbm"
        return "sklearn"

    @staticmethod
    def _compute_sample_weights(n_samples: int, half_life: int = 200) -> np.ndarray:
        positions = np.arange(n_samples)
        return np.power(2.0, (positions - n_samples + 1) / half_life)

    def tune_classifier(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        n_trials: int = 100,
        sample_weighting: bool = True,
    ) -> dict:
        """Tune classifier hyperparameters to minimize CV log loss."""
        logger.info("Tuning classifier (%d trials, %d-fold CV, engine=%s)...",
                     n_trials, self.n_splits, self.engine)

        all_weights = self._compute_sample_weights(len(y)) if sample_weighting else None

        def objective(trial):
            params = {
                "max_depth": trial.suggest_int("max_depth", 3, 6),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 0.9),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 0.8),
                "min_child_weight": trial.suggest_int("min_child_weight", 3, 15),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 1.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 5.0, log=True),
                "gamma": trial.suggest_float("gamma", 0.0, 2.0),
            }

            tscv = TimeSeriesSplit(n_splits=self.n_splits)
            scores = []

            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                sw = all_weights[train_idx] if all_weights is not None else None

                model = self._make_classifier(params)
                if model is None:
                    return float("inf")

                fit_kw = self._fit_kwargs(model, X_val, y_val, sw)
                model.fit(X_train, y_train, **fit_kw)
                proba = model.predict_proba(X_val)[:, 1]
                scores.append(log_loss(y_val, proba))

            return np.mean(scores)

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        best = study.best_params
        best["n_estimators"] = 1000
        best["early_stopping_rounds"] = 30
        logger.info("Best classifier log_loss: %.4f", study.best_value)
        logger.info("Best params: %s", best)
        return best

    def tune_regressor(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        n_trials: int = 50,
        sample_weighting: bool = True,
    ) -> dict:
        """Tune regressor hyperparameters to minimize CV MAE."""
        logger.info("Tuning regressor (%d trials, %d-fold CV, engine=%s)...",
                     n_trials, self.n_splits, self.engine)

        all_weights = self._compute_sample_weights(len(y)) if sample_weighting else None

        def objective(trial):
            params = {
                "max_depth": trial.suggest_int("max_depth", 3, 6),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 0.9),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 0.8),
                "min_child_weight": trial.suggest_int("min_child_weight", 3, 15),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 1.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 5.0, log=True),
                "gamma": trial.suggest_float("gamma", 0.0, 2.0),
            }

            tscv = TimeSeriesSplit(n_splits=self.n_splits)
            scores = []

            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                sw = all_weights[train_idx] if all_weights is not None else None

                model = self._make_regressor(params)
                if model is None:
                    return float("inf")

                fit_kw = self._fit_kwargs(model, X_val, y_val, sw)
                model.fit(X_train, y_train, **fit_kw)
                preds = model.predict(X_val)
                scores.append(mean_absolute_error(y_val, preds))

            return np.mean(scores)

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        best = study.best_params
        best["n_estimators"] = 1000
        best["early_stopping_rounds"] = 20
        logger.info("Best regressor MAE: %.4f", study.best_value)
        logger.info("Best params: %s", best)
        return best

    # ── Model factories ────────────────────────────────────────────

    def _make_classifier(self, params):
        if self.engine == "catboost" and cb is not None:
            return cb.CatBoostClassifier(
                iterations=1000, depth=params.get("max_depth", 4),
                learning_rate=params.get("learning_rate", 0.03),
                subsample=params.get("subsample", 0.7),
                rsm=params.get("colsample_bytree", 0.7),
                l2_leaf_reg=params.get("reg_lambda", 2.0),
                early_stopping_rounds=30, random_seed=42, verbose=0,
            )
        if self.engine == "xgboost" and xgb is not None:
            xgb_params = {k: v for k, v in params.items()}
            xgb_params["n_estimators"] = 1000
            xgb_params["early_stopping_rounds"] = 30
            return xgb.XGBClassifier(
                **xgb_params, use_label_encoder=False,
                eval_metric="logloss", random_state=42, verbosity=0,
            )
        if self.engine == "lightgbm" and lgb is not None:
            lgb_params = {k: v for k, v in params.items()
                          if k not in ("gamma",)}
            lgb_params["n_estimators"] = 1000
            return lgb.LGBMClassifier(
                **lgb_params, random_state=42, verbose=-1,
                callbacks=[lgb.early_stopping(30, verbose=False)],
            )
        # Fallback
        if xgb is not None:
            return xgb.XGBClassifier(
                **params, n_estimators=1000, early_stopping_rounds=30,
                use_label_encoder=False, eval_metric="logloss",
                random_state=42, verbosity=0,
            )
        return None

    def _make_regressor(self, params):
        if self.engine == "catboost" and cb is not None:
            return cb.CatBoostRegressor(
                iterations=1000, depth=params.get("max_depth", 4),
                learning_rate=params.get("learning_rate", 0.03),
                subsample=params.get("subsample", 0.7),
                rsm=params.get("colsample_bytree", 0.7),
                l2_leaf_reg=params.get("reg_lambda", 2.0),
                early_stopping_rounds=20, random_seed=42, verbose=0,
            )
        if self.engine == "xgboost" and xgb is not None:
            xgb_params = {k: v for k, v in params.items()}
            xgb_params["n_estimators"] = 1000
            xgb_params["early_stopping_rounds"] = 20
            return xgb.XGBRegressor(
                **xgb_params, random_state=42, verbosity=0,
            )
        if self.engine == "lightgbm" and lgb is not None:
            lgb_params = {k: v for k, v in params.items()
                          if k not in ("gamma",)}
            lgb_params["n_estimators"] = 1000
            return lgb.LGBMRegressor(
                **lgb_params, random_state=42, verbose=-1,
                callbacks=[lgb.early_stopping(20, verbose=False)],
            )
        if xgb is not None:
            return xgb.XGBRegressor(
                **params, n_estimators=1000, early_stopping_rounds=20,
                random_state=42, verbosity=0,
            )
        return None

    def _fit_kwargs(self, model, X_eval, y_eval, sample_weights=None) -> dict:
        """Build fit() kwargs appropriate for the model engine."""
        if cb is not None and isinstance(model, (cb.CatBoostClassifier, cb.CatBoostRegressor)):
            kwargs = {"eval_set": (X_eval, y_eval), "verbose": False}
            if sample_weights is not None:
                kwargs["sample_weight"] = sample_weights
            return kwargs
        kwargs = {"eval_set": [(X_eval, y_eval)], "verbose": False}
        if sample_weights is not None:
            kwargs["sample_weight"] = sample_weights
        return kwargs

    @staticmethod
    def save_params(classifier_params: dict, regressor_params: dict, path: str = None):
        """Save tuned parameters to JSON."""
        path = path or PARAMS_FILE
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {
            "classifier": classifier_params,
            "regressor": regressor_params,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("Tuned params saved to %s", path)

    @staticmethod
    def load_params(path: str = None) -> dict | None:
        """Load tuned parameters from JSON. Returns None if not found."""
        path = path or PARAMS_FILE
        if not os.path.exists(path):
            return None
        with open(path, "r") as f:
            return json.load(f)
