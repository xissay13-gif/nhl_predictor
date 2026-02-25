"""
Hyperparameter Tuning with Optuna.

Optimizes XGBoost/LightGBM parameters via Bayesian search,
using TimeSeriesSplit cross-validation to prevent look-ahead bias.
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

from config import cfg

logger = logging.getLogger("nhl_predictor.models.tuner")

PARAMS_FILE = os.path.join(cfg.models_dir, "best_params.json")


class HyperparamTuner:
    """Optuna-based hyperparameter optimization for NHL prediction models."""

    def __init__(self, n_splits: int = 5):
        if optuna is None:
            raise ImportError("optuna is required for tuning: pip install optuna")
        self.n_splits = n_splits

    def tune_classifier(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        n_trials: int = 100,
    ) -> dict:
        """Tune classifier hyperparameters to minimize CV log loss."""
        logger.info("Tuning classifier (%d trials, %d-fold CV)...", n_trials, self.n_splits)

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 1.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 5.0, log=True),
            }

            tscv = TimeSeriesSplit(n_splits=self.n_splits)
            scores = []

            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                if xgb is not None:
                    model = xgb.XGBClassifier(
                        **params,
                        use_label_encoder=False,
                        eval_metric="logloss",
                        random_state=42,
                        verbosity=0,
                    )
                elif lgb is not None:
                    model = lgb.LGBMClassifier(
                        **params, random_state=42, verbose=-1,
                    )
                else:
                    return float("inf")

                model.fit(X_train, y_train)
                proba = model.predict_proba(X_val)[:, 1]
                scores.append(log_loss(y_val, proba))

            return np.mean(scores)

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        best = study.best_params
        logger.info("Best classifier log_loss: %.4f", study.best_value)
        logger.info("Best params: %s", best)
        return best

    def tune_regressor(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        n_trials: int = 50,
    ) -> dict:
        """Tune regressor hyperparameters to minimize CV MAE."""
        logger.info("Tuning regressor (%d trials, %d-fold CV)...", n_trials, self.n_splits)

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 400, step=50),
                "max_depth": trial.suggest_int("max_depth", 3, 7),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 1.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 5.0, log=True),
            }

            tscv = TimeSeriesSplit(n_splits=self.n_splits)
            scores = []

            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                if xgb is not None:
                    model = xgb.XGBRegressor(
                        **params, random_state=42, verbosity=0,
                    )
                elif lgb is not None:
                    model = lgb.LGBMRegressor(
                        **params, random_state=42, verbose=-1,
                    )
                else:
                    return float("inf")

                model.fit(X_train, y_train)
                preds = model.predict(X_val)
                scores.append(mean_absolute_error(y_val, preds))

            return np.mean(scores)

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        best = study.best_params
        logger.info("Best regressor MAE: %.4f", study.best_value)
        logger.info("Best params: %s", best)
        return best

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
