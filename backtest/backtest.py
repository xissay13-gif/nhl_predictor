"""
Walk-forward Backtest Engine with Comprehensive Error Analysis.

Simulates the full prediction pipeline on historical data:
  - Trains on data before each game, predicts forward
  - Records ML, Poisson, Elo, and blended predictions
  - Produces detailed error analysis by multiple dimensions:
    team, home/away, favorite/underdog, season phase, rest,
    division matchups, model agreement, temporal drift, totals
"""

import logging
import os
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss, mean_absolute_error

from config import cfg
from models.elo import EloSystem
from models.poisson import PoissonGoalModel, estimate_xg_from_features
from features.engineer import FeatureEngineer
from data.collectors.nhl_api import get_team_game_log, get_standings
from data.collectors.moneypuck import (
    get_team_xg_summary, get_goalie_all_situations, get_skater_stats,
)

logger = logging.getLogger("nhl_predictor.backtest")

# Teams in same division for division-game detection
DIVISIONS = {
    "Atlantic": ["BOS", "BUF", "DET", "FLA", "MTL", "OTT", "TBL", "TOR"],
    "Metropolitan": ["CAR", "CBJ", "NJD", "NYI", "NYR", "PHI", "PIT", "WSH"],
    "Central": ["ARI", "CHI", "COL", "DAL", "MIN", "NSH", "STL", "WPG", "UTA"],
    "Pacific": ["ANA", "CGY", "EDM", "LAK", "SEA", "SJS", "VAN", "VGK"],
}

TEAM_TO_DIVISION = {}
for div, teams in DIVISIONS.items():
    for t in teams:
        TEAM_TO_DIVISION[t] = div


class BacktestEngine:
    """
    Walk-forward backtest that mirrors the real prediction pipeline.

    For each game in chronological order:
      1. Build features from data available BEFORE the game
      2. Generate OOF ML prediction, Poisson prediction, Elo prediction
      3. Record all predictions alongside actual results
      4. Update Elo after the game

    Then runs comprehensive error analysis on the collected predictions.
    """

    def __init__(self):
        self.results_df = None  # DataFrame of all backtest predictions

    def run(
        self,
        seasons: list[str] = None,
        n_splits: int = 5,
        min_train_games: int = 50,
    ) -> pd.DataFrame:
        """
        Execute walk-forward backtest.

        Uses TimeSeriesSplit OOF predictions for the ML model (same as
        calibration_analysis) but enriches each row with Poisson, Elo,
        and contextual metadata for segmented analysis.

        Returns DataFrame with one row per game.
        """
        if seasons is None:
            seasons = cfg.training_seasons

        logger.info("Starting backtest over seasons: %s", ", ".join(seasons))

        all_features = []
        all_results = []
        all_meta = []  # metadata for segmentation

        for season in seasons:
            season_year = season[:4]
            logger.info("Loading data for season %s...", season)

            standings = get_standings()
            mp_team = get_team_xg_summary(season_year)
            goalies_df = get_goalie_all_situations(season_year)
            skaters = get_skater_stats(season_year)

            goalie_lookup = {}
            if not goalies_df.empty:
                team_col = "team" if "team" in goalies_df.columns else goalies_df.columns[0]
                for team in goalies_df[team_col].unique():
                    tg = goalies_df[goalies_df[team_col] == team]
                    gp_col = next((c for c in ("games_played", "gamesPlayed", "GP")
                                   if c in tg.columns), None)
                    if gp_col and not tg.empty:
                        best = tg.sort_values(gp_col, ascending=False).iloc[0]
                        goalie_lookup[team] = best.to_dict()

            teams = list(cfg.team_abbrevs.keys())
            logs = []
            for team in teams:
                gl = get_team_game_log(team, season)
                if not gl.empty:
                    logs.append(gl)

            if not logs:
                logger.warning("No game logs for season %s — skipping", season)
                continue

            combined = pd.concat(logs, ignore_index=True)
            combined["date"] = pd.to_datetime(combined["date"])
            combined = combined.sort_values("date").reset_index(drop=True)

            elo = EloSystem()
            engineer = FeatureEngineer(elo_system=elo)

            home_games = combined[combined["is_home"]].copy().sort_values("date")
            logger.info("  Processing %d home games for season %s...",
                         len(home_games), season)

            for _, row in home_games.iterrows():
                home = row["team"]
                away = row["opponent"]
                game_date = row["date"].strftime("%Y-%m-%d")
                past_log = combined[combined["date"] < row["date"]]

                home_goals = int(row["goals_for"])
                away_goals = int(row["goals_against"])
                is_ot = bool(row.get("ot", False))

                if len(past_log) < min_train_games:
                    elo.update(home, away, home_goals, away_goals, is_ot)
                    continue

                features = engineer.build_features(
                    home_team=home, away_team=away, game_date=game_date,
                    standings=standings, game_log=past_log,
                    mp_team_data=mp_team, goalie_data=goalie_lookup,
                    skaters_df=skaters,
                )

                # Elo prediction (before update)
                elo_pred = elo.expected_score(home, away, a_is_home=True)

                # Poisson xG estimate
                home_xg, away_xg = estimate_xg_from_features(features)

                poisson = PoissonGoalModel(home_xg, away_xg)
                poisson_pred = poisson.win_probabilities()
                poisson_total = home_xg + away_xg

                all_features.append(features)
                all_results.append({
                    "home_win": home_goals > away_goals,
                    "total_goals": home_goals + away_goals,
                    "home_covered": (home_goals - away_goals) >= 2,
                    "home_goals": home_goals,
                    "away_goals": away_goals,
                })

                # Game-level metadata for segmentation
                month = row["date"].month
                day_of_season = (row["date"] - home_games["date"].min()).days

                all_meta.append({
                    "date": game_date,
                    "season": season,
                    "home_team": home,
                    "away_team": away,
                    "home_goals": home_goals,
                    "away_goals": away_goals,
                    "is_ot": is_ot,
                    "goal_diff": home_goals - away_goals,
                    "month": month,
                    "day_of_season": day_of_season,
                    # Elo
                    "elo_home_prob": elo_pred,
                    "elo_home_rating": elo._get(home),
                    "elo_away_rating": elo._get(away),
                    # Poisson
                    "poisson_home_prob": poisson_pred["home_win_prob"],
                    "poisson_total": poisson_total,
                    "poisson_home_xg": home_xg,
                    "poisson_away_xg": away_xg,
                    # Context features for segmentation
                    "home_rest_days": features.get("home_rest_days", np.nan),
                    "away_rest_days": features.get("away_rest_days", np.nan),
                    "home_b2b": features.get("home_back_to_back", 0),
                    "away_b2b": features.get("away_back_to_back", 0),
                    "division_game": _is_division_game(home, away),
                    "diff_win_pct": features.get("diff_win_pct", 0),
                    "diff_elo": features.get("elo_diff", elo._get(home) - elo._get(away)),
                })

                # Update Elo after game
                elo.update(home, away, home_goals, away_goals, is_ot)

        if not all_features:
            logger.error("No games collected for backtest")
            return pd.DataFrame()

        features_df = pd.DataFrame(all_features)
        results_df = pd.DataFrame(all_results)
        meta_df = pd.DataFrame(all_meta)

        logger.info("Backtest data: %d games across %d seasons", len(features_df), len(seasons))

        # ── OOF ML predictions via TimeSeriesSplit ────────────────────
        from models.predictor import _prepare_features

        X, feature_names = _prepare_features(features_df)
        y_win = results_df["home_win"].astype(int).values
        y_total = results_df["total_goals"].values

        tscv = TimeSeriesSplit(n_splits=n_splits)
        oof_win_probs = np.full(len(y_win), np.nan)
        oof_total_preds = np.full(len(y_total), np.nan)

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            logger.info("  OOF fold %d/%d: train=%d, val=%d",
                         fold + 1, n_splits, len(train_idx), len(val_idx))

            X_train_raw = X.iloc[train_idx]
            X_val_raw = X.iloc[val_idx]
            y_train_w = y_win[train_idx]
            y_train_t = y_total[train_idx]

            scaler = StandardScaler()
            X_tr = pd.DataFrame(scaler.fit_transform(X_train_raw), columns=feature_names)
            X_vl = pd.DataFrame(scaler.transform(X_val_raw), columns=feature_names)

            # Win classifier
            from models.predictor import NHLPredictor
            tmp_pred = NHLPredictor()
            clf = tmp_pred._build_classifier("backtest_oof")
            clf.fit(X_tr, y_train_w,
                    eval_set=[(X_vl, y_win[val_idx])], verbose=False)
            oof_win_probs[val_idx] = clf.predict_proba(X_vl)[:, 1]

            # Total regressor
            reg = tmp_pred._build_regressor("backtest_oof")
            reg.fit(X_tr, y_train_t,
                    eval_set=[(X_vl, y_total[val_idx])], verbose=False)
            oof_total_preds[val_idx] = reg.predict(X_vl)

        # ── Assemble results DataFrame ────────────────────────────────
        valid = ~np.isnan(oof_win_probs)
        df = meta_df[valid].copy().reset_index(drop=True)
        df["actual_home_win"] = y_win[valid]
        df["actual_total"] = y_total[valid]

        df["ml_home_prob"] = oof_win_probs[valid]
        df["ml_total_pred"] = oof_total_preds[valid]

        # Blended probability: simple 50/30/20 ML/Poisson/Elo (no meta-model in backtest)
        df["blended_home_prob"] = (
            0.50 * df["ml_home_prob"]
            + 0.30 * df["poisson_home_prob"]
            + 0.20 * df["elo_home_prob"]
        )

        # Predictions
        df["ml_pred_home_win"] = (df["ml_home_prob"] > 0.5).astype(int)
        df["poisson_pred_home_win"] = (df["poisson_home_prob"] > 0.5).astype(int)
        df["elo_pred_home_win"] = (df["elo_home_prob"] > 0.5).astype(int)
        df["blended_pred_home_win"] = (df["blended_home_prob"] > 0.5).astype(int)

        # Correct flags
        for model in ["ml", "poisson", "elo", "blended"]:
            df[f"{model}_correct"] = (df[f"{model}_pred_home_win"] == df["actual_home_win"]).astype(int)

        # Confidence
        df["ml_confidence"] = (df["ml_home_prob"] - 0.5).abs() * 2
        df["blended_confidence"] = (df["blended_home_prob"] - 0.5).abs() * 2

        # Favorite/underdog
        df["ml_is_favorite"] = df["ml_home_prob"] > 0.5
        df["actual_favorite_won"] = (
            ((df["ml_home_prob"] > 0.5) & (df["actual_home_win"] == 1))
            | ((df["ml_home_prob"] <= 0.5) & (df["actual_home_win"] == 0))
        )

        # Season phase: early (Oct-Nov), mid (Dec-Feb), late (Mar-Apr)
        df["season_phase"] = df["month"].map(
            lambda m: "early" if m in (9, 10, 11) else ("mid" if m in (12, 1, 2) else "late")
        )

        # Total error
        df["ml_total_error"] = df["ml_total_pred"] - df["actual_total"]
        df["poisson_total_error"] = df["poisson_total"] - df["actual_total"]

        # Close vs blowout
        df["game_type"] = df["goal_diff"].abs().map(
            lambda d: "1-goal" if d <= 1 else ("2-goal" if d == 2 else "blowout")
        )

        # Model agreement
        df["all_agree"] = (
            (df["ml_pred_home_win"] == df["poisson_pred_home_win"])
            & (df["ml_pred_home_win"] == df["elo_pred_home_win"])
        )
        df["ml_poisson_agree"] = df["ml_pred_home_win"] == df["poisson_pred_home_win"]
        df["ml_elo_agree"] = df["ml_pred_home_win"] == df["elo_pred_home_win"]

        self.results_df = df
        logger.info("Backtest complete: %d games with OOF predictions", len(df))
        return df

    def error_analysis(self) -> dict:
        """
        Comprehensive error analysis on backtest results.

        Returns dict with analysis sections, each containing summary stats
        and breakdowns.
        """
        df = self.results_df
        if df is None or df.empty:
            return {}

        report = {}

        # ── 1. Overall metrics per model ──────────────────────────────
        report["overall"] = self._overall_metrics(df)

        # ── 2. By team ───────────────────────────────────────────────
        report["by_home_team"] = self._accuracy_by_group(df, "home_team")
        report["by_away_team"] = self._accuracy_by_group(df, "away_team", pred_col="blended")

        # ── 3. By season phase ───────────────────────────────────────
        report["by_season_phase"] = self._accuracy_by_group(df, "season_phase")

        # ── 4. By game type (close vs blowout) ───────────────────────
        report["by_game_type"] = self._accuracy_by_group(df, "game_type")

        # ── 5. Favorite vs underdog ──────────────────────────────────
        report["favorite_underdog"] = self._favorite_underdog(df)

        # ── 6. Division games ────────────────────────────────────────
        report["division_games"] = self._division_analysis(df)

        # ── 7. Rest / back-to-back ───────────────────────────────────
        report["rest_analysis"] = self._rest_analysis(df)

        # ── 8. Model agreement ───────────────────────────────────────
        report["model_agreement"] = self._agreement_analysis(df)

        # ── 9. Calibration per model ─────────────────────────────────
        report["calibration"] = self._calibration_per_model(df)

        # ── 10. Temporal drift ───────────────────────────────────────
        report["temporal_drift"] = self._temporal_drift(df)

        # ── 11. Total goals analysis ─────────────────────────────────
        report["totals"] = self._totals_analysis(df)

        # ── 12. Confidence bands ─────────────────────────────────────
        report["confidence_bands"] = self._confidence_bands(df)

        return report

    # ── Analysis helpers ──────────────────────────────────────────────

    @staticmethod
    def _overall_metrics(df: pd.DataFrame) -> dict:
        y = df["actual_home_win"].values
        metrics = {}
        for model in ["ml", "poisson", "elo", "blended"]:
            prob_col = f"{model}_home_prob"
            pred_col = f"{model}_pred_home_win"
            probs = df[prob_col].clip(0.01, 0.99).values
            preds = df[pred_col].values
            metrics[model] = {
                "accuracy": float(accuracy_score(y, preds)),
                "log_loss": float(log_loss(y, probs)),
                "brier": float(brier_score_loss(y, probs)),
                "n_games": len(y),
            }
        # Totals
        for model, tcol in [("ml", "ml_total_pred"), ("poisson", "poisson_total")]:
            valid = df.dropna(subset=[tcol, "actual_total"])
            if not valid.empty:
                metrics[model]["total_mae"] = float(
                    mean_absolute_error(valid["actual_total"], valid[tcol])
                )
        return metrics

    @staticmethod
    def _accuracy_by_group(df: pd.DataFrame, group_col: str,
                           pred_col: str = "blended") -> list[dict]:
        results = []
        for name, grp in df.groupby(group_col):
            if len(grp) < 5:
                continue
            y = grp["actual_home_win"].values
            preds = grp[f"{pred_col}_pred_home_win"].values
            probs = grp[f"{pred_col}_home_prob"].clip(0.01, 0.99).values
            results.append({
                "group": str(name),
                "n_games": len(grp),
                "accuracy": float(accuracy_score(y, preds)),
                "log_loss": float(log_loss(y, probs)),
                "brier": float(brier_score_loss(y, probs)),
            })
        results.sort(key=lambda x: x["accuracy"])
        return results

    @staticmethod
    def _favorite_underdog(df: pd.DataFrame) -> dict:
        # ML favorite = home team when ml_home_prob > 0.5, away otherwise
        fav_mask = df["ml_home_prob"] > 0.5
        und_mask = ~fav_mask

        fav_correct = df.loc[fav_mask, "ml_correct"].mean() if fav_mask.sum() > 0 else 0
        und_correct = df.loc[und_mask, "ml_correct"].mean() if und_mask.sum() > 0 else 0

        # By confidence bucket for favorites
        strong_fav = df["ml_home_prob"] > 0.6
        mild_fav = (df["ml_home_prob"] > 0.5) & (df["ml_home_prob"] <= 0.6)
        mild_und = (df["ml_home_prob"] >= 0.4) & (df["ml_home_prob"] <= 0.5)
        strong_und = df["ml_home_prob"] < 0.4

        buckets = []
        for label, mask in [("Strong fav (>60%)", strong_fav),
                            ("Mild fav (50-60%)", mild_fav),
                            ("Mild und (40-50%)", mild_und),
                            ("Strong und (<40%)", strong_und)]:
            n = mask.sum()
            if n >= 5:
                buckets.append({
                    "bucket": label,
                    "n_games": int(n),
                    "accuracy": float(df.loc[mask, "blended_correct"].mean()),
                    "actual_home_win_rate": float(df.loc[mask, "actual_home_win"].mean()),
                })

        return {
            "favorite_accuracy": float(fav_correct),
            "favorite_n": int(fav_mask.sum()),
            "underdog_accuracy": float(und_correct),
            "underdog_n": int(und_mask.sum()),
            "buckets": buckets,
        }

    @staticmethod
    def _division_analysis(df: pd.DataFrame) -> dict:
        div = df[df["division_game"]]
        non_div = df[~df["division_game"]]

        result = {}
        for label, grp in [("division", div), ("non_division", non_div)]:
            if len(grp) < 10:
                continue
            y = grp["actual_home_win"].values
            preds = grp["blended_pred_home_win"].values
            probs = grp["blended_home_prob"].clip(0.01, 0.99).values
            result[label] = {
                "n_games": len(grp),
                "accuracy": float(accuracy_score(y, preds)),
                "log_loss": float(log_loss(y, probs)),
                "home_win_rate": float(y.mean()),
            }
        return result

    @staticmethod
    def _rest_analysis(df: pd.DataFrame) -> dict:
        result = {}

        # Back-to-back
        for col, label in [("home_b2b", "home_b2b"), ("away_b2b", "away_b2b")]:
            if col not in df.columns:
                continue
            b2b = df[df[col] == 1]
            rested = df[df[col] == 0]
            if len(b2b) >= 10 and len(rested) >= 10:
                result[label] = {
                    "b2b_n": len(b2b),
                    "b2b_accuracy": float(b2b["blended_correct"].mean()),
                    "b2b_home_win_rate": float(b2b["actual_home_win"].mean()),
                    "rested_n": len(rested),
                    "rested_accuracy": float(rested["blended_correct"].mean()),
                    "rested_home_win_rate": float(rested["actual_home_win"].mean()),
                }

        # Rest advantage (home rest > away rest)
        if "home_rest_days" in df.columns and "away_rest_days" in df.columns:
            valid = df.dropna(subset=["home_rest_days", "away_rest_days"])
            if not valid.empty:
                rest_diff = valid["home_rest_days"] - valid["away_rest_days"]
                for label, mask in [("home_rest_adv", rest_diff > 0),
                                    ("equal_rest", rest_diff == 0),
                                    ("away_rest_adv", rest_diff < 0)]:
                    grp = valid[mask]
                    if len(grp) >= 10:
                        result[label] = {
                            "n_games": len(grp),
                            "accuracy": float(grp["blended_correct"].mean()),
                            "home_win_rate": float(grp["actual_home_win"].mean()),
                        }

        return result

    @staticmethod
    def _agreement_analysis(df: pd.DataFrame) -> dict:
        result = {}

        for label, col in [("all_three", "all_agree"),
                           ("ml_poisson", "ml_poisson_agree"),
                           ("ml_elo", "ml_elo_agree")]:
            agree = df[df[col]]
            disagree = df[~df[col]]
            if len(agree) >= 10 and len(disagree) >= 10:
                result[label] = {
                    "agree_n": len(agree),
                    "agree_accuracy": float(agree["blended_correct"].mean()),
                    "disagree_n": len(disagree),
                    "disagree_accuracy": float(disagree["blended_correct"].mean()),
                    "delta": float(agree["blended_correct"].mean() - disagree["blended_correct"].mean()),
                }
        return result

    @staticmethod
    def _calibration_per_model(df: pd.DataFrame, n_bins: int = 10) -> dict:
        result = {}
        bin_edges = np.linspace(0, 1, n_bins + 1)

        for model in ["ml", "poisson", "elo", "blended"]:
            prob_col = f"{model}_home_prob"
            probs = df[prob_col].values
            actuals = df["actual_home_win"].values

            bins = []
            ece = 0.0
            total = len(probs)

            for i in range(n_bins):
                lo, hi = bin_edges[i], bin_edges[i + 1]
                mask = (probs >= lo) & (probs < hi) if i < n_bins - 1 else (probs >= lo) & (probs <= hi)
                count = mask.sum()
                if count > 0:
                    pred_mean = float(probs[mask].mean())
                    actual_mean = float(actuals[mask].mean())
                    gap = abs(pred_mean - actual_mean)
                    ece += (count / total) * gap
                    bins.append({
                        "range": f"{lo:.0%}-{hi:.0%}",
                        "predicted": round(pred_mean, 3),
                        "actual": round(actual_mean, 3),
                        "gap": round(pred_mean - actual_mean, 3),
                        "n": int(count),
                    })

            result[model] = {
                "ece": round(float(ece), 4),
                "bins": bins,
            }

        return result

    @staticmethod
    def _temporal_drift(df: pd.DataFrame, window: int = 100) -> dict:
        """Rolling accuracy over time to detect drift."""
        df_sorted = df.sort_values("date").reset_index(drop=True)
        n = len(df_sorted)

        if n < window:
            return {"warning": f"Not enough games ({n}) for drift analysis (need {window})"}

        windows = []
        step = max(window // 2, 25)

        for start in range(0, n - window + 1, step):
            chunk = df_sorted.iloc[start:start + window]
            date_start = chunk["date"].iloc[0]
            date_end = chunk["date"].iloc[-1]

            entry = {
                "date_start": date_start,
                "date_end": date_end,
                "n_games": len(chunk),
            }

            for model in ["ml", "blended"]:
                y = chunk["actual_home_win"].values
                preds = chunk[f"{model}_pred_home_win"].values
                probs = chunk[f"{model}_home_prob"].clip(0.01, 0.99).values
                entry[f"{model}_accuracy"] = float(accuracy_score(y, preds))
                entry[f"{model}_log_loss"] = float(log_loss(y, probs))

            windows.append(entry)

        # Trend: compare first half vs second half
        mid = len(windows) // 2
        if mid > 0:
            first_half = np.mean([w["blended_accuracy"] for w in windows[:mid]])
            second_half = np.mean([w["blended_accuracy"] for w in windows[mid:]])
            trend = second_half - first_half
        else:
            trend = 0.0

        return {
            "windows": windows,
            "trend": round(float(trend), 4),
            "trend_label": "improving" if trend > 0.02 else ("degrading" if trend < -0.02 else "stable"),
        }

    @staticmethod
    def _totals_analysis(df: pd.DataFrame) -> dict:
        result = {}

        for model, pred_col in [("ml", "ml_total_pred"), ("poisson", "poisson_total")]:
            valid = df.dropna(subset=[pred_col, "actual_total"])
            if valid.empty:
                continue

            preds = valid[pred_col].values
            actuals = valid["actual_total"].values
            errors = preds - actuals

            result[model] = {
                "mae": round(float(np.abs(errors).mean()), 3),
                "mean_error": round(float(errors.mean()), 3),  # positive = over-predicting
                "std_error": round(float(errors.std()), 3),
                "n_games": len(valid),
            }

            # By actual total bucket
            buckets = []
            for label, lo, hi in [("low (<=4)", 0, 4), ("medium (5-6)", 5, 6),
                                   ("high (7+)", 7, 99)]:
                mask = (actuals >= lo) & (actuals <= hi)
                if mask.sum() >= 10:
                    buckets.append({
                        "bucket": label,
                        "n_games": int(mask.sum()),
                        "mae": round(float(np.abs(errors[mask]).mean()), 3),
                        "mean_error": round(float(errors[mask].mean()), 3),
                    })
            result[model]["by_total_range"] = buckets

        # Over/under accuracy at 5.5 line
        if "poisson_total" in df.columns:
            valid = df.dropna(subset=["poisson_total", "actual_total"])
            if not valid.empty:
                pred_over = valid["poisson_total"] > 5.5
                actual_over = valid["actual_total"] > 5.5
                ou_acc = float((pred_over == actual_over).mean())
                result["over_under_accuracy_5_5"] = round(ou_acc, 4)

        return result

    @staticmethod
    def _confidence_bands(df: pd.DataFrame) -> list[dict]:
        """Accuracy by model confidence level."""
        bands = []
        thresholds = [(0.0, 0.1, "Very low (0-10%)"),
                      (0.1, 0.2, "Low (10-20%)"),
                      (0.2, 0.4, "Medium (20-40%)"),
                      (0.4, 0.6, "High (40-60%)"),
                      (0.6, 1.0, "Very high (60%+)")]

        for lo, hi, label in thresholds:
            mask = (df["blended_confidence"] >= lo) & (df["blended_confidence"] < hi)
            n = mask.sum()
            if n >= 5:
                grp = df[mask]
                bands.append({
                    "band": label,
                    "n_games": int(n),
                    "pct_of_total": round(float(n / len(df) * 100), 1),
                    "accuracy": round(float(grp["blended_correct"].mean()), 4),
                    "home_win_rate": round(float(grp["actual_home_win"].mean()), 4),
                })
        return bands

    # ── Report printing ───────────────────────────────────────────────

    def print_report(self, report: dict = None):
        """Print comprehensive backtest report."""
        if report is None:
            report = self.error_analysis()

        if not report:
            print("\nNo backtest results available.\n")
            return

        print(f"\n{'='*70}")
        print("  BACKTEST ERROR ANALYSIS REPORT")
        print(f"{'='*70}")

        # 1. Overall
        self._print_overall(report.get("overall", {}))

        # 2. Calibration
        self._print_calibration(report.get("calibration", {}))

        # 3. By season phase
        self._print_group("ACCURACY BY SEASON PHASE", report.get("by_season_phase", []))

        # 4. By game type
        self._print_group("ACCURACY BY GAME TYPE", report.get("by_game_type", []))

        # 5. Favorite/underdog
        self._print_fav_und(report.get("favorite_underdog", {}))

        # 6. Division
        self._print_division(report.get("division_games", {}))

        # 7. Rest
        self._print_rest(report.get("rest_analysis", {}))

        # 8. Model agreement
        self._print_agreement(report.get("model_agreement", {}))

        # 9. Temporal drift
        self._print_drift(report.get("temporal_drift", {}))

        # 10. Totals
        self._print_totals(report.get("totals", {}))

        # 11. Confidence bands
        self._print_confidence(report.get("confidence_bands", []))

        # 12. Worst teams
        self._print_worst_teams(report.get("by_home_team", []))

        print(f"\n{'='*70}\n")

    # ── Print helpers ─────────────────────────────────────────────────

    @staticmethod
    def _print_overall(overall: dict):
        if not overall:
            return
        print(f"\n  1. OVERALL MODEL COMPARISON")
        print(f"  {'─'*65}")
        print(f"  {'Model':<12} {'Accuracy':>10} {'Log Loss':>10} {'Brier':>10} {'Total MAE':>10} {'Games':>7}")
        print(f"  {'─'*65}")

        for model in ["ml", "poisson", "elo", "blended"]:
            m = overall.get(model, {})
            mae_str = f"{m.get('total_mae', 0):.3f}" if "total_mae" in m else "—"
            print(f"  {model:<12} {m.get('accuracy', 0):>10.1%} "
                  f"{m.get('log_loss', 0):>10.4f} {m.get('brier', 0):>10.4f} "
                  f"{mae_str:>10} {m.get('n_games', 0):>7}")

    @staticmethod
    def _print_calibration(calibration: dict):
        if not calibration:
            return
        print(f"\n  2. CALIBRATION PER MODEL (ECE)")
        print(f"  {'─'*65}")
        print(f"  {'Model':<12} {'ECE':>8}   Reliability detail")
        print(f"  {'─'*65}")

        for model in ["ml", "poisson", "elo", "blended"]:
            cal = calibration.get(model, {})
            ece = cal.get("ece", 0)
            status = "GOOD" if ece < 0.05 else ("FAIR" if ece < 0.08 else "POOR")
            print(f"  {model:<12} {ece:>7.4f}   [{status}]")

            for b in cal.get("bins", []):
                if b["n"] < 5:
                    continue
                gap = b["gap"]
                flag = ""
                if abs(gap) > 0.05:
                    flag = " OVERCONF" if gap > 0 else " UNDERCONF"
                elif abs(gap) <= 0.02:
                    flag = " ok"
                print(f"    {b['range']:<10} pred={b['predicted']:.1%}  actual={b['actual']:.1%}  "
                      f"gap={gap:+.1%}  n={b['n']}{flag}")

    @staticmethod
    def _print_group(title: str, groups: list):
        if not groups:
            return
        num = title[0:2].strip()
        print(f"\n  {title}")
        print(f"  {'─'*65}")
        print(f"  {'Group':<20} {'Accuracy':>10} {'Log Loss':>10} {'Brier':>10} {'Games':>7}")
        print(f"  {'─'*65}")
        for g in groups:
            print(f"  {g['group']:<20} {g['accuracy']:>10.1%} {g['log_loss']:>10.4f} "
                  f"{g['brier']:>10.4f} {g['n_games']:>7}")

    @staticmethod
    def _print_fav_und(fav: dict):
        if not fav:
            return
        print(f"\n  5. FAVORITE vs UNDERDOG")
        print(f"  {'─'*65}")
        print(f"  Favorites: {fav.get('favorite_accuracy', 0):.1%} accuracy ({fav.get('favorite_n', 0)} games)")
        print(f"  Underdogs:  {fav.get('underdog_accuracy', 0):.1%} accuracy ({fav.get('underdog_n', 0)} games)")

        buckets = fav.get("buckets", [])
        if buckets:
            print(f"\n  {'Bucket':<25} {'Accuracy':>10} {'Home Win%':>10} {'Games':>7}")
            print(f"  {'─'*55}")
            for b in buckets:
                print(f"  {b['bucket']:<25} {b['accuracy']:>10.1%} "
                      f"{b['actual_home_win_rate']:>10.1%} {b['n_games']:>7}")

    @staticmethod
    def _print_division(div: dict):
        if not div:
            return
        print(f"\n  6. DIVISION vs NON-DIVISION GAMES")
        print(f"  {'─'*65}")
        for label in ["division", "non_division"]:
            d = div.get(label, {})
            if d:
                print(f"  {label:<15} acc={d.get('accuracy', 0):.1%}  "
                      f"log_loss={d.get('log_loss', 0):.4f}  "
                      f"home_win_rate={d.get('home_win_rate', 0):.1%}  "
                      f"n={d.get('n_games', 0)}")

    @staticmethod
    def _print_rest(rest: dict):
        if not rest:
            return
        print(f"\n  7. REST / BACK-TO-BACK ANALYSIS")
        print(f"  {'─'*65}")

        for label in ["home_b2b", "away_b2b"]:
            r = rest.get(label, {})
            if r:
                side = "Home" if "home" in label else "Away"
                print(f"  {side} B2B:    acc={r.get('b2b_accuracy', 0):.1%}  "
                      f"home_win_rate={r.get('b2b_home_win_rate', 0):.1%}  "
                      f"n={r.get('b2b_n', 0)}")
                print(f"  {side} Rested: acc={r.get('rested_accuracy', 0):.1%}  "
                      f"home_win_rate={r.get('rested_home_win_rate', 0):.1%}  "
                      f"n={r.get('rested_n', 0)}")

        for label in ["home_rest_adv", "equal_rest", "away_rest_adv"]:
            r = rest.get(label, {})
            if r:
                print(f"  {label:<18} acc={r.get('accuracy', 0):.1%}  "
                      f"home_win_rate={r.get('home_win_rate', 0):.1%}  "
                      f"n={r.get('n_games', 0)}")

    @staticmethod
    def _print_agreement(agreement: dict):
        if not agreement:
            return
        print(f"\n  8. MODEL AGREEMENT")
        print(f"  {'─'*65}")
        for label, data in agreement.items():
            print(f"  {label}:")
            print(f"    Agree:    {data.get('agree_accuracy', 0):.1%} ({data.get('agree_n', 0)} games)")
            print(f"    Disagree: {data.get('disagree_accuracy', 0):.1%} ({data.get('disagree_n', 0)} games)")
            print(f"    Delta:    {data.get('delta', 0):+.1%}")

    @staticmethod
    def _print_drift(drift: dict):
        if not drift:
            return
        print(f"\n  9. TEMPORAL DRIFT")
        print(f"  {'─'*65}")

        if "warning" in drift:
            print(f"  {drift['warning']}")
            return

        print(f"  Trend: {drift.get('trend_label', '?')} ({drift.get('trend', 0):+.4f})")
        windows = drift.get("windows", [])
        if windows:
            print(f"\n  {'Period':<25} {'Blend Acc':>10} {'Blend LL':>10} {'ML Acc':>10} {'Games':>7}")
            print(f"  {'─'*65}")
            for w in windows:
                period = f"{w['date_start'][:7]}..{w['date_end'][:7]}"
                print(f"  {period:<25} {w.get('blended_accuracy', 0):>10.1%} "
                      f"{w.get('blended_log_loss', 0):>10.4f} "
                      f"{w.get('ml_accuracy', 0):>10.1%} {w['n_games']:>7}")

    @staticmethod
    def _print_totals(totals: dict):
        if not totals:
            return
        print(f"\n  10. TOTAL GOALS PREDICTION")
        print(f"  {'─'*65}")

        for model in ["ml", "poisson"]:
            t = totals.get(model, {})
            if not t:
                continue
            bias = "over" if t.get("mean_error", 0) > 0 else "under"
            print(f"  {model}: MAE={t.get('mae', 0):.3f}  bias={t.get('mean_error', 0):+.3f} "
                  f"({bias}-predicting)  std={t.get('std_error', 0):.3f}")

            for b in t.get("by_total_range", []):
                print(f"    {b['bucket']:<15} MAE={b['mae']:.3f}  bias={b['mean_error']:+.3f}  "
                      f"n={b['n_games']}")

        ou = totals.get("over_under_accuracy_5_5")
        if ou is not None:
            print(f"\n  Over/Under 5.5 accuracy (Poisson): {ou:.1%}")

    @staticmethod
    def _print_confidence(bands: list):
        if not bands:
            return
        print(f"\n  11. ACCURACY BY CONFIDENCE")
        print(f"  {'─'*65}")
        print(f"  {'Band':<25} {'Accuracy':>10} {'Home Win%':>10} {'% Total':>10} {'Games':>7}")
        print(f"  {'─'*65}")
        for b in bands:
            print(f"  {b['band']:<25} {b['accuracy']:>10.1%} "
                  f"{b['home_win_rate']:>10.1%} {b['pct_of_total']:>9.1f}% "
                  f"{b['n_games']:>7}")

    @staticmethod
    def _print_worst_teams(teams: list):
        if not teams or len(teams) < 6:
            return
        print(f"\n  12. HARDEST TO PREDICT (home team)")
        print(f"  {'─'*65}")
        worst = teams[:5]
        best = teams[-5:]
        print(f"  Worst:")
        for t in worst:
            print(f"    {t['group']:<5} acc={t['accuracy']:.1%}  log_loss={t['log_loss']:.4f}  n={t['n_games']}")
        print(f"  Best:")
        for t in reversed(best):
            print(f"    {t['group']:<5} acc={t['accuracy']:.1%}  log_loss={t['log_loss']:.4f}  n={t['n_games']}")


def _is_division_game(home: str, away: str) -> bool:
    return TEAM_TO_DIVISION.get(home, "") == TEAM_TO_DIVISION.get(away, "") and TEAM_TO_DIVISION.get(home, "") != ""
