#!/usr/bin/env python3
"""
NHL Match Predictor — Main Entry Point

Usage:
  python main.py predict [--date YYYY-MM-DD]   Predict today's / given date's games
  python main.py train [--seasons 2024,2025]    Train model on historical data
  python main.py evaluate                       Evaluate model performance
  python main.py elo                            Show current Elo rankings
  python main.py value [--date YYYY-MM-DD]      Find value bets
  python main.py shap                           Show SHAP feature importance
  python main.py tune [--trials 100]            Tune hyperparameters with Optuna
  python main.py calibrate [--seasons ...]       Calibration analysis on OOF predictions
  python main.py track                          Show prediction tracking report
  python main.py pipeline [--trials 150]        Full pipeline: tune → train → calibrate → sweetspot
"""

import sys
import os
import argparse
import logging
from datetime import date, timedelta

import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import cfg

# Data collectors
from data.collectors.nhl_api import (
    get_schedule, get_standings, get_team_game_log, get_roster,
)
from data.collectors.moneypuck import (
    get_team_xg_summary, get_goalie_stats, get_skater_stats,
    get_goalie_5v5, get_team_all_situations,
)
from data.collectors.odds_api import (
    get_consensus_odds, get_totals, get_moneyline,
)
from data.collectors.pinnacle import (
    get_pinnacle_consensus, get_pinnacle_totals, get_pinnacle_odds,
)
from data.collectors.dailyfaceoff import get_starting_goalies, get_injuries

# Models
from models.elo import EloSystem
from models.poisson import PoissonGoalModel, estimate_xg_from_features
from models.predictor import NHLPredictor

# Features
from features.engineer import FeatureEngineer

# Value
from value.detector import ValueDetector, format_value_report

# Tracking
from tracking.tracker import PredictionTracker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("nhl_predictor")


class NHLPredictionPipeline:
    """
    Orchestrates the full prediction pipeline:
      1. Collect data from all sources
      2. Build Elo ratings from season game log
      3. Engineer features for each matchup
      4. Run ML model + Poisson model
      5. Detect value bets
    """

    def __init__(self, engine: str = "auto"):
        self.elo = EloSystem()
        self.predictor = NHLPredictor(engine=engine)
        self.engineer = FeatureEngineer(elo_system=self.elo)
        self.value_detector = ValueDetector(predictor=self.predictor)
        self.tracker = PredictionTracker()

        # Cached data
        self._standings = None
        self._game_logs = {}         # {team: DataFrame}
        self._combined_log = None
        self._mp_team_data = None
        self._goalie_data = None
        self._skaters_df = None
        self._injuries_df = None

    # ── Data Loading ─────────────────────────────────────────────────

    def load_data(self, season: str = None):
        """Load all data sources for the current season."""
        season = season or cfg.current_season
        season_year = season[:4]

        logger.info("Loading data for season %s...", season)

        # 1. Standings
        logger.info("  Fetching standings...")
        self._standings = get_standings()
        if not self._standings.empty:
            logger.info("    Got standings for %d teams", len(self._standings))
        else:
            logger.warning("    Standings unavailable")

        # 2. Game logs for all teams
        logger.info("  Fetching team game logs...")
        teams = list(cfg.team_abbrevs.keys())
        if not self._standings.empty:
            teams = self._standings["team"].tolist()

        logs = []
        for team in teams:
            gl = get_team_game_log(team, season)
            if not gl.empty:
                self._game_logs[team] = gl
                logs.append(gl)

        if logs:
            self._combined_log = pd.concat(logs, ignore_index=True)
            self._combined_log["date"] = pd.to_datetime(self._combined_log["date"])
            self._combined_log = self._combined_log.sort_values("date").reset_index(drop=True)
            logger.info("    Got %d total game records", len(self._combined_log))
        else:
            self._combined_log = pd.DataFrame()
            logger.warning("    No game logs available")

        # 3. MoneyPuck advanced stats
        logger.info("  Fetching MoneyPuck team xG data...")
        self._mp_team_data = get_team_xg_summary(season_year)
        logger.info("    Got advanced stats for %d teams", len(self._mp_team_data))

        # 4. Goalie stats
        logger.info("  Fetching goalie stats...")
        goalie_df = get_goalie_5v5(season_year)
        self._goalie_data = {}
        if not goalie_df.empty:
            # Get starting goalies
            starters = get_starting_goalies()

            # Build goalie data per team: use best goalie by games played
            team_col = "team" if "team" in goalie_df.columns else goalie_df.columns[0]
            for team in teams:
                team_goalies = goalie_df[goalie_df[team_col] == team]
                if not team_goalies.empty:
                    gp_col = None
                    for c in ("games_played", "gamesPlayed", "GP"):
                        if c in team_goalies.columns:
                            gp_col = c
                            break
                    if gp_col:
                        best = team_goalies.sort_values(gp_col, ascending=False).iloc[0]
                    else:
                        best = team_goalies.iloc[0]
                    self._goalie_data[team] = best.to_dict()

            logger.info("    Got goalie data for %d teams", len(self._goalie_data))

        # 5. Skater stats
        logger.info("  Fetching skater stats...")
        self._skaters_df = get_skater_stats(season_year)
        if not self._skaters_df.empty:
            logger.info("    Got stats for %d skaters", len(self._skaters_df))

        # 6. Injuries
        logger.info("  Fetching injury data...")
        self._injuries_df = get_injuries()
        if not self._injuries_df.empty:
            logger.info("    Got %d injury records", len(self._injuries_df))

        # 7. Build Elo from game log
        if not self._combined_log.empty:
            logger.info("  Building Elo ratings...")
            self._build_elo()

    def _build_elo(self):
        """Build Elo ratings from the combined game log."""
        # Deduplicate: each game appears twice (once per team), we need unique games
        if self._combined_log.empty:
            return

        # Keep only home team rows to avoid double-counting
        home_games = self._combined_log[self._combined_log["is_home"]].copy()
        home_games = home_games.sort_values("date")

        for _, row in home_games.iterrows():
            self.elo.update(
                home=row["team"],
                away=row["opponent"],
                home_goals=int(row["goals_for"]),
                away_goals=int(row["goals_against"]),
                ot=bool(row.get("ot", False)),
            )

        logger.info("    Elo ratings built from %d games", len(home_games))

    # ── Prediction ───────────────────────────────────────────────────

    def predict_games(self, target_date: str = None) -> list[dict]:
        """
        Predict all games for a given date.
        Returns list of prediction dicts.
        """
        target_date = target_date or date.today().isoformat()

        logger.info("Predicting games for %s...", target_date)

        # Get schedule
        schedule = get_schedule(target_date)
        if schedule.empty:
            logger.warning("No games found for %s", target_date)
            return []

        # Filter to actual games (not completed)
        upcoming = schedule[schedule["status"].isin(["FUT", "PRE", ""])]
        if upcoming.empty:
            # Maybe all games are today — include all
            upcoming = schedule

        logger.info("Found %d games", len(upcoming))

        # Get odds — prefer Pinnacle (sharp, no API key needed), fallback to Odds API
        logger.info("Fetching Pinnacle odds...")
        consensus = get_pinnacle_consensus()
        totals_df = get_pinnacle_totals()

        if consensus.empty:
            logger.info("Pinnacle unavailable — falling back to Odds API")
            consensus = get_consensus_odds(target_date=target_date)
            totals_df = get_totals(target_date=target_date)
        else:
            logger.info("Using Pinnacle odds (%d games)", len(consensus))

        results = []

        for _, game in upcoming.iterrows():
            home = game["home_team"]
            away = game["away_team"]

            logger.info("  Predicting %s @ %s...", away, home)

            # Build features
            features = self.engineer.build_features(
                home_team=home,
                away_team=away,
                game_date=target_date,
                standings=self._standings,
                game_log=self._combined_log,
                mp_team_data=self._mp_team_data,
                goalie_data=self._goalie_data,
                skaters_df=self._skaters_df,
                injuries_df=self._injuries_df,
                consensus_odds=consensus,
                totals_df=totals_df,
            )

            # ML prediction
            ml_pred = self.predictor.predict(features)

            # Pass Elo prob through for meta-model blending
            ml_pred["elo_home_win_prob"] = features.get("elo_home_win_prob", 0.5)

            # Poisson prediction
            home_xg, away_xg = estimate_xg_from_features(features)
            poisson = PoissonGoalModel(home_xg, away_xg)
            total_line = features.get("market_total", 5.5)
            poisson_pred = poisson.full_prediction(total_line=total_line)

            # Pass context to enhanced meta-model blending
            ml_pred["meta_context"] = {
                "rest_diff": features.get("rest_diff", 0),
                "diff_momentum": features.get("diff_momentum", 0),
                "market_home_true_prob": features.get("market_home_true_prob", 0.5),
                "diff_win_pct": features.get("diff_win_pct", 0),
                "diff_goal_diff_pg": features.get("diff_goal_diff_pg", 0),
            }

            # Value analysis
            odds_fmt = features.get("odds_format", "american")
            market_data = {
                "best_home_odds": features.get("best_home_odds", 0),
                "best_away_odds": features.get("best_away_odds", 0),
                "market_total": total_line,
                "spread_odds": 1.909 if odds_fmt == "decimal" else -110,
                "odds_format": odds_fmt,
            }
            value = self.value_detector.full_analysis(
                ml_pred, poisson_pred, market_data, home, away,
            )

            prediction = {
                "game_id": game.get("game_id"),
                "date": target_date,
                "home_team": home,
                "away_team": away,
                "venue": game.get("venue", ""),
                # Probabilities
                "home_win_prob": value["blended_home_prob"],
                "away_win_prob": value["blended_away_prob"],
                "expected_total": value["blended_total"],
                "home_cover_prob": value["blended_cover_prob"],
                # Sub-model details
                "ml_home_prob": ml_pred["ml_home_win_prob"],
                "poisson_home_prob": poisson_pred["home_win_prob"],
                "elo_home_prob": features.get("elo_home_win_prob", 0.5),
                # Poisson details
                "poisson_home_xg": home_xg,
                "poisson_away_xg": away_xg,
                "poisson_over_prob": poisson_pred.get("over_prob", 0.5),
                "poisson_under_prob": poisson_pred.get("under_prob", 0.5),
                "most_likely_score": poisson_pred.get("most_likely_scores", [{}])[0],
                # Market odds (decimal format)
                "best_home_odds": features.get("best_home_odds", 0),
                "best_away_odds": features.get("best_away_odds", 0),
                "odds_format": odds_fmt,
                "reg_draw_prob": poisson_pred.get("regulation_draw_prob", 0),
                # Elo
                "home_elo": features.get("elo_home", cfg.elo_initial),
                "away_elo": features.get("elo_away", cfg.elo_initial),
                # Value bets
                "value_bets": value["value_bets"],
                "total_value_bets": value["total_value_bets"],
                # Key features
                "home_rest_days": features.get("home_rest_days", "?"),
                "away_rest_days": features.get("away_rest_days", "?"),
                "home_last5_win_pct": features.get("home_last5_win_pct", "?"),
                "away_last5_win_pct": features.get("away_last5_win_pct", "?"),
                "home_momentum": features.get("home_momentum", 0),
                "away_momentum": features.get("away_momentum", 0),
            }

            results.append(prediction)

        return results

    # ── Training ─────────────────────────────────────────────────────

    def train_model(
        self,
        seasons: list[str] = None,
        sample_weighting: bool = True,
        auto_select_top_n: bool = True,
    ):
        """
        Train the ML model on historical game data.
        Builds features for each past game and trains on outcomes.

        seasons: list of season strings (e.g. ["20232024", "20242025"])
                 Supports multi-season training for more data.
        """
        if seasons is None:
            seasons = cfg.training_seasons

        logger.info("Training model on seasons: %s (multi-season=%s)",
                     seasons, len(seasons) > 1)

        all_features = []
        all_results = []

        for season in seasons:
            season_year = season[:4]
            logger.info("Processing season %s...", season)

            # Load previous season data
            prev_standings = get_standings()  # current standings as proxy
            prev_mp = get_team_xg_summary(season_year)
            prev_goalies_df = get_goalie_5v5(season_year)
            prev_skaters = get_skater_stats(season_year)

            # Build goalie lookup
            goalie_lookup = {}
            if not prev_goalies_df.empty:
                team_col = "team" if "team" in prev_goalies_df.columns else prev_goalies_df.columns[0]
                for team in prev_goalies_df[team_col].unique():
                    tg = prev_goalies_df[prev_goalies_df[team_col] == team]
                    gp_col = None
                    for c in ("games_played", "gamesPlayed", "GP"):
                        if c in tg.columns:
                            gp_col = c
                            break
                    if gp_col and not tg.empty:
                        best = tg.sort_values(gp_col, ascending=False).iloc[0]
                        goalie_lookup[team] = best.to_dict()

            # Get game logs for all teams
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

            # Build Elo for this season
            train_elo = EloSystem()
            train_engineer = FeatureEngineer(elo_system=train_elo)

            # Process each unique game (home team perspective)
            home_games = combined[combined["is_home"]].copy()
            home_games = home_games.sort_values("date")

            logger.info("  Processing %d games for feature engineering...", len(home_games))

            for idx, row in home_games.iterrows():
                home = row["team"]
                away = row["opponent"]
                game_date = row["date"].strftime("%Y-%m-%d")

                # Build features using data available UP TO this game
                past_log = combined[combined["date"] < row["date"]]

                if len(past_log) < 50:  # skip early season, not enough data
                    # But still update Elo
                    train_elo.update(home, away, int(row["goals_for"]),
                                     int(row["goals_against"]), bool(row.get("ot", False)))
                    continue

                features = train_engineer.build_features(
                    home_team=home,
                    away_team=away,
                    game_date=game_date,
                    standings=prev_standings,
                    game_log=past_log,
                    mp_team_data=prev_mp,
                    goalie_data=goalie_lookup,
                    skaters_df=prev_skaters,
                )

                all_features.append(features)
                all_results.append({
                    "home_win": row["goals_for"] > row["goals_against"],
                    "total_goals": row["goals_for"] + row["goals_against"],
                    "home_covered": (row["goals_for"] - row["goals_against"]) >= 2,
                    "home_goals": row["goals_for"],
                    "away_goals": row["goals_against"],
                })

                # Update Elo after the game
                train_elo.update(home, away, int(row["goals_for"]),
                                 int(row["goals_against"]), bool(row.get("ot", False)))

        if not all_features:
            logger.error("No training data available")
            return

        features_df = pd.DataFrame(all_features)
        results_df = pd.DataFrame(all_results)

        logger.info("Training on %d games with %d features (seasons: %s)",
                     len(features_df), len(features_df.columns), ", ".join(seasons))

        # Train with new options
        self.predictor.train(
            features_df, results_df,
            sample_weighting=sample_weighting,
            auto_select_top_n=auto_select_top_n,
        )

        # Evaluate
        metrics = self.predictor.evaluate(features_df, results_df)
        logger.info("Training metrics: %s", metrics)

        # Cross-validate
        cv = self.predictor.cross_validate(features_df, results_df)
        logger.info("CV metrics: %s", cv)

        # Save
        self.predictor.save()
        logger.info("Model saved")

    def _prepare_training_data(self, seasons: list[str] = None):
        """Prepare training data and return (features_df, results_df). Used by tune."""
        if seasons is None:
            seasons = [cfg.previous_season]

        all_features = []
        all_results = []

        for season in seasons:
            season_year = season[:4]
            logger.info("Preparing data for season %s...", season)

            prev_standings = get_standings()
            prev_mp = get_team_xg_summary(season_year)
            prev_goalies_df = get_goalie_5v5(season_year)
            prev_skaters = get_skater_stats(season_year)

            goalie_lookup = {}
            if not prev_goalies_df.empty:
                team_col = "team" if "team" in prev_goalies_df.columns else prev_goalies_df.columns[0]
                for team in prev_goalies_df[team_col].unique():
                    tg = prev_goalies_df[prev_goalies_df[team_col] == team]
                    gp_col = None
                    for c in ("games_played", "gamesPlayed", "GP"):
                        if c in tg.columns:
                            gp_col = c
                            break
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
                continue

            combined = pd.concat(logs, ignore_index=True)
            combined["date"] = pd.to_datetime(combined["date"])
            combined = combined.sort_values("date").reset_index(drop=True)

            train_elo = EloSystem()
            train_engineer = FeatureEngineer(elo_system=train_elo)

            home_games = combined[combined["is_home"]].copy()
            home_games = home_games.sort_values("date")

            for idx, row in home_games.iterrows():
                home = row["team"]
                away = row["opponent"]
                game_date = row["date"].strftime("%Y-%m-%d")
                past_log = combined[combined["date"] < row["date"]]

                if len(past_log) < 50:
                    train_elo.update(home, away, int(row["goals_for"]),
                                     int(row["goals_against"]), bool(row.get("ot", False)))
                    continue

                features = train_engineer.build_features(
                    home_team=home, away_team=away, game_date=game_date,
                    standings=prev_standings, game_log=past_log,
                    mp_team_data=prev_mp, goalie_data=goalie_lookup,
                    skaters_df=prev_skaters,
                )
                all_features.append(features)
                all_results.append({
                    "home_win": row["goals_for"] > row["goals_against"],
                    "total_goals": row["goals_for"] + row["goals_against"],
                    "home_covered": (row["goals_for"] - row["goals_against"]) >= 2,
                })

                train_elo.update(home, away, int(row["goals_for"]),
                                 int(row["goals_against"]), bool(row.get("ot", False)))

        if not all_features:
            return None, None

        return pd.DataFrame(all_features), pd.DataFrame(all_results)

    # ── Output ───────────────────────────────────────────────────────

    def print_predictions(self, predictions: list[dict]):
        """Pretty-print predictions to console."""
        if not predictions:
            print("\nNo games to predict.\n")
            return

        print(f"\n{'='*70}")
        print(f"  NHL PREDICTIONS — {predictions[0].get('date', 'Today')}")
        print(f"{'='*70}\n")

        for pred in predictions:
            home = pred["home_team"]
            away = pred["away_team"]
            hwp = pred["home_win_prob"]
            awp = pred["away_win_prob"]

            fav = home if hwp > awp else away
            fav_prob = max(hwp, awp)

            print(f"  {away} @ {home}")
            print(f"  {'─'*50}")
            print(f"    Prediction:  {fav} ({fav_prob:.1%})")
            print(f"    Home:  {hwp:.1%}  |  Away:  {awp:.1%}")

            # Market odds (P1 / X / P2) — decimal format
            ho = pred.get("best_home_odds", 0)
            ao = pred.get("best_away_odds", 0)
            draw_p = pred.get("reg_draw_prob", 0)
            odds_fmt = pred.get("odds_format", "decimal")
            if ho and ao:
                if odds_fmt == "decimal":
                    print(f"    Odds:  P1 {ho:.2f}  |  X {draw_p:.1%}  |  P2 {ao:.2f}")
                else:
                    print(f"    Odds:  P1 {ho:+d}  |  X {draw_p:.1%}  |  P2 {ao:+d}")
            elif draw_p:
                print(f"    Reg Draw:  {draw_p:.1%}")

            print(f"    Expected Total:  {pred['expected_total']:.1f}")
            print(f"    Poisson xG:  {pred['poisson_home_xg']:.2f} — {pred['poisson_away_xg']:.2f}")
            print(f"    Elo:  {pred['home_elo']:.0f} vs {pred['away_elo']:.0f}")

            mls = pred.get("most_likely_score", {})
            if mls:
                print(f"    Most likely score:  {mls.get('home_goals', '?')}-{mls.get('away_goals', '?')} "
                      f"({mls.get('probability', 0):.1%})")

            print(f"    Rest: Home {pred['home_rest_days']}d / Away {pred['away_rest_days']}d")
            print(f"    Form L5: Home {pred.get('home_last5_win_pct', '?')}"
                  f" / Away {pred.get('away_last5_win_pct', '?')}")

            # Value bets
            vbs = pred.get("value_bets", [])
            if vbs:
                print(f"\n    VALUE BETS ({len(vbs)}):")
                for vb in vbs:
                    side_str = vb.get("team", vb.get("side", ""))
                    print(f"      [{vb['confidence']}] {vb['market'].upper()} {vb['side'].upper()}"
                          f" — Edge: {vb['edge_pct']:.1f}% | EV: {vb['expected_value']:.1f}%"
                          f" | Kelly: {vb['kelly_size']:.2%}")

            print()

        print(f"{'='*70}\n")

    def print_elo_rankings(self):
        """Print Elo rankings."""
        rankings = self.elo.get_rankings()
        if rankings.empty:
            print("No Elo ratings available. Load data first.\n")
            return

        print(f"\n{'='*60}")
        print(f"  NHL ELO RANKINGS")
        print(f"{'='*60}\n")
        print(f"  {'Rank':<5} {'Team':<5} {'Elo':>7} {'Off':>7} {'Def':>7} {'Adj':>7}")
        print(f"  {'─'*45}")

        for i, (_, row) in enumerate(rankings.iterrows(), 1):
            print(f"  {i:<5} {row['team']:<5} {row['elo']:>7.1f} "
                  f"{row['off_elo']:>7.1f} {row['def_elo']:>7.1f} "
                  f"{row['adjusted_elo']:>7.1f}")

        print()


# ── CLI ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="NHL Match Predictor")
    subparsers = parser.add_subparsers(dest="command")

    # predict
    pred_parser = subparsers.add_parser("predict", help="Predict games")
    pred_parser.add_argument("--date", type=str, default=None, help="Date YYYY-MM-DD")
    pred_parser.add_argument("--min-confidence", type=float, default=None,
                             help="Only show picks with confidence >= this %% (e.g. 60)")

    # train
    train_parser = subparsers.add_parser("train", help="Train model")
    train_parser.add_argument("--seasons", type=str, default=None,
                              help="Comma-separated seasons (e.g. 20232024,20242025)")
    train_parser.add_argument("--engine", type=str, default="auto",
                              choices=["auto", "xgboost", "lightgbm", "catboost"],
                              help="ML engine (default: auto)")
    train_parser.add_argument("--no-weighting", action="store_true",
                              help="Disable sample weighting by recency")
    train_parser.add_argument("--no-auto-topn", action="store_true",
                              help="Disable automatic top_n_features optimization")

    # evaluate
    subparsers.add_parser("evaluate", help="Evaluate model")

    # elo
    subparsers.add_parser("elo", help="Show Elo rankings")

    # value
    val_parser = subparsers.add_parser("value", help="Find value bets")
    val_parser.add_argument("--date", type=str, default=None, help="Date YYYY-MM-DD")

    # shap
    subparsers.add_parser("shap", help="Show SHAP feature importance")

    # tune
    tune_parser = subparsers.add_parser("tune", help="Tune hyperparameters with Optuna")
    tune_parser.add_argument("--trials", type=int, default=100,
                             help="Number of Optuna trials (default: 100)")
    tune_parser.add_argument("--seasons", type=str, default=None,
                             help="Comma-separated seasons for tuning data")
    tune_parser.add_argument("--engine", type=str, default="auto",
                             choices=["auto", "xgboost", "lightgbm", "catboost"],
                             help="ML engine for tuning (default: auto)")

    # calibrate
    cal_parser = subparsers.add_parser("calibrate", help="Calibration analysis on OOF predictions")
    cal_parser.add_argument("--seasons", type=str, default=None,
                            help="Comma-separated seasons for calibration data")

    # sweetspot
    ss_parser = subparsers.add_parser("sweetspot",
        help="Find conditions where model accuracy reaches 75%%+")
    ss_parser.add_argument("--seasons", type=str, default=None,
                           help="Comma-separated seasons for analysis")
    ss_parser.add_argument("--target", type=float, default=0.75,
                           help="Target accuracy (default: 0.75)")
    ss_parser.add_argument("--min-sample", type=int, default=30,
                           help="Minimum sample size per condition (default: 30)")

    # track
    subparsers.add_parser("track", help="Show prediction tracking report")

    # pipeline — full training pipeline: tune → train → calibrate → sweetspot
    pipe_parser = subparsers.add_parser("pipeline",
        help="Run full pipeline: tune → train → calibrate → sweetspot")
    pipe_parser.add_argument("--trials", type=int, default=150,
                             help="Number of Optuna trials (default: 150)")
    pipe_parser.add_argument("--seasons", type=str, default=None,
                             help="Comma-separated seasons (e.g. 20232024,20242025)")
    pipe_parser.add_argument("--engine", type=str, default="auto",
                             choices=["auto", "xgboost", "lightgbm", "catboost"],
                             help="ML engine (default: auto)")
    pipe_parser.add_argument("--skip-tune", action="store_true",
                             help="Skip tuning (use existing best_params.json)")
    pipe_parser.add_argument("--skip-calibrate", action="store_true",
                             help="Skip calibration and sweetspot analysis")
    pipe_parser.add_argument("--no-weighting", action="store_true",
                             help="Disable sample weighting by recency")
    pipe_parser.add_argument("--no-auto-topn", action="store_true",
                             help="Disable automatic top_n_features optimization")

    args = parser.parse_args()

    engine = getattr(args, "engine", "auto")
    pipeline = NHLPredictionPipeline(engine=engine)

    if args.command == "train":
        seasons = args.seasons.split(",") if args.seasons else None
        # Override engine if specified
        if hasattr(args, "engine") and args.engine != "auto":
            pipeline.predictor.engine = pipeline.predictor._resolve_engine(args.engine)
        pipeline.train_model(
            seasons,
            sample_weighting=not getattr(args, "no_weighting", False),
            auto_select_top_n=not getattr(args, "no_auto_topn", False),
        )

    elif args.command == "predict":
        pipeline.load_data()

        # Try to load pre-trained model
        try:
            pipeline.predictor.load()
            logger.info("Loaded pre-trained model")
        except FileNotFoundError:
            logger.info("No pre-trained model found — predictions will use Elo + Poisson only")

        predictions = pipeline.predict_games(args.date)

        # Filter by minimum confidence if specified
        min_conf = args.min_confidence
        if min_conf is not None:
            threshold = min_conf / 100.0 if min_conf > 1 else min_conf
            filtered = [p for p in predictions
                        if max(p["home_win_prob"], p["away_win_prob"]) >= threshold]
            logger.info("Confidence filter >= %.0f%%: %d/%d games pass",
                        threshold * 100, len(filtered), len(predictions))
            predictions = filtered

        pipeline.print_predictions(predictions)

        # Log predictions for tracking
        pipeline.tracker.log_predictions(predictions)

    elif args.command == "elo":
        pipeline.load_data()
        pipeline.print_elo_rankings()

    elif args.command == "value":
        pipeline.load_data()

        try:
            pipeline.predictor.load()
        except FileNotFoundError:
            logger.info("No pre-trained model — using Elo + Poisson")

        predictions = pipeline.predict_games(args.date)

        # Print detailed value report
        for pred in predictions:
            if pred.get("value_bets"):
                analysis = {
                    "blended_home_prob": pred["home_win_prob"],
                    "blended_away_prob": pred["away_win_prob"],
                    "blended_total": pred["expected_total"],
                    "blended_cover_prob": pred["home_cover_prob"],
                    "value_bets": pred["value_bets"],
                    "total_value_bets": pred["total_value_bets"],
                    "best_edge": pred["value_bets"][0] if pred["value_bets"] else None,
                }
                report = format_value_report(analysis, pred["home_team"], pred["away_team"])
                print(report)

        # Summary
        total_vb = sum(p.get("total_value_bets", 0) for p in predictions)
        print(f"\nTotal value bets found: {total_vb} across {len(predictions)} games\n")

        # Log predictions for tracking
        pipeline.tracker.log_predictions(predictions)

    elif args.command == "shap":
        # Load model and show SHAP report
        try:
            pipeline.predictor.load()
            report = pipeline.predictor.get_shap_report()
            print(report)
        except FileNotFoundError:
            logger.error("No trained model. Run 'train' first.")

    elif args.command == "tune":
        from models.tuner import HyperparamTuner
        from models.predictor import _prepare_features

        seasons = args.seasons.split(",") if args.seasons else None
        tune_engine = getattr(args, "engine", "auto")
        logger.info("Preparing training data for hyperparameter tuning (engine=%s)...", tune_engine)

        features_df, results_df = pipeline._prepare_training_data(seasons)
        if features_df is None:
            logger.error("No training data available for tuning")
            return

        X, feature_names = _prepare_features(features_df)
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_names)

        y_win = results_df["home_win"].astype(int).values
        y_total = results_df["total_goals"].values

        tuner = HyperparamTuner(engine=tune_engine)

        # Tune classifier (with sample weighting)
        clf_params = tuner.tune_classifier(X_scaled, y_win, n_trials=args.trials,
                                           sample_weighting=True)

        # Tune regressor (with sample weighting)
        reg_params = tuner.tune_regressor(X_scaled, y_total,
                                          n_trials=max(args.trials // 2, 20),
                                          sample_weighting=True)

        # Save params
        tuner.save_params(clf_params, reg_params)

        print(f"\nBest classifier params: {clf_params}")
        print(f"Best regressor params:  {reg_params}")
        print(f"\nParams saved. Run 'train' to retrain with optimized parameters.\n")

    elif args.command == "track":
        logger.info("Updating prediction results...")
        updated = pipeline.tracker.update_results()
        if updated:
            logger.info("Updated %d game results", updated)
        pipeline.tracker.print_report()

    elif args.command == "calibrate":
        seasons = args.seasons.split(",") if args.seasons else None
        logger.info("Preparing data for calibration analysis...")

        features_df, results_df = pipeline._prepare_training_data(seasons)
        if features_df is None:
            logger.error("No training data available for calibration")
            return

        logger.info("Running OOF calibration analysis on %d games...", len(features_df))
        cal = pipeline.predictor.calibration_analysis(features_df, results_df)

        # Print report
        print(f"\n{'='*65}")
        print("  CALIBRATION ANALYSIS (Out-of-Fold Predictions)")
        print(f"{'='*65}\n")

        print(f"  OOF Samples:   {cal['total_oof_samples']}")
        print(f"  OOF Accuracy:  {cal['oof_accuracy']:.1%}")
        print(f"  OOF Log Loss:  {cal['oof_log_loss']:.4f}")
        print(f"  OOF Brier:     {cal['oof_brier']:.4f}")

        print(f"\n  ECE (Expected Calibration Error):  {cal['ece']:.4f}")
        print(f"  MCE (Maximum Calibration Error):   {cal['mce']:.4f}")

        if cal['overconfidence'] > 0.02:
            print(f"  Overconfidence (high prob bins):    +{cal['overconfidence']:.1%}")
        if cal['underconfidence'] > 0.02:
            print(f"  Underconfidence (low prob bins):    +{cal['underconfidence']:.1%}")

        print(f"\n  RELIABILITY TABLE")
        print(f"  {'─'*60}")
        print(f"  {'Bin':^12} {'Predicted':>10} {'Actual':>10} {'Gap':>8} {'N':>6}  {'Status'}")
        print(f"  {'─'*60}")

        for b in cal['calibration_bins']:
            if b['count'] == 0:
                continue
            label = f"{b['bin_low']:.0%}-{b['bin_high']:.0%}"
            pred = b['predicted_mean']
            actual = b['actual_mean']
            if np.isnan(actual):
                print(f"  {label:^12} {pred:>10.1%} {'N/A':>10} {'':>8} {b['count']:>6}")
            else:
                gap = pred - actual
                status = ""
                if abs(gap) > 0.05:
                    status = "OVERCONF" if gap > 0 else "UNDERCONF"
                elif abs(gap) <= 0.02:
                    status = "OK"
                else:
                    status = "~ok"
                print(f"  {label:^12} {pred:>10.1%} {actual:>10.1%} {gap:>+8.1%} {b['count']:>6}  {status}")

        # Recommendation
        print(f"\n  RECOMMENDATION FOR VALUE BETTING")
        print(f"  {'─'*60}")
        ece = cal['ece']
        overconf = cal['overconfidence']

        if ece < 0.03:
            print("  Calibration: EXCELLENT (ECE < 3%)")
            print("  -> Safe to use min_edge 3% threshold")
        elif ece < 0.05:
            print("  Calibration: GOOD (ECE < 5%)")
            print("  -> Recommended min_edge 5% to compensate for calibration noise")
        elif ece < 0.08:
            print("  Calibration: FAIR (ECE < 8%)")
            print("  -> Recommended min_edge 6-8% — model probabilities have notable gaps")
        else:
            print("  Calibration: POOR (ECE >= 8%)")
            print("  -> NOT recommended for value betting — retrain with isotonic calibration")

        if overconf > 0.05:
            print(f"  WARNING: Model is overconfident by ~{overconf:.1%} in high-prob bins")
            print("  -> Value detector may find FALSE edges on favorites")

        print(f"\n{'='*65}\n")

    elif args.command == "sweetspot":
        from models.predictor import _prepare_features

        seasons = args.seasons.split(",") if args.seasons else None
        target = args.target
        min_sample = args.min_sample

        logger.info("Preparing data for sweet spot analysis (target=%.0f%%)...", target * 100)

        features_df, results_df = pipeline._prepare_training_data(seasons)
        if features_df is None:
            logger.error("No training data available")
            return

        logger.info("Analyzing %d games for conditions with %.0f%%+ accuracy...",
                     len(features_df), target * 100)

        analysis = pipeline.predictor.sweetspot_analysis(
            features_df, results_df,
            target_accuracy=target,
            min_sample=min_sample,
        )

        # ── Print report ─────────────────────────────────────────────
        print(f"\n{'='*70}")
        print(f"  SWEET SPOT ANALYSIS — When does the model hit {target:.0%}+ accuracy?")
        print(f"{'='*70}")
        print(f"\n  Overall OOF Accuracy: {analysis['overall_accuracy']:.1%}  ({analysis['total_games']} games)\n")

        # 1. Confidence thresholds
        print(f"  {'─'*65}")
        print(f"  1. CONFIDENCE THRESHOLDS")
        print(f"  {'─'*65}")
        print(f"  {'Threshold':>12} {'Accuracy':>10} {'Games':>8} {'% Total':>10} {'Target?':>10}")
        print(f"  {'─'*65}")
        for cr in analysis["confidence_analysis"]:
            marker = " <<<" if cr["above_target"] else ""
            thr_label = f">= {cr['threshold']:.0%}"
            target_hit = "YES" if cr["above_target"] else "no"
            print(f"  {thr_label:>12} "
                  f"{cr['accuracy']:>10.1%} {cr['n_games']:>8} "
                  f"{cr['pct_of_total']:>9.1f}% "
                  f"{target_hit:>10}{marker}")

        # 2. Model agreement
        if analysis["model_agreement"]:
            print(f"\n  {'─'*65}")
            print(f"  2. MODEL AGREEMENT (when models agree)")
            print(f"  {'─'*65}")
            for ma in analysis["model_agreement"]:
                print(f"  {ma['condition']}:")
                print(f"    Agree:    {ma['accuracy_agree']:.1%} accuracy  ({ma['n_agree']} games)")
                print(f"    Disagree: {ma['accuracy_disagree']:.1%} accuracy  ({ma['n_disagree']} games)")

        # 3. Feature filters
        if analysis["feature_filters"]:
            print(f"\n  {'─'*65}")
            print(f"  3. SINGLE FEATURE FILTERS (sorted by accuracy)")
            print(f"  {'─'*65}")
            print(f"  {'Feature':<30} {'Condition':<18} {'Accuracy':>9} {'Games':>7} {'Impr':>7}")
            print(f"  {'─'*65}")
            shown = set()
            for ff in analysis["feature_filters"][:15]:
                key = ff["feature"]
                if key in shown:
                    continue
                shown.add(key)
                marker = " <<<" if ff["above_target"] else ""
                print(f"  {ff['feature']:<30} {ff['condition']:<18} "
                      f"{ff['accuracy']:>8.1%} {ff['n_games']:>7} "
                      f"{ff['improvement']:>+6.1%}{marker}")

        # 4. Combined rules
        if analysis["combined_rules"]:
            print(f"\n  {'─'*65}")
            print(f"  4. COMBINED RULES (confidence + feature)")
            print(f"  {'─'*65}")
            for i, cr in enumerate(analysis["combined_rules"][:10], 1):
                print(f"  #{i}: {cr['accuracy']:.1%} accuracy ({cr['n_games']} games, "
                      f"{cr['pct_of_total']:.1f}% of total)")
                print(f"      Rule: {cr['rule']}")

        # 5. Decision tree rules
        if analysis["tree_rules"]:
            print(f"\n  {'─'*65}")
            print(f"  5. AUTO-DISCOVERED RULES (decision tree)")
            print(f"  {'─'*65}")
            for i, tr in enumerate(analysis["tree_rules"][:8], 1):
                print(f"  #{i}: {tr['accuracy']:.1%} accuracy ({tr['n_games']} games)")
                print(f"      IF {tr['rule_text']}")

        # Summary
        print(f"\n  {'='*65}")
        print(f"  SUMMARY")
        print(f"  {'='*65}")

        # Find best achievable accuracy with reasonable sample size
        best_conf = max(analysis["confidence_analysis"],
                        key=lambda x: x["accuracy"]) if analysis["confidence_analysis"] else None
        best_combo = analysis["combined_rules"][0] if analysis["combined_rules"] else None

        if best_conf:
            print(f"  Best via confidence alone: {best_conf['accuracy']:.1%} "
                  f"(threshold {best_conf['threshold']:.0%}, {best_conf['n_games']} games)")

        if best_combo:
            print(f"  Best combined rule:        {best_combo['accuracy']:.1%} "
                  f"({best_combo['n_games']} games)")

        # Can we reach the target?
        target_reached = any(cr["above_target"] for cr in analysis["confidence_analysis"])
        combo_reached = any(cr["accuracy"] >= target for cr in analysis.get("combined_rules", []))
        tree_reached = any(tr["accuracy"] >= target for tr in analysis.get("tree_rules", []))

        if target_reached or combo_reached or tree_reached:
            print(f"\n  {target:.0%} accuracy IS achievable under specific conditions.")
            if target_reached:
                hits = [cr for cr in analysis["confidence_analysis"] if cr["above_target"]]
                best_hit = max(hits, key=lambda x: x["n_games"])
                print(f"  -> Best: confidence >= {best_hit['threshold']:.0%} → "
                      f"{best_hit['accuracy']:.1%} on {best_hit['n_games']} games "
                      f"({best_hit['pct_of_total']:.0f}% of all games)")
        else:
            # What's the best we can do?
            all_accs = [cr["accuracy"] for cr in analysis["confidence_analysis"]]
            if analysis["combined_rules"]:
                all_accs += [cr["accuracy"] for cr in analysis["combined_rules"]]
            best_possible = max(all_accs) if all_accs else analysis["overall_accuracy"]
            print(f"\n  {target:.0%} accuracy NOT reliably achievable with current model.")
            print(f"  Best achievable: {best_possible:.1%}")
            gap = target - best_possible
            print(f"  Gap to target: {gap:+.1%}")

        print(f"\n{'='*70}\n")

    elif args.command == "evaluate":
        pipeline.load_data()
        try:
            pipeline.predictor.load()
        except FileNotFoundError:
            logger.error("No trained model to evaluate. Run 'train' first.")
            return
        logger.info("Run training with --seasons to get evaluation metrics.")

    elif args.command == "pipeline":
        import time
        from models.tuner import HyperparamTuner
        from models.predictor import _prepare_features

        seasons = args.seasons.split(",") if args.seasons else None
        pipe_engine = getattr(args, "engine", "auto")
        total_start = time.time()

        stages = []
        if not args.skip_tune:
            stages.append("tune")
        stages.append("train")
        if not args.skip_calibrate:
            stages += ["calibrate", "sweetspot"]

        print(f"\n{'='*65}")
        print(f"  FULL PIPELINE: {' → '.join(stages)}")
        print(f"{'='*65}\n")

        # ── Stage 1: Tune ──────────────────────────────────────────
        if not args.skip_tune:
            stage_start = time.time()
            print(f"  [1/{len(stages)}] TUNING HYPERPARAMETERS ({args.trials} trials)...")
            print(f"  {'─'*60}")

            features_df, results_df = pipeline._prepare_training_data(seasons)
            if features_df is None:
                logger.error("No training data available — aborting pipeline")
                return

            X, feature_names = _prepare_features(features_df)
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_names)

            y_win = results_df["home_win"].astype(int).values
            y_total = results_df["total_goals"].values

            tuner = HyperparamTuner(engine=pipe_engine)
            clf_params = tuner.tune_classifier(X_scaled, y_win, n_trials=args.trials,
                                               sample_weighting=True)
            reg_params = tuner.tune_regressor(X_scaled, y_total,
                                              n_trials=max(args.trials // 2, 20),
                                              sample_weighting=True)
            tuner.save_params(clf_params, reg_params)

            elapsed = time.time() - stage_start
            print(f"  Tune completed in {elapsed:.0f}s")
            print(f"  Best classifier log_loss params saved")
            print(f"  Best regressor MAE params saved\n")
        else:
            print(f"  [skip] Tuning skipped — using existing best_params.json\n")

        # ── Stage 2: Train ─────────────────────────────────────────
        stage_idx = 2 if not args.skip_tune else 1
        stage_start = time.time()
        print(f"  [{stage_idx}/{len(stages)}] TRAINING MODEL...")
        print(f"  {'─'*60}")

        if hasattr(args, "engine") and args.engine != "auto":
            pipeline.predictor.engine = pipeline.predictor._resolve_engine(args.engine)

        pipeline.train_model(
            seasons,
            sample_weighting=not getattr(args, "no_weighting", False),
            auto_select_top_n=not getattr(args, "no_auto_topn", False),
        )

        elapsed = time.time() - stage_start
        print(f"  Train completed in {elapsed:.0f}s\n")

        # ── Stage 3: Calibrate ─────────────────────────────────────
        if not args.skip_calibrate:
            stage_idx += 1
            stage_start = time.time()
            print(f"  [{stage_idx}/{len(stages)}] CALIBRATION ANALYSIS...")
            print(f"  {'─'*60}")

            features_df, results_df = pipeline._prepare_training_data(seasons)
            if features_df is not None:
                cal = pipeline.predictor.calibration_analysis(features_df, results_df)
                print(f"  OOF Accuracy:  {cal['oof_accuracy']:.1%}")
                print(f"  OOF Log Loss:  {cal['oof_log_loss']:.4f}")
                print(f"  OOF Brier:     {cal['oof_brier']:.4f}")
                print(f"  ECE:           {cal['ece']:.4f}")
                if cal['overconfidence'] > 0.02:
                    print(f"  WARNING: Overconfident by {cal['overconfidence']:.1%}")
            else:
                logger.warning("Could not run calibration — no data")

            elapsed = time.time() - stage_start
            print(f"  Calibrate completed in {elapsed:.0f}s\n")

            # ── Stage 4: Sweetspot ─────────────────────────────────
            stage_idx += 1
            stage_start = time.time()
            print(f"  [{stage_idx}/{len(stages)}] SWEETSPOT ANALYSIS...")
            print(f"  {'─'*60}")

            if features_df is not None:
                analysis = pipeline.predictor.sweetspot_analysis(
                    features_df, results_df, target_accuracy=0.75, min_sample=30)
                print(f"  Overall OOF Accuracy: {analysis['overall_accuracy']:.1%} "
                      f"({analysis['total_games']} games)")

                best_conf = max(analysis["confidence_analysis"],
                                key=lambda x: x["accuracy"]) if analysis["confidence_analysis"] else None
                if best_conf:
                    print(f"  Best via confidence: {best_conf['accuracy']:.1%} "
                          f"(>= {best_conf['threshold']:.0%}, {best_conf['n_games']} games)")

                best_combo = analysis["combined_rules"][0] if analysis["combined_rules"] else None
                if best_combo:
                    print(f"  Best combined rule:  {best_combo['accuracy']:.1%} "
                          f"({best_combo['n_games']} games)")
            else:
                logger.warning("Could not run sweetspot — no data")

            elapsed = time.time() - stage_start
            print(f"  Sweetspot completed in {elapsed:.0f}s\n")

        # ── Summary ────────────────────────────────────────────────
        total_elapsed = time.time() - total_start
        minutes = int(total_elapsed // 60)
        seconds = int(total_elapsed % 60)
        print(f"{'='*65}")
        print(f"  PIPELINE COMPLETE — {minutes}m {seconds}s total")
        print(f"  Stages: {' → '.join(stages)}")
        print(f"{'='*65}\n")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
