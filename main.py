#!/usr/bin/env python3
"""
NHL Match Predictor — Main Entry Point

Usage:
  python main.py predict [--date YYYY-MM-DD]   Predict today's / given date's games
  python main.py train [--seasons 2024,2025]    Train model on historical data
  python main.py evaluate                       Evaluate model performance
  python main.py elo                            Show current Elo rankings
  python main.py value [--date YYYY-MM-DD]      Find value bets
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
from data.collectors.dailyfaceoff import get_starting_goalies, get_injuries

# Models
from models.elo import EloSystem
from models.poisson import PoissonGoalModel, estimate_xg_from_features
from models.predictor import NHLPredictor

# Features
from features.engineer import FeatureEngineer

# Value
from value.detector import ValueDetector, format_value_report

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

    def __init__(self):
        self.elo = EloSystem()
        self.predictor = NHLPredictor()
        self.engineer = FeatureEngineer(elo_system=self.elo)
        self.value_detector = ValueDetector()

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

        # Get odds
        consensus = get_consensus_odds()
        totals_df = get_totals()

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

            # Poisson prediction
            home_xg, away_xg = estimate_xg_from_features(features)
            poisson = PoissonGoalModel(home_xg, away_xg)
            total_line = features.get("market_total", 5.5)
            poisson_pred = poisson.full_prediction(total_line=total_line)

            # Value analysis
            market_data = {
                "best_home_odds": features.get("best_home_odds", 0),
                "best_away_odds": features.get("best_away_odds", 0),
                "market_total": total_line,
                "spread_odds": -110,
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
                # Market odds
                "best_home_odds": features.get("best_home_odds", 0),
                "best_away_odds": features.get("best_away_odds", 0),
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

    def train_model(self, seasons: list[str] = None):
        """
        Train the ML model on historical game data.
        Builds features for each past game and trains on outcomes.
        """
        if seasons is None:
            seasons = [cfg.previous_season]

        logger.info("Training model on seasons: %s", seasons)

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

        logger.info("Training on %d games with %d features", len(features_df), len(features_df.columns))

        # Train
        self.predictor.train(features_df, results_df)

        # Evaluate
        metrics = self.predictor.evaluate(features_df, results_df)
        logger.info("Training metrics: %s", metrics)

        # Cross-validate
        cv = self.predictor.cross_validate(features_df, results_df)
        logger.info("CV metrics: %s", cv)

        # Save
        self.predictor.save()
        logger.info("Model saved")

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

            # Market odds (P1 / X / P2)
            ho = pred.get("best_home_odds", 0)
            ao = pred.get("best_away_odds", 0)
            draw_p = pred.get("reg_draw_prob", 0)
            if ho and ao:
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

    # train
    train_parser = subparsers.add_parser("train", help="Train model")
    train_parser.add_argument("--seasons", type=str, default=None,
                              help="Comma-separated seasons (e.g. 20242025)")

    # evaluate
    subparsers.add_parser("evaluate", help="Evaluate model")

    # elo
    subparsers.add_parser("elo", help="Show Elo rankings")

    # value
    val_parser = subparsers.add_parser("value", help="Find value bets")
    val_parser.add_argument("--date", type=str, default=None, help="Date YYYY-MM-DD")

    args = parser.parse_args()

    pipeline = NHLPredictionPipeline()

    if args.command == "train":
        seasons = args.seasons.split(",") if args.seasons else None
        pipeline.train_model(seasons)

    elif args.command == "predict":
        pipeline.load_data()

        # Try to load pre-trained model
        try:
            pipeline.predictor.load()
            logger.info("Loaded pre-trained model")
        except FileNotFoundError:
            logger.info("No pre-trained model found — predictions will use Elo + Poisson only")

        predictions = pipeline.predict_games(args.date)
        pipeline.print_predictions(predictions)

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

    elif args.command == "evaluate":
        pipeline.load_data()
        try:
            pipeline.predictor.load()
        except FileNotFoundError:
            logger.error("No trained model to evaluate. Run 'train' first.")
            return
        logger.info("Run training with --seasons to get evaluation metrics.")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
