"""
Prediction Tracker.

Logs every prediction with timestamp and compares to actual outcomes.
Tracks accuracy, log loss, calibration, and value bet ROI over time.
"""

import os
import json
import logging
from datetime import datetime

import numpy as np
import pandas as pd

from config import cfg
from data.collectors.nhl_api import get_schedule

logger = logging.getLogger("nhl_predictor.tracking")

PREDICTIONS_LOG = os.path.join(cfg.data_dir, "predictions_log.csv")

COLUMNS = [
    "timestamp", "date", "home_team", "away_team",
    "home_win_prob", "away_win_prob", "expected_total",
    "best_home_odds", "best_away_odds",
    "ml_home_prob", "poisson_home_prob", "elo_home_prob",
    "value_bets_json",
    # Filled after game completes
    "actual_home_goals", "actual_away_goals", "actual_home_win",
    "actual_total", "result_updated",
]


class PredictionTracker:
    """Log predictions and track accuracy against actual results."""

    def __init__(self, log_path: str = None):
        self.log_path = log_path or PREDICTIONS_LOG

    def log_predictions(self, predictions: list[dict]):
        """Append predictions to the CSV log."""
        if not predictions:
            return

        rows = []
        ts = datetime.now().isoformat(timespec="seconds")

        for pred in predictions:
            vb = pred.get("value_bets", [])
            rows.append({
                "timestamp": ts,
                "date": pred.get("date", ""),
                "home_team": pred.get("home_team", ""),
                "away_team": pred.get("away_team", ""),
                "home_win_prob": pred.get("home_win_prob", 0.5),
                "away_win_prob": pred.get("away_win_prob", 0.5),
                "expected_total": pred.get("expected_total", 0),
                "best_home_odds": pred.get("best_home_odds", 0),
                "best_away_odds": pred.get("best_away_odds", 0),
                "ml_home_prob": pred.get("ml_home_prob", 0.5),
                "poisson_home_prob": pred.get("poisson_home_prob", 0.5),
                "elo_home_prob": pred.get("elo_home_prob", 0.5),
                "value_bets_json": json.dumps(vb) if vb else "",
                "actual_home_goals": np.nan,
                "actual_away_goals": np.nan,
                "actual_home_win": np.nan,
                "actual_total": np.nan,
                "result_updated": False,
            })

        new_df = pd.DataFrame(rows)

        if os.path.exists(self.log_path):
            existing = pd.read_csv(self.log_path)
            # Avoid duplicates: same date + same matchup
            for _, row in new_df.iterrows():
                mask = (
                    (existing["date"] == row["date"])
                    & (existing["home_team"] == row["home_team"])
                    & (existing["away_team"] == row["away_team"])
                )
                if mask.any():
                    # Update probabilities for existing entry
                    existing.loc[mask, "home_win_prob"] = row["home_win_prob"]
                    existing.loc[mask, "away_win_prob"] = row["away_win_prob"]
                    existing.loc[mask, "expected_total"] = row["expected_total"]
                    existing.loc[mask, "timestamp"] = row["timestamp"]
                else:
                    existing = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
            existing.to_csv(self.log_path, index=False)
        else:
            new_df.to_csv(self.log_path, index=False)

        logger.info("Logged %d predictions to %s", len(rows), self.log_path)

    def update_results(self):
        """Fetch actual results for past predictions that haven't been updated."""
        if not os.path.exists(self.log_path):
            logger.warning("No predictions log found at %s", self.log_path)
            return 0

        df = pd.read_csv(self.log_path)
        if df.empty:
            return 0

        # Find rows needing update
        needs_update = df[
            (df["result_updated"] != True)
            & (pd.to_datetime(df["date"]) < pd.Timestamp.now().normalize())
        ]

        if needs_update.empty:
            logger.info("All past predictions already have results")
            return 0

        updated = 0
        dates_to_check = needs_update["date"].unique()

        for game_date in dates_to_check:
            schedule = get_schedule(game_date)
            if schedule.empty:
                continue

            # Look for completed games
            for _, game in schedule.iterrows():
                if game.get("status") not in ("FINAL", "OFF"):
                    continue

                home = game.get("home_team", "")
                away = game.get("away_team", "")
                home_score = game.get("home_score", np.nan)
                away_score = game.get("away_score", np.nan)

                if pd.isna(home_score) or pd.isna(away_score):
                    continue

                mask = (
                    (df["date"] == game_date)
                    & (df["home_team"] == home)
                    & (df["away_team"] == away)
                )
                if mask.any():
                    df.loc[mask, "actual_home_goals"] = int(home_score)
                    df.loc[mask, "actual_away_goals"] = int(away_score)
                    df.loc[mask, "actual_home_win"] = int(home_score) > int(away_score)
                    df.loc[mask, "actual_total"] = int(home_score) + int(away_score)
                    df.loc[mask, "result_updated"] = True
                    updated += 1

        df.to_csv(self.log_path, index=False)
        logger.info("Updated results for %d games", updated)
        return updated

    def get_stats(self) -> dict:
        """Compute accuracy statistics from logged predictions."""
        if not os.path.exists(self.log_path):
            return {}

        df = pd.read_csv(self.log_path)
        completed = df[df["result_updated"] == True].copy()

        if completed.empty:
            return {"total_predictions": len(df), "completed_games": 0}

        completed["predicted_home_win"] = completed["home_win_prob"] > 0.5
        completed["correct"] = completed["predicted_home_win"] == completed["actual_home_win"]

        stats = {
            "total_predictions": len(df),
            "completed_games": len(completed),
            "pending_games": len(df) - len(completed),
        }

        # Overall accuracy
        stats["accuracy"] = completed["correct"].mean()

        # Log loss
        probs = completed["home_win_prob"].clip(0.01, 0.99).values
        actuals = completed["actual_home_win"].astype(int).values
        stats["log_loss"] = float(-(
            actuals * np.log(probs) + (1 - actuals) * np.log(1 - probs)
        ).mean())

        # Brier score
        stats["brier_score"] = float(((probs - actuals) ** 2).mean())

        # Total prediction accuracy
        if "expected_total" in completed.columns:
            valid_total = completed.dropna(subset=["expected_total", "actual_total"])
            if not valid_total.empty:
                stats["total_mae"] = float(
                    (valid_total["expected_total"] - valid_total["actual_total"]).abs().mean()
                )

        # Recent accuracy (last 7 and 30 days)
        completed["date_dt"] = pd.to_datetime(completed["date"])
        now = pd.Timestamp.now()
        for days, label in [(7, "7d"), (30, "30d")]:
            recent = completed[completed["date_dt"] >= now - pd.Timedelta(days=days)]
            if len(recent) >= 3:
                stats[f"accuracy_{label}"] = recent["correct"].mean()
                stats[f"games_{label}"] = len(recent)

        # Calibration bins
        bins = [0.0, 0.35, 0.45, 0.55, 0.65, 1.0]
        labels_cal = ["<35%", "35-45%", "45-55%", "55-65%", ">65%"]
        completed["prob_bin"] = pd.cut(completed["home_win_prob"], bins=bins, labels=labels_cal)
        cal = completed.groupby("prob_bin", observed=True).agg(
            predicted_mean=("home_win_prob", "mean"),
            actual_mean=("actual_home_win", "mean"),
            count=("actual_home_win", "count"),
        )
        stats["calibration"] = cal.to_dict("index")

        # Value bet ROI
        stats.update(self._compute_value_roi(completed))

        return stats

    def _compute_value_roi(self, completed: pd.DataFrame) -> dict:
        """Compute ROI on value bets."""
        from utils.helpers import american_to_decimal

        total_staked = 0
        total_returned = 0
        value_bets_count = 0
        value_wins = 0

        for _, row in completed.iterrows():
            vb_json = row.get("value_bets_json", "")
            if not vb_json:
                continue

            try:
                vbs = json.loads(vb_json)
            except (json.JSONDecodeError, TypeError):
                continue

            for vb in vbs:
                if vb.get("market") != "moneyline":
                    continue

                value_bets_count += 1
                stake = 100  # Flat $100 per bet
                total_staked += stake

                odds = vb.get("odds", 0)
                side = vb.get("side", "")
                actual_hw = row.get("actual_home_win")

                if pd.isna(actual_hw):
                    continue

                won = (side == "home" and actual_hw) or (side == "away" and not actual_hw)
                if won:
                    dec = american_to_decimal(odds)
                    total_returned += stake * dec
                    value_wins += 1

        result = {"value_bets_total": value_bets_count}
        if value_bets_count > 0:
            result["value_bets_wins"] = value_wins
            result["value_bets_win_pct"] = value_wins / value_bets_count
            result["value_bets_roi"] = (total_returned - total_staked) / total_staked if total_staked else 0
            result["value_bets_profit"] = total_returned - total_staked
        return result

    def print_report(self):
        """Print formatted tracking report."""
        stats = self.get_stats()
        if not stats:
            print("\nNo prediction data available.\n")
            return

        print(f"\n{'='*60}")
        print("  PREDICTION TRACKING REPORT")
        print(f"{'='*60}\n")

        print(f"  Total predictions:  {stats.get('total_predictions', 0)}")
        print(f"  Completed games:    {stats.get('completed_games', 0)}")
        print(f"  Pending games:      {stats.get('pending_games', 0)}")

        if stats.get("completed_games", 0) == 0:
            print("\n  No completed games to analyze yet.\n")
            return

        print(f"\n  ACCURACY METRICS")
        print(f"  {'─'*45}")
        print(f"    Accuracy:     {stats.get('accuracy', 0):.1%}")
        print(f"    Log Loss:     {stats.get('log_loss', 0):.4f}")
        print(f"    Brier Score:  {stats.get('brier_score', 0):.4f}")
        if "total_mae" in stats:
            print(f"    Total MAE:    {stats['total_mae']:.2f}")

        # Recent
        for label in ["7d", "30d"]:
            if f"accuracy_{label}" in stats:
                print(f"    Last {label}:      {stats[f'accuracy_{label}']:.1%} ({stats[f'games_{label}']} games)")

        # Calibration
        if "calibration" in stats:
            print(f"\n  CALIBRATION")
            print(f"  {'─'*45}")
            print(f"    {'Bin':<10} {'Predicted':>10} {'Actual':>10} {'Count':>7}")
            for bin_name, data in stats["calibration"].items():
                print(f"    {bin_name:<10} {data['predicted_mean']:>10.1%} "
                      f"{data['actual_mean']:>10.1%} {data['count']:>7}")

        # Value bets
        if stats.get("value_bets_total", 0) > 0:
            print(f"\n  VALUE BET PERFORMANCE")
            print(f"  {'─'*45}")
            print(f"    Total bets:   {stats['value_bets_total']}")
            print(f"    Wins:         {stats.get('value_bets_wins', 0)}")
            print(f"    Win rate:     {stats.get('value_bets_win_pct', 0):.1%}")
            print(f"    ROI:          {stats.get('value_bets_roi', 0):.1%}")
            print(f"    Profit ($100):${stats.get('value_bets_profit', 0):.0f}")

        print(f"\n{'='*60}\n")
