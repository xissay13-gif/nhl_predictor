"""
Goalie feature extraction.

Covers:
  - Save %
  - Goals Saved Above Expected (GSAx)
  - High-danger save %
  - Rest / fatigue
  - Starter vs backup
  - Career vs opponent splits
  - Back-to-back starts
"""

import logging
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import numpy as np

from utils.helpers import safe_div

logger = logging.getLogger("nhl_predictor.features.goalie")


def compute_goalie_features(
    goalie_row: dict,
    game_log: Optional[pd.DataFrame] = None,
    opponent: Optional[str] = None,
) -> dict:
    """
    Extract goalie features from MoneyPuck goalie data row.
    goalie_row is a dict with keys from MoneyPuck CSV.
    """
    saves = goalie_row.get("saves", goalie_row.get("onGoalFor", 0))
    shots_against = goalie_row.get("shotsAgainst",
                                   goalie_row.get("onGoalAgainst",
                                   goalie_row.get("shotsOnGoalAgainst", 0)))
    goals_against = goalie_row.get("goalsAgainst", 0)
    xga = goalie_row.get("xGoalsAgainst", goalie_row.get("xOnGoalAgainst", 0))
    hd_shots = goalie_row.get("highDangerShotsAgainst",
                              goalie_row.get("highDangerxGoalsAgainst", 0))
    hd_goals = goalie_row.get("highDangerGoalsAgainst", 0)
    gp = max(goalie_row.get("games_played", goalie_row.get("gamesPlayed", 1)), 1)

    save_pct = safe_div(shots_against - goals_against, shots_against, 0.910)
    gsax = xga - goals_against   # positive = saved more than expected
    hd_save_pct = safe_div(hd_shots - hd_goals, hd_shots, 0.850) if hd_shots > 0 else 0.850

    features = {
        "goalie_save_pct": save_pct,
        "goalie_gsax": gsax,
        "goalie_gsax_pg": gsax / gp,
        "goalie_hd_save_pct": hd_save_pct,
        "goalie_gaa": goals_against / gp,
        "goalie_games_played": gp,
        "goalie_xga_pg": xga / gp,
    }

    return features


def compute_goalie_rest(
    goalie_name: str,
    game_date: str,
    game_log: pd.DataFrame,
) -> dict:
    """
    Compute rest-related features for a goalie.
    game_log should have: goalie, date columns.
    """
    today = pd.to_datetime(game_date)

    goalie_games = game_log[
        game_log["goalie"].str.contains(goalie_name, case=False, na=False)
    ].sort_values("date")

    if goalie_games.empty:
        return {
            "goalie_rest_days": 3,
            "goalie_back_to_back": False,
            "goalie_starts_last_7": 0,
            "goalie_starts_last_14": 0,
            "goalie_fatigue_index": 0.0,
        }

    last_game_date = pd.to_datetime(goalie_games["date"].iloc[-1])
    rest_days = (today - last_game_date).days

    # Back-to-back: played yesterday
    btb = rest_days <= 1

    # Workload
    week_ago = today - timedelta(days=7)
    two_weeks_ago = today - timedelta(days=14)
    starts_7 = len(goalie_games[pd.to_datetime(goalie_games["date"]) >= week_ago])
    starts_14 = len(goalie_games[pd.to_datetime(goalie_games["date"]) >= two_weeks_ago])

    # Fatigue index: weighted recent starts
    fatigue = 0.0
    for _, row in goalie_games.iterrows():
        days_ago = (today - pd.to_datetime(row["date"])).days
        if 0 < days_ago <= 14:
            fatigue += 1.0 / days_ago  # more recent = more fatigue

    return {
        "goalie_rest_days": rest_days,
        "goalie_back_to_back": btb,
        "goalie_starts_last_7": starts_7,
        "goalie_starts_last_14": starts_14,
        "goalie_fatigue_index": round(fatigue, 3),
    }


def compute_goalie_vs_opponent(
    goalie_stats_df: pd.DataFrame,
    goalie_name: str,
    opponent: str,
) -> dict:
    """
    Goalie's career stats vs a specific opponent.
    Requires a game-level goalie log with opponent column.
    """
    if goalie_stats_df.empty:
        return {
            "goalie_vs_opp_sv_pct": 0.910,
            "goalie_vs_opp_gaa": 2.80,
            "goalie_vs_opp_games": 0,
        }

    mask = (
        goalie_stats_df["goalie"].str.contains(goalie_name, case=False, na=False)
        & goalie_stats_df["opponent"].str.contains(opponent, case=False, na=False)
    )
    subset = goalie_stats_df[mask]

    if subset.empty:
        return {
            "goalie_vs_opp_sv_pct": 0.910,
            "goalie_vs_opp_gaa": 2.80,
            "goalie_vs_opp_games": 0,
        }

    total_sa = subset["shots_against"].sum()
    total_ga = subset["goals_against"].sum()
    gp = len(subset)

    return {
        "goalie_vs_opp_sv_pct": safe_div(total_sa - total_ga, total_sa, 0.910),
        "goalie_vs_opp_gaa": total_ga / max(gp, 1),
        "goalie_vs_opp_games": gp,
    }


def classify_starter_backup(
    goalie_games_played: int,
    team_games_played: int,
) -> dict:
    """Determine if goalie is starter or backup based on games played."""
    start_pct = safe_div(goalie_games_played, team_games_played, 0.5)
    is_starter = start_pct > 0.55

    return {
        "goalie_is_starter": is_starter,
        "goalie_start_share": start_pct,
    }
