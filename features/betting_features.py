"""
Betting market features.

The market is a strong aggregator of information —
even if not betting, market lines contain signal.
"""

import logging

import pandas as pd
import numpy as np

from utils.helpers import implied_prob, safe_div

logger = logging.getLogger("nhl_predictor.features.betting")


def compute_market_features(
    consensus: pd.DataFrame,
    home_team: str,
    away_team: str,
) -> dict:
    """
    Extract betting market features for a specific matchup.

    consensus: output of odds_api.get_consensus_odds() or similar.
    """
    if consensus.empty:
        return _empty_market()

    # Match by team names (Odds API uses full names, need flexible matching)
    row = _find_matchup(consensus, home_team, away_team)
    if row is None:
        return _empty_market()

    home_imp = row.get("home_implied_mean", 0.5)
    away_imp = row.get("away_implied_mean", 0.5)

    # Remove vig for true implied probabilities
    total_imp = home_imp + away_imp
    home_true = home_imp / total_imp if total_imp > 0 else 0.5
    away_true = away_imp / total_imp if total_imp > 0 else 0.5

    # Market spread (how wide the line is)
    home_range = row.get("home_implied_max", 0.5) - row.get("home_implied_min", 0.5)

    return {
        "market_home_implied": home_imp,
        "market_away_implied": away_imp,
        "market_home_true_prob": home_true,
        "market_away_true_prob": away_true,
        "market_vig": total_imp - 1.0,
        "market_spread": home_range,  # wider = more disagreement
        "market_favorite": "home" if home_true > 0.5 else "away",
        "market_favorite_prob": max(home_true, away_true),
        "num_bookmakers": row.get("num_books", 0),
        "best_home_odds": row.get("home_best_odds", 0),
        "best_away_odds": row.get("away_best_odds", 0),
    }


def compute_line_movement(
    opening_odds: dict,
    current_odds: dict,
) -> dict:
    """
    Detect line movement between opening and current odds.
    Sharp money often moves lines.
    """
    if not opening_odds or not current_odds:
        return {
            "line_movement_home": 0,
            "line_movement_away": 0,
            "sharp_indicator": 0,
        }

    open_home = opening_odds.get("home_implied", 0.5)
    curr_home = current_odds.get("home_implied", 0.5)
    open_away = opening_odds.get("away_implied", 0.5)
    curr_away = current_odds.get("away_implied", 0.5)

    move_home = curr_home - open_home
    move_away = curr_away - open_away

    # Sharp indicator: significant movement suggests informed money
    sharp = 1 if abs(move_home) > 0.03 else 0

    return {
        "line_movement_home": round(move_home, 4),
        "line_movement_away": round(move_away, 4),
        "sharp_indicator": sharp,
    }


def compute_totals_features(
    totals_df: pd.DataFrame,
    home_team: str,
    away_team: str,
) -> dict:
    """Extract over/under consensus features."""
    if totals_df.empty:
        return {"market_total": 5.5, "market_over_implied": 0.5, "market_under_implied": 0.5}

    row = _find_matchup_totals(totals_df, home_team, away_team)
    if row is None:
        return {"market_total": 5.5, "market_over_implied": 0.5, "market_under_implied": 0.5}

    return {
        "market_total": row.get("total", 5.5),
        "market_over_implied": row.get("over_implied", 0.5),
        "market_under_implied": row.get("under_implied", 0.5),
    }


def _find_matchup(df: pd.DataFrame, home: str, away: str):
    """Find a matchup row, trying various name formats."""
    for _, row in df.iterrows():
        h = str(row.get("home_team", "")).lower()
        a = str(row.get("away_team", "")).lower()
        if (home.lower() in h or h in home.lower()) and \
           (away.lower() in a or a in away.lower()):
            return row
    return None


def _find_matchup_totals(df: pd.DataFrame, home: str, away: str):
    """Find totals for a matchup."""
    for game_id in df["game_id"].unique():
        game = df[df["game_id"] == game_id]
        h = str(game.iloc[0].get("home_team", "")).lower()
        a = str(game.iloc[0].get("away_team", "")).lower()
        if (home.lower() in h or h in home.lower()) and \
           (away.lower() in a or a in away.lower()):
            over = game[game["side"] == "over"]
            under = game[game["side"] == "under"]
            return {
                "total": over.iloc[0]["total"] if not over.empty else 5.5,
                "over_implied": over["implied"].mean() if not over.empty else 0.5,
                "under_implied": under["implied"].mean() if not under.empty else 0.5,
            }
    return None


def _empty_market() -> dict:
    return {
        "market_home_implied": 0.5,
        "market_away_implied": 0.5,
        "market_home_true_prob": 0.5,
        "market_away_true_prob": 0.5,
        "market_vig": 0,
        "market_spread": 0,
        "market_favorite": "home",
        "market_favorite_prob": 0.5,
        "num_bookmakers": 0,
        "best_home_odds": 0,
        "best_away_odds": 0,
    }
