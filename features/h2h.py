"""
Head-to-head feature extraction.

Used with low weight — hockey is high-variance.
"""

import logging
from typing import Optional

import pandas as pd

from utils.helpers import safe_div

logger = logging.getLogger("nhl_predictor.features.h2h")


def compute_h2h_features(
    game_log: pd.DataFrame,
    team: str,
    opponent: str,
    n_meetings: int = 5,
) -> dict:
    """
    Head-to-head features from last N meetings.

    game_log must have: team, opponent, date, win, goals_for, goals_against, is_home.
    """
    matchups = game_log[
        (game_log["team"] == team) & (game_log["opponent"] == opponent)
    ].sort_values("date")

    if matchups.empty:
        return _empty_h2h()

    recent = matchups.tail(n_meetings)
    gp = len(recent)

    wins = recent["win"].sum()
    gf = recent["goals_for"].sum()
    ga = recent["goals_against"].sum()

    # Home vs away in H2H
    home_games = recent[recent["is_home"]]
    away_games = recent[~recent["is_home"]]

    return {
        "h2h_games": gp,
        "h2h_wins": int(wins),
        "h2h_losses": int(gp - wins),
        "h2h_win_pct": wins / gp,
        "h2h_gf_pg": gf / gp,
        "h2h_ga_pg": ga / gp,
        "h2h_goal_diff": gf - ga,
        "h2h_goal_diff_pg": (gf - ga) / gp,
        "h2h_home_wins": int(home_games["win"].sum()) if not home_games.empty else 0,
        "h2h_away_wins": int(away_games["win"].sum()) if not away_games.empty else 0,
    }


def _empty_h2h() -> dict:
    return {
        "h2h_games": 0,
        "h2h_wins": 0,
        "h2h_losses": 0,
        "h2h_win_pct": 0.5,
        "h2h_gf_pg": 2.8,
        "h2h_ga_pg": 2.8,
        "h2h_goal_diff": 0,
        "h2h_goal_diff_pg": 0.0,
        "h2h_home_wins": 0,
        "h2h_away_wins": 0,
    }


def compute_division_rivalry(
    team: str,
    opponent: str,
    standings: pd.DataFrame,
) -> dict:
    """
    Detect division/conference rivalry and playoff pressure.
    """
    if standings.empty:
        return {"is_division_rival": False, "is_conference_rival": False}

    team_row = standings[standings["team"] == team]
    opp_row = standings[standings["team"] == opponent]

    if team_row.empty or opp_row.empty:
        return {"is_division_rival": False, "is_conference_rival": False}

    team_div = team_row.iloc[0].get("division", "")
    opp_div = opp_row.iloc[0].get("division", "")
    team_conf = team_row.iloc[0].get("conference", "")
    opp_conf = opp_row.iloc[0].get("conference", "")

    return {
        "is_division_rival": team_div == opp_div and team_div != "",
        "is_conference_rival": team_conf == opp_conf and team_conf != "",
    }


def compute_playoff_pressure(
    team: str,
    standings: pd.DataFrame,
) -> dict:
    """
    How much playoff pressure is a team under?
    Based on wildcard position and points from playoff line.
    """
    if standings.empty or "team" not in standings.columns:
        return {"playoff_pressure": 0.5, "is_playoff_team": False, "is_bubble_team": False}

    team_row = standings[standings["team"] == team]
    if team_row.empty:
        return {"playoff_pressure": 0.5, "is_playoff_team": False, "is_bubble_team": False}

    row = team_row.iloc[0]
    wc_seq = row.get("wildcard_sequence", 99)
    pts_pct = row.get("points_pct", 0.5)
    gp = row.get("games_played", 0)

    # In playoff position
    is_playoff = wc_seq <= 8 if wc_seq < 99 else pts_pct > 0.55

    # Bubble team: close to cutoff
    is_bubble = 7 <= wc_seq <= 12 if wc_seq < 99 else 0.48 < pts_pct < 0.58

    # Pressure increases later in season
    season_urgency = min(gp / 82.0, 1.0)  # 0 to 1 over season

    if is_bubble:
        pressure = 0.7 + 0.3 * season_urgency
    elif is_playoff:
        pressure = 0.5 + 0.2 * season_urgency
    else:
        # Out of it
        pressure = max(0.1, 0.4 - 0.3 * season_urgency)

    return {
        "playoff_pressure": round(pressure, 3),
        "is_playoff_team": is_playoff,
        "is_bubble_team": is_bubble,
    }
