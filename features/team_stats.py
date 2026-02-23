"""
Team-level feature extraction.

Covers:
  - Basic stats (W/L/OTL, points, GF/GA, shots, faceoff%)
  - Special teams (PP%, PK%)
  - Home/away splits
  - Recent form (last 5 / last 10)
  - Streaks
  - Discipline (PIM, hits, blocks)
  - PDO / luck indicators
"""

import logging

import numpy as np
import pandas as pd

from config import cfg
from utils.helpers import safe_div

logger = logging.getLogger("nhl_predictor.features.team_stats")


def compute_basic_stats(standings: pd.DataFrame) -> pd.DataFrame:
    """
    Derive basic features from standings DataFrame.
    Input must have: team, games_played, wins, losses, ot_losses, points,
                     goals_for, goals_against, goal_diff.
    """
    df = standings.copy()
    gp = df["games_played"].clip(lower=1)

    df["win_pct"] = df["wins"] / gp
    df["loss_pct"] = df["losses"] / gp
    df["ot_loss_pct"] = df["ot_losses"] / gp
    df["points_pct"] = df["points"] / (gp * 2)
    df["gf_pg"] = df["goals_for"] / gp
    df["ga_pg"] = df["goals_against"] / gp
    df["goal_diff_pg"] = df["goal_diff"] / gp

    return df


def compute_home_away_splits(standings: pd.DataFrame) -> pd.DataFrame:
    """Home vs away records and goal differential."""
    df = standings.copy()

    home_gp = (df["home_wins"] + df["home_losses"] + df["home_ot_losses"]).clip(lower=1)
    away_gp = (df["away_wins"] + df["away_losses"] + df["away_ot_losses"]).clip(lower=1)

    df["home_win_pct"] = df["home_wins"] / home_gp
    df["away_win_pct"] = df["away_wins"] / away_gp
    df["home_away_win_diff"] = df["home_win_pct"] - df["away_win_pct"]

    return df


def compute_recent_form(game_log: pd.DataFrame, team: str, n_games: int = 5) -> dict:
    """
    Compute form metrics from last N games for a team.
    game_log must have: team, date, win, goals_for, goals_against, is_home.
    """
    team_games = game_log[game_log["team"] == team].sort_values("date")
    if team_games.empty:
        return _empty_form(n_games)

    recent = team_games.tail(n_games)
    gp = len(recent)
    if gp == 0:
        return _empty_form(n_games)

    wins = recent["win"].sum()
    gf = recent["goals_for"].sum()
    ga = recent["goals_against"].sum()

    return {
        f"last{n_games}_wins": int(wins),
        f"last{n_games}_losses": int(gp - wins),
        f"last{n_games}_win_pct": wins / gp,
        f"last{n_games}_gf_pg": gf / gp,
        f"last{n_games}_ga_pg": ga / gp,
        f"last{n_games}_goal_diff_pg": (gf - ga) / gp,
    }


def _empty_form(n: int) -> dict:
    return {
        f"last{n}_wins": 0, f"last{n}_losses": 0, f"last{n}_win_pct": 0.5,
        f"last{n}_gf_pg": 2.8, f"last{n}_ga_pg": 2.8, f"last{n}_goal_diff_pg": 0.0,
    }


def compute_streak(game_log: pd.DataFrame, team: str) -> dict:
    """Current win/loss streak."""
    team_games = game_log[game_log["team"] == team].sort_values("date")
    if team_games.empty:
        return {"streak_type": "N", "streak_count": 0}

    streak_type = None
    streak_count = 0

    for _, row in team_games.iloc[::-1].iterrows():
        result = "W" if row["win"] else "L"
        if streak_type is None:
            streak_type = result
            streak_count = 1
        elif result == streak_type:
            streak_count += 1
        else:
            break

    return {
        "streak_type": streak_type or "N",
        "streak_count": streak_count,
        "streak_value": streak_count if streak_type == "W" else -streak_count,
    }


def compute_rolling_stats(game_log: pd.DataFrame, team: str,
                           windows: list[int] = None) -> dict:
    """
    Rolling averages for goals for/against over multiple windows.
    Returns dict with rolling_N_gf, rolling_N_ga, rolling_N_goal_diff.
    """
    if windows is None:
        windows = [cfg.rolling_window_short, cfg.rolling_window_long]

    team_games = game_log[game_log["team"] == team].sort_values("date")
    result = {}

    for w in windows:
        if len(team_games) >= w:
            recent = team_games.tail(w)
            result[f"rolling_{w}_gf"] = recent["goals_for"].mean()
            result[f"rolling_{w}_ga"] = recent["goals_against"].mean()
            result[f"rolling_{w}_goal_diff"] = (recent["goals_for"] - recent["goals_against"]).mean()
        else:
            result[f"rolling_{w}_gf"] = team_games["goals_for"].mean() if len(team_games) > 0 else 2.8
            result[f"rolling_{w}_ga"] = team_games["goals_against"].mean() if len(team_games) > 0 else 2.8
            result[f"rolling_{w}_goal_diff"] = 0.0

    return result


def compute_momentum_index(game_log: pd.DataFrame, team: str, window: int = 10) -> float:
    """
    Weighted momentum index: recent games weighted more heavily.
    Range roughly [-1, 1] where 1 = all wins with increasing margins.
    """
    team_games = game_log[game_log["team"] == team].sort_values("date")
    recent = team_games.tail(window)
    if recent.empty:
        return 0.0

    n = len(recent)
    weights = np.arange(1, n + 1, dtype=float)
    weights /= weights.sum()

    # Score each game: win = +1 scaled by goal diff, loss = -1 scaled
    scores = []
    for _, row in recent.iterrows():
        gd = row["goals_for"] - row["goals_against"]
        s = np.tanh(gd / 3.0)  # smooth to [-1, 1]
        scores.append(s)

    return float(np.dot(scores, weights))


def compute_discipline_features(team_mp_data: dict) -> dict:
    """
    Discipline and style features from MoneyPuck or similar data.
    """
    gp = max(team_mp_data.get("gamesPlayed", 1), 1)
    return {
        "hits_pg": team_mp_data.get("hitsFor", 0) / gp,
        "hits_against_pg": team_mp_data.get("hitsAgainst", 0) / gp,
        "blocks_pg": team_mp_data.get("blockedShotAttemptsFor", 0) / gp,
        "pim_pg": team_mp_data.get("penalityMinutesFor", team_mp_data.get("penaltyMinutesFor", 0)) / gp,
        "penalties_drawn_pg": team_mp_data.get("penaltiesDrawn", 0) / gp,
    }


def compute_pdo(team_mp_data: dict) -> dict:
    """
    PDO = Shooting% + Save%.
    Above 100 suggests luck, below 100 suggests bad luck.
    Strong regressor to mean.
    """
    shots_for = max(team_mp_data.get("shotsOnGoalFor", 1), 1)
    shots_against = max(team_mp_data.get("shotsOnGoalAgainst", 1), 1)
    goals_for = team_mp_data.get("goalsFor", 0)
    goals_against = team_mp_data.get("goalsAgainst", 0)

    shooting_pct = goals_for / shots_for * 100
    save_pct = (1 - goals_against / shots_against) * 100
    pdo = shooting_pct + save_pct

    return {
        "shooting_pct": shooting_pct,
        "save_pct_team": save_pct,
        "pdo": pdo,
        "pdo_luck": pdo - 100.0,  # positive = running hot
    }
