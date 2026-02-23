"""
Schedule and fatigue feature extraction.

Covers:
  - Back-to-back games
  - 3-in-4 nights
  - Travel distance
  - Time zone changes
  - Road trip length
  - Rest differential
  - Last opponent strength
  - Season phase
"""

import logging
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import numpy as np

from config import cfg
from utils.helpers import haversine_km, timezone_offset

logger = logging.getLogger("nhl_predictor.features.schedule")


def compute_schedule_features(
    team: str,
    game_date: str,
    game_log: pd.DataFrame,
    is_home: bool,
    opponent: str,
) -> dict:
    """
    Comprehensive schedule/fatigue features.

    game_log must have: team, date, opponent, is_home.
    """
    today = pd.to_datetime(game_date)
    team_games = game_log[game_log["team"] == team].sort_values("date").copy()
    team_games["date"] = pd.to_datetime(team_games["date"])

    past = team_games[team_games["date"] < today]

    features = {}

    # ── Rest days ────────────────────────────────────────────────────
    if not past.empty:
        last_game = past.iloc[-1]
        rest_days = (today - last_game["date"]).days
    else:
        rest_days = 3  # default for season start
    features["rest_days"] = rest_days

    # ── Back-to-back ─────────────────────────────────────────────────
    features["is_back_to_back"] = rest_days <= 1

    # Second of back-to-back (more fatiguing)
    features["is_second_btb"] = rest_days <= 1

    # ── 3 in 4 nights ────────────────────────────────────────────────
    four_days_ago = today - timedelta(days=4)
    games_in_4 = len(past[past["date"] >= four_days_ago])
    features["games_in_4_nights"] = games_in_4
    features["is_3_in_4"] = games_in_4 >= 2  # this will be 3rd game

    # ── 4 in 6 nights ────────────────────────────────────────────────
    six_days_ago = today - timedelta(days=6)
    games_in_6 = len(past[past["date"] >= six_days_ago])
    features["games_in_6_nights"] = games_in_6
    features["is_4_in_6"] = games_in_6 >= 3

    # ── Travel distance ──────────────────────────────────────────────
    travel_km = 0.0
    if not past.empty:
        last_game = past.iloc[-1]
        # Where did last game happen?
        if last_game["is_home"]:
            last_city = team
        else:
            last_city = last_game["opponent"]

        # Where is this game?
        current_city = team if is_home else opponent

        last_coords = cfg.arena_coords.get(last_city)
        curr_coords = cfg.arena_coords.get(current_city)

        if last_coords and curr_coords:
            travel_km = haversine_km(
                last_coords[0], last_coords[1],
                curr_coords[0], curr_coords[1],
            )

    features["travel_km"] = round(travel_km, 1)
    features["travel_fatigue"] = min(travel_km / 4000.0, 1.0)  # normalized 0-1

    # ── Time zone change ─────────────────────────────────────────────
    tz_change = 0
    if not past.empty:
        last_game = past.iloc[-1]
        last_city = team if last_game["is_home"] else last_game["opponent"]
        current_city = team if is_home else opponent

        last_coords = cfg.arena_coords.get(last_city)
        curr_coords = cfg.arena_coords.get(current_city)

        if last_coords and curr_coords:
            tz_change = abs(timezone_offset(curr_coords[1]) - timezone_offset(last_coords[1]))

    features["timezone_change"] = tz_change

    # ── Road trip length ─────────────────────────────────────────────
    road_trip_len = 0
    if not is_home and not past.empty:
        # Count consecutive away games going backwards
        for _, row in past.iloc[::-1].iterrows():
            if not row["is_home"]:
                road_trip_len += 1
            else:
                break
    features["road_trip_length"] = road_trip_len

    # ── Home stand length ────────────────────────────────────────────
    home_stand_len = 0
    if is_home and not past.empty:
        for _, row in past.iloc[::-1].iterrows():
            if row["is_home"]:
                home_stand_len += 1
            else:
                break
    features["home_stand_length"] = home_stand_len

    # ── Season phase ─────────────────────────────────────────────────
    month = today.month
    if month in (10, 11):
        season_phase = "early"
        phase_num = 0
    elif month in (12, 1):
        season_phase = "mid"
        phase_num = 1
    elif month in (2, 3):
        season_phase = "late"
        phase_num = 2
    else:
        season_phase = "playoff_push"
        phase_num = 3

    features["season_phase"] = season_phase
    features["season_phase_num"] = phase_num

    # ── Games played (for cold-start detection) ──────────────────────
    features["team_games_played"] = len(past)

    return features


def compute_rest_differential(
    home_features: dict,
    away_features: dict,
) -> dict:
    """Rest advantage: positive = home team more rested."""
    return {
        "rest_diff": home_features.get("rest_days", 2) - away_features.get("rest_days", 2),
        "btb_diff": int(away_features.get("is_back_to_back", False))
                    - int(home_features.get("is_back_to_back", False)),
        "travel_diff": away_features.get("travel_km", 0) - home_features.get("travel_km", 0),
    }


def compute_last_opponent_strength(
    game_log: pd.DataFrame,
    team: str,
    game_date: str,
    standings: pd.DataFrame,
) -> dict:
    """
    How strong was the last opponent? Useful for detecting letdown games.
    """
    today = pd.to_datetime(game_date)
    team_games = game_log[game_log["team"] == team].sort_values("date")
    team_games["date"] = pd.to_datetime(team_games["date"])
    past = team_games[team_games["date"] < today]

    if past.empty:
        return {"last_opp_points_pct": 0.5, "last_opp_was_strong": False}

    last_opp = past.iloc[-1]["opponent"]

    if standings.empty or "team" not in standings.columns:
        return {"last_opp_points_pct": 0.5, "last_opp_was_strong": False}

    opp_row = standings[standings["team"] == last_opp]
    if opp_row.empty:
        return {"last_opp_points_pct": 0.5, "last_opp_was_strong": False}

    pts_pct = opp_row.iloc[0].get("points_pct", 0.5)
    return {
        "last_opp_points_pct": pts_pct,
        "last_opp_was_strong": pts_pct > 0.6,
    }
