"""
Player-level feature aggregation.

Focus on top-6 forwards, top-4 defensemen per team.
Aggregate to team-level features.
"""

import logging
from typing import Optional

import pandas as pd
import numpy as np

from utils.helpers import safe_div

logger = logging.getLogger("nhl_predictor.features.player")


def aggregate_skater_features(
    skaters_df: pd.DataFrame,
    team: str,
    top_fwd: int = 6,
    top_def: int = 4,
) -> dict:
    """
    Aggregate top-N skater stats into team-level features.

    skaters_df: MoneyPuck skater data with columns including:
        team, position, I_F_points, I_F_goals, I_F_shots, icetime, ...
    """
    team_col = "team" if "team" in skaters_df.columns else skaters_df.columns[0]
    team_sk = skaters_df[skaters_df[team_col] == team].copy()

    if team_sk.empty:
        return _empty_player_features()

    # Determine position
    pos_col = "position" if "position" in team_sk.columns else None
    if pos_col is None:
        # Fallback: try to infer from other columns
        return _aggregate_all(team_sk, top_fwd + top_def)

    forwards = team_sk[team_sk[pos_col].isin(["C", "L", "R", "LW", "RW", "F"])]
    defensemen = team_sk[team_sk[pos_col].isin(["D"])]

    # Sort by ice time or points
    sort_col = _pick_sort_col(team_sk)
    forwards = forwards.sort_values(sort_col, ascending=False).head(top_fwd)
    defensemen = defensemen.sort_values(sort_col, ascending=False).head(top_def)

    features = {}

    # ── Forward features ─────────────────────────────────────────────
    if not forwards.empty:
        gp = forwards.get("games_played", forwards.get("gamesPlayed", pd.Series([1]))).clip(lower=1)
        features["top_fwd_pts_pg"] = safe_div(
            _sum(forwards, "I_F_points", "points"),
            gp.sum(), 0
        )
        features["top_fwd_goals_pg"] = safe_div(
            _sum(forwards, "I_F_goals", "goals"),
            gp.sum(), 0
        )
        features["top_fwd_shots_pg"] = safe_div(
            _sum(forwards, "I_F_shots", "shots"),
            gp.sum(), 0
        )
        features["top_fwd_avg_toi"] = _mean(forwards, "icetime", "timeOnIce") / 60 if _mean(forwards, "icetime", "timeOnIce") > 60 else _mean(forwards, "icetime", "timeOnIce")
        features["top_fwd_xgf"] = _sum(forwards, "I_F_xGoals", "xGoals")
    else:
        features.update({
            "top_fwd_pts_pg": 0, "top_fwd_goals_pg": 0,
            "top_fwd_shots_pg": 0, "top_fwd_avg_toi": 0,
            "top_fwd_xgf": 0,
        })

    # ── Defenseman features ──────────────────────────────────────────
    if not defensemen.empty:
        gp = defensemen.get("games_played", defensemen.get("gamesPlayed", pd.Series([1]))).clip(lower=1)
        features["top_def_pts_pg"] = safe_div(
            _sum(defensemen, "I_F_points", "points"),
            gp.sum(), 0
        )
        features["top_def_blocks_pg"] = safe_div(
            _sum(defensemen, "I_F_shotsBlockedByPlayer", "blockedShots"),
            gp.sum(), 0
        )
        features["top_def_avg_toi"] = _mean(defensemen, "icetime", "timeOnIce") / 60 if _mean(defensemen, "icetime", "timeOnIce") > 60 else _mean(defensemen, "icetime", "timeOnIce")
        features["top_def_xgf"] = _sum(defensemen, "I_F_xGoals", "xGoals")
        features["top_def_plus_minus"] = _sum(defensemen, "plusMinus", "I_F_plusMinus")
    else:
        features.update({
            "top_def_pts_pg": 0, "top_def_blocks_pg": 0,
            "top_def_avg_toi": 0, "top_def_xgf": 0,
            "top_def_plus_minus": 0,
        })

    # ── Depth metrics ────────────────────────────────────────────────
    all_skaters = pd.concat([forwards, defensemen])
    if not all_skaters.empty:
        toi_vals = all_skaters.get("icetime", all_skaters.get("timeOnIce", pd.Series()))
        if not toi_vals.empty:
            features["lineup_toi_std"] = toi_vals.std()
        else:
            features["lineup_toi_std"] = 0
    else:
        features["lineup_toi_std"] = 0

    return features


def _sum(df: pd.DataFrame, *col_candidates) -> float:
    for col in col_candidates:
        if col in df.columns:
            return df[col].sum()
    return 0


def _mean(df: pd.DataFrame, *col_candidates) -> float:
    for col in col_candidates:
        if col in df.columns:
            return df[col].mean()
    return 0


def _pick_sort_col(df: pd.DataFrame) -> str:
    for col in ("icetime", "timeOnIce", "I_F_points", "points"):
        if col in df.columns:
            return col
    return df.columns[-1]


def _aggregate_all(df: pd.DataFrame, top_n: int) -> dict:
    sort_col = _pick_sort_col(df)
    top = df.sort_values(sort_col, ascending=False).head(top_n)
    gp = top.get("games_played", top.get("gamesPlayed", pd.Series([1]))).clip(lower=1)
    return {
        "top_fwd_pts_pg": safe_div(_sum(top, "I_F_points", "points"), gp.sum(), 0),
        "top_fwd_goals_pg": safe_div(_sum(top, "I_F_goals", "goals"), gp.sum(), 0),
        "top_fwd_shots_pg": safe_div(_sum(top, "I_F_shots", "shots"), gp.sum(), 0),
        "top_fwd_avg_toi": 0, "top_fwd_xgf": 0,
        "top_def_pts_pg": 0, "top_def_blocks_pg": 0,
        "top_def_avg_toi": 0, "top_def_xgf": 0,
        "top_def_plus_minus": 0, "lineup_toi_std": 0,
    }


def _empty_player_features() -> dict:
    return {
        "top_fwd_pts_pg": 0, "top_fwd_goals_pg": 0,
        "top_fwd_shots_pg": 0, "top_fwd_avg_toi": 0, "top_fwd_xgf": 0,
        "top_def_pts_pg": 0, "top_def_blocks_pg": 0,
        "top_def_avg_toi": 0, "top_def_xgf": 0,
        "top_def_plus_minus": 0, "lineup_toi_std": 0,
    }


def compute_injury_impact(
    injuries_df: pd.DataFrame,
    team: str,
    skaters_df: pd.DataFrame,
) -> dict:
    """
    Estimate impact of injuries on team strength.
    Weights by player's TOI/points contribution.
    """
    if injuries_df.empty:
        return {"injury_toi_lost": 0, "injury_impact_score": 0, "key_injuries": 0}

    team_injuries = injuries_df[
        injuries_df["team"].str.contains(team, case=False, na=False)
    ]

    if team_injuries.empty:
        return {"injury_toi_lost": 0, "injury_impact_score": 0, "key_injuries": 0}

    # Try to match injured players to their stats
    impact_score = 0
    toi_lost = 0
    key_injuries = 0

    for _, inj in team_injuries.iterrows():
        player_name = inj.get("player_name", "")
        status = str(inj.get("status", "")).lower()

        # Skip day-to-day with low impact
        if "day" in status and "long" not in status:
            weight = 0.5
        else:
            weight = 1.0

        # Find player in skater stats
        if not skaters_df.empty and player_name:
            match = skaters_df[
                skaters_df.apply(
                    lambda r: player_name.lower() in str(r.get("name", "")).lower()
                    or player_name.lower() in str(r.get("playerName", "")).lower(),
                    axis=1,
                )
            ]
            if not match.empty:
                player = match.iloc[0]
                toi = player.get("icetime", player.get("timeOnIce", 0))
                pts = player.get("I_F_points", player.get("points", 0))
                toi_lost += toi * weight
                impact_score += (pts * weight)

                if toi > 900 or pts > 20:  # significant player
                    key_injuries += 1

    return {
        "injury_toi_lost": round(toi_lost, 1),
        "injury_impact_score": round(impact_score, 2),
        "key_injuries": key_injuries,
    }
