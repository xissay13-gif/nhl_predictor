"""
MoneyPuck data collector.

MoneyPuck provides CSV downloads with xGoals, shot quality, goalie stats,
and advanced team metrics.

Data files at:
  https://moneypuck.com/moneypuck/playerData/seasonSummary/{season}/regular/teams.csv
  https://moneypuck.com/moneypuck/playerData/seasonSummary/{season}/regular/skaters.csv
  https://moneypuck.com/moneypuck/playerData/seasonSummary/{season}/regular/goalies.csv
  https://moneypuck.com/moneypuck/playerData/games/...   (game-level)
"""

import io
import logging
from typing import Optional

import pandas as pd

from config import cfg
from utils.helpers import safe_request

logger = logging.getLogger("nhl_predictor.moneypuck")

MP_BASE = "https://moneypuck.com/moneypuck/playerData/seasonSummary"


def _fetch_csv(url: str) -> pd.DataFrame:
    resp = safe_request(url, timeout=60)
    if resp is None:
        return pd.DataFrame()
    try:
        return pd.read_csv(io.StringIO(resp.text))
    except Exception as exc:
        logger.error("Failed to parse CSV from %s: %s", url, exc)
        return pd.DataFrame()


# ── Team-level advanced stats ────────────────────────────────────────

def get_team_advanced(season: Optional[str] = None) -> pd.DataFrame:
    """
    MoneyPuck team-level stats: xGoals, Corsi, Fenwick, shot quality, etc.
    Returns one row per team-situation combo (all, 5on5, pp, pk, etc.).
    """
    season = season or cfg.current_season[:4]  # MoneyPuck uses starting year
    # Try multiple URL formats
    for fmt in (
        f"{MP_BASE}/{season}/regular/teams.csv",
        f"https://moneypuck.com/moneypuck/playerData/seasonSummary/{season}/regular/teams.csv",
    ):
        df = _fetch_csv(fmt)
        if not df.empty:
            return df
    return pd.DataFrame()


def get_team_5v5(season: Optional[str] = None) -> pd.DataFrame:
    """Filtered to 5v5 situation only — most analytically clean."""
    df = get_team_advanced(season)
    if df.empty:
        return df
    if "situation" in df.columns:
        return df[df["situation"] == "5on5"].copy()
    return df


def get_team_all_situations(season: Optional[str] = None) -> pd.DataFrame:
    """Filtered to 'all' situations."""
    df = get_team_advanced(season)
    if df.empty:
        return df
    if "situation" in df.columns:
        return df[df["situation"] == "all"].copy()
    return df


# ── Goalie stats ─────────────────────────────────────────────────────

def get_goalie_stats(season: Optional[str] = None) -> pd.DataFrame:
    """
    Goalie stats: save %, GSAx, xGoals against, high-danger save %, etc.
    """
    season = season or cfg.current_season[:4]
    for fmt in (
        f"{MP_BASE}/{season}/regular/goalies.csv",
        f"https://moneypuck.com/moneypuck/playerData/seasonSummary/{season}/regular/goalies.csv",
    ):
        df = _fetch_csv(fmt)
        if not df.empty:
            return df
    return pd.DataFrame()


def get_goalie_5v5(season: Optional[str] = None) -> pd.DataFrame:
    """Goalie stats at 5v5."""
    df = get_goalie_stats(season)
    if df.empty:
        return df
    if "situation" in df.columns:
        return df[df["situation"] == "5on5"].copy()
    return df


# ── Skater stats ─────────────────────────────────────────────────────

def get_skater_stats(season: Optional[str] = None) -> pd.DataFrame:
    """Individual skater stats with xGoals, ice-time, on-ice metrics."""
    season = season or cfg.current_season[:4]
    for fmt in (
        f"{MP_BASE}/{season}/regular/skaters.csv",
        f"https://moneypuck.com/moneypuck/playerData/seasonSummary/{season}/regular/skaters.csv",
    ):
        df = _fetch_csv(fmt)
        if not df.empty:
            return df
    return pd.DataFrame()


def get_skater_5v5(season: Optional[str] = None) -> pd.DataFrame:
    """Skater stats at 5v5."""
    df = get_skater_stats(season)
    if df.empty:
        return df
    if "situation" in df.columns:
        return df[df["situation"] == "5on5"].copy()
    return df


# ── Convenience: key team metrics as dict ────────────────────────────

def get_team_xg_summary(season: Optional[str] = None) -> dict:
    """
    Returns {team_abbrev: {xGF, xGA, CF%, FF%, hdCF%, ...}} for all teams.
    Tries 5v5 first, falls back to all-situations.
    """
    df = get_team_5v5(season)
    if df.empty:
        df = get_team_all_situations(season)
    if df.empty:
        return {}

    result = {}
    team_col = "team" if "team" in df.columns else df.columns[0]
    for _, row in df.iterrows():
        team = row.get(team_col, "")
        result[team] = {
            "xGoalsFor": row.get("xGoalsFor", 0),
            "xGoalsAgainst": row.get("xGoalsAgainst", 0),
            "corsiFor": row.get("corsiForAfterShifts", row.get("corsiFor", 0)),
            "corsiAgainst": row.get("corsiAgainstAfterShifts", row.get("corsiAgainst", 0)),
            "fenwickFor": row.get("fenwickForAfterShifts", row.get("fenwickFor", 0)),
            "fenwickAgainst": row.get("fenwickAgainstAfterShifts", row.get("fenwickAgainst", 0)),
            "highDangerGoalsFor": row.get("highDangerGoalsFor", 0),
            "highDangerGoalsAgainst": row.get("highDangerGoalsAgainst", 0),
            "shotsOnGoalFor": row.get("shotsOnGoalFor", 0),
            "shotsOnGoalAgainst": row.get("shotsOnGoalAgainst", 0),
            "goalsFor": row.get("goalsFor", 0),
            "goalsAgainst": row.get("goalsAgainst", 0),
            "gamesPlayed": row.get("games_played", row.get("gamesPlayed", 0)),
        }
    return result
