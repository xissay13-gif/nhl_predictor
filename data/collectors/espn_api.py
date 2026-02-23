"""
ESPN API collector — fallback for schedule/scores/team stats.

Endpoints:
  - /scoreboard            → today's games
  - /scoreboard?dates=…    → specific date
  - /teams                 → team list + records
  - /teams/{id}/statistics → team season stats
"""

import logging
from typing import Optional

import pandas as pd

from config import cfg
from utils.helpers import safe_request

logger = logging.getLogger("nhl_predictor.espn_api")
BASE = cfg.espn_base_url


def get_scoreboard(target_date: Optional[str] = None) -> pd.DataFrame:
    """
    Fetch ESPN scoreboard. target_date in YYYYMMDD format.
    Returns schedule-like DataFrame.
    """
    params = {}
    if target_date:
        params["dates"] = target_date.replace("-", "")
    resp = safe_request(f"{BASE}/scoreboard", params=params)
    if resp is None:
        return pd.DataFrame()

    data = resp.json()
    rows = []
    for event in data.get("events", []):
        competition = event.get("competitions", [{}])[0]
        competitors = competition.get("competitors", [])
        home = next((c for c in competitors if c.get("homeAway") == "home"), {})
        away = next((c for c in competitors if c.get("homeAway") == "away"), {})
        rows.append({
            "game_id": event.get("id"),
            "date": event.get("date", "")[:10],
            "home_team": home.get("team", {}).get("abbreviation", ""),
            "away_team": away.get("team", {}).get("abbreviation", ""),
            "home_score": int(home.get("score", 0)) if home.get("score") else None,
            "away_score": int(away.get("score", 0)) if away.get("score") else None,
            "status": event.get("status", {}).get("type", {}).get("name", ""),
            "venue": competition.get("venue", {}).get("fullName", ""),
        })
    return pd.DataFrame(rows)


def get_teams() -> pd.DataFrame:
    """Team list with basic records."""
    resp = safe_request(f"{BASE}/teams")
    if resp is None:
        return pd.DataFrame()

    data = resp.json()
    rows = []
    for t in data.get("sports", [{}])[0].get("leagues", [{}])[0].get("teams", []):
        team = t.get("team", {})
        rows.append({
            "espn_id": team.get("id"),
            "abbrev": team.get("abbreviation", ""),
            "name": team.get("displayName", ""),
            "location": team.get("location", ""),
        })
    return pd.DataFrame(rows)


def get_team_stats_espn(team_id: int) -> dict:
    """Team season statistics from ESPN."""
    resp = safe_request(f"{BASE}/teams/{team_id}/statistics")
    if resp is None:
        return {}
    return resp.json()
