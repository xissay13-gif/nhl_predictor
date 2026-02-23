"""
NHL Edge API collector.

Endpoints used (unofficial but stable, no auth required):
  - /v1/schedule/{date}           → schedule
  - /v1/standings/now              → standings
  - /v1/club-stats/{team}/{season} → team season stats
  - /v1/score/now                  → live scores
  - /v1/club-schedule-season/{team}/{season} → full team schedule
  - /v1/player/{id}/landing        → player card
  - /v1/roster/{team}/current      → roster
"""

import logging
from datetime import date, timedelta
from typing import Optional

import pandas as pd

from config import cfg
from utils.helpers import safe_request, safe_div

logger = logging.getLogger("nhl_predictor.nhl_api")
BASE = cfg.nhl_base_url


# ── Schedule ─────────────────────────────────────────────────────────

def get_schedule(target_date: Optional[str] = None) -> pd.DataFrame:
    """
    Fetch NHL schedule for a given date (YYYY-MM-DD) or today.
    Returns DataFrame with columns:
        game_id, date, home_team, away_team, start_time, game_type, venue, status
    """
    endpoint = f"{BASE}/schedule/{target_date}" if target_date else f"{BASE}/schedule/now"
    resp = safe_request(endpoint)
    if resp is None:
        return pd.DataFrame()

    data = resp.json()
    rows = []
    for day in data.get("gameWeek", []):
        d = day.get("date", "")
        for g in day.get("games", []):
            rows.append({
                "game_id": g.get("id"),
                "date": d,
                "home_team": g.get("homeTeam", {}).get("abbrev", ""),
                "away_team": g.get("awayTeam", {}).get("abbrev", ""),
                "home_team_id": g.get("homeTeam", {}).get("id"),
                "away_team_id": g.get("awayTeam", {}).get("id"),
                "start_time": g.get("startTimeUTC", ""),
                "game_type": g.get("gameType"),           # 2=regular, 3=playoff
                "venue": g.get("venue", {}).get("default", ""),
                "status": g.get("gameState", ""),
            })

    return pd.DataFrame(rows)


def get_schedule_range(start: str, end: str) -> pd.DataFrame:
    """Fetch schedule for a date range. Dates are YYYY-MM-DD strings."""
    frames = []
    current = date.fromisoformat(start)
    end_dt = date.fromisoformat(end)
    while current <= end_dt:
        df = get_schedule(current.isoformat())
        if not df.empty:
            frames.append(df)
        current += timedelta(days=1)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ── Standings ────────────────────────────────────────────────────────

def get_standings() -> pd.DataFrame:
    """Current standings with W/L/OTL, points, goal diff, home/away records."""
    resp = safe_request(f"{BASE}/standings/now")
    if resp is None:
        return pd.DataFrame()

    data = resp.json()
    rows = []
    for t in data.get("standings", []):
        rows.append({
            "team": t.get("teamAbbrev", {}).get("default", ""),
            "team_name": t.get("teamName", {}).get("default", ""),
            "conference": t.get("conferenceName", ""),
            "division": t.get("divisionName", ""),
            "games_played": t.get("gamesPlayed", 0),
            "wins": t.get("wins", 0),
            "losses": t.get("losses", 0),
            "ot_losses": t.get("otLosses", 0),
            "points": t.get("points", 0),
            "points_pct": t.get("pointPctg", 0),
            "goals_for": t.get("goalFor", 0),
            "goals_against": t.get("goalAgainst", 0),
            "goal_diff": t.get("goalDifferential", 0),
            "goals_for_pg": t.get("goalFor", 0) / max(t.get("gamesPlayed", 1), 1),
            "goals_against_pg": t.get("goalAgainst", 0) / max(t.get("gamesPlayed", 1), 1),
            "regulation_wins": t.get("regulationWins", 0),
            "home_wins": t.get("homeWins", 0),
            "home_losses": t.get("homeLosses", 0),
            "home_ot_losses": t.get("homeOtLosses", 0),
            "away_wins": t.get("roadWins", 0),
            "away_losses": t.get("roadLosses", 0),
            "away_ot_losses": t.get("roadOtLosses", 0),
            "streak_code": t.get("streakCode", ""),
            "streak_count": t.get("streakCount", 0),
            "l10_wins": t.get("l10Wins", 0),
            "l10_losses": t.get("l10Losses", 0),
            "l10_ot_losses": t.get("l10OtLosses", 0),
            "wildcard_sequence": t.get("wildcardSequence", 0),
        })
    return pd.DataFrame(rows)


# ── Team Season Stats ────────────────────────────────────────────────

def get_team_stats(team: str, season: Optional[str] = None) -> dict:
    """
    Detailed team stats for a season.
    Returns dict with faceoff%, PP%, PK%, shots, etc.
    """
    season = season or cfg.current_season
    resp = safe_request(f"{BASE}/club-stats/{team}/{season}/2")
    if resp is None:
        return {}
    return resp.json()


# ── Roster ───────────────────────────────────────────────────────────

def get_roster(team: str) -> pd.DataFrame:
    """Current roster for a team."""
    resp = safe_request(f"{BASE}/roster/{team}/current")
    if resp is None:
        return pd.DataFrame()

    data = resp.json()
    rows = []
    for position_group in ("forwards", "defensemen", "goalies"):
        for p in data.get(position_group, []):
            rows.append({
                "player_id": p.get("id"),
                "first_name": p.get("firstName", {}).get("default", ""),
                "last_name": p.get("lastName", {}).get("default", ""),
                "position": p.get("positionCode", ""),
                "jersey": p.get("sweaterNumber"),
                "shoots_catches": p.get("shootsCatches", ""),
                "height_inches": p.get("heightInInches"),
                "weight_pounds": p.get("weightInPounds"),
                "birth_date": p.get("birthDate", ""),
                "team": team,
            })
    return pd.DataFrame(rows)


# ── Team Game Log (full season results) ──────────────────────────────

def get_team_game_log(team: str, season: Optional[str] = None) -> pd.DataFrame:
    """
    Full season game log for a team.
    Includes game results, scores, opponents.
    """
    season = season or cfg.current_season
    resp = safe_request(f"{BASE}/club-schedule-season/{team}/{season}")
    if resp is None:
        return pd.DataFrame()

    data = resp.json()
    rows = []
    for g in data.get("games", []):
        if g.get("gameState") not in ("OFF", "FINAL"):
            continue
        home = g.get("homeTeam", {})
        away = g.get("awayTeam", {})
        is_home = home.get("abbrev") == team
        rows.append({
            "game_id": g.get("id"),
            "date": g.get("gameDate", ""),
            "team": team,
            "opponent": away.get("abbrev") if is_home else home.get("abbrev"),
            "is_home": is_home,
            "goals_for": home.get("score", 0) if is_home else away.get("score", 0),
            "goals_against": away.get("score", 0) if is_home else home.get("score", 0),
            "game_type": g.get("gameType"),
            "period": g.get("periodDescriptor", {}).get("number"),
            "ot": g.get("periodDescriptor", {}).get("number", 3) > 3,
            "shootout": g.get("periodDescriptor", {}).get("periodType") == "SO",
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df["win"] = df["goals_for"] > df["goals_against"]
        df["loss"] = df["goals_for"] < df["goals_against"]
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
    return df


# ── Player Stats ─────────────────────────────────────────────────────

def get_player_stats(player_id: int) -> dict:
    """Get player season stats from landing page."""
    resp = safe_request(f"{BASE}/player/{player_id}/landing")
    if resp is None:
        return {}
    return resp.json()


# ── Scoreboard (live / recent) ───────────────────────────────────────

def get_scores() -> pd.DataFrame:
    """Today's scoreboard."""
    resp = safe_request(f"{BASE}/score/now")
    if resp is None:
        return pd.DataFrame()

    data = resp.json()
    rows = []
    for g in data.get("games", []):
        rows.append({
            "game_id": g.get("id"),
            "home_team": g.get("homeTeam", {}).get("abbrev"),
            "away_team": g.get("awayTeam", {}).get("abbrev"),
            "home_score": g.get("homeTeam", {}).get("score"),
            "away_score": g.get("awayTeam", {}).get("score"),
            "status": g.get("gameState"),
            "period": g.get("periodDescriptor", {}).get("number"),
        })
    return pd.DataFrame(rows)
