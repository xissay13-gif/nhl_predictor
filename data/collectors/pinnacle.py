"""
Pinnacle odds collector.

Fetches NHL odds from Pinnacle's guest API (arcadia) in decimal (European) format.
Uses the same public endpoint that powers pinnacle.com.
No API key required — uses the guest key embedded in the site frontend.
"""

import logging
from datetime import datetime, timezone

import pandas as pd

from utils.helpers import safe_request

logger = logging.getLogger("nhl_predictor.pinnacle")

BASE_URL = "https://guest.api.arcadia.pinnacle.com/0.1"

# Public guest API key (used by pinnacle.com frontend)
API_KEY = "CmX2KcMrXuFmNg6YFbmTxE0y9CIrOi0R"

HEADERS = {
    "X-API-Key": API_KEY,
    "Referer": "https://www.pinnacle.com/",
    "Accept": "application/json",
}

NHL_LEAGUE_ID = 1456

# Pinnacle full name → NHL API abbreviation
_TEAM_ABBREV = {
    "Anaheim Ducks": "ANA",
    "Arizona Coyotes": "ARI",
    "Boston Bruins": "BOS",
    "Buffalo Sabres": "BUF",
    "Calgary Flames": "CGY",
    "Carolina Hurricanes": "CAR",
    "Chicago Blackhawks": "CHI",
    "Colorado Avalanche": "COL",
    "Columbus Blue Jackets": "CBJ",
    "Dallas Stars": "DAL",
    "Detroit Red Wings": "DET",
    "Edmonton Oilers": "EDM",
    "Florida Panthers": "FLA",
    "Los Angeles Kings": "LAK",
    "Minnesota Wild": "MIN",
    "Montreal Canadiens": "MTL",
    "Nashville Predators": "NSH",
    "New Jersey Devils": "NJD",
    "New York Islanders": "NYI",
    "New York Rangers": "NYR",
    "Ottawa Senators": "OTT",
    "Philadelphia Flyers": "PHI",
    "Pittsburgh Penguins": "PIT",
    "San Jose Sharks": "SJS",
    "Seattle Kraken": "SEA",
    "St. Louis Blues": "STL",
    "Tampa Bay Lightning": "TBL",
    "Toronto Maple Leafs": "TOR",
    "Utah Hockey Club": "UTA",
    "Utah Mammoth": "UTA",
    "Vancouver Canucks": "VAN",
    "Vegas Golden Knights": "VGK",
    "Washington Capitals": "WSH",
    "Winnipeg Jets": "WPG",
}


def _normalize_team(name: str) -> str:
    """Convert Pinnacle team name to NHL abbreviation."""
    return _TEAM_ABBREV.get(name, name)


def _american_to_decimal(american_odds: float) -> float:
    """Convert American odds to decimal."""
    if american_odds > 0:
        return round(american_odds / 100 + 1, 4)
    else:
        return round(-100 / american_odds + 1, 4)


def _decimal_to_implied(decimal_odds: float) -> float:
    """Convert decimal odds to implied probability."""
    if decimal_odds <= 0:
        return 0.0
    return 1.0 / decimal_odds


def _get_matchups() -> list[dict]:
    """Fetch NHL matchups from Pinnacle."""
    resp = safe_request(
        f"{BASE_URL}/leagues/{NHL_LEAGUE_ID}/matchups",
        headers=HEADERS,
    )
    if resp is None:
        return []
    return resp.json()


def _get_straight_odds() -> list[dict]:
    """Fetch straight market odds for NHL."""
    resp = safe_request(
        f"{BASE_URL}/leagues/{NHL_LEAGUE_ID}/markets/straight",
        headers=HEADERS,
    )
    if resp is None:
        return []
    return resp.json()


def get_pinnacle_odds() -> list[dict]:
    """
    Main function: fetch NHL odds from Pinnacle.
    Returns list of dicts with moneyline, spread, totals in DECIMAL format.
    """
    matchups_raw = _get_matchups()

    matchups = {}
    for m in matchups_raw:
        if m.get("type") != "matchup":
            continue
        if m.get("isLive", False):
            continue

        participants = m.get("participants", [])
        if len(participants) < 2:
            continue

        home = away = None
        for p in participants:
            if p.get("alignment") == "home":
                home = p.get("name")
            elif p.get("alignment") == "away":
                away = p.get("name")

        if not home or not away:
            continue

        matchups[m["id"]] = {
            "id": m["id"],
            "home": _normalize_team(home),
            "away": _normalize_team(away),
            "start_time": m.get("startTime"),
        }

    odds_raw = _get_straight_odds()

    odds_by_matchup = {}
    for o in odds_raw:
        matchup_id = o.get("matchupId")
        if matchup_id not in matchups:
            continue

        if matchup_id not in odds_by_matchup:
            odds_by_matchup[matchup_id] = {
                "moneyline": None,
                "spread": None,
                "total": None,
            }

        market_type = o.get("type")
        period = o.get("period", 0)

        # Full game only
        if period != 0:
            continue

        prices = o.get("prices", [])
        if not prices:
            continue

        if market_type == "moneyline":
            ml = {}
            for p in prices:
                designation = p.get("designation")
                price = p.get("price")
                if designation and price is not None:
                    ml[designation] = _american_to_decimal(price)
            odds_by_matchup[matchup_id]["moneyline"] = ml

        elif market_type == "spread":
            spread_data = {}
            for p in prices:
                designation = p.get("designation")
                price = p.get("price")
                points = p.get("points")
                if designation and price is not None:
                    spread_data[designation] = {
                        "line": points,
                        "odds": _american_to_decimal(price),
                    }
            if spread_data:
                odds_by_matchup[matchup_id]["spread"] = spread_data

        elif market_type == "total":
            total_data = {}
            for p in prices:
                designation = p.get("designation")
                price = p.get("price")
                points = p.get("points")
                if designation and price is not None:
                    total_data[designation] = {
                        "line": points,
                        "odds": _american_to_decimal(price),
                    }
            if total_data:
                odds_by_matchup[matchup_id]["total"] = total_data

    results = []
    for mid, info in matchups.items():
        odds = odds_by_matchup.get(mid, {})
        results.append({
            "matchup_id": mid,
            "home": info["home"],
            "away": info["away"],
            "start_time": info["start_time"],
            "moneyline": odds.get("moneyline"),
            "spread": odds.get("spread"),
            "total": odds.get("total"),
        })

    results.sort(key=lambda x: x.get("start_time") or "")
    return results


def get_pinnacle_moneyline() -> pd.DataFrame:
    """
    Pinnacle moneyline odds as DataFrame.
    Returns: home_team, away_team, home_odds, away_odds (decimal),
             home_implied, away_implied.
    """
    games = get_pinnacle_odds()
    rows = []
    for g in games:
        ml = g.get("moneyline")
        if not ml:
            continue
        home_odds = ml.get("home")
        away_odds = ml.get("away")
        if home_odds and away_odds:
            rows.append({
                "home_team": g["home"],
                "away_team": g["away"],
                "commence_time": g["start_time"],
                "bookmaker": "pinnacle",
                "home_odds": home_odds,
                "away_odds": away_odds,
                "home_implied": _decimal_to_implied(home_odds),
                "away_implied": _decimal_to_implied(away_odds),
            })
    return pd.DataFrame(rows)


def get_pinnacle_totals() -> pd.DataFrame:
    """
    Pinnacle totals (over/under) as DataFrame.
    Returns: home_team, away_team, side, total, odds (decimal), implied.
    """
    games = get_pinnacle_odds()
    rows = []
    for g in games:
        total = g.get("total")
        if not total:
            continue
        for side in ["over", "under"]:
            data = total.get(side)
            if data:
                rows.append({
                    "home_team": g["home"],
                    "away_team": g["away"],
                    "commence_time": g["start_time"],
                    "bookmaker": "pinnacle",
                    "side": side,
                    "total": data.get("line", 0),
                    "odds": data.get("odds", 0),
                    "implied": _decimal_to_implied(data.get("odds", 0)),
                })
    return pd.DataFrame(rows)


def get_pinnacle_spreads() -> pd.DataFrame:
    """
    Pinnacle spread (puck line) odds as DataFrame.
    Returns: home_team, away_team, side, spread, odds (decimal), implied.
    """
    games = get_pinnacle_odds()
    rows = []
    for g in games:
        spread = g.get("spread")
        if not spread:
            continue
        for side_key, side_name in [("home", "home"), ("away", "away")]:
            data = spread.get(side_key)
            if data:
                rows.append({
                    "home_team": g["home"],
                    "away_team": g["away"],
                    "commence_time": g["start_time"],
                    "bookmaker": "pinnacle",
                    "side": side_name,
                    "spread": data.get("line", 0),
                    "odds": data.get("odds", 0),
                    "implied": _decimal_to_implied(data.get("odds", 0)),
                })
    return pd.DataFrame(rows)


def get_pinnacle_consensus() -> pd.DataFrame:
    """
    Pinnacle consensus odds in the same format as odds_api.get_consensus_odds().
    Since Pinnacle is a single sharp book, we return one row per game
    with Pinnacle's line as the consensus.
    """
    ml = get_pinnacle_moneyline()
    if ml.empty:
        return pd.DataFrame()

    # Pinnacle is one bookmaker — build consensus-like output
    rows = []
    for _, row in ml.iterrows():
        rows.append({
            "home_team": row["home_team"],
            "away_team": row["away_team"],
            "commence_time": row["commence_time"],
            "home_implied_mean": row["home_implied"],
            "away_implied_mean": row["away_implied"],
            "home_implied_min": row["home_implied"],
            "home_implied_max": row["home_implied"],
            "away_implied_min": row["away_implied"],
            "away_implied_max": row["away_implied"],
            "num_books": 1,
            "home_best_odds": row["home_odds"],
            "away_best_odds": row["away_odds"],
            "source": "pinnacle",
        })
    return pd.DataFrame(rows)
