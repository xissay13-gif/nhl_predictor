"""
The Odds API collector.

Provides ML (moneyline), spreads (puck line), totals from multiple bookmakers.
Requires an API key (free tier = 500 requests/month).

Docs: https://the-odds-api.com/liveapi/guides/v4/
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd

from config import cfg
from utils.helpers import safe_request, implied_prob

logger = logging.getLogger("nhl_predictor.odds_api")

BASE = cfg.odds_base_url


def _get_odds(markets: str = "h2h,spreads,totals", regions: str = "us,eu",
              target_date: str = None) -> list:
    """Raw odds response from The Odds API, optionally filtered to a single date."""
    if not cfg.odds_api_key:
        logger.warning("ODDS_API_KEY not set — skipping odds fetch")
        return []

    params = {
        "apiKey": cfg.odds_api_key,
        "regions": regions,
        "markets": markets,
        "oddsFormat": "american",
    }

    # Filter to a specific date window if provided
    if target_date:
        dt = datetime.strptime(target_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        params["commenceTimeFrom"] = dt.strftime("%Y-%m-%dT00:00:00Z")
        params["commenceTimeTo"] = (dt + timedelta(days=1)).strftime("%Y-%m-%dT00:00:00Z")

    resp = safe_request(f"{BASE}/odds", params=params)
    if resp is None:
        return []
    return resp.json()


def get_moneyline(target_date: str = None) -> pd.DataFrame:
    """
    Moneyline odds for upcoming NHL games.
    Returns: game_id, home_team, away_team, bookmaker, home_odds, away_odds,
             home_implied, away_implied.
    """
    events = _get_odds(markets="h2h", target_date=target_date)
    rows = []
    for ev in events:
        home = ev.get("home_team", "")
        away = ev.get("away_team", "")
        game_id = ev.get("id", "")
        commence = ev.get("commence_time", "")

        for bm in ev.get("bookmakers", []):
            for market in bm.get("markets", []):
                if market.get("key") != "h2h":
                    continue
                outcomes = {o["name"]: o["price"] for o in market.get("outcomes", [])}
                home_odds = outcomes.get(home)
                away_odds = outcomes.get(away)
                if home_odds is not None and away_odds is not None:
                    rows.append({
                        "game_id": game_id,
                        "commence_time": commence,
                        "home_team": home,
                        "away_team": away,
                        "bookmaker": bm.get("key", ""),
                        "home_odds": home_odds,
                        "away_odds": away_odds,
                        "home_implied": implied_prob(home_odds),
                        "away_implied": implied_prob(away_odds),
                    })
    return pd.DataFrame(rows)


def get_spreads(target_date: str = None) -> pd.DataFrame:
    """Puck line (spread) odds."""
    events = _get_odds(markets="spreads", target_date=target_date)
    rows = []
    for ev in events:
        home = ev.get("home_team", "")
        away = ev.get("away_team", "")
        game_id = ev.get("id", "")
        commence = ev.get("commence_time", "")

        for bm in ev.get("bookmakers", []):
            for market in bm.get("markets", []):
                if market.get("key") != "spreads":
                    continue
                for o in market.get("outcomes", []):
                    is_home = o.get("name") == home
                    rows.append({
                        "game_id": game_id,
                        "commence_time": commence,
                        "home_team": home,
                        "away_team": away,
                        "bookmaker": bm.get("key", ""),
                        "side": "home" if is_home else "away",
                        "spread": o.get("point", 0),
                        "odds": o.get("price", 0),
                        "implied": implied_prob(o.get("price", -110)),
                    })
    return pd.DataFrame(rows)


def get_totals(target_date: str = None) -> pd.DataFrame:
    """Over/Under totals odds."""
    events = _get_odds(markets="totals", target_date=target_date)
    rows = []
    for ev in events:
        home = ev.get("home_team", "")
        away = ev.get("away_team", "")
        game_id = ev.get("id", "")
        commence = ev.get("commence_time", "")

        for bm in ev.get("bookmakers", []):
            for market in bm.get("markets", []):
                if market.get("key") != "totals":
                    continue
                for o in market.get("outcomes", []):
                    rows.append({
                        "game_id": game_id,
                        "commence_time": commence,
                        "home_team": home,
                        "away_team": away,
                        "bookmaker": bm.get("key", ""),
                        "side": o.get("name", "").lower(),  # "over" or "under"
                        "total": o.get("point", 0),
                        "odds": o.get("price", 0),
                        "implied": implied_prob(o.get("price", -110)),
                    })
    return pd.DataFrame(rows)


def get_consensus_odds(target_date: str = None) -> pd.DataFrame:
    """
    Aggregate consensus odds across bookmakers.
    Returns one row per game with mean implied probabilities and opening/closing.
    """
    ml = get_moneyline(target_date=target_date)
    if ml.empty:
        return pd.DataFrame()

    consensus = ml.groupby(["game_id", "home_team", "away_team", "commence_time"]).agg(
        home_implied_mean=("home_implied", "mean"),
        away_implied_mean=("away_implied", "mean"),
        home_implied_min=("home_implied", "min"),
        home_implied_max=("home_implied", "max"),
        away_implied_min=("away_implied", "min"),
        away_implied_max=("away_implied", "max"),
        num_books=("bookmaker", "nunique"),
        home_best_odds=("home_odds", "max"),
        away_best_odds=("away_odds", "max"),
    ).reset_index()

    return consensus
