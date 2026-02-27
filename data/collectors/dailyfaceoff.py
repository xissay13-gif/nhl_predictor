"""
DailyFaceoff scraper for starting goalies, line combinations, and injuries.

Note: web scraping — structure may change. Fallback gracefully.
"""

import logging
import re
from typing import Optional

import pandas as pd

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None

from config import cfg
from utils.helpers import safe_request

logger = logging.getLogger("nhl_predictor.dailyfaceoff")

DFO_BASE = cfg.dailyfaceoff_url
HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; NHLPredictor/1.0)"
}


def get_starting_goalies() -> pd.DataFrame:
    """
    Scrape confirmed / expected starting goalies for today's games.
    Returns: team, goalie_name, status (confirmed/expected/unconfirmed).
    """
    if BeautifulSoup is None:
        logger.warning("beautifulsoup4 not installed — cannot scrape DailyFaceoff")
        return pd.DataFrame()

    resp = safe_request(f"{DFO_BASE}/starting-goalies/", headers=HEADERS)
    if resp is None:
        return pd.DataFrame()

    soup = BeautifulSoup(resp.text, "lxml")
    rows = []

    # DailyFaceoff uses cards for each matchup
    cards = soup.select(".starting-goalies-card, .goalie-card, [class*='matchup']")
    if not cards:
        # Fallback: try parsing any goalie-related containers
        cards = soup.find_all("div", class_=re.compile(r"goalie|starter|matchup", re.I))

    for card in cards:
        # Try to extract team abbreviation and goalie name
        team_els = card.select("[class*='team-abbr'], [class*='team-name'], .team-logo")
        goalie_els = card.select("[class*='goalie-name'], [class*='player-name'], h4, h5")
        status_els = card.select("[class*='status'], [class*='confirmed'], [class*='label']")

        teams = [el.get_text(strip=True) for el in team_els]
        goalies = [el.get_text(strip=True) for el in goalie_els]
        statuses = [el.get_text(strip=True).lower() for el in status_els]

        for i, (team, goalie) in enumerate(zip(teams, goalies)):
            status = statuses[i] if i < len(statuses) else "unconfirmed"
            if "confirm" in status:
                status = "confirmed"
            elif "expect" in status or "likely" in status:
                status = "expected"
            elif "unconfirm" in status:
                status = "unconfirmed"

            rows.append({
                "team": team,
                "goalie_name": goalie,
                "status": status,
            })

    return pd.DataFrame(rows)


def get_line_combinations(team: str) -> dict:
    """
    Scrape projected line combinations for a team.
    Returns dict with keys: forwards (lines 1-4), defense (pairs 1-3), pp, pk.
    """
    if BeautifulSoup is None:
        return {}

    team_slug = team.lower().replace(" ", "-")
    resp = safe_request(f"{DFO_BASE}/teams/{team_slug}/line-combinations/", headers=HEADERS)
    if resp is None:
        return {}

    soup = BeautifulSoup(resp.text, "lxml")
    result = {"forwards": [], "defense": [], "pp1": [], "pp2": [], "pk1": [], "pk2": []}

    # Extract forward lines
    for line_div in soup.select("[class*='forward-line'], [class*='fwd-line']"):
        players = [el.get_text(strip=True) for el in line_div.select("[class*='player']")]
        if players:
            result["forwards"].append(players)

    # Extract defense pairs
    for pair_div in soup.select("[class*='defense-pair'], [class*='def-pair']"):
        players = [el.get_text(strip=True) for el in pair_div.select("[class*='player']")]
        if players:
            result["defense"].append(players)

    return result


def get_injuries() -> pd.DataFrame:
    """
    Scrape current injury report.
    Returns: team, player_name, injury, status, estimated_return.
    """
    import json

    resp = safe_request(f"{DFO_BASE}/hockey-player-news/injuries", headers=HEADERS)
    if resp is None:
        logger.warning("Could not reach DailyFaceoff injury report")
        return pd.DataFrame()

    rows = []

    # Page embeds JSON in __NEXT_DATA__ script tag
    soup = BeautifulSoup(resp.text, "html.parser") if BeautifulSoup else None
    if soup is None:
        return pd.DataFrame()

    script = soup.find("script", id="__NEXT_DATA__")
    if script and script.string:
        try:
            data = json.loads(script.string)
            page_data = data["props"]["pageProps"]["data"]
            news_items = page_data.get("data", [])

            for item in news_items:
                rows.append({
                    "team": item.get("teamName", ""),
                    "player_name": item.get("playerName", ""),
                    "injury": item.get("details", ""),
                    "status": item.get("newsCategoryName", ""),
                    "estimated_return": item.get("date", ""),
                })
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"Failed to parse __NEXT_DATA__ JSON: {e}")

    if not rows:
        logger.warning("No injury data found on DailyFaceoff page")

    return pd.DataFrame(rows)
