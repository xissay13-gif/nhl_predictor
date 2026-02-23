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
    if BeautifulSoup is None:
        return pd.DataFrame()

    # Try multiple URL patterns since DailyFaceoff changes structure
    urls = [
        f"{DFO_BASE}/injuries",
        f"{DFO_BASE}/teams/injury-report/",
        f"{DFO_BASE}/injury-report/",
    ]
    resp = None
    for url in urls:
        resp = safe_request(url, headers=HEADERS, retries=1)
        if resp is not None:
            break
    if resp is None:
        logger.warning("Could not reach DailyFaceoff injury report")
        return pd.DataFrame()
    if resp is None:
        return pd.DataFrame()

    soup = BeautifulSoup(resp.text, "lxml")
    rows = []

    # Parse injury tables
    tables = soup.find_all("table")
    for table in tables:
        # Try to get team name from header above table
        header = table.find_previous(["h2", "h3", "h4"])
        team_name = header.get_text(strip=True) if header else "Unknown"

        for tr in table.find_all("tr")[1:]:  # skip header
            cells = tr.find_all("td")
            if len(cells) >= 3:
                rows.append({
                    "team": team_name,
                    "player_name": cells[0].get_text(strip=True),
                    "injury": cells[1].get_text(strip=True) if len(cells) > 1 else "",
                    "status": cells[2].get_text(strip=True) if len(cells) > 2 else "",
                    "estimated_return": cells[3].get_text(strip=True) if len(cells) > 3 else "",
                })

    return pd.DataFrame(rows)
