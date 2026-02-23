"""
Natural Stat Trick scraper for Corsi, Fenwick, high-danger chances, zone data.

NST provides excellent underlying possession and shot quality metrics.
Data is scraped from their report pages.
"""

import io
import logging
from typing import Optional

import pandas as pd

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None

from utils.helpers import safe_request

logger = logging.getLogger("nhl_predictor.nst")

NST_BASE = "https://www.naturalstattrick.com"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; NHLPredictor/1.0)"}


def get_team_table(
    season: str = "20252026",
    sit: str = "5v5",        # "5v5", "all", "pp", "pk", "sva"
    score: str = "all",      # "all", "leading", "trailing", "close"
    rate: bool = False,      # per-60 rates
) -> pd.DataFrame:
    """
    Team-level Corsi/Fenwick/xG table from NST.
    """
    if BeautifulSoup is None:
        return pd.DataFrame()

    from_season = season[:4] + season[4:]  # e.g. "20252026"
    params = {
        "fromseason": from_season,
        "thruseason": from_season,
        "stype": "2",         # regular season
        "sit": sit,
        "score": score,
        "rate": "y" if rate else "n",
        "team": "all",
        "loc": "B",           # both home and away
        "gpf": "410",         # games played filter
        "fd": "",
        "td": "",
    }

    resp = safe_request(f"{NST_BASE}/teamtable.php", params=params, headers=HEADERS)
    if resp is None:
        return pd.DataFrame()

    soup = BeautifulSoup(resp.text, "lxml")
    table = soup.find("table", {"id": "teams"})
    if table is None:
        # Try any table
        tables = soup.find_all("table")
        if tables:
            table = tables[0]
        else:
            return pd.DataFrame()

    try:
        df = pd.read_html(io.StringIO(str(table)))[0]
        return df
    except Exception as exc:
        logger.error("Failed to parse NST team table: %s", exc)
        return pd.DataFrame()


def get_team_lines(team: str, season: str = "20252026") -> pd.DataFrame:
    """Line combination stats from NST."""
    if BeautifulSoup is None:
        return pd.DataFrame()

    params = {
        "fromseason": season,
        "thruseason": season,
        "stype": "2",
        "sit": "5v5",
        "team": team,
    }

    resp = safe_request(f"{NST_BASE}/linestats.php", params=params, headers=HEADERS)
    if resp is None:
        return pd.DataFrame()

    try:
        dfs = pd.read_html(io.StringIO(resp.text))
        return dfs[0] if dfs else pd.DataFrame()
    except Exception:
        return pd.DataFrame()
