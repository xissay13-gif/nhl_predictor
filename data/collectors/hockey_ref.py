"""
Hockey-Reference scraper for historical stats, game logs, and archival data.

Used primarily for:
  - Historical team stats
  - Player career stats vs specific opponents
  - Goalie career splits
"""

import io
import logging
import re
from typing import Optional

import pandas as pd

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None

from utils.helpers import safe_request

logger = logging.getLogger("nhl_predictor.hockey_ref")

HR_BASE = "https://www.hockey-reference.com"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; NHLPredictor/1.0)"}


def get_team_stats_page(season_end_year: int = 2026) -> pd.DataFrame:
    """
    Parse team stats table from Hockey-Reference.
    E.g., /leagues/NHL_2026.html
    """
    if BeautifulSoup is None:
        return pd.DataFrame()

    url = f"{HR_BASE}/leagues/NHL_{season_end_year}.html"
    resp = safe_request(url, headers=HEADERS)
    if resp is None:
        return pd.DataFrame()

    soup = BeautifulSoup(resp.text, "lxml")
    table = soup.find("table", {"id": "stats"})
    if table is None:
        return pd.DataFrame()

    try:
        return pd.read_html(io.StringIO(str(table)))[0]
    except Exception as exc:
        logger.error("Failed to parse HR stats table: %s", exc)
        return pd.DataFrame()


def get_team_game_log_hr(team_abbrev: str, season_end_year: int = 2026) -> pd.DataFrame:
    """
    Team game log from Hockey-Reference.
    /teams/{abbrev}/{year}_gamelog.html
    """
    if BeautifulSoup is None:
        return pd.DataFrame()

    url = f"{HR_BASE}/teams/{team_abbrev}/{season_end_year}_gamelog.html"
    resp = safe_request(url, headers=HEADERS)
    if resp is None:
        return pd.DataFrame()

    soup = BeautifulSoup(resp.text, "lxml")
    table = soup.find("table", {"id": "tm_gamelog_rs"})
    if table is None:
        return pd.DataFrame()

    try:
        df = pd.read_html(io.StringIO(str(table)))[0]
        # Flatten multi-level columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ["_".join(str(c) for c in col).strip("_") for col in df.columns]
        return df
    except Exception as exc:
        logger.error("Failed to parse HR game log for %s: %s", team_abbrev, exc)
        return pd.DataFrame()
