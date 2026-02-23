"""
Shared utility functions.
"""

import time
import math
import logging
import requests
import pandas as pd
from functools import wraps
from typing import Optional

logger = logging.getLogger("nhl_predictor")


def safe_request(
    url: str,
    params: Optional[dict] = None,
    headers: Optional[dict] = None,
    retries: int = 3,
    backoff: float = 2.0,
    timeout: int = 30,
) -> Optional[requests.Response]:
    """HTTP GET with exponential backoff."""
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=timeout)
            resp.raise_for_status()
            return resp
        except requests.RequestException as exc:
            wait = backoff * (2 ** attempt)
            logger.warning("Request to %s failed (attempt %d/%d): %s — retrying in %.1fs",
                           url, attempt + 1, retries, exc, wait)
            time.sleep(wait)
    logger.error("All %d attempts to %s failed", retries, url)
    return None


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two points in km."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def timezone_offset(lon: float) -> int:
    """Rough timezone offset (hours from UTC) based on longitude."""
    return round(lon / 15)


def safe_div(a, b, default=0.0):
    """Safe division, returns default if b is zero."""
    if b == 0:
        return default
    return a / b


def implied_prob(american_odds: int) -> float:
    """Convert American odds to implied probability."""
    if american_odds < 0:
        return abs(american_odds) / (abs(american_odds) + 100)
    else:
        return 100 / (american_odds + 100)


def american_to_decimal(american_odds: int) -> float:
    """Convert American odds to decimal odds."""
    if american_odds < 0:
        return 1 + 100 / abs(american_odds)
    else:
        return 1 + american_odds / 100


def kelly_criterion(prob: float, decimal_odds: float, fraction: float = 0.25) -> float:
    """
    Fractional Kelly bet sizing.
    Returns recommended fraction of bankroll to wager.
    """
    b = decimal_odds - 1
    q = 1 - prob
    edge = (b * prob - q) / b
    if edge <= 0:
        return 0.0
    return edge * fraction
