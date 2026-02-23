"""
NHL Predictor — Configuration
"""

import os
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv
load_dotenv()


@dataclass
class Config:
    # ── API Keys ──────────────────────────────────────────────────────
    odds_api_key: str = os.environ.get("ODDS_API_KEY", "")

    # ── NHL Edge API ──────────────────────────────────────────────────
    nhl_base_url: str = "https://api-web.nhle.com/v1"
    nhl_stats_url: str = "https://api.nhle.com/stats/rest/en"

    # ── ESPN API ──────────────────────────────────────────────────────
    espn_base_url: str = "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl"

    # ── MoneyPuck ─────────────────────────────────────────────────────
    moneypuck_base_url: str = "https://moneypuck.com/moneypuck/playerData/seasonSummary"

    # ── The Odds API ──────────────────────────────────────────────────
    odds_base_url: str = "https://api.the-odds-api.com/v4/sports/icehockey_nhl"

    # ── DailyFaceoff ──────────────────────────────────────────────────
    dailyfaceoff_url: str = "https://www.dailyfaceoff.com"

    # ── Data paths ────────────────────────────────────────────────────
    data_dir: str = os.path.join(os.path.dirname(__file__), "data_cache")
    models_dir: str = os.path.join(os.path.dirname(__file__), "trained_models")

    # ── Season settings ───────────────────────────────────────────────
    current_season: str = "20252026"
    previous_season: str = "20242025"

    # ── Model parameters ──────────────────────────────────────────────
    rolling_window_short: int = 5
    rolling_window_long: int = 10
    elo_k_factor: float = 8.0
    elo_home_advantage: float = 18.0
    elo_initial: float = 1500.0
    decay_factor: float = 0.75       # weight of last-season stats for cold start
    roster_continuity_weight: float = 0.8

    # ── Value betting thresholds ──────────────────────────────────────
    min_edge_pct: float = 3.0        # minimum edge % to flag a value bet
    kelly_fraction: float = 0.25     # quarter-Kelly for bankroll sizing

    # ── Team abbreviations mapping ────────────────────────────────────
    team_abbrevs: dict = field(default_factory=lambda: {
        "ANA": "Anaheim Ducks", "ARI": "Arizona Coyotes", "BOS": "Boston Bruins",
        "BUF": "Buffalo Sabres", "CGY": "Calgary Flames", "CAR": "Carolina Hurricanes",
        "CHI": "Chicago Blackhawks", "COL": "Colorado Avalanche",
        "CBJ": "Columbus Blue Jackets", "DAL": "Dallas Stars",
        "DET": "Detroit Red Wings", "EDM": "Edmonton Oilers",
        "FLA": "Florida Panthers", "LAK": "Los Angeles Kings",
        "MIN": "Minnesota Wild", "MTL": "Montreal Canadiens",
        "NSH": "Nashville Predators", "NJD": "New Jersey Devils",
        "NYI": "New York Islanders", "NYR": "New York Rangers",
        "OTT": "Ottawa Senators", "PHI": "Philadelphia Flyers",
        "PIT": "Pittsburgh Penguins", "SJS": "San Jose Sharks",
        "SEA": "Seattle Kraken", "STL": "St. Louis Blues",
        "TBL": "Tampa Bay Lightning", "TOR": "Toronto Maple Leafs",
        "UTA": "Utah Hockey Club", "VAN": "Vancouver Canucks",
        "VGK": "Vegas Golden Knights", "WSH": "Washington Capitals",
        "WPG": "Winnipeg Jets",
    })

    # ── Arena coordinates (lat, lon) for travel distance ──────────────
    arena_coords: dict = field(default_factory=lambda: {
        "ANA": (33.8078, -117.8765), "ARI": (33.4456, -112.0712),
        "BOS": (42.3662, -71.0621), "BUF": (42.8750, -78.8764),
        "CGY": (51.0374, -114.0519), "CAR": (35.8032, -78.7219),
        "CHI": (41.8807, -87.6742), "COL": (39.7487, -105.0077),
        "CBJ": (39.9691, -83.0060), "DAL": (32.7905, -96.8103),
        "DET": (42.3411, -83.0550), "EDM": (53.5469, -113.4979),
        "FLA": (26.1584, -80.3256), "LAK": (34.0430, -118.2673),
        "MIN": (44.9448, -93.1011), "MTL": (45.4961, -73.5693),
        "NSH": (36.1591, -86.7786), "NJD": (40.7335, -74.1711),
        "NYI": (40.6895, -73.9754), "NYR": (40.7505, -73.9934),
        "OTT": (45.2969, -75.9272), "PHI": (39.9012, -75.1720),
        "PIT": (40.4395, -79.9892), "SJS": (37.3328, -121.9013),
        "SEA": (47.6221, -122.3540), "STL": (38.6268, -90.2025),
        "TBL": (27.9427, -82.4519), "TOR": (43.6435, -79.3791),
        "UTA": (40.7683, -111.9011), "VAN": (49.2778, -123.1089),
        "VGK": (36.1029, -115.1785), "WSH": (38.8981, -77.0209),
        "WPG": (49.8928, -97.1438),
    })

    def __post_init__(self):
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)


# Singleton
cfg = Config()
