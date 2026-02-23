"""
Elo Rating System for NHL teams.

Features:
  - Separate offensive / defensive Elo
  - Home-ice advantage adjustment
  - Margin-of-victory multiplier
  - Season carryover with regression to mean
  - Goalie-adjusted Elo
"""

import math
import logging
from typing import Optional

import pandas as pd

from config import cfg

logger = logging.getLogger("nhl_predictor.elo")


class EloSystem:
    """NHL Elo rating system with offensive/defensive decomposition."""

    def __init__(
        self,
        k_factor: float = cfg.elo_k_factor,
        home_advantage: float = cfg.elo_home_advantage,
        initial_rating: float = cfg.elo_initial,
        season_revert: float = 0.6,          # revert 40% toward mean each season
        mov_exponent: float = 0.8,           # margin of victory exponent
    ):
        self.k = k_factor
        self.home_adv = home_advantage
        self.initial = initial_rating
        self.season_revert = season_revert
        self.mov_exp = mov_exponent

        # {team: rating}
        self.ratings: dict[str, float] = {}
        self.off_ratings: dict[str, float] = {}     # offensive Elo
        self.def_ratings: dict[str, float] = {}     # defensive Elo
        self.goalie_adj: dict[str, float] = {}      # goalie adjustment

        self.history: list[dict] = []                # log of every update

    def _get(self, team: str) -> float:
        return self.ratings.get(team, self.initial)

    def _get_off(self, team: str) -> float:
        return self.off_ratings.get(team, self.initial)

    def _get_def(self, team: str) -> float:
        return self.def_ratings.get(team, self.initial)

    def expected_score(self, team_a: str, team_b: str, a_is_home: bool = True) -> float:
        """
        Expected win probability for team_a.
        Accounts for home advantage if a_is_home.
        """
        r_a = self._get(team_a)
        r_b = self._get(team_b)
        if a_is_home:
            r_a += self.home_adv
        else:
            r_b += self.home_adv
        return 1.0 / (1.0 + 10.0 ** ((r_b - r_a) / 400.0))

    def expected_score_detailed(self, home: str, away: str) -> dict:
        """Detailed expected scores including offensive/defensive decomposition."""
        overall = self.expected_score(home, away, a_is_home=True)

        # Offensive Elo of home vs Defensive Elo of away, and vice-versa
        home_off = self._get_off(home) + self.home_adv / 2
        away_def = self._get_def(away)
        away_off = self._get_off(away)
        home_def = self._get_def(home) + self.home_adv / 2

        # Higher home_off relative to away_def => home should score more
        home_attack_edge = 1.0 / (1.0 + 10.0 ** ((away_def - home_off) / 400.0))
        away_attack_edge = 1.0 / (1.0 + 10.0 ** ((home_def - away_off) / 400.0))

        return {
            "home_win_prob": overall,
            "away_win_prob": 1.0 - overall,
            "home_attack_edge": home_attack_edge,
            "away_attack_edge": away_attack_edge,
            "home_elo": self._get(home),
            "away_elo": self._get(away),
            "home_off_elo": self._get_off(home),
            "home_def_elo": self._get_def(home),
            "away_off_elo": self._get_off(away),
            "away_def_elo": self._get_def(away),
            "home_goalie_adj": self.goalie_adj.get(home, 0),
            "away_goalie_adj": self.goalie_adj.get(away, 0),
        }

    def _mov_multiplier(self, goal_diff: int) -> float:
        """Margin-of-victory multiplier: log-scaled to avoid runaway updates."""
        return math.log(abs(goal_diff) + 1) ** self.mov_exp

    def update(
        self,
        home: str,
        away: str,
        home_goals: int,
        away_goals: int,
        ot: bool = False,
    ):
        """
        Update Elo ratings after a game result.
        OT/SO losses are weighted at 0.6 instead of 0.0.
        """
        # Actual score: 1 = win, 0.6 = OT loss, 0 = reg loss
        if home_goals > away_goals:
            s_home = 1.0
            s_away = 0.6 if ot else 0.0
        else:
            s_away = 1.0
            s_home = 0.6 if ot else 0.0

        e_home = self.expected_score(home, away, a_is_home=True)
        e_away = 1.0 - e_home

        goal_diff = abs(home_goals - away_goals)
        mov = self._mov_multiplier(goal_diff)

        # Overall update
        delta_home = self.k * mov * (s_home - e_home)
        delta_away = self.k * mov * (s_away - e_away)

        self.ratings[home] = self._get(home) + delta_home
        self.ratings[away] = self._get(away) + delta_away

        # Offensive/Defensive decomposition
        # Offensive Elo goes up when team scores, down when they don't
        off_k = self.k * 0.5
        self.off_ratings[home] = self._get_off(home) + off_k * (home_goals - away_goals) / max(goal_diff, 1)
        self.off_ratings[away] = self._get_off(away) + off_k * (away_goals - home_goals) / max(goal_diff, 1)
        self.def_ratings[home] = self._get_def(home) + off_k * (away_goals - home_goals) / max(goal_diff, 1) * -1
        self.def_ratings[away] = self._get_def(away) + off_k * (home_goals - away_goals) / max(goal_diff, 1) * -1

        self.history.append({
            "home": home, "away": away,
            "home_goals": home_goals, "away_goals": away_goals,
            "ot": ot,
            "home_elo_before": self._get(home) - delta_home,
            "away_elo_before": self._get(away) - delta_away,
            "home_elo_after": self._get(home),
            "away_elo_after": self._get(away),
            "delta_home": delta_home,
            "delta_away": delta_away,
        })

    def new_season(self):
        """
        Regress ratings toward the mean at start of a new season.
        """
        for team in list(self.ratings.keys()):
            self.ratings[team] = (
                self.initial * (1 - self.season_revert)
                + self.ratings[team] * self.season_revert
            )
            self.off_ratings[team] = (
                self.initial * (1 - self.season_revert)
                + self.off_ratings.get(team, self.initial) * self.season_revert
            )
            self.def_ratings[team] = (
                self.initial * (1 - self.season_revert)
                + self.def_ratings.get(team, self.initial) * self.season_revert
            )

        logger.info("Season reset applied (%.0f%% revert to mean)", (1 - self.season_revert) * 100)

    def set_goalie_adjustment(self, team: str, adj: float):
        """Set a goalie-based Elo adjustment for a team (e.g., based on GSAx)."""
        self.goalie_adj[team] = adj

    def adjusted_rating(self, team: str) -> float:
        """Rating with goalie adjustment."""
        return self._get(team) + self.goalie_adj.get(team, 0)

    def build_from_game_log(self, game_log: pd.DataFrame):
        """
        Process a full season game log to build Elo ratings.
        Expects columns: home_team, away_team, home_goals, away_goals, ot
        Rows must be sorted by date.
        """
        for _, row in game_log.iterrows():
            self.update(
                home=row["home_team"],
                away=row["away_team"],
                home_goals=int(row["home_goals"]),
                away_goals=int(row["away_goals"]),
                ot=bool(row.get("ot", False)),
            )

    def get_rankings(self) -> pd.DataFrame:
        """Current Elo rankings as a DataFrame."""
        rows = []
        for team in sorted(self.ratings, key=lambda t: self.ratings[t], reverse=True):
            rows.append({
                "team": team,
                "elo": round(self.ratings[team], 1),
                "off_elo": round(self.off_ratings.get(team, self.initial), 1),
                "def_elo": round(self.def_ratings.get(team, self.initial), 1),
                "goalie_adj": round(self.goalie_adj.get(team, 0), 1),
                "adjusted_elo": round(self.adjusted_rating(team), 1),
            })
        return pd.DataFrame(rows)
