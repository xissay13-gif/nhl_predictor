"""
Poisson Goal Model.

Models the number of goals scored by each team using Poisson distributions.
Key for:
  - Predicting totals (over/under)
  - Spread probabilities
  - Regulation win probability
  - Correct score probabilities
"""

import logging
from typing import Optional

import numpy as np
from scipy.stats import poisson
from scipy.optimize import minimize

from utils.helpers import safe_div

logger = logging.getLogger("nhl_predictor.models.poisson")

# Maximum goals to consider in probability matrix
MAX_GOALS = 12


class PoissonGoalModel:
    """
    Independent Poisson model for NHL goal scoring.

    Given expected goals for each side, computes:
      - Win/draw/loss probabilities
      - Total goals distribution
      - Spread probabilities
      - Score matrix
    """

    def __init__(self, home_xg: float = 3.0, away_xg: float = 2.7):
        self.home_xg = max(home_xg, 0.5)  # floor to avoid degenerate distributions
        self.away_xg = max(away_xg, 0.5)

    def set_xg(self, home_xg: float, away_xg: float):
        self.home_xg = max(home_xg, 0.5)
        self.away_xg = max(away_xg, 0.5)

    def score_matrix(self) -> np.ndarray:
        """
        Compute probability matrix P[home_goals, away_goals].
        Returns (MAX_GOALS+1, MAX_GOALS+1) array.
        """
        home_probs = poisson.pmf(range(MAX_GOALS + 1), self.home_xg)
        away_probs = poisson.pmf(range(MAX_GOALS + 1), self.away_xg)
        return np.outer(home_probs, away_probs)

    def regulation_probabilities(self) -> dict:
        """
        P(home wins in regulation), P(away wins in regulation), P(draw at end of regulation).
        """
        matrix = self.score_matrix()
        home_win = np.sum(np.tril(matrix, -1))   # below diagonal: home > away
        away_win = np.sum(np.triu(matrix, 1))    # above diagonal: away > home
        draw = np.sum(np.diag(matrix))             # diagonal: tied

        return {
            "reg_home_win": home_win,
            "reg_away_win": away_win,
            "reg_draw": draw,
        }

    def win_probabilities(self, home_ot_edge: float = 0.52) -> dict:
        """
        Full win probabilities including OT.
        home_ot_edge: probability home wins in OT/SO (slight home advantage).
        """
        reg = self.regulation_probabilities()
        draw_prob = reg["reg_draw"]

        home_win = reg["reg_home_win"] + draw_prob * home_ot_edge
        away_win = reg["reg_away_win"] + draw_prob * (1 - home_ot_edge)

        return {
            "home_win_prob": home_win,
            "away_win_prob": away_win,
            "regulation_draw_prob": draw_prob,
            "home_reg_win_prob": reg["reg_home_win"],
            "away_reg_win_prob": reg["reg_away_win"],
        }

    def total_goals_distribution(self) -> dict:
        """
        Distribution of total goals scored.
        """
        matrix = self.score_matrix()
        total_probs = {}
        for t in range(MAX_GOALS * 2 + 1):
            prob = 0.0
            for h in range(min(t + 1, MAX_GOALS + 1)):
                a = t - h
                if 0 <= a <= MAX_GOALS:
                    prob += matrix[h, a]
            total_probs[t] = prob

        return total_probs

    def over_under_prob(self, line: float = 5.5) -> dict:
        """
        Probability of total goals going over/under a given line.
        """
        total_dist = self.total_goals_distribution()
        over = sum(p for g, p in total_dist.items() if g > line)
        under = sum(p for g, p in total_dist.items() if g < line)
        push = sum(p for g, p in total_dist.items() if g == line)

        return {
            "over_prob": over,
            "under_prob": under,
            "push_prob": push,
            "total_line": line,
            "expected_total": self.home_xg + self.away_xg,
        }

    def spread_prob(self, line: float = -1.5) -> dict:
        """
        Probability of home team covering a spread.
        line = -1.5 means home must win by 2+.
        line = +1.5 means home can lose by 1 and still cover.
        """
        matrix = self.score_matrix()
        cover = 0.0
        no_cover = 0.0
        push = 0.0

        for h in range(MAX_GOALS + 1):
            for a in range(MAX_GOALS + 1):
                diff = h - a  # positive = home winning
                adjusted = diff + line  # line is from home perspective
                if adjusted > 0:
                    cover += matrix[h, a]
                elif adjusted < 0:
                    no_cover += matrix[h, a]
                else:
                    push += matrix[h, a]

        return {
            "cover_prob": cover,
            "no_cover_prob": no_cover,
            "push_prob": push,
            "spread_line": line,
        }

    def most_likely_scores(self, n: int = 5) -> list[dict]:
        """Top N most likely final scores."""
        matrix = self.score_matrix()
        scores = []
        for h in range(MAX_GOALS + 1):
            for a in range(MAX_GOALS + 1):
                scores.append({
                    "home_goals": h,
                    "away_goals": a,
                    "probability": matrix[h, a],
                })
        scores.sort(key=lambda x: x["probability"], reverse=True)
        return scores[:n]

    def simulate(self, n_sims: int = 10000) -> dict:
        """
        Monte Carlo simulation for validation and variance estimation.
        """
        rng = np.random.default_rng()
        home_goals = rng.poisson(self.home_xg, n_sims)
        away_goals = rng.poisson(self.away_xg, n_sims)

        home_wins = np.sum(home_goals > away_goals)
        away_wins = np.sum(away_goals > home_goals)
        draws = np.sum(home_goals == away_goals)
        totals = home_goals + away_goals

        return {
            "sim_home_win_pct": home_wins / n_sims,
            "sim_away_win_pct": away_wins / n_sims,
            "sim_draw_pct": draws / n_sims,
            "sim_avg_total": totals.mean(),
            "sim_total_std": totals.std(),
            "sim_avg_home": home_goals.mean(),
            "sim_avg_away": away_goals.mean(),
        }

    def full_prediction(self, total_line: float = 5.5, spread_line: float = -1.5) -> dict:
        """Complete prediction package."""
        result = {}
        result.update(self.win_probabilities())
        result.update(self.over_under_prob(total_line))
        result.update(self.spread_prob(spread_line))
        result["most_likely_scores"] = self.most_likely_scores(5)
        result["expected_home_goals"] = self.home_xg
        result["expected_away_goals"] = self.away_xg
        return result


def estimate_xg_from_features(features: dict) -> tuple[float, float]:
    """
    Estimate expected goals from feature vector.
    Uses a blend of xG data, recent form, and Elo.
    """
    # Primary: xG per game data
    home_xg = features.get("home_xgf_pg", 0)
    away_xg = features.get("away_xgf_pg", 0)

    # Fallback: actual goals per game
    if home_xg == 0:
        home_xg = features.get("home_gf_pg", 2.9)
    if away_xg == 0:
        away_xg = features.get("away_gf_pg", 2.9)

    # Adjust for opponent defense
    home_opp_xga = features.get("away_xga_pg", 2.9)
    away_opp_xga = features.get("home_xga_pg", 2.9)
    league_avg = 3.0

    # Opponent adjustment: if opponent allows more than average, scale up
    home_adj = home_xg * (home_opp_xga / league_avg)
    away_adj = away_xg * (away_opp_xga / league_avg)

    # Blend with rolling averages for recency
    home_rolling = features.get("home_rolling_5_gf", home_xg)
    away_rolling = features.get("away_rolling_5_gf", away_xg)

    # Weighted blend: 60% season xG, 25% opponent-adjusted, 15% recent form
    home_final = 0.60 * home_xg + 0.25 * home_adj + 0.15 * home_rolling
    away_final = 0.60 * away_xg + 0.25 * away_adj + 0.15 * away_rolling

    # Home advantage bump (~0.15 goals)
    home_final += 0.15

    # Goalie adjustment
    home_sv = features.get("home_goalie_save_pct", 0.910)
    away_sv = features.get("away_goalie_save_pct", 0.910)
    avg_sv = 0.910

    # Better goalie => fewer goals against
    away_final *= (avg_sv / max(home_sv, 0.85))  # home goalie affects away scoring
    home_final *= (avg_sv / max(away_sv, 0.85))  # away goalie affects home scoring

    return max(home_final, 0.5), max(away_final, 0.5)
