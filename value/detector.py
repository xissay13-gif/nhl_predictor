"""
Value Bet Detector.

Compares model predictions against market odds to find +EV opportunities.
Supports:
  - Moneyline value
  - Spread/puck line value
  - Totals value
  - Kelly criterion sizing
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from config import cfg
from utils.helpers import (
    implied_prob, american_to_decimal, kelly_criterion, safe_div,
)


def _decimal_to_implied(decimal_odds: float) -> float:
    """Convert decimal odds to implied probability."""
    if decimal_odds <= 0:
        return 0.0
    return 1.0 / decimal_odds

logger = logging.getLogger("nhl_predictor.value")


class ValueDetector:
    """
    Detect value bets by comparing model probabilities vs market odds.
    """

    def __init__(
        self,
        min_edge_pct: float = cfg.min_edge_pct,
        kelly_frac: float = cfg.kelly_fraction,
        predictor=None,
    ):
        self.min_edge = min_edge_pct / 100.0
        self.kelly_frac = kelly_frac
        self.predictor = predictor  # NHLPredictor instance for meta-model blending

    def analyze_moneyline(
        self,
        model_home_prob: float,
        model_away_prob: float,
        best_home_odds,
        best_away_odds,
        home_team: str = "",
        away_team: str = "",
        odds_format: str = "american",
    ) -> list[dict]:
        """
        Find ML value bets.
        Returns list of value opportunities (could be 0, 1, or 2).

        odds_format: "american" (-110, +120) or "decimal" (1.91, 2.20)
        """
        values = []

        # Home ML
        if odds_format == "decimal":
            home_implied = _decimal_to_implied(best_home_odds)
            dec_home = best_home_odds
        else:
            home_implied = implied_prob(best_home_odds)
            dec_home = american_to_decimal(best_home_odds)

        home_edge = model_home_prob - home_implied
        if home_edge >= self.min_edge:
            kelly = kelly_criterion(model_home_prob, dec_home, self.kelly_frac)
            values.append({
                "market": "moneyline",
                "side": "home",
                "team": home_team,
                "model_prob": round(model_home_prob, 4),
                "market_implied": round(home_implied, 4),
                "edge": round(home_edge, 4),
                "edge_pct": round(home_edge * 100, 2),
                "decimal_odds": round(dec_home, 3),
                "kelly_size": round(kelly, 4),
                "expected_value": round((model_home_prob * dec_home - 1) * 100, 2),
                "confidence": _confidence_level(home_edge),
            })

        # Away ML
        if odds_format == "decimal":
            away_implied = _decimal_to_implied(best_away_odds)
            dec_away = best_away_odds
        else:
            away_implied = implied_prob(best_away_odds)
            dec_away = american_to_decimal(best_away_odds)

        away_edge = model_away_prob - away_implied
        if away_edge >= self.min_edge:
            kelly = kelly_criterion(model_away_prob, dec_away, self.kelly_frac)
            values.append({
                "market": "moneyline",
                "side": "away",
                "team": away_team,
                "model_prob": round(model_away_prob, 4),
                "market_implied": round(away_implied, 4),
                "edge": round(away_edge, 4),
                "edge_pct": round(away_edge * 100, 2),
                "decimal_odds": round(dec_away, 3),
                "kelly_size": round(kelly, 4),
                "expected_value": round((model_away_prob * dec_away - 1) * 100, 2),
                "confidence": _confidence_level(away_edge),
            })

        return values

    def analyze_spread(
        self,
        model_cover_prob: float,
        spread_line: float,
        spread_odds=None,
        side: str = "home",
        team: str = "",
        odds_format: str = "american",
    ) -> Optional[dict]:
        """Analyze puck line / spread value."""
        if spread_odds is None:
            spread_odds = -110 if odds_format == "american" else 1.909

        if odds_format == "decimal":
            market_implied = _decimal_to_implied(spread_odds)
            dec = spread_odds
        else:
            market_implied = implied_prob(spread_odds)
            dec = american_to_decimal(spread_odds)

        edge = model_cover_prob - market_implied

        if edge >= self.min_edge:
            kelly = kelly_criterion(model_cover_prob, dec, self.kelly_frac)
            return {
                "market": "spread",
                "side": side,
                "team": team,
                "spread_line": spread_line,
                "model_prob": round(model_cover_prob, 4),
                "market_implied": round(market_implied, 4),
                "edge": round(edge, 4),
                "edge_pct": round(edge * 100, 2),
                "decimal_odds": round(dec, 3),
                "kelly_size": round(kelly, 4),
                "expected_value": round((model_cover_prob * dec - 1) * 100, 2),
                "confidence": _confidence_level(edge),
            }
        return None

    def analyze_total(
        self,
        model_over_prob: float,
        model_under_prob: float,
        total_line: float,
        over_odds=None,
        under_odds=None,
        odds_format: str = "american",
    ) -> list[dict]:
        """Analyze over/under value."""
        if over_odds is None:
            over_odds = -110 if odds_format == "american" else 1.909
        if under_odds is None:
            under_odds = -110 if odds_format == "american" else 1.909

        values = []

        # Over
        if odds_format == "decimal":
            over_implied = _decimal_to_implied(over_odds)
            dec_over = over_odds
        else:
            over_implied = implied_prob(over_odds)
            dec_over = american_to_decimal(over_odds)

        over_edge = model_over_prob - over_implied
        if over_edge >= self.min_edge:
            kelly = kelly_criterion(model_over_prob, dec_over, self.kelly_frac)
            values.append({
                "market": "total",
                "side": "over",
                "total_line": total_line,
                "model_prob": round(model_over_prob, 4),
                "market_implied": round(over_implied, 4),
                "edge": round(over_edge, 4),
                "edge_pct": round(over_edge * 100, 2),
                "decimal_odds": round(dec_over, 3),
                "kelly_size": round(kelly, 4),
                "expected_value": round((model_over_prob * dec_over - 1) * 100, 2),
                "confidence": _confidence_level(over_edge),
            })

        # Under
        if odds_format == "decimal":
            under_implied = _decimal_to_implied(under_odds)
            dec_under = under_odds
        else:
            under_implied = implied_prob(under_odds)
            dec_under = american_to_decimal(under_odds)

        under_edge = model_under_prob - under_implied
        if under_edge >= self.min_edge:
            kelly = kelly_criterion(model_under_prob, dec_under, self.kelly_frac)
            values.append({
                "market": "total",
                "side": "under",
                "total_line": total_line,
                "model_prob": round(model_under_prob, 4),
                "market_implied": round(under_implied, 4),
                "edge": round(under_edge, 4),
                "edge_pct": round(under_edge * 100, 2),
                "decimal_odds": round(dec_under, 3),
                "kelly_size": round(kelly, 4),
                "expected_value": round((model_under_prob * dec_under - 1) * 100, 2),
                "confidence": _confidence_level(under_edge),
            })

        return values

    def full_analysis(
        self,
        model_predictions: dict,
        poisson_predictions: dict,
        market_data: dict,
        home_team: str = "",
        away_team: str = "",
    ) -> dict:
        """
        Complete value analysis combining ML model and Poisson model predictions.

        model_predictions: from NHLPredictor.predict()
        poisson_predictions: from PoissonGoalModel.full_prediction()
        market_data: odds data dict
        """
        # ── Blend ML + Poisson for final probabilities ───────────────
        ml_home = model_predictions.get("ml_home_win_prob", 0.5)
        poisson_home = poisson_predictions.get("home_win_prob", 0.5)
        elo_home = model_predictions.get("elo_home_win_prob", 0.5)

        # Use enhanced meta-model if available (with context features)
        meta_context = model_predictions.get("meta_context", None)
        if self.predictor is not None and hasattr(self.predictor, "blend_predictions"):
            blended_home = self.predictor.blend_predictions(
                ml_home, poisson_home, elo_home, context=meta_context
            )
        else:
            blended_home = 0.60 * ml_home + 0.40 * poisson_home
        blended_away = 1.0 - blended_home

        # Total: blend ML predicted total with Poisson expected total
        ml_total = model_predictions.get("ml_total_pred", 5.8)
        poisson_total = poisson_predictions.get("expected_total",
                        poisson_predictions.get("expected_home_goals", 3.0)
                        + poisson_predictions.get("expected_away_goals", 2.8))
        blended_total = 0.55 * ml_total + 0.45 * poisson_total

        # Spread: use Poisson's detailed probability + ML signal
        ml_cover = model_predictions.get("ml_home_cover_prob", 0.45)
        poisson_cover = poisson_predictions.get("cover_prob", 0.40)
        blended_cover = 0.55 * ml_cover + 0.45 * poisson_cover

        # ── Collect all value opportunities ─────────────────────────
        all_values = []
        odds_fmt = market_data.get("odds_format", "american")

        # Moneyline
        best_home = market_data.get("best_home_odds", 0)
        best_away = market_data.get("best_away_odds", 0)
        if best_home and best_away:
            all_values.extend(self.analyze_moneyline(
                blended_home, blended_away, best_home, best_away,
                home_team, away_team, odds_format=odds_fmt,
            ))

        # Totals (over/under)
        over_odds = market_data.get("over_odds", 0)
        under_odds = market_data.get("under_odds", 0)
        total_line = market_data.get("market_total", 5.5)
        if over_odds and under_odds:
            # Blend Poisson over/under with ML total prediction
            poisson_over = poisson_predictions.get("over_prob", 0.5)
            poisson_under = poisson_predictions.get("under_prob", 0.5)

            # ML-based over/under: if blended_total > line → lean over
            total_diff = blended_total - total_line
            # Convert total difference to probability via logistic
            ml_over = 1.0 / (1.0 + np.exp(-1.5 * total_diff))
            ml_under = 1.0 - ml_over

            # Blend: Poisson is better at totals, give it more weight
            blended_over = 0.40 * ml_over + 0.60 * poisson_over
            blended_under = 0.40 * ml_under + 0.60 * poisson_under

            all_values.extend(self.analyze_total(
                blended_over, blended_under, total_line,
                over_odds, under_odds, odds_format=odds_fmt,
            ))

        # Sort by edge descending
        all_values.sort(key=lambda x: x.get("edge_pct", 0), reverse=True)

        return {
            "blended_home_prob": round(blended_home, 4),
            "blended_away_prob": round(blended_away, 4),
            "blended_total": round(blended_total, 2),
            "blended_cover_prob": round(blended_cover, 4),
            "value_bets": all_values,
            "total_value_bets": len(all_values),
            "best_edge": all_values[0] if all_values else None,
        }


def _confidence_level(edge: float) -> str:
    """Classify edge into confidence tiers."""
    if edge >= 0.10:
        return "HIGH"
    elif edge >= 0.06:
        return "MEDIUM"
    elif edge >= 0.03:
        return "LOW"
    return "NONE"


def format_value_report(analysis: dict, home_team: str, away_team: str) -> str:
    """Format value analysis into a readable report."""
    lines = []
    lines.append(f"{'='*60}")
    lines.append(f"  {away_team} @ {home_team}")
    lines.append(f"{'='*60}")
    lines.append("")
    lines.append(f"  Model Probabilities:")
    lines.append(f"    Home win:  {analysis['blended_home_prob']:.1%}")
    lines.append(f"    Away win:  {analysis['blended_away_prob']:.1%}")
    lines.append(f"    Exp Total: {analysis['blended_total']:.1f}")
    lines.append("")

    if analysis["value_bets"]:
        lines.append(f"  VALUE BETS FOUND: {analysis['total_value_bets']}")
        lines.append(f"  {'-'*50}")

        for vb in analysis["value_bets"]:
            lines.append(f"    [{vb['confidence']}] {vb['market'].upper()} — {vb.get('side', '').upper()}")
            if vb.get("team"):
                lines.append(f"      Team: {vb['team']}")
            if vb.get("total_line"):
                lines.append(f"      Line: {vb['total_line']}")
            if vb.get("spread_line"):
                lines.append(f"      Spread: {vb['spread_line']}")
            lines.append(f"      Model: {vb['model_prob']:.1%}  |  Market: {vb['market_implied']:.1%}")
            lines.append(f"      Edge:  {vb['edge_pct']:.1f}%  |  EV: {vb['expected_value']:.1f}%")
            lines.append(f"      Odds:  {vb['decimal_odds']:.3f}  |  Kelly: {vb['kelly_size']:.2%}")
            lines.append("")
    else:
        lines.append("  No value bets detected.")
        lines.append("")

    lines.append(f"{'='*60}")
    return "\n".join(lines)
