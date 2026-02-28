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

logger = logging.getLogger("nhl_predictor.value")


class ValueDetector:
    """
    Detect value bets by comparing model probabilities vs market odds.
    """

    # Tiered Kelly: higher confidence → more aggressive sizing
    TIERED_KELLY = {
        "HIGH": 1.0,       # full kelly_frac for edge ≥ 10%
        "MEDIUM": 0.75,    # 75% of kelly_frac for edge 6-10%
        "LOW": 0.50,       # 50% of kelly_frac for edge 3-6%
        "NONE": 0.0,
    }

    def __init__(
        self,
        min_edge_pct: float = cfg.min_edge_pct,
        kelly_frac: float = cfg.kelly_fraction,
        max_bet_pct: float = cfg.max_bet_pct,
        predictor=None,
    ):
        self.min_edge = min_edge_pct / 100.0
        self.kelly_frac = kelly_frac
        self.max_bet_pct = max_bet_pct
        self.predictor = predictor  # NHLPredictor instance for meta-model blending

    def _sized_kelly(self, prob: float, dec_odds: float, confidence: str) -> float:
        """Kelly sizing with tiered multiplier and max bet cap."""
        tier_mult = self.TIERED_KELLY.get(confidence, 0.5)
        frac = self.kelly_frac * tier_mult
        raw = kelly_criterion(prob, dec_odds, frac)
        return min(raw, self.max_bet_pct)

    def analyze_moneyline(
        self,
        model_home_prob: float,
        model_away_prob: float,
        best_home_odds: int,
        best_away_odds: int,
        home_team: str = "",
        away_team: str = "",
    ) -> list[dict]:
        """
        Find ML value bets.
        Returns list of value opportunities (could be 0, 1, or 2).
        """
        values = []

        # Home ML
        home_implied = implied_prob(best_home_odds)
        home_edge = model_home_prob - home_implied
        if home_edge >= self.min_edge:
            dec = american_to_decimal(best_home_odds)
            conf = _confidence_level(home_edge)
            kelly = self._sized_kelly(model_home_prob, dec, conf)
            values.append({
                "market": "moneyline",
                "side": "home",
                "team": home_team,
                "model_prob": round(model_home_prob, 4),
                "market_implied": round(home_implied, 4),
                "edge": round(home_edge, 4),
                "edge_pct": round(home_edge * 100, 2),
                "odds": best_home_odds,
                "decimal_odds": round(dec, 3),
                "kelly_size": round(kelly, 4),
                "expected_value": round((model_home_prob * dec - 1) * 100, 2),
                "confidence": conf,
            })

        # Away ML
        away_implied = implied_prob(best_away_odds)
        away_edge = model_away_prob - away_implied
        if away_edge >= self.min_edge:
            dec = american_to_decimal(best_away_odds)
            conf = _confidence_level(away_edge)
            kelly = self._sized_kelly(model_away_prob, dec, conf)
            values.append({
                "market": "moneyline",
                "side": "away",
                "team": away_team,
                "model_prob": round(model_away_prob, 4),
                "market_implied": round(away_implied, 4),
                "edge": round(away_edge, 4),
                "edge_pct": round(away_edge * 100, 2),
                "odds": best_away_odds,
                "decimal_odds": round(dec, 3),
                "kelly_size": round(kelly, 4),
                "expected_value": round((model_away_prob * dec - 1) * 100, 2),
                "confidence": conf,
            })

        return values

    def analyze_spread(
        self,
        model_cover_prob: float,
        spread_line: float,
        spread_odds: int,
        side: str = "home",
        team: str = "",
    ) -> Optional[dict]:
        """Analyze puck line / spread value."""
        market_implied = implied_prob(spread_odds)
        edge = model_cover_prob - market_implied

        if edge >= self.min_edge:
            dec = american_to_decimal(spread_odds)
            conf = _confidence_level(edge)
            kelly = self._sized_kelly(model_cover_prob, dec, conf)
            return {
                "market": "spread",
                "side": side,
                "team": team,
                "spread_line": spread_line,
                "model_prob": round(model_cover_prob, 4),
                "market_implied": round(market_implied, 4),
                "edge": round(edge, 4),
                "edge_pct": round(edge * 100, 2),
                "odds": spread_odds,
                "decimal_odds": round(dec, 3),
                "kelly_size": round(kelly, 4),
                "expected_value": round((model_cover_prob * dec - 1) * 100, 2),
                "confidence": conf,
            }
        return None

    def analyze_total(
        self,
        model_over_prob: float,
        model_under_prob: float,
        total_line: float,
        over_odds: int = -110,
        under_odds: int = -110,
    ) -> list[dict]:
        """Analyze over/under value."""
        values = []

        # Over
        over_implied = implied_prob(over_odds)
        over_edge = model_over_prob - over_implied
        if over_edge >= self.min_edge:
            dec = american_to_decimal(over_odds)
            conf = _confidence_level(over_edge)
            kelly = self._sized_kelly(model_over_prob, dec, conf)
            values.append({
                "market": "total",
                "side": "over",
                "total_line": total_line,
                "model_prob": round(model_over_prob, 4),
                "market_implied": round(over_implied, 4),
                "edge": round(over_edge, 4),
                "edge_pct": round(over_edge * 100, 2),
                "odds": over_odds,
                "decimal_odds": round(dec, 3),
                "kelly_size": round(kelly, 4),
                "expected_value": round((model_over_prob * dec - 1) * 100, 2),
                "confidence": conf,
            })

        # Under
        under_implied = implied_prob(under_odds)
        under_edge = model_under_prob - under_implied
        if under_edge >= self.min_edge:
            dec = american_to_decimal(under_odds)
            conf = _confidence_level(under_edge)
            kelly = self._sized_kelly(model_under_prob, dec, conf)
            values.append({
                "market": "total",
                "side": "under",
                "total_line": total_line,
                "model_prob": round(model_under_prob, 4),
                "market_implied": round(under_implied, 4),
                "edge": round(under_edge, 4),
                "edge_pct": round(under_edge * 100, 2),
                "odds": under_odds,
                "decimal_odds": round(dec, 3),
                "kelly_size": round(kelly, 4),
                "expected_value": round((model_under_prob * dec - 1) * 100, 2),
                "confidence": conf,
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

        # ── Collect all value opportunities ────────────────────────────
        all_values = []

        # Moneyline
        best_home = market_data.get("best_home_odds", 0)
        best_away = market_data.get("best_away_odds", 0)
        if best_home and best_away:
            all_values.extend(self.analyze_moneyline(
                blended_home, blended_away, best_home, best_away,
                home_team, away_team,
            ))

        # Spread / puck line
        spread_line = market_data.get("spread_line", -1.5)
        home_spread_odds = market_data.get("home_spread_odds", 0)
        away_spread_odds = market_data.get("away_spread_odds", 0)
        if home_spread_odds:
            val = self.analyze_spread(
                blended_cover, spread_line, home_spread_odds,
                side="home", team=home_team,
            )
            if val:
                all_values.append(val)
        if away_spread_odds:
            away_cover = 1.0 - blended_cover
            val = self.analyze_spread(
                away_cover, -spread_line, away_spread_odds,
                side="away", team=away_team,
            )
            if val:
                all_values.append(val)

        # Totals (over/under)
        total_line = market_data.get("total_line", 0)
        over_odds = market_data.get("over_odds", 0)
        under_odds = market_data.get("under_odds", 0)
        if total_line and over_odds and under_odds:
            over_prob = poisson_predictions.get("over_prob", 0.5)
            under_prob = poisson_predictions.get("under_prob", 0.5)
            all_values.extend(self.analyze_total(
                over_prob, under_prob, total_line, over_odds, under_odds,
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
            lines.append(f"      Odds:  {vb['odds']}  |  Kelly: {vb['kelly_size']:.2%}")
            lines.append("")
    else:
        lines.append("  No value bets detected.")
        lines.append("")

    lines.append(f"{'='*60}")
    return "\n".join(lines)
