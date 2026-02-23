"""
Master Feature Engineering Pipeline.

Assembles all feature modules into a single feature vector per game.
Handles cold-start, decay, and differential features.
"""

import logging
from typing import Optional

import pandas as pd
import numpy as np

from config import cfg
from features.team_stats import (
    compute_basic_stats, compute_home_away_splits, compute_recent_form,
    compute_streak, compute_rolling_stats, compute_momentum_index,
    compute_pdo, compute_discipline_features,
)
from features.advanced_stats import (
    compute_possession_features, compute_expected_goals_features,
    compute_shot_quality_features, compute_pace_factor,
    compute_special_teams_features,
)
from features.goalie_stats import (
    compute_goalie_features, compute_goalie_rest,
    classify_starter_backup,
)
from features.schedule_fatigue import (
    compute_schedule_features, compute_rest_differential,
    compute_last_opponent_strength,
)
from features.h2h import (
    compute_h2h_features, compute_division_rivalry,
    compute_playoff_pressure,
)
from features.player_stats import (
    aggregate_skater_features, compute_injury_impact,
)
from features.betting_features import (
    compute_market_features, compute_totals_features,
)

logger = logging.getLogger("nhl_predictor.features.engineer")


class FeatureEngineer:
    """
    Orchestrates feature extraction for a single matchup.
    Call build_features() with all available data sources.
    """

    def __init__(self, elo_system=None):
        self.elo = elo_system

    def build_features(
        self,
        home_team: str,
        away_team: str,
        game_date: str,
        # Data sources (all optional — graceful degradation)
        standings: Optional[pd.DataFrame] = None,
        game_log: Optional[pd.DataFrame] = None,
        mp_team_data: Optional[dict] = None,       # MoneyPuck {team: {...stats}}
        goalie_data: Optional[dict] = None,         # {team: goalie_row_dict}
        skaters_df: Optional[pd.DataFrame] = None,
        injuries_df: Optional[pd.DataFrame] = None,
        consensus_odds: Optional[pd.DataFrame] = None,
        totals_df: Optional[pd.DataFrame] = None,
        prev_season_data: Optional[dict] = None,    # for cold-start
    ) -> dict:
        """
        Build complete feature vector for a home vs away matchup.
        Returns flat dict of features, prefixed with home_/away_ or diff_.
        """
        features = {"home_team": home_team, "away_team": away_team, "date": game_date}

        # Initialize defaults
        if standings is None:
            standings = pd.DataFrame()
        if game_log is None:
            game_log = pd.DataFrame()
        if mp_team_data is None:
            mp_team_data = {}
        if goalie_data is None:
            goalie_data = {}
        if skaters_df is None:
            skaters_df = pd.DataFrame()
        if injuries_df is None:
            injuries_df = pd.DataFrame()
        if consensus_odds is None:
            consensus_odds = pd.DataFrame()
        if totals_df is None:
            totals_df = pd.DataFrame()
        if prev_season_data is None:
            prev_season_data = {}

        # ── 1. Basic team stats ──────────────────────────────────────
        if not standings.empty:
            enriched = compute_basic_stats(standings)
            enriched = compute_home_away_splits(enriched)
            for side, team in [("home", home_team), ("away", away_team)]:
                row = enriched[enriched["team"] == team]
                if not row.empty:
                    r = row.iloc[0]
                    for col in ["win_pct", "points_pct", "gf_pg", "ga_pg",
                                "goal_diff_pg", "home_win_pct", "away_win_pct",
                                "home_away_win_diff"]:
                        features[f"{side}_{col}"] = r.get(col, 0)

        # ── 2. Recent form ───────────────────────────────────────────
        if not game_log.empty:
            for side, team in [("home", home_team), ("away", away_team)]:
                for window in [5, 10]:
                    form = compute_recent_form(game_log, team, window)
                    for k, v in form.items():
                        features[f"{side}_{k}"] = v

                streak = compute_streak(game_log, team)
                features[f"{side}_streak_value"] = streak["streak_value"]

                rolling = compute_rolling_stats(game_log, team)
                for k, v in rolling.items():
                    features[f"{side}_{k}"] = v

                features[f"{side}_momentum"] = compute_momentum_index(game_log, team)

        # ── 3. Advanced stats (xG, Corsi, Fenwick) ───────────────────
        for side, team in [("home", home_team), ("away", away_team)]:
            td = mp_team_data.get(team, {})
            if td:
                poss = compute_possession_features(td)
                xg = compute_expected_goals_features(td)
                sq = compute_shot_quality_features(td)
                pace = compute_pace_factor(td)
                pdo = compute_pdo(td)
                disc = compute_discipline_features(td)

                for d in [poss, xg, sq, pace, pdo, disc]:
                    for k, v in d.items():
                        features[f"{side}_{k}"] = v
            elif prev_season_data.get(team):
                # Cold-start: use previous season with decay
                prev = prev_season_data[team]
                decay = cfg.decay_factor
                for k, v in prev.items():
                    if isinstance(v, (int, float)):
                        features[f"{side}_{k}"] = v * decay

        # ── 4. Special teams ─────────────────────────────────────────
        for side, team in [("home", home_team), ("away", away_team)]:
            td = mp_team_data.get(team, {})
            stnd = {}
            if not standings.empty:
                row = standings[standings["team"] == team]
                if not row.empty:
                    stnd = row.iloc[0].to_dict()
            st = compute_special_teams_features(stnd, td)
            for k, v in st.items():
                features[f"{side}_{k}"] = v

        # ── 5. Goalie features ───────────────────────────────────────
        for side, team in [("home", home_team), ("away", away_team)]:
            gd = goalie_data.get(team, {})
            if gd:
                gf = compute_goalie_features(gd)
                for k, v in gf.items():
                    features[f"{side}_{k}"] = v

        # ── 6. Schedule / Fatigue ────────────────────────────────────
        if not game_log.empty:
            home_sched = compute_schedule_features(
                home_team, game_date, game_log, is_home=True, opponent=away_team)
            away_sched = compute_schedule_features(
                away_team, game_date, game_log, is_home=False, opponent=home_team)

            for k, v in home_sched.items():
                if isinstance(v, (int, float, bool, np.bool_)):
                    features[f"home_{k}"] = v
            for k, v in away_sched.items():
                if isinstance(v, (int, float, bool, np.bool_)):
                    features[f"away_{k}"] = v

            rest_diff = compute_rest_differential(home_sched, away_sched)
            features.update(rest_diff)

            # Last opponent strength
            if not standings.empty:
                for side, team in [("home", home_team), ("away", away_team)]:
                    los = compute_last_opponent_strength(game_log, team, game_date, standings)
                    for k, v in los.items():
                        if isinstance(v, (int, float, bool, np.bool_)):
                            features[f"{side}_{k}"] = v

        # ── 7. H2H ──────────────────────────────────────────────────
        if not game_log.empty:
            h2h = compute_h2h_features(game_log, home_team, away_team)
            features.update(h2h)

        if not standings.empty:
            div = compute_division_rivalry(home_team, away_team, standings)
            features.update(div)

            for side, team in [("home", home_team), ("away", away_team)]:
                pp = compute_playoff_pressure(team, standings)
                for k, v in pp.items():
                    if isinstance(v, (int, float, bool, np.bool_)):
                        features[f"{side}_{k}"] = v

        # ── 8. Player aggregates ─────────────────────────────────────
        if not skaters_df.empty:
            for side, team in [("home", home_team), ("away", away_team)]:
                pf = aggregate_skater_features(skaters_df, team)
                for k, v in pf.items():
                    features[f"{side}_{k}"] = v

        # ── 9. Injury impact ────────────────────────────────────────
        if not injuries_df.empty:
            for side, team in [("home", home_team), ("away", away_team)]:
                inj = compute_injury_impact(injuries_df, team, skaters_df)
                for k, v in inj.items():
                    features[f"{side}_{k}"] = v

        # ── 10. Betting market features ──────────────────────────────
        mkt = compute_market_features(consensus_odds, home_team, away_team)
        features.update(mkt)

        tots = compute_totals_features(totals_df, home_team, away_team)
        features.update(tots)

        # ── 11. Elo ratings ──────────────────────────────────────────
        if self.elo:
            elo_detail = self.elo.expected_score_detailed(home_team, away_team)
            features.update({
                "elo_home_win_prob": elo_detail["home_win_prob"],
                "elo_away_win_prob": elo_detail["away_win_prob"],
                "elo_home": elo_detail["home_elo"],
                "elo_away": elo_detail["away_elo"],
                "elo_diff": elo_detail["home_elo"] - elo_detail["away_elo"],
                "elo_home_off": elo_detail["home_off_elo"],
                "elo_home_def": elo_detail["home_def_elo"],
                "elo_away_off": elo_detail["away_off_elo"],
                "elo_away_def": elo_detail["away_def_elo"],
                "elo_home_attack_edge": elo_detail["home_attack_edge"],
                "elo_away_attack_edge": elo_detail["away_attack_edge"],
            })

        # ── 12. Engineered differential features ─────────────────────
        features.update(self._compute_differentials(features))

        # ── 13. Temporal features ────────────────────────────────────
        features.update(self._compute_temporal(game_date))

        return features

    def _compute_differentials(self, f: dict) -> dict:
        """Key differential features (home minus away)."""
        diffs = {}

        pairs = [
            ("win_pct", "win_pct"),
            ("points_pct", "points_pct"),
            ("gf_pg", "gf_pg"),
            ("ga_pg", "ga_pg"),
            ("goal_diff_pg", "goal_diff_pg"),
            ("xg_diff_pg", "xg_diff_pg"),
            ("corsi_for_pct", "corsi_for_pct"),
            ("fenwick_for_pct", "fenwick_for_pct"),
            ("goalie_save_pct", "goalie_save_pct"),
            ("goalie_gsax_pg", "goalie_gsax_pg"),
            ("pp_pct", "pp_pct"),
            ("pk_pct", "pk_pct"),
            ("special_teams_index", "special_teams_index"),
            ("momentum", "momentum"),
            ("pdo_luck", "pdo_luck"),
            ("playoff_pressure", "playoff_pressure"),
        ]

        for home_suffix, away_suffix in pairs:
            h = f.get(f"home_{home_suffix}")
            a = f.get(f"away_{away_suffix}")
            if h is not None and a is not None:
                diffs[f"diff_{home_suffix}"] = h - a

        # Compound features
        h_xgf = f.get("home_xgf_pg", 0)
        a_xga = f.get("away_xga_pg", 0)
        a_xgf = f.get("away_xgf_pg", 0)
        h_xga = f.get("home_xga_pg", 0)
        h_sv = f.get("home_goalie_save_pct", 0.910)
        a_sv = f.get("away_goalie_save_pct", 0.910)

        # Home xGF vs Away goalie save%
        diffs["home_xgf_vs_away_sv"] = h_xgf * (1 - a_sv) if a_sv else 0
        diffs["away_xgf_vs_home_sv"] = a_xgf * (1 - h_sv) if h_sv else 0

        # Rest differential (already computed, but ensure)
        if "rest_diff" not in f:
            diffs["rest_diff"] = f.get("home_rest_days", 2) - f.get("away_rest_days", 2)

        return diffs

    def _compute_temporal(self, game_date: str) -> dict:
        """Temporal features."""
        try:
            dt = pd.to_datetime(game_date)
            day_of_week = dt.dayofweek
            month = dt.month
            is_weekend = day_of_week >= 5
        except Exception:
            day_of_week = 3
            month = 1
            is_weekend = False

        return {
            "day_of_week": day_of_week,
            "month": month,
            "is_weekend": is_weekend,
        }

    def get_feature_names(self) -> list[str]:
        """
        Return the list of numeric feature names used by the model.
        Non-numeric columns (team names, date) are excluded.
        """
        # Build a dummy to get all keys
        dummy = self.build_features("BOS", "TOR", "2025-01-15")
        return [k for k, v in dummy.items()
                if isinstance(v, (int, float, bool, np.bool_))
                and k not in ("home_team", "away_team", "date")]
