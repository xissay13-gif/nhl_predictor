"""
Advanced analytics features.

Covers:
  - Possession: Corsi For %, Fenwick %, shot attempt differential
  - Shot quality: xGoals For/Against, high-danger chances %
  - Pace factor
  - Expected goals differential
"""

import logging

from utils.helpers import safe_div

logger = logging.getLogger("nhl_predictor.features.advanced")


def compute_possession_features(team_data: dict) -> dict:
    """
    Corsi, Fenwick, and shot attempt metrics.
    Expects MoneyPuck-style team data dict.
    """
    cf = team_data.get("corsiFor", 0)
    ca = team_data.get("corsiAgainst", 0)
    ff = team_data.get("fenwickFor", 0)
    fa = team_data.get("fenwickAgainst", 0)
    sf = team_data.get("shotsOnGoalFor", 0)
    sa = team_data.get("shotsOnGoalAgainst", 0)

    cf_pct = safe_div(cf, cf + ca, 0.5) * 100
    ff_pct = safe_div(ff, ff + fa, 0.5) * 100
    sf_pct = safe_div(sf, sf + sa, 0.5) * 100

    gp = max(team_data.get("gamesPlayed", 1), 1)

    return {
        "corsi_for": cf,
        "corsi_against": ca,
        "corsi_for_pct": cf_pct,
        "corsi_diff_pg": (cf - ca) / gp,
        "fenwick_for": ff,
        "fenwick_against": fa,
        "fenwick_for_pct": ff_pct,
        "fenwick_diff_pg": (ff - fa) / gp,
        "shots_for": sf,
        "shots_against": sa,
        "shots_for_pct": sf_pct,
        "shots_diff_pg": (sf - sa) / gp,
    }


def compute_expected_goals_features(team_data: dict) -> dict:
    """
    Expected goals (xG) features.
    """
    xgf = team_data.get("xGoalsFor", 0)
    xga = team_data.get("xGoalsAgainst", 0)
    gf = team_data.get("goalsFor", 0)
    ga = team_data.get("goalsAgainst", 0)
    gp = max(team_data.get("gamesPlayed", 1), 1)

    xgf_pct = safe_div(xgf, xgf + xga, 0.5) * 100

    return {
        "xgf": xgf,
        "xga": xga,
        "xgf_pg": xgf / gp,
        "xga_pg": xga / gp,
        "xg_diff": xgf - xga,
        "xg_diff_pg": (xgf - xga) / gp,
        "xgf_pct": xgf_pct,
        # Goals above expected — team finishing ability
        "goals_above_xg": gf - xgf,
        "goals_saved_above_xg_team": xga - ga,
        # Luck indicator: how much actual results diverge from expected
        "gf_vs_xgf_ratio": safe_div(gf, xgf, 1.0),
        "ga_vs_xga_ratio": safe_div(ga, xga, 1.0),
    }


def compute_shot_quality_features(team_data: dict) -> dict:
    """
    High-danger chances and shot quality.
    """
    hdgf = team_data.get("highDangerGoalsFor", 0)
    hdga = team_data.get("highDangerGoalsAgainst", 0)
    gp = max(team_data.get("gamesPlayed", 1), 1)

    return {
        "hd_goals_for": hdgf,
        "hd_goals_against": hdga,
        "hd_goals_for_pg": hdgf / gp,
        "hd_goals_against_pg": hdga / gp,
        "hd_goal_diff_pg": (hdgf - hdga) / gp,
        "hd_goals_pct": safe_div(hdgf, hdgf + hdga, 0.5) * 100,
    }


def compute_pace_factor(team_data: dict) -> dict:
    """
    Pace = total shot attempts per game.
    High pace teams generate more variance.
    """
    cf = team_data.get("corsiFor", 0)
    ca = team_data.get("corsiAgainst", 0)
    gp = max(team_data.get("gamesPlayed", 1), 1)

    total_corsi_pg = (cf + ca) / gp

    return {
        "pace_corsi_pg": total_corsi_pg,
        "pace_shots_pg": (
            team_data.get("shotsOnGoalFor", 0) + team_data.get("shotsOnGoalAgainst", 0)
        ) / gp,
    }


def compute_special_teams_features(standings: dict, team_data: dict) -> dict:
    """
    Special teams: PP%, PK%, opportunities.
    Takes data from both standings and MoneyPuck.
    """
    # These might come from different sources; be flexible
    return {
        "pp_pct": team_data.get("powerPlayPct", standings.get("pp_pct", 20.0)),
        "pk_pct": team_data.get("penaltyKillPct", standings.get("pk_pct", 80.0)),
        "pp_opportunities_pg": team_data.get("ppOpportunitiesPerGame",
                                              standings.get("pp_opp_pg", 3.0)),
        "pk_times_pg": team_data.get("timesShorthanded",
                                     standings.get("pk_times_pg", 3.0)),
        "special_teams_index": (
            team_data.get("powerPlayPct", standings.get("pp_pct", 20.0))
            + team_data.get("penaltyKillPct", standings.get("pk_pct", 80.0))
        ),
    }
