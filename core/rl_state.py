"""
State discretization utilities shared by all RL algorithms.

Converts continuous Rocket League game state into discrete keys
for Q-table lookups and pattern matching.
"""

from __future__ import annotations

# Actions the Q-learner can choose for role selection
ROLE_ACTIONS = ["attack", "defense", "balanced", "rotate_back", "challenge", "shadow"]

# Observable opponent behavior categories
OPP_ACTIONS = ["rush_ball", "rotate_back", "boost_grab", "shadow", "demolish", "aerial"]


def _zone(x: float, y: float, push_dir: float) -> str:
    """Convert (x, y) to a zone string for Q-table lookup."""
    if x < -1500:
        xz = "L"
    elif x > 1500:
        xz = "R"
    else:
        xz = "C"
    rel_y = y * push_dir
    if rel_y > 2500:
        yz = "opp_deep"
    elif rel_y > 500:
        yz = "opp_half"
    elif rel_y > -500:
        yz = "mid"
    elif rel_y > -2500:
        yz = "our_half"
    else:
        yz = "our_deep"
    return f"{xz}_{yz}"


def _speed_zone(speed: float) -> str:
    if speed < 500:
        return "slow"
    elif speed < 1200:
        return "med"
    elif speed < 1800:
        return "fast"
    return "supersonic"


def _boost_zone(boost: float) -> str:
    if boost < 15:
        return "empty"
    elif boost < 40:
        return "low"
    elif boost < 70:
        return "mid"
    return "full"


def _score_zone(diff: int) -> str:
    if diff <= -3:
        return "losing_bad"
    elif diff < 0:
        return "losing"
    elif diff == 0:
        return "tied"
    elif diff <= 2:
        return "winning"
    return "winning_big"


def build_state_key(
    situation: str,
    ball_zone: str,
    car_zone: str,
    speed_z: str,
    boost_z: str,
    score_z: str,
    possession: str,
) -> str:
    """Build a hashable state key for Q-table."""
    return f"{situation}|{ball_zone}|{car_zone}|{speed_z}|{boost_z}|{score_z}|{possession}"
