"""
game_logic/kickoff_roles.py
============================
Assigns kickoff roles to team-mates based on distance to the ball.

Roles
-----
KICKOFF_PLAYER  — closest to ball; attacks immediately
BOOST_COLLECTOR — second closest; collects boost pad on the way
DEFENDER        — farthest; holds position near own goal

Works for 1-v-1 (only KICKOFF_PLAYER assigned), 2-v-2, and 3-v-3.

Author: medo dyaa
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple


class KickoffRole:
    KICKOFF_PLAYER  = "kickoff_player"
    BOOST_COLLECTOR = "boost_collector"
    DEFENDER        = "defender"


def _dist2d(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def assign_kickoff_roles(
    bot_positions: Dict[int, Tuple[float, float]],
    ball_pos: Tuple[float, float] = (0.0, 0.0),
) -> Dict[int, str]:
    """
    Assign kickoff roles to bots based on distance to the ball.

    Parameters
    ----------
    bot_positions : mapping of {bot_id: (x, y)} for every friendly bot.
    ball_pos      : (x, y) of the ball — typically (0, 0) at kickoff.

    Returns
    -------
    Dict mapping bot_id → KickoffRole string.

    Author: medo dyaa
    """
    if not bot_positions:
        return {}

    sorted_bots = sorted(
        bot_positions.items(),
        key=lambda item: _dist2d(item[1], ball_pos),
    )

    roles: Dict[int, str] = {}
    role_list = [
        KickoffRole.KICKOFF_PLAYER,
        KickoffRole.BOOST_COLLECTOR,
        KickoffRole.DEFENDER,
    ]
    for idx, (bot_id, _) in enumerate(sorted_bots):
        roles[bot_id] = role_list[min(idx, len(role_list) - 1)]

    return roles


class KickoffRoleManager:
    """
    Stateful wrapper — remembers last assignment for the current kickoff.

    Author: medo dyaa
    """

    def __init__(self) -> None:
        self._roles: Dict[int, str] = {}

    def update(
        self,
        bot_positions: Dict[int, Tuple[float, float]],
        ball_pos: Tuple[float, float] = (0.0, 0.0),
    ) -> Dict[int, str]:
        self._roles = assign_kickoff_roles(bot_positions, ball_pos)
        return self._roles

    def get_role(self, bot_id: int) -> str:
        return self._roles.get(bot_id, KickoffRole.KICKOFF_PLAYER)

    def reset(self) -> None:
        self._roles = {}
