"""
game_logic/positioning_ai.py
=============================
Computes the optimal field position for a bot based on its role, the
current ball position, goal positions, and team-mate locations.

Roles understood
----------------
"attacker" / "first_man"   — stay near ball, threaten goal
"support"  / "second_man"  — midfield, ready for rebounds
"defender" / "third_man"   — between ball and own goal

The module is intentionally simple and physics-free — it returns a 2-D
(x, y) target that higher-level path planners can route towards.

Author: medo dyaa
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple


_FIELD_HALF_WIDTH = 4096.0   # uu
_FIELD_HALF_LEN   = 5120.0   # uu
_GOAL_WIDTH_HALF  = 893.0    # uu

# How far behind the ball a support player shadows (uu toward own goal)
_SUPPORT_SHADOW_DIST = 600.0
# How far toward own goal a defender shadows (uu beyond ball's Y projection)
_DEFENDER_SHADOW_DIST = 900.0
# Maximum X offset a defender is allowed to use (stay central-ish)
_DEFENDER_MAX_X = 500.0


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


class PositioningAI:
    """
    Computes an (x, y) target position for one bot.

    Author: medo dyaa
    """

    def get_target_position(
        self,
        role: str,
        ball_pos: Tuple[float, float, float],
        own_goal_y: float,
        opp_goal_y: float,
        my_pos: Tuple[float, float, float],
        teammate_positions: Optional[List[Tuple[float, float]]] = None,
    ) -> Tuple[float, float]:
        """
        Returns (x, y) recommended world position for this tick.

        Parameters
        ----------
        role               : "attacker"/"first_man", "support"/"second_man",
                             "defender"/"third_man"
        ball_pos           : (x, y, z) ball world position
        own_goal_y         : Y-coordinate of this bot's own goal line
        opp_goal_y         : Y-coordinate of the opponent's goal line
        my_pos             : (x, y, z) own car position
        teammate_positions : optional list of (x, y) team-mates (for spacing)

        Author: medo dyaa
        """
        bx, by = ball_pos[0], ball_pos[1]
        role_lower = role.lower()

        if role_lower in ("attacker", "first_man"):
            return self._attacker_pos(bx, by, opp_goal_y)

        if role_lower in ("support", "second_man"):
            return self._support_pos(bx, by, own_goal_y, opp_goal_y)

        # defender / third_man — default
        return self._defender_pos(bx, by, own_goal_y)

    # ------------------------------------------------------------------
    # Role-specific helpers
    # ------------------------------------------------------------------

    def _attacker_pos(
        self, bx: float, by: float, opp_goal_y: float
    ) -> Tuple[float, float]:
        """
        Stay beside the ball, slightly toward the opponent's goal, angled
        to threaten the shooting lane.

        Author: medo dyaa
        """
        # Nudge a little toward the goal to get first touch
        goal_sign = 1.0 if opp_goal_y > 0 else -1.0
        tx = _clamp(bx, -_FIELD_HALF_WIDTH + 200, _FIELD_HALF_WIDTH - 200)
        ty = by + goal_sign * 50.0
        ty = _clamp(ty, -_FIELD_HALF_LEN + 200, _FIELD_HALF_LEN - 200)
        return (tx, ty)

    def _support_pos(
        self,
        bx: float,
        by: float,
        own_goal_y: float,
        opp_goal_y: float,
    ) -> Tuple[float, float]:
        """
        Midfield shadow — halfway between ball and own goal, offset in X
        to cover a different angle.

        Author: medo dyaa
        """
        goal_dir = 1.0 if own_goal_y < 0 else -1.0

        # Shadow ball position, pulled toward own half
        mid_x = _lerp(bx, 0.0, 0.35)
        mid_y = by + goal_dir * _SUPPORT_SHADOW_DIST
        mid_y = _clamp(mid_y, -_FIELD_HALF_LEN + 200, _FIELD_HALF_LEN - 200)

        # Offset X to open passing lane (opposite side to ball)
        x_offset = -300.0 if bx > 0 else 300.0
        mid_x = _clamp(mid_x + x_offset, -_FIELD_HALF_WIDTH + 200, _FIELD_HALF_WIDTH - 200)

        return (mid_x, mid_y)

    def _defender_pos(
        self,
        bx: float,
        by: float,
        own_goal_y: float,
    ) -> Tuple[float, float]:
        """
        Stay between the ball and own goal, near the goal line.

        Author: medo dyaa
        """
        goal_sign = 1.0 if own_goal_y > 0 else -1.0

        # X: track ball X slightly but stay near centre
        tx = _clamp(bx * 0.4, -_DEFENDER_MAX_X, _DEFENDER_MAX_X)

        # Y: between ball and own goal, weighted toward goal
        ty = _lerp(by, own_goal_y, 0.6)
        ty = _clamp(ty, -_FIELD_HALF_LEN + 100, _FIELD_HALF_LEN - 100)

        return (tx, ty)
