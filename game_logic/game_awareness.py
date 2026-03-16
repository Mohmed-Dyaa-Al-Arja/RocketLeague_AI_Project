"""
game_logic/game_awareness.py
=============================
Classifies the current game situation into a high-level ``GameState``
that higher-level decision modules (passing, positioning, rotation) can
branch on.

States
------
KICKOFF        — ball is centred and a kickoff is in progress
ATTACK         — we possess the ball and are in the attacking half
DEFENSE        — opponent is closer to the ball than we are
COUNTER_ATTACK — we just regained possession in our own half
POSSESSION     — we control the ball in the middle of the field
GOAL_DEFENSE   — opponent is in our box; priority is clearing the ball

Author: medo dyaa
"""

from __future__ import annotations

import math
from typing import Tuple


class GameState:
    KICKOFF        = "Kickoff"
    ATTACK         = "Attack"
    DEFENSE        = "Defense"
    COUNTER_ATTACK = "Counter Attack"
    POSSESSION     = "Possession"
    GOAL_DEFENSE   = "Goal Defense"


_FIELD_HALF_LEN      = 5120.0   # uu — half field length
_OWN_BOX_DIST        = 1500.0   # uu — "own box" radius from goal line
_BALL_IN_OWN_HALF    = 500.0    # uu — ball is "in own half" below this offset from centre
_COUNTER_VEL_THRESH  = 400.0    # uu/s — we need to be moving toward opp goal for counter


class GameAwareness:
    """
    Call ``update()`` once per tick; read ``current_state`` property.

    Author: medo dyaa
    """

    def __init__(self) -> None:
        self._state: str = GameState.POSSESSION
        self._prev_state: str = GameState.POSSESSION

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        ball_pos: Tuple[float, float, float],
        car_pos:  Tuple[float, float, float],
        opp_pos:  Tuple[float, float, float],   # closest opponent
        own_goal_y: float,
        opp_goal_y: float,
        is_kickoff: bool,
        we_have_ball: bool,
    ) -> str:
        """
        Classify the current game situation.

        Returns the new ``GameState`` string (also stored in
        ``self.current_state``).

        Author: medo dyaa
        """
        self._prev_state = self._state
        self._state = self._classify(
            ball_pos, car_pos, opp_pos,
            own_goal_y, opp_goal_y,
            is_kickoff, we_have_ball,
        )
        return self._state

    @property
    def current_state(self) -> str:
        return self._state

    @property
    def previous_state(self) -> str:
        return self._prev_state

    # ------------------------------------------------------------------
    # Classification logic
    # ------------------------------------------------------------------

    def _classify(
        self,
        ball_pos:    Tuple[float, float, float],
        car_pos:     Tuple[float, float, float],
        opp_pos:     Tuple[float, float, float],
        own_goal_y:  float,
        opp_goal_y:  float,
        is_kickoff:  bool,
        we_have_ball: bool,
    ) -> str:
        if is_kickoff:
            return GameState.KICKOFF

        bx, by, bz = ball_pos
        cx, cy, cz = car_pos
        ox, oy, oz = opp_pos

        own_goal_dir  = 1.0 if own_goal_y > 0 else -1.0
        opp_goal_dir  = -own_goal_dir

        # Distance helpers (2-D)
        ball_2d    = (bx, by)
        car_2d     = (cx, cy)
        opp_2d     = (ox, oy)
        own_goal2d = (0.0, own_goal_y)
        opp_goal2d = (0.0, opp_goal_y)

        car_ball_dist = _dist2d(car_2d, ball_2d)
        opp_ball_dist = _dist2d(opp_2d, ball_2d)
        opp_own_goal  = _dist2d(opp_2d, own_goal2d)

        # 1. Goal defense: opponent is in our box with the ball
        if (
            not we_have_ball
            and opp_own_goal < _OWN_BOX_DIST
            and opp_ball_dist < car_ball_dist
        ):
            return GameState.GOAL_DEFENSE

        # 2. Defense: opponent clearly closer to ball
        if not we_have_ball and opp_ball_dist < car_ball_dist - 200:
            return GameState.DEFENSE

        # 3. Attack: we have ball and are in attacking half
        if we_have_ball:
            # "Attacking half" = Y closer to opponent goal
            car_opp_dist = _dist2d(car_2d, opp_goal2d)
            car_own_dist = _dist2d(car_2d, (0.0, own_goal_y))
            if car_opp_dist < car_own_dist:
                return GameState.ATTACK

            # Counter attack: we have ball but are still in own half
            return GameState.COUNTER_ATTACK

        # 4. Possession: we're contesting equally near midfield
        return GameState.POSSESSION


def _dist2d(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])
