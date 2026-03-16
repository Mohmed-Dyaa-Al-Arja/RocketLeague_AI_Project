"""
core/opponent_prediction.py
============================
Predicts future opponent position and estimates threat level using a
simple first-order kinematic model (constant velocity extrapolation).

Designed to run every tick; lightweight, no external deps.

Author: medo dyaa
"""

from __future__ import annotations

import math
from typing import Tuple


# Distance from ball at which an opponent is considered to be "challenging"
_CHALLENGE_DIST_THRESH = 600.0    # uu
# If opponent is this much faster toward the ball, they're definitely first
_FIRST_TO_BALL_BUFFER  = 200.0    # uu


def _dist3d(
    a: Tuple[float, float, float],
    b: Tuple[float, float, float],
) -> float:
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)


class OpponentPredictor:
    """
    Maintain a short history of opponent positions and velocities to
    produce:
      * extrapolated future positions
      * estimated time until the opponent reaches the ball
      * a boolean ``is_challenging`` flag

    Author: medo dyaa
    """

    def __init__(self) -> None:
        self._pos: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._vel: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._prev_pos: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._challenging: bool = False
        self._speed: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        opp_pos: Tuple[float, float, float],
        opp_vel: Tuple[float, float, float],
        ball_pos: Tuple[float, float, float],
        dt: float = 1 / 60,
    ) -> None:
        """
        Call once per tick with the latest opponent state.

        Author: medo dyaa
        """
        self._prev_pos = self._pos
        self._pos      = opp_pos
        self._vel      = opp_vel
        self._speed    = math.sqrt(opp_vel[0]**2 + opp_vel[1]**2 + opp_vel[2]**2)

        dist_to_ball   = _dist3d(opp_pos, ball_pos)
        self._challenging = dist_to_ball < _CHALLENGE_DIST_THRESH

    def predict_position(
        self,
        steps_ahead: int = 10,
        dt: float = 1 / 60,
    ) -> Tuple[float, float]:
        """
        Returns estimated (x, y) position ``steps_ahead`` ticks in the
        future using constant-velocity extrapolation.

        Author: medo dyaa
        """
        t  = steps_ahead * dt
        px = self._pos[0] + self._vel[0] * t
        py = self._pos[1] + self._vel[1] * t
        return (px, py)

    def predict_position_3d(
        self,
        steps_ahead: int = 10,
        dt: float = 1 / 60,
    ) -> Tuple[float, float, float]:
        """
        Same as ``predict_position`` but returns (x, y, z).

        Author: medo dyaa
        """
        t  = steps_ahead * dt
        px = self._pos[0] + self._vel[0] * t
        py = self._pos[1] + self._vel[1] * t
        pz = self._pos[2] + self._vel[2] * t
        return (px, py, pz)

    def predict_challenge_time(
        self,
        opp_pos:  Tuple[float, float, float],
        ball_pos: Tuple[float, float, float],
    ) -> float:
        """
        Estimate seconds until this opponent reaches the ball, assuming
        they continue at their current speed toward it.

        Author: medo dyaa
        """
        dist = _dist3d(opp_pos, ball_pos)
        if self._speed <= 0.0:
            return float("inf")
        return dist / self._speed

    def is_challenging(self) -> bool:
        """
        Returns ``True`` if the opponent is within ``_CHALLENGE_DIST_THRESH``
        of the ball.

        Author: medo dyaa
        """
        return self._challenging

    @property
    def current_pos(self) -> Tuple[float, float, float]:
        return self._pos

    @property
    def current_vel(self) -> Tuple[float, float, float]:
        return self._vel

    @property
    def speed(self) -> float:
        return self._speed
