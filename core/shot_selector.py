"""
core/shot_selector.py
=====================
Evaluates the quality of a potential shot and selects the best shooting strategy.

Criteria
--------
- Distance to opponent goal
- Shooting angle (open corridor width)
- Defender positions (shot blocked?)
- Ball velocity direction (relative to goal)

Author: medo dyaa
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

# ── Constants ─────────────────────────────────────────────────────────────────
_GOAL_HALF_WIDTH   = 893.0   # uu — half-width of a standard goal mouth
_MAX_SHOT_DIST     = 6000.0  # uu — beyond this, reward for shooting is low
_CORNER_OFFSET     = 600.0   # uu — aim this far from centre when aiming corner
_BLOCK_RADIUS      = 400.0   # uu — defender within this cone blocks centre shot
_AIM_THRESHOLD_DEG = 20.0    # degrees — shot angle below this = poor shot


class ShotResult:
    """Named outcome returned by `ShotSelector.evaluate()`."""
    SHOOT_CENTRE  = "shoot_centre"
    AIM_CORNER    = "aim_corner"
    WAIT          = "wait"
    FAKE_SHOT     = "fake_shot"


class ShotSelector:
    """
    Evaluates shooting opportunity and returns a recommended action.

    Usage
    -----
    ::
        sel = ShotSelector()
        result = sel.evaluate(ball_pos, ball_vel, car_pos, opp_goal_y,
                              defender_positions)
        aim = sel.get_aim_point()
    """

    def __init__(self) -> None:
        self._last_result: str = ShotResult.WAIT
        self._last_aim: Optional[Tuple[float, float]] = None

    # ── Public API ────────────────────────────────────────────────────────────

    def evaluate(
        self,
        ball_pos: Tuple[float, float, float],
        ball_vel: Tuple[float, float, float],
        car_pos: Tuple[float, float, float],
        opp_goal_y: float,
        defender_positions: Optional[List[Tuple[float, float]]] = None,
        boost_amount: float = 100.0,
    ) -> str:
        """
        Evaluate the shot situation and return a `ShotResult` constant.

        Parameters
        ----------
        ball_pos           : current ball position (x, y, z).
        ball_vel           : current ball velocity (vx, vy, vz).
        car_pos            : bot car position (x, y, z).
        opp_goal_y         : Y-coordinate of opponent goal line.
        defender_positions : list of (x, y) positions of opposing cars.
        boost_amount       : current boost level (0–100).

        Returns
        -------
        One of ``ShotResult.SHOOT_CENTRE``, ``AIM_CORNER``, ``WAIT``,
        ``FAKE_SHOT``.
        """
        bx, by, bz = ball_pos
        dist_to_goal = abs(opp_goal_y - by)

        # Too far to shoot meaningfully
        if dist_to_goal > _MAX_SHOT_DIST:
            self._last_result = ShotResult.WAIT
            self._last_aim = None
            return ShotResult.WAIT

        # Shooting angle: arc from ball to goal
        dx = _GOAL_HALF_WIDTH
        angle_deg = math.degrees(math.atan2(dx, max(1.0, dist_to_goal))) * 2.0

        if angle_deg < _AIM_THRESHOLD_DEG:
            self._last_result = ShotResult.WAIT
            self._last_aim = None
            return ShotResult.WAIT

        # Analyse defender coverage
        centre_blocked = False
        if defender_positions:
            for dpx, dpy in defender_positions:
                # proximity to the ball–goal line
                t = (dpy - by) / (opp_goal_y - by + 1e-6)
                if 0.0 < t < 1.0:
                    line_x = bx + t * (0.0 - bx)  # goal centre x = 0
                    if abs(dpx - line_x) < _BLOCK_RADIUS:
                        centre_blocked = True
                        break

        if centre_blocked:
            # Aim for the far post (opposite side of defender)
            if defender_positions:
                avg_def_x = sum(d[0] for d in defender_positions) / len(defender_positions)
                corner_x = -math.copysign(_CORNER_OFFSET, avg_def_x)
            else:
                corner_x = _CORNER_OFFSET
            self._last_aim = (corner_x, opp_goal_y)
            self._last_result = ShotResult.AIM_CORNER
            return ShotResult.AIM_CORNER

        # Open shot at centre goal
        self._last_aim = (0.0, opp_goal_y)
        self._last_result = ShotResult.SHOOT_CENTRE
        return ShotResult.SHOOT_CENTRE

    def get_aim_point(self) -> Optional[Tuple[float, float]]:
        """Return the last computed aim point (x, y), or None if waiting."""
        return self._last_aim

    def get_last_result(self) -> str:
        """Return the last `ShotResult` constant."""
        return self._last_result

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def open_shot_angle(
        ball_pos: Tuple[float, float, float],
        opp_goal_y: float,
        defender_positions: Optional[List[Tuple[float, float]]] = None,
    ) -> float:
        """
        Return the effective open arc (degrees) toward the opponent goal from
        ``ball_pos``.  Reduced by 0 when defenders block the full corridor.
        """
        bx, by, _ = ball_pos
        dist = max(1.0, abs(opp_goal_y - by))
        full_angle = math.degrees(math.atan2(_GOAL_HALF_WIDTH, dist)) * 2.0

        if not defender_positions:
            return full_angle

        # Simple block fraction
        blocked_fraction = 0.0
        for dpx, dpy in defender_positions:
            if abs(dpy - opp_goal_y) < 1200:
                block_w = _BLOCK_RADIUS / max(1.0, _GOAL_HALF_WIDTH)
                blocked_fraction = min(1.0, blocked_fraction + block_w)

        return full_angle * max(0.0, 1.0 - blocked_fraction)
