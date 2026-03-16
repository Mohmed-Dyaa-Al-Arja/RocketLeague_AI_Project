"""
core/wall_play_ai.py
====================
Wall-riding and wall-shot AI module.

Conditions for activation
--------------------------
- Ball is near a side wall (|ball_x| > 3200) or ceiling area
- Ball is rolling/travelling along the wall
- Ball height < wall_play_limit

Behaviours
----------
- Drive up the wall to intercept
- Wall shot / clear toward field centre
- Pass back to ground when ball returns to floor

Author: medo dyaa
"""

from __future__ import annotations

import math
from typing import Optional, Tuple


# ── Constants ─────────────────────────────────────────────────────────────────
_FIELD_HALF_X    = 4096.0   # uu — wall boundary start
_WALL_THRESHOLD  = 3200.0   # uu — ball x or y beyond this = wall zone
_WALL_HEIGHT_MAX = 1500.0   # uu — maximum ball height for wall play
_WALL_SHOT_DIST  = 2200.0   # uu — max distance from car to ball for wall shot


class WallPlayState:
    IDLE        = "idle"
    DRIVING_UP  = "driving_up"
    INTERCEPTING = "intercepting"
    SHOOTING    = "shooting"
    RETURNING   = "returning"


class WallPlayAI:
    """
    Controls wall-riding interception and wall-shot sequences.

    Usage
    -----
    Call `should_engage_wall()` each tick; if True, call `update()`.

    Author: medo dyaa
    """

    def __init__(self) -> None:
        self._state: str = WallPlayState.IDLE
        self._tick: int = 0

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def is_active(self) -> bool:
        return self._state != WallPlayState.IDLE

    # ── Public API ────────────────────────────────────────────────────────────

    def should_engage_wall(
        self,
        ball_pos: Tuple[float, float, float],
        car_pos: Tuple[float, float, float],
        car_on_ground: bool,
    ) -> bool:
        """
        Return True when wall play should begin.

        Activates when ball is in wall zone, within reachable height, and car
        is still close enough to make the intercept.
        """
        bx, by, bz = ball_pos
        cx, cy, cz = car_pos

        ball_near_wall = (abs(bx) > _WALL_THRESHOLD or abs(by) > _WALL_THRESHOLD)
        if not ball_near_wall:
            if self._state != WallPlayState.IDLE:
                self._state = WallPlayState.RETURNING
            else:
                return False

        if bz > _WALL_HEIGHT_MAX:
            return False

        dist = math.hypot(bx - cx, by - cy)
        if dist > _WALL_SHOT_DIST * 2:
            return False

        if self._state == WallPlayState.IDLE:
            self._state = WallPlayState.DRIVING_UP
            self._tick = 0

        return True

    def update(
        self,
        ball_pos: Tuple[float, float, float],
        car_pos: Tuple[float, float, float],
        car_vel: Tuple[float, float, float],
        car_yaw: float,
        opp_goal_y: float,
        dt: float = 1 / 60,
    ) -> dict:
        """
        Return a control dict: throttle, steer, boost, jump, pitch, active.
        """
        self._tick += 1
        bx, by, bz = ball_pos
        cx, cy, cz = car_pos

        # Direction to ball
        dx = bx - cx
        dy = by - cy
        dist = max(1.0, math.hypot(dx, dy))
        ball_angle  = math.atan2(dy, dx)
        yaw_err     = _normalize_angle(ball_angle - car_yaw)
        steer       = max(-1.0, min(1.0, yaw_err * 2.0))

        if self._state == WallPlayState.DRIVING_UP:
            # Charge toward wall / ball
            result = {"throttle": 1.0, "boost": True, "jump": False,
                      "pitch": 0.0, "steer": steer, "active": True,
                      "state": self._state}
            if dist < 500:
                self._state = WallPlayState.INTERCEPTING
            return result

        elif self._state == WallPlayState.INTERCEPTING:
            # On the wall — aim toward goal direction
            goal_dx = 0.0 - cx
            goal_dy = opp_goal_y - cy
            goal_angle = math.atan2(goal_dy, goal_dx)
            steer = max(-1.0, min(1.0, _normalize_angle(goal_angle - car_yaw) * 2.5))
            result = {"throttle": 1.0, "boost": False, "jump": False,
                      "pitch": 0.0, "steer": steer, "active": True,
                      "state": self._state}
            if dist < 150:
                self._state = WallPlayState.SHOOTING
            return result

        elif self._state == WallPlayState.SHOOTING:
            # Jump / redirect toward centre or goal
            if self._tick % 20 < 5:
                result = {"throttle": 1.0, "boost": True, "jump": True,
                          "pitch": -0.5, "steer": 0.0, "active": True,
                          "state": self._state}
            else:
                result = {"throttle": 1.0, "boost": True, "jump": False,
                          "pitch": 0.0, "steer": steer, "active": True,
                          "state": self._state}
            # After shot, transition back
            if self._tick > 40:
                self._state = WallPlayState.RETURNING
            return result

        elif self._state == WallPlayState.RETURNING:
            self._state = WallPlayState.IDLE
            return {"active": False, "state": WallPlayState.RETURNING}

        # Default / IDLE
        return {"active": False, "state": WallPlayState.IDLE}

    def cancel(self) -> None:
        self._state = WallPlayState.IDLE


def _normalize_angle(a: float) -> float:
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a
