"""
core/dribbling_ai.py
====================
Ball-control dribbling system.

The bot maintains the ball on top of the car while advancing toward the
opponent goal.  Speed is modulated to keep the ball balanced, and the system
detects when the ball is lost so it can deactivate cleanly.

Dribbling conditions
--------------------
- Ball close to car (< ``DRIBBLE_RANGE``)
- Ball height within dribble envelope (< ``CONTROL_HEIGHT``)
- Car is on the ground

Author: medo dyaa
"""

from __future__ import annotations

import math
from typing import Tuple


# ── Constants ─────────────────────────────────────────────────────────────────
DRIBBLE_RANGE   = 350.0   # uu horizontal distance ball must be within
CONTROL_HEIGHT  = 270.0   # uu ball height upper limit for active dribble
_TARGET_OFFSET  = 80.0    # uu — nudge car slightly behind ball centre
_BASE_THROTTLE  = 0.65    # normal dribble speed
_BOOST_THROTTLE = 0.90    # throttle when boosting out of dribble
_MAX_SPEED      = 1400.0  # uu/s limit while dribbling


class DribblingAI:
    """
    Maintains ball possession by modulating throttle and steering.

    Call `should_dribble()` each tick; if True, call `update()` to get control
    overrides.

    Author: medo dyaa
    """

    def __init__(self) -> None:
        self._active: bool = False
        self._tick: int = 0

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def is_active(self) -> bool:
        return self._active

    # ── Public API ────────────────────────────────────────────────────────────

    def should_dribble(
        self,
        ball_pos: Tuple[float, float, float],
        car_pos: Tuple[float, float, float],
        car_on_ground: bool,
    ) -> bool:
        """
        Return True if dribble mode should activate / stay active.
        Automatically deactivates when ball leaves the control envelope.
        """
        if not car_on_ground:
            self._active = False
            return False

        bx, by, bz = ball_pos
        cx, cy, _ = car_pos
        horiz_dist = math.hypot(bx - cx, by - cy)

        if horiz_dist < DRIBBLE_RANGE and bz < CONTROL_HEIGHT:
            self._active = True
            return True

        self._active = False
        return False

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
        Return a control dict: throttle, steer, boost, active.

        ``car_yaw`` is in radians (rl rotation.yaw).
        """
        self._tick += 1
        bx, by, bz = ball_pos
        cx, cy, _ = car_pos
        cvx, cvy, _ = car_vel

        speed = math.hypot(cvx, cvy)

        # Direction from car to ball (flatten)
        dx = bx - cx
        dy = by - cy
        dist = max(1.0, math.hypot(dx, dy))
        ball_angle = math.atan2(dy, dx)
        yaw_err    = _normalize_angle(ball_angle - car_yaw)

        # Steering: keep car pointed at ball
        steer = max(-1.0, min(1.0, yaw_err * 2.5))

        # Throttle: slow if ball is drifting away → catch up; else cruise
        forward_component = cvx * math.cos(car_yaw) + cvy * math.sin(car_yaw)
        if dist > 150:
            throttle = min(_BOOST_THROTTLE, _BASE_THROTTLE + 0.1 * (dist / DRIBBLE_RANGE))
        else:
            throttle = max(0.2, _BASE_THROTTLE - 0.15 * (1.0 - dist / DRIBBLE_RANGE))

        # Cap speed
        if speed > _MAX_SPEED:
            throttle = 0.0

        # Don't boost while dribbling (jostles ball)
        boost = False

        return {
            "throttle": throttle,
            "steer":    steer,
            "boost":    boost,
            "active":   True,
        }

    def deactivate(self) -> None:
        self._active = False


def _normalize_angle(a: float) -> float:
    """Wrap angle to [-π, π]."""
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a
