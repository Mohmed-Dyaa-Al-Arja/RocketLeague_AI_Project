"""
core/aerial_controller.py
=========================
Aerial (air-shot) mechanics controller for the Rocket League AI bot.

Detects when the ball is airborne and calculates jump timing, boost direction,
and car tilt to perform aerial interceptions.

Conditions for attempting an aerial
------------------------------------
- ball_height > 200 units
- distance_to_ball < 2500 units
- car has sufficient boost (> 25)
- car is on the ground (pre-jump phase)

State machine
-------------
IDLE  →  JUMPING  →  BOOSTING  →  COMPLETE

The controller outputs a recommended action dict consumed by the bot tick loop.

Author: medo dyaa
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple


# ── Constants ──────────────────────────────────────────────────────────────────
_MIN_BALL_HEIGHT   = 200.0   # uu — ball must be this high to consider aerial
_MAX_BALL_HEIGHT   = 1900.0  # uu — above ceiling is impossible
_MAX_DIST_TO_BALL  = 2500.0  # uu — too far away → don't commit
_MIN_BOOST         = 25.0    # units of boost required before committing
_JUMP_DURATION     = 0.2     # seconds of first jump hold
_BOOST_TILT_GAIN   = 0.8     # how aggressively to tilt toward ball (0–1)
_AIR_MAX_SPEED     = 2300.0  # uu/s in air with boost


class AerialState:
    IDLE      = "idle"
    JUMPING   = "jumping"
    BOOSTING  = "boosting"
    COMPLETE  = "complete"


class AerialController:
    """
    Manages aerial interception logic.

    Call `update()` every tick; if an aerial is recommended the returned dict
    will contain the control overrides to apply.
    """

    def __init__(self) -> None:
        self._state: str = AerialState.IDLE
        self._jump_timer: float = 0.0
        self._target: Optional[Tuple[float, float, float]] = None
        self._boost_timer: float = 0.0

    # ── Public API ────────────────────────────────────────────────────────────

    def should_attempt_aerial(
        self,
        ball_pos: Tuple[float, float, float],
        car_pos: Tuple[float, float, float],
        car_boost: float,
        car_on_ground: bool,
    ) -> bool:
        """
        Return True when conditions for an aerial are met.
        Only recommends starting a NEW aerial when IDLE.
        """
        if self._state != AerialState.IDLE:
            return False  # already in flight

        bz = ball_pos[2]
        if bz < _MIN_BALL_HEIGHT or bz > _MAX_BALL_HEIGHT:
            return False

        dist = math.hypot(
            ball_pos[0] - car_pos[0],
            ball_pos[1] - car_pos[1],
        )
        if dist > _MAX_DIST_TO_BALL:
            return False

        if car_boost < _MIN_BOOST:
            return False

        return True   # conditions satisfied; aerial is feasible

    def update(
        self,
        ball_pos: Tuple[float, float, float],
        car_pos: Tuple[float, float, float],
        car_vel: Tuple[float, float, float],
        car_boost: float,
        car_on_ground: bool,
        dt: float = 1.0 / 60.0,
    ) -> Dict[str, object]:
        """
        Compute aerial control output for this tick.

        Returns a dict with keys:
        - ``"jump"``    (bool)
        - ``"boost"``   (bool)
        - ``"pitch"``   (float, -1..+1)
        - ``"yaw"``     (float, -1..+1)
        - ``"roll"``    (float, -1..+1)
        - ``"active"``  (bool) — True while aerial is ongoing
        - ``"state"``   (str)
        """
        default = {"jump": False, "boost": False,
                   "pitch": 0.0, "yaw": 0.0, "roll": 0.0,
                   "active": False, "state": self._state}

        if self._state == AerialState.IDLE:
            if self.should_attempt_aerial(ball_pos, car_pos, car_boost, car_on_ground):
                self._target = ball_pos
                self._state = AerialState.JUMPING
                self._jump_timer = 0.0
                self._boost_timer = 0.0
            else:
                return default

        bx, by, bz = (self._target or ball_pos)
        cx, cy, cz = car_pos

        # Direction vector from car to ball
        dx = bx - cx
        dy = by - cy
        dz = bz - cz
        dist3d = max(1.0, math.sqrt(dx*dx + dy*dy + dz*dz))

        # ── JUMPING phase: first jump ──────────────────────────────────────
        if self._state == AerialState.JUMPING:
            self._jump_timer += dt
            ctrl: Dict[str, object] = {
                "jump":   True,
                "boost":  False,
                "pitch":  0.0,
                "yaw":    0.0,
                "roll":   0.0,
                "active": True,
                "state":  self._state,
            }
            if self._jump_timer >= _JUMP_DURATION:
                self._state = AerialState.BOOSTING
            return ctrl

        # ── BOOSTING phase: tilt car toward ball and boost ─────────────────
        if self._state == AerialState.BOOSTING:
            self._boost_timer += dt

            # Pitch: tilt nose up toward ball (clamped)
            pitch_needed = math.atan2(dz, math.hypot(dx, dy))
            pitch_norm   = math.degrees(pitch_needed) / 90.0
            pitch_ctrl   = max(-1.0, min(1.0, pitch_norm * _BOOST_TILT_GAIN))

            # Yaw: steer toward ball horizontally
            yaw_needed = math.atan2(dx, dy)
            # Convert to -1..+1 by rough normalisation
            yaw_ctrl = math.sin(yaw_needed) * _BOOST_TILT_GAIN
            yaw_ctrl = max(-1.0, min(1.0, yaw_ctrl))

            # Check if we've reached the ball (within 200 units) or run out of boost
            reached = dist3d < 200.0
            no_boost = car_boost < 5 and self._boost_timer > 0.5
            timeout  = self._boost_timer > 3.0   # give up after 3 seconds

            if reached or no_boost or timeout:
                self._state = AerialState.COMPLETE

            ctrl = {
                "jump":   False,
                "boost":  car_boost > 5,
                "pitch":  pitch_ctrl,
                "yaw":    yaw_ctrl,
                "roll":   0.0,
                "active": True,
                "state":  self._state,
            }
            return ctrl

        # ── COMPLETE — reset for next opportunity ───────────────────────────
        self._state = AerialState.IDLE
        self._target = None
        return {**default, "active": False, "state": AerialState.IDLE}

    def cancel(self) -> None:
        """Abandon the current aerial and return to IDLE."""
        self._state = AerialState.IDLE
        self._target = None
        self._jump_timer = 0.0
        self._boost_timer = 0.0

    @property
    def is_active(self) -> bool:
        return self._state not in (AerialState.IDLE, AerialState.COMPLETE)

    @property
    def state(self) -> str:
        return self._state
