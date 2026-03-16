"""
core/kickoff_ai.py
==================
Professional kickoff controller with multiple selectable strategies.

Kickoff detection
-----------------
- Ball at centre field (|x| < 200, |y| < 200)
- Match just started or goal was scored (app calls `notify_kickoff()`)

Strategies
----------
Speed Flip Kickoff  — wave-dash + flip toward ball; fastest
Diagonal Kickoff    — angled approach with front-flip at ball
Delayed Kickoff     — hang back, force opponent to commit first
Fake Kickoff        — rush then brake; opponent over-commits

Strategy chosen automatically based on spawn position distance.

Author: medo dyaa
"""

from __future__ import annotations

import math
from typing import Optional, Tuple


# ── Constants ─────────────────────────────────────────────────────────────────
_BALL_CENTRE_RADIUS = 200.0   # uu — ball within this = kickoff
_BOOST_PAD_EARLY    = 2300.0  # uu — grab side boost pad if close enough
_SPEED_FLIP_DIST    = 2500.0  # uu — use Speed Flip when closer than this
_BOOST_THROTTLE     = 1.0
_KICKOFF_PITCH_FLIP = -1.0    # nose-forward flip pitch


class KickoffStrategy:
    SPEED_FLIP = "speed_flip"
    DIAGONAL   = "diagonal"
    DELAYED    = "delayed"
    FAKE       = "fake"


class KickoffAI:
    """
    Manages a single kickoff sequence.

    Call `notify_kickoff()` when a kickoff is detected; then call `update()`
    every tick.  The controller becomes inactive after hitting the ball or on
    timeout.

    Author: medo dyaa
    """

    _TIMEOUT_TICKS = 180  # 3 s — give up if we haven't hit in this time

    def __init__(self) -> None:
        self._active: bool = False
        self._strategy: str = KickoffStrategy.DIAGONAL
        self._tick: int = 0
        self._phase: int = 0   # internal sub-step
        self._flip_done: bool = False

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def is_active(self) -> bool:
        return self._active

    # ── Public API ────────────────────────────────────────────────────────────

    def notify_kickoff(
        self,
        car_pos: Tuple[float, float, float],
        push_dir: float,
    ) -> None:
        """
        Arm the kickoff controller.

        Parameters
        ----------
        car_pos   : current car position (x, y, z).
        push_dir  : +1 or -1 (team direction).
        """
        self._active = True
        self._tick = 0
        self._phase = 0
        self._flip_done = False

        spawn_dist = math.hypot(car_pos[0], car_pos[1])
        if spawn_dist < _SPEED_FLIP_DIST:
            self._strategy = KickoffStrategy.SPEED_FLIP
        elif abs(car_pos[0]) > 1500:
            self._strategy = KickoffStrategy.DIAGONAL
        else:
            self._strategy = KickoffStrategy.DELAYED

    def cancel(self) -> None:
        """Abort kickoff (e.g. after successfully hitting ball)."""
        self._active = False

    def update(
        self,
        car_pos: Tuple[float, float, float],
        ball_pos: Tuple[float, float, float],
        car_on_ground: bool,
        dt: float = 1 / 60,
    ) -> dict:
        """
        Return control dict: throttle, boost, jump, pitch, steer, active.

        Call every tick while ``is_active``.
        """
        if not self._active:
            return {"active": False}

        self._tick += 1
        if self._tick > self._TIMEOUT_TICKS:
            self._active = False
            return {"active": False}

        # Auto-cancel when ball is no longer at centre
        bx, by, _ = ball_pos
        if self._tick > 20 and math.hypot(bx, by) > _BALL_CENTRE_RADIUS * 3:
            self._active = False
            return {"active": False}

        t = self._tick

        if self._strategy == KickoffStrategy.SPEED_FLIP:
            return self._speed_flip(t, car_pos, ball_pos, car_on_ground)
        elif self._strategy == KickoffStrategy.DIAGONAL:
            return self._diagonal(t, car_pos, ball_pos, car_on_ground)
        elif self._strategy == KickoffStrategy.DELAYED:
            return self._delayed(t, car_pos, ball_pos, car_on_ground)
        else:
            return self._diagonal(t, car_pos, ball_pos, car_on_ground)

    # ── Strategy implementations ──────────────────────────────────────────────

    def _speed_flip(self, t: int, car_pos, ball_pos, on_ground: bool) -> dict:
        """Fast wave-dash + front-flip toward ball."""
        bx, by, _ = ball_pos
        cx, cy, _ = car_pos
        steer = math.copysign(0.3, bx - cx)   # slight correction

        if t < 5:
            return {"throttle": 1.0, "boost": True, "jump": False,
                    "pitch": 0.0, "steer": steer, "active": True}
        if t < 10:
            # First jump
            return {"throttle": 1.0, "boost": True, "jump": True,
                    "pitch": 0.0, "steer": steer, "active": True}
        if t < 13:
            return {"throttle": 1.0, "boost": True, "jump": False,
                    "pitch": -0.3, "steer": steer, "active": True}
        if t < 18 and not self._flip_done:
            self._flip_done = True
            return {"throttle": 1.0, "boost": True, "jump": True,
                    "pitch": _KICKOFF_PITCH_FLIP, "steer": steer, "active": True}
        return {"throttle": 1.0, "boost": True, "jump": False,
                "pitch": 0.0, "steer": steer, "active": True}

    def _diagonal(self, t: int, car_pos, ball_pos, on_ground: bool) -> dict:
        """Angled approach with flip at ball."""
        bx, by, _ = ball_pos
        cx, cy, _ = car_pos
        steer = math.tanh((bx - cx) * 0.001)

        if t < 60:
            return {"throttle": 1.0, "boost": True, "jump": False,
                    "pitch": 0.0, "steer": steer, "active": True}
        if t < 65:
            return {"throttle": 1.0, "boost": True, "jump": True,
                    "pitch": 0.0, "steer": steer, "active": True}
        if t < 70 and not self._flip_done:
            self._flip_done = True
            return {"throttle": 1.0, "boost": True, "jump": True,
                    "pitch": _KICKOFF_PITCH_FLIP, "steer": 0.0, "active": True}
        return {"throttle": 1.0, "boost": False, "jump": False,
                "pitch": 0.0, "steer": steer, "active": True}

    def _delayed(self, t: int, car_pos, ball_pos, on_ground: bool) -> dict:
        """Hang back until opponent commits, then rush."""
        bx, by, _ = ball_pos
        cx, cy, _ = car_pos
        steer = math.tanh((bx - cx) * 0.001)

        if t < 30:
            # Slow approach
            return {"throttle": 0.5, "boost": False, "jump": False,
                    "pitch": 0.0, "steer": steer, "active": True}
        # Full rush after delay
        if t < 90:
            return {"throttle": 1.0, "boost": True, "jump": False,
                    "pitch": 0.0, "steer": steer, "active": True}
        if t < 95:
            return {"throttle": 1.0, "boost": True, "jump": True,
                    "pitch": 0.0, "steer": steer, "active": True}
        if t < 100 and not self._flip_done:
            self._flip_done = True
            return {"throttle": 1.0, "boost": True, "jump": True,
                    "pitch": _KICKOFF_PITCH_FLIP, "steer": 0.0, "active": True}
        return {"throttle": 1.0, "boost": False, "jump": False,
                "pitch": 0.0, "steer": steer, "active": True}
