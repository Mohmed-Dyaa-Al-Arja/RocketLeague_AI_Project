"""
core/fake_shot_ai.py
====================
Deception system: simulates a shot approach, reads the defender's commitment,
then redirects to finish around the sliding keeper.

Trigger conditions
------------------
- Defender between ball and goal (shot block likely)
- Opponent rushing toward ball / closing fast
- Bot has sufficient approach speed and angle

State machine
-------------
DORMANT → APPROACHING → COMMITTING → REDIRECTING → DONE

Author: medo dyaa
"""

from __future__ import annotations

import math
from typing import Optional, Tuple


# ── Constants ─────────────────────────────────────────────────────────────────
_APPROACH_DIST    = 800.0   # uu — start fake when ball this close
_DEFENDER_RADIUS  = 600.0   # uu — defender within this considered "blocking"
_COMMIT_SPEED     = 600.0   # uu/s closing speed that counts as "committed"
_SLOW_THROTTLE    = 0.3
_FAST_THROTTLE    = 1.0
_REDIRECT_DEG     = 35.0    # degrees to steer away from original line


class FakeShotState:
    DORMANT     = "dormant"
    APPROACHING = "approaching"
    COMMITTING  = "committing"
    REDIRECTING = "redirecting"
    DONE        = "done"


class FakeShotResult:
    """Action dict keys returned by update()."""
    THROTTLE = "throttle"
    STEER    = "steer"
    BOOST    = "boost"
    ACTIVE   = "active"
    STATE    = "state"


class FakeShotAI:
    """
    Controls a deceptive shot-fake sequence.

    Usage
    -----
    Call `check_trigger()` each tick; if it returns True, the state machine has
    armed.  Call `update()` to get throttle/steer overrides while `is_active`.
    """

    def __init__(self) -> None:
        self._state: str = FakeShotState.DORMANT
        self._tick: int = 0
        self._commit_tick: int = 0
        self._redirect_dir: float = 1.0   # +1 left, -1 right

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def is_active(self) -> bool:
        return self._state not in (FakeShotState.DORMANT, FakeShotState.DONE)

    # ── Public API ────────────────────────────────────────────────────────────

    def check_trigger(
        self,
        ball_pos: Tuple[float, float, float],
        car_pos: Tuple[float, float, float],
        car_vel: Tuple[float, float, float],
        defender_pos: Optional[Tuple[float, float]] = None,
        defender_vel: Optional[Tuple[float, float]] = None,
        opp_goal_y: float = 5120.0,
    ) -> bool:
        """
        Decide whether to start a fake shot.

        Returns True (and arms the state machine) when:
        - Defender directly between ball and goal
        - Defender is closing (rushing toward ball)
        - We are within approach distance

        Call only when bot is already in possession / has-ball context.
        """
        if self._state != FakeShotState.DORMANT:
            return self.is_active

        bx, by, _ = ball_pos
        cx, cy, _ = car_pos

        dist_car_ball = math.hypot(cx - bx, cy - by)
        if dist_car_ball > _APPROACH_DIST * 2:
            return False

        if defender_pos is None:
            return False

        dpx, dpy = defender_pos

        # Is defender between ball and goal?
        goal_dir_y = opp_goal_y - by
        if abs(goal_dir_y) < 1.0:
            return False
        t = (dpy - by) / goal_dir_y
        if not (0.05 < t < 0.95):
            return False
        line_x_at_def = bx + t * (0.0 - bx)
        if abs(dpx - line_x_at_def) > _DEFENDER_RADIUS:
            return False

        # Is defender closing fast?
        if defender_vel:
            dvx, dvy = defender_vel
            closing = -math.hypot(dvx, dvy) if dvy * (by - dpy) > 0 else math.hypot(dvx, dvy)
            if closing > _COMMIT_SPEED:
                pass  # defender is charging — good candidate for fake

        # Arm the fake: redirect to the side with more space
        self._redirect_dir = -math.copysign(1.0, dpx)   # away from defender
        self._state = FakeShotState.APPROACHING
        self._tick = 0
        return True

    def update(
        self,
        ball_pos: Tuple[float, float, float],
        car_pos: Tuple[float, float, float],
        defender_vel: Optional[Tuple[float, float]] = None,
        dt: float = 1 / 60,
    ) -> dict:
        """
        Return a control dict with throttle / steer overrides.

        Must call every tick while `is_active`.
        Returns ``{"active": False}`` when the sequence ends.
        """
        self._tick += 1
        result: dict = {
            FakeShotResult.THROTTLE: _FAST_THROTTLE,
            FakeShotResult.STEER:    0.0,
            FakeShotResult.BOOST:    False,
            FakeShotResult.ACTIVE:   True,
            FakeShotResult.STATE:    self._state,
        }

        if self._state == FakeShotState.APPROACHING:
            # Drive at full speed toward ball as if shooting
            result[FakeShotResult.THROTTLE] = _FAST_THROTTLE
            result[FakeShotResult.BOOST] = True
            if self._tick > 45:      # ~0.75 s approach
                self._state = FakeShotState.COMMITTING
                self._commit_tick = self._tick

        elif self._state == FakeShotState.COMMITTING:
            # Slow briefly to bait defender
            result[FakeShotResult.THROTTLE] = _SLOW_THROTTLE
            result[FakeShotResult.BOOST] = False
            elapsed = self._tick - self._commit_tick
            if elapsed > 15:         # 0.25 s bait window
                self._state = FakeShotState.REDIRECTING

        elif self._state == FakeShotState.REDIRECTING:
            # Burst away from keeper
            result[FakeShotResult.THROTTLE] = _FAST_THROTTLE
            result[FakeShotResult.STEER] = self._redirect_dir * 0.8
            result[FakeShotResult.BOOST] = True
            if self._tick > self._commit_tick + 45:
                self._state = FakeShotState.DONE

        elif self._state == FakeShotState.DONE:
            self._state = FakeShotState.DORMANT
            return {FakeShotResult.ACTIVE: False, FakeShotResult.STATE: FakeShotState.DONE}

        return result

    def cancel(self) -> None:
        """Abort the current fake shot sequence."""
        self._state = FakeShotState.DORMANT
        self._tick = 0
