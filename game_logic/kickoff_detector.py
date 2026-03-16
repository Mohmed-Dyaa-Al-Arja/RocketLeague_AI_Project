"""
game_logic/kickoff_detector.py
===============================
Detects kickoff events from live game packets.

A kickoff is active when:
- ball is within _CENTRE_RADIUS of field centre (0, 0)
- ball velocity magnitude is below _VEL_THRESH
- packet.game_info.is_kickoff_pause is True   (confirmed via game flag)

The detector also fires a "new_kickoff" edge event exactly once per
kickoff so callers can perform one-shot resets.

Author: medo dyaa
"""

from __future__ import annotations

from typing import Tuple


_CENTRE_RADIUS: float = 120.0   # uu  — ball within this = centre
_VEL_THRESH:    float = 30.0    # uu/s — ball "nearly still"


class KickoffDetector:
    """
    Stateful kickoff detector.

    Call `update()` every tick.  Use `is_kickoff` to read current state
    and `just_started` to detect the rising edge (first tick of new kickoff).

    Author: medo dyaa
    """

    def __init__(self) -> None:
        self._in_kickoff: bool = False
        self._just_started: bool = False

    # ── Per-tick update ───────────────────────────────────────────────────────

    def update(
        self,
        ball_pos: Tuple[float, float, float],
        ball_vel: Tuple[float, float, float],
        is_kickoff_pause: bool,
    ) -> None:
        """
        Update detector state from current packet data.

        Parameters
        ----------
        ball_pos        : (x, y, z) ball position in unreal units.
        ball_vel        : (vx, vy, vz) ball velocity.
        is_kickoff_pause: ``packet.game_info.is_kickoff_pause``
        """
        bx, by, _ = ball_pos
        bvx, bvy, _ = ball_vel

        ball_at_centre = (abs(bx) < _CENTRE_RADIUS and abs(by) < _CENTRE_RADIUS)
        ball_still = (abs(bvx) < _VEL_THRESH and abs(bvy) < _VEL_THRESH)
        currently_kickoff = ball_at_centre and ball_still and is_kickoff_pause

        self._just_started = currently_kickoff and not self._in_kickoff
        self._in_kickoff = currently_kickoff

    # ── Accessors ─────────────────────────────────────────────────────────────

    @property
    def is_kickoff(self) -> bool:
        """True while the kickoff pause is active."""
        return self._in_kickoff

    @property
    def just_started(self) -> bool:
        """True only on the very first tick of a new kickoff."""
        return self._just_started

    def reset(self) -> None:
        self._in_kickoff = False
        self._just_started = False
