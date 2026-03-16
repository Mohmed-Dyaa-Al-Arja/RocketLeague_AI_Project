"""
core/boost_manager.py
=====================
Boost management system for the Rocket League AI bot.

Responsibilities
----------------
- Track the current boost level
- Identify the nearest available boost pads (large pads preferred)
- Decide when it is appropriate to collect boost vs. stay in play

Decision rules
--------------
- If boost < 20 → go to nearest boost pad (any pad)
- If attacking and boost < 40 → prefer large boost pads
- If defending → collect nearby pads opportunistically while rotating back

Author: medo dyaa
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

# Boost pad types (must check pad.is_large to distinguish)
_LARGE_PAD_BOOST = 100.0   # full-boost pad restores to 100
_SMALL_PAD_BOOST = 12.0    # small pad adds 12

# Thresholds
_CRITICAL_BOOST  = 20.0    # below this → collect urgently
_LOW_BOOST_ATK   = 40.0    # attacking threshold for large pad preference
_LOW_BOOST_DEF   = 30.0    # defending threshold for opportunistic pickup

# Weight bonus for large pads in scoring
_LARGE_PAD_BONUS = 500.0


class BoostPadInfo:
    """Lightweight value object for a single boost pad."""
    __slots__ = ("pos", "is_large", "timer")

    def __init__(self, pos: Tuple[float, float], is_large: bool) -> None:
        self.pos      = pos
        self.is_large = is_large
        self.timer    = 0.0   # seconds until pad respawns (0 = available)


class BoostManager:
    """
    Manages boost collection decisions for the AI bot.

    Call `update()` each tick to refresh pad availability and the car's current
    boost level.  Then call `get_target_pad()` to get the recommended pad
    position (or ``None`` if no action is needed).
    """

    def __init__(self) -> None:
        self._pads: List[BoostPadInfo] = []
        self._current_boost: float = 50.0
        self._mode: str = "balanced"   # "attack", "defense", "balanced"

    # ── Setup ──────────────────────────────────────────────────────────────────

    def set_pads(self, pad_list: List[Dict]) -> None:
        """
        Load pad positions from field info.

        Parameters
        ----------
        pad_list : list of dicts with keys ``"pos"`` (x, y) and ``"is_large"`` (bool).
        """
        self._pads = [
            BoostPadInfo(
                pos=tuple(p["pos"]),       # type: ignore[arg-type]
                is_large=bool(p.get("is_large", False)),
            )
            for p in pad_list
        ]

    # ── Per-tick update ───────────────────────────────────────────────────────

    def update(
        self,
        current_boost: float,
        mode: str,
        car_pos: Tuple[float, float],
        dt: float = 1.0 / 60.0,
    ) -> None:
        """
        Refresh boost level and pad availability estimate.

        Parameters
        ----------
        current_boost : current boost value [0, 100].
        mode          : current bot mode ("attack", "defense", "balanced").
        car_pos       : (x, y) car position (used to detect pad pickup).
        dt            : seconds since last update.
        """
        self._current_boost = max(0.0, min(100.0, float(current_boost)))
        self._mode = mode

        # Tick down pad respawn timers
        for pad in self._pads:
            if pad.timer > 0:
                pad.timer = max(0.0, pad.timer - dt)
                continue
            # Approximate pickup: if car is within 150 uu of an available pad
            dist = math.hypot(car_pos[0] - pad.pos[0], car_pos[1] - pad.pos[1])
            if dist < 150:
                # Assume if the car passed over it recently, it picked it up
                pad.timer = 10.0 if pad.is_large else 4.0

    # ── Decision ──────────────────────────────────────────────────────────────

    def needs_boost(self) -> bool:
        """Return True when boost is low enough to warrant detouring for a pad."""
        if self._current_boost < _CRITICAL_BOOST:
            return True
        if self._mode == "attack" and self._current_boost < _LOW_BOOST_ATK:
            return True
        if self._mode == "defense" and self._current_boost < _LOW_BOOST_DEF:
            return True
        return False

    def get_target_pad(
        self,
        car_pos: Tuple[float, float],
        prefer_large: bool = False,
    ) -> Optional[Tuple[float, float]]:
        """
        Return the best boost pad position to target, or ``None`` if boost is
        sufficient / no pads are available.

        Parameters
        ----------
        car_pos      : (x, y) current car position.
        prefer_large : if True, prioritise large pads (e.g. when attacking).
        """
        if not self.needs_boost():
            return None

        available = [p for p in self._pads if p.timer <= 0.0]
        if not available:
            return None

        def _score(pad: BoostPadInfo) -> float:
            dist = math.hypot(pad.pos[0] - car_pos[0], pad.pos[1] - car_pos[1])
            bonus = _LARGE_PAD_BONUS if pad.is_large else 0.0
            # Lower score = better (we minimise)
            return dist - (bonus if prefer_large else 0.0)

        best = min(available, key=_score)
        return best.pos

    def get_nearest_large_pad(
        self,
        car_pos: Tuple[float, float],
    ) -> Optional[Tuple[float, float]]:
        """Return the nearest available large boost pad position."""
        large_available = [p for p in self._pads if p.is_large and p.timer <= 0.0]
        if not large_available:
            return None
        nearest = min(
            large_available,
            key=lambda p: math.hypot(p.pos[0] - car_pos[0], p.pos[1] - car_pos[1]),
        )
        return nearest.pos

    def get_nearest_pad_within(
        self,
        car_pos: Tuple[float, float],
        max_dist: float = 1500.0,
    ) -> Optional[Tuple[float, float]]:
        """Return any available pad within `max_dist` units (for opportunistic pickup)."""
        reachable = [
            p for p in self._pads
            if p.timer <= 0.0
            and math.hypot(p.pos[0] - car_pos[0], p.pos[1] - car_pos[1]) <= max_dist
        ]
        if not reachable:
            return None
        return min(
            reachable,
            key=lambda p: math.hypot(p.pos[0] - car_pos[0], p.pos[1] - car_pos[1]),
        ).pos

    def summary(self) -> Dict[str, object]:
        return {
            "current_boost": self._current_boost,
            "mode":          self._mode,
            "needs_boost":   self.needs_boost(),
            "available_pads": sum(1 for p in self._pads if p.timer <= 0.0),
            "total_pads":    len(self._pads),
        }
