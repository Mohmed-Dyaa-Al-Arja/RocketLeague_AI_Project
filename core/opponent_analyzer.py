"""
core/opponent_analyzer.py
=========================
Tracks and analyses opponent behaviour to allow the bot to adapt its strategy.

Metrics tracked
---------------
- Movement patterns (aggression, average attack speed)
- Shot frequency and average shot distance
- Goal-defence frequency (how often goalkeeper stays in goal area)
- Aggression level (ratio of offensive vs defensive ticks)

The `OpponentAnalyzer.get_style()` method returns a human-readable label
("aggressive", "defensive", "possessive", "unknown") that the strategy layer
can use to switch tactics.

Author: medo dyaa
"""

from __future__ import annotations

import math
from collections import deque
from typing import Deque, Dict, Optional, Tuple


class OpponentAnalyzer:
    """
    Accumulate per-tick observations about the opponent and expose derived
    behaviour metrics.

    Usage
    -----
    Create once per match, then call `update()` every tick with current
    opponent data.  Query `get_style()` to get a strategy recommendation.
    """

    _WINDOW = 300           # rolling window: last N ticks (~5 seconds at 60 Hz)
    _GOAL_ZONE_Y = 4000.0   # abs(y) threshold — "near their goal"
    _ATTACK_SPEED_THRESH = 800.0   # speed considered "charging"

    def __init__(self) -> None:
        self._tick: int = 0

        # Positions in the last N ticks
        self._opp_y_hist: Deque[float] = deque(maxlen=self._WINDOW)
        # Speed samples
        self._speed_hist: Deque[float] = deque(maxlen=self._WINDOW)
        # Fraction of ticks opponent was in attack half
        self._attack_ticks: int = 0
        self._defend_ticks: int = 0
        # Shot events (appended when a shot is detected)
        self._shot_distances: Deque[float] = deque(maxlen=50)
        self._shot_count: int = 0
        # Demo/bump counter
        self._demo_count: int = 0
        # Ball-possession ticks (opponent near ball)
        self._poss_ticks: int = 0
        # Cumulative ticks
        self._total_ticks: int = 0

    # ── Per-tick update ───────────────────────────────────────────────────────

    def update(
        self,
        opp_pos: Tuple[float, float],
        opp_vel: Tuple[float, float],
        ball_pos: Tuple[float, float],
        push_dir: float,
        opp_demolished: bool = False,
    ) -> None:
        """
        Call every game tick with the latest opponent state.

        Parameters
        ----------
        opp_pos      : (x, y) opponent position.
        opp_vel      : (vx, vy) opponent velocity.
        ball_pos     : (x, y) ball position.
        push_dir     : +1 if we are team-0 (attack toward +Y), -1 otherwise.
        opp_demolished: True if the opponent was just demolished this tick.
        """
        self._tick += 1
        self._total_ticks += 1

        ox, oy = opp_pos
        ovx, ovy = opp_vel
        speed = math.hypot(ovx, ovy)

        self._opp_y_hist.append(oy)
        self._speed_hist.append(speed)

        # Attacker: opponent is on our half (past midfield toward our goal)
        # i.e. oy * push_dir < 0  →  opponent is advancing toward our goal
        if oy * push_dir < -500:
            self._attack_ticks += 1
        elif oy * push_dir > 500:
            self._defend_ticks += 1

        # Possession: opp is close to the ball
        if math.hypot(ox - ball_pos[0], oy - ball_pos[1]) < 400:
            self._poss_ticks += 1

        if opp_demolished:
            self._demo_count += 1

    def record_shot(self, opp_pos: Tuple[float, float], opp_goal_pos: Tuple[float, float]) -> None:
        """Call when an opponent shot is detected (e.g. ball velocity spiked toward our goal)."""
        dist = math.hypot(opp_pos[0] - opp_goal_pos[0], opp_pos[1] - opp_goal_pos[1])
        self._shot_distances.append(dist)
        self._shot_count += 1

    # ── Derived metrics ───────────────────────────────────────────────────────

    @property
    def aggression_ratio(self) -> float:
        """Value in [0, 1]; higher means more attacking ticks."""
        total = self._attack_ticks + self._defend_ticks
        if total == 0:
            return 0.5
        return self._attack_ticks / total

    @property
    def average_attack_speed(self) -> float:
        """Average opponent speed over the recent rolling window."""
        if not self._speed_hist:
            return 0.0
        return sum(self._speed_hist) / len(self._speed_hist)

    @property
    def average_shot_distance(self) -> float:
        """Average distance from which opponent shoots."""
        if not self._shot_distances:
            return 3000.0
        return sum(self._shot_distances) / len(self._shot_distances)

    @property
    def goal_defence_frequency(self) -> float:
        """Fraction of ticks opponent spent near their own goal [0, 1]."""
        if self._total_ticks == 0:
            return 0.0
        return self._defend_ticks / self._total_ticks

    @property
    def possession_rate(self) -> float:
        """Fraction of ticks opponent had ball possession [0, 1]."""
        if self._total_ticks == 0:
            return 0.0
        return self._poss_ticks / self._total_ticks

    # ── Style classification ───────────────────────────────────────────────────

    def get_style(self) -> str:
        """
        Return a behaviour label for the opponent:
        - ``"aggressive"``  : high aggression ratio, fast attack
        - ``"defensive"``   : spends most time near their goal
        - ``"possessive"``  : high ball-possession rate
        - ``"balanced"``    : mixed play
        - ``"unknown"``     : not enough data yet
        """
        if self._total_ticks < 120:   # need at least 2 seconds
            return "unknown"

        if self.aggression_ratio > 0.65 and self.average_attack_speed > self._ATTACK_SPEED_THRESH:
            return "aggressive"
        if self.goal_defence_frequency > 0.50:
            return "defensive"
        if self.possession_rate > 0.35:
            return "possessive"
        return "balanced"

    def get_counter_strategy(self) -> str:
        """
        Map opponent style to recommended bot strategy.

        Returns
        -------
        One of: "defensive", "aggressive_attack", "possession", "counter", "balanced"
        """
        style = self.get_style()
        return {
            "aggressive":  "defensive",
            "defensive":   "aggressive_attack",
            "possessive":  "counter",
            "balanced":    "balanced",
            "unknown":     "balanced",
        }.get(style, "balanced")

    def summary(self) -> Dict[str, object]:
        """Return a dict with all key metrics for debug display."""
        return {
            "style":              self.get_style(),
            "counter_strategy":   self.get_counter_strategy(),
            "aggression_ratio":   round(self.aggression_ratio, 3),
            "avg_attack_speed":   round(self.average_attack_speed, 1),
            "avg_shot_distance":  round(self.average_shot_distance, 1),
            "defence_frequency":  round(self.goal_defence_frequency, 3),
            "possession_rate":    round(self.possession_rate, 3),
            "shot_count":         self._shot_count,
            "demo_count":         self._demo_count,
            "ticks_observed":     self._total_ticks,
        }

    def reset(self) -> None:
        """Clear all accumulated data (call at match start)."""
        self.__init__()

    # ── Weakness detection ────────────────────────────────────────────────────

    def detect_weaknesses(self) -> Dict[str, bool]:
        """
        Analyse opponent behaviour and return a dict of detected weaknesses.

        Keys
        ----
        slow_defense      : Gets caught out of position defending (low speed + low defence).
        poor_boost        : Frequently attacks from long range → likely low boost.
        over_aggressive   : High aggression but concedes counter-attacks easily.
        predictable_shots : Shoots from similar distances every time (low variance).
        passive_midfield  : Rarely crosses midfield (low aggression ratio).
        """
        if self._total_ticks < 180:
            return {}

        weaknesses: Dict[str, bool] = {}

        # Slow defense: defends a lot but at low speed when in own half
        avg_speed = self.average_attack_speed
        weaknesses["slow_defense"] = (
            self.goal_defence_frequency > 0.40 and avg_speed < 600.0
        )

        # Poor boost: shoots from far away (likely not boosting toward goal)
        weaknesses["poor_boost"] = self.average_shot_distance > 3500.0

        # Over-aggressive: very high aggression ratio
        weaknesses["over_aggressive"] = self.aggression_ratio > 0.75

        # Predictable shots: low variance in shot distances
        if len(self._shot_distances) >= 5:
            mean = self.average_shot_distance
            variance = sum((d - mean) ** 2 for d in self._shot_distances) / len(self._shot_distances)
            weaknesses["predictable_shots"] = variance < 200_000.0   # std-dev < ~450
        else:
            weaknesses["predictable_shots"] = False

        # Passive midfield: barely ventures out of own half
        weaknesses["passive_midfield"] = self.aggression_ratio < 0.25

        return weaknesses

    def get_adaptive_difficulty_adjustment(self) -> Dict[str, float]:
        """
        Return multipliers that the bot can apply to its own behaviour to exploit
        detected opponent weaknesses.

        Returns
        -------
        dict with keys: ``throttle``, ``boost``, ``aggression``
        All values default to 1.0 (no change).
        """
        weaknesses = self.detect_weaknesses()
        throttle = 1.0
        boost = 1.0
        aggression = 1.0

        if weaknesses.get("slow_defense"):
            # Rush them — they recover slowly
            throttle   *= 1.10
            boost      *= 1.15
            aggression *= 1.20

        if weaknesses.get("poor_boost"):
            # Commit more — they can't contest aerials
            boost      *= 1.10
            aggression *= 1.10

        if weaknesses.get("over_aggressive"):
            # Counter-attack — let them over-commit, then strike
            aggression *= 0.85
            throttle   *= 1.05

        if weaknesses.get("passive_midfield"):
            # Apply constant forward pressure
            aggression *= 1.15
            throttle   *= 1.05

        return {
            "throttle":   round(min(throttle,   1.40), 3),
            "boost":      round(min(boost,       1.40), 3),
            "aggression": round(min(aggression,  1.40), 3),
        }
