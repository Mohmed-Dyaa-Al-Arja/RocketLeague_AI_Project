"""
game_logic/match_strategy_analyzer.py
======================================
Analyses ongoing match performance and adjusts tactics dynamically.

Tracks
------
- Goal opportunities (near-goal possession ticks)
- Missed shots (we had clear shot but no goal followed in next N ticks)
- Opponent attack patterns
- Cumulative team performance (goals, saves, conceded)

Outputs
-------
- `get_recommended_adjustment()` → dict with tactic hints consumed by
  StrategyManager or DecisionEngine.
- `summary()` → dict for GUI display.

Author: medo dyaa
"""

from __future__ import annotations

import math
from collections import deque
from typing import Deque, Dict, Optional, Tuple


class MatchStrategyAnalyzer:
    """
    Real-time match performance analyzer.

    Call `update()` every tick.  Call `notify_goal()`, `notify_conceded()`,
    `notify_save()` on those events.

    Author: medo dyaa
    """

    _WINDOW = 600          # rolling window (10 s at 60 Hz)
    _SHOT_TRAIL = 90       # ticks after which a missed shot is confirmed
    _OPP_GOAL_Y = 5120.0  # default; overridden by caller if needed

    def __init__(self) -> None:
        # Performance counters
        self._goals_scored:   int = 0
        self._goals_conceded: int = 0
        self._saves:          int = 0

        # Rolling history
        self._our_near_goal_ticks: Deque[int] = deque(maxlen=self._WINDOW)
        self._opp_near_goal_ticks: Deque[int] = deque(maxlen=self._WINDOW)

        # Shot opportunity tracking
        self._pending_shots: Deque[Tuple[int, Tuple[float,float,float]]] = deque(maxlen=20)
        self._missed_shots:  int = 0

        # Game tick counter
        self._tick: int = 0

        # Tactic hints
        self._current_hint: Dict[str, str] = {}

    # ── Event hooks ──────────────────────────────────────────────────────────

    def notify_goal(self) -> None:
        self._goals_scored += 1
        self._pending_shots.clear()
        self._current_hint["momentum"] = "positive"

    def notify_conceded(self) -> None:
        self._goals_conceded += 1
        self._current_hint["momentum"] = "negative"

    def notify_save(self) -> None:
        self._saves += 1

    # ── Per-tick update ───────────────────────────────────────────────────────

    def update(
        self,
        ball_pos: Tuple[float, float, float],
        car_pos: Tuple[float, float, float],
        opp_pos: Tuple[float, float, float],
        opp_goal_y: float = 5120.0,
        own_goal_y: float = -5120.0,
        we_have_ball: bool = False,
    ) -> None:
        """Call every game tick."""
        self._tick += 1

        bx, by, bz = ball_pos
        dist_ball_to_opp_goal = abs(opp_goal_y - by)
        dist_ball_to_own_goal = abs(own_goal_y - by)

        # Near-goal presence
        self._our_near_goal_ticks.append(1 if dist_ball_to_opp_goal < 2500 and we_have_ball else 0)
        self._opp_near_goal_ticks.append(1 if dist_ball_to_own_goal < 2500 and not we_have_ball else 0)

        # Shot opportunity: if we're in opp half with ball and no goal follows
        if we_have_ball and dist_ball_to_opp_goal < 3000:
            self._pending_shots.append((self._tick, ball_pos))

        # Expire un-converted shots as missed
        expired = []
        for entry in list(self._pending_shots):
            shot_tick, shot_pos = entry
            if self._tick - shot_tick > self._SHOT_TRAIL:
                self._missed_shots += 1
                expired.append(entry)
        for e in expired:
            try:
                self._pending_shots.remove(e)
            except ValueError:
                pass

        self._update_hints()

    # ── Analysis ─────────────────────────────────────────────────────────────

    def _update_hints(self) -> None:
        hints: Dict[str, str] = {}

        # Opportunity rate
        near_goal_frac = (
            sum(self._our_near_goal_ticks) / max(1, len(self._our_near_goal_ticks))
        )
        if near_goal_frac > 0.3:
            hints["attack_pressure"] = "high"
        elif near_goal_frac > 0.1:
            hints["attack_pressure"] = "medium"
        else:
            hints["attack_pressure"] = "low"

        # Defensive pressure
        opp_near_frac = (
            sum(self._opp_near_goal_ticks) / max(1, len(self._opp_near_goal_ticks))
        )
        if opp_near_frac > 0.25:
            hints["defense_pressure"] = "high"
        else:
            hints["defense_pressure"] = "low"

        # Shot conversion feedback
        if self._missed_shots > 5 and self._goals_scored == 0:
            hints["shot_quality"] = "poor"
        elif self._goals_scored > self._missed_shots:
            hints["shot_quality"] = "good"
        else:
            hints["shot_quality"] = "average"

        self._current_hint.update(hints)

    def get_recommended_adjustment(self) -> Dict[str, str]:
        """
        Return a dict of tactic hints for the strategy manager.

        Keys: ``attack_pressure``, ``defense_pressure``, ``shot_quality``,
        ``momentum`` (optional).
        """
        return dict(self._current_hint)

    def summary(self) -> Dict:
        return {
            "goals_scored":   self._goals_scored,
            "goals_conceded": self._goals_conceded,
            "saves":          self._saves,
            "missed_shots":   self._missed_shots,
            "hints":          dict(self._current_hint),
        }
