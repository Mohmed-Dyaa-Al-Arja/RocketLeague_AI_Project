"""game_logic/match_state_detector.py — Lightweight match-state change detector.

Tracks score and ball position each tick to fire registered callbacks
whenever a meaningful match event (kickoff, goal, replay, match end) occurs.
"""
from __future__ import annotations

import math
from enum import Enum, auto
from typing import Callable, List

# Ball must be within this radius of (0, 0) to count as a kickoff position.
_KICKOFF_RADIUS = 120.0


class MatchState(Enum):
    UNKNOWN       = auto()
    KICKOFF       = auto()
    ACTIVE        = auto()
    GOAL_SCORED   = auto()
    GOAL_CONCEDED = auto()
    REPLAY        = auto()
    MATCH_END     = auto()


class MatchStateDetector:
    """
    Feed it every game tick via ``update(packet, my_team)``; it calls each
    registered callback with the new ``MatchState`` on every transition.
    """

    def __init__(self) -> None:
        self._callbacks: List[Callable[[MatchState], None]] = []
        self._prev_my_score:  int       = 0
        self._prev_opp_score: int       = 0
        self._state:          MatchState = MatchState.UNKNOWN
        self._replay_frames:  int       = 0

    # ── Public API ────────────────────────────────────────────────────────────

    def register_callback(self, fn: Callable[[MatchState], None]) -> None:
        self._callbacks.append(fn)

    def reset_for_new_match(self) -> None:
        self._prev_my_score  = 0
        self._prev_opp_score = 0
        self._state          = MatchState.UNKNOWN
        self._replay_frames  = 0

    def update(self, packet: object, my_team: int) -> MatchState:
        """Call once per game tick.  Returns the current MatchState."""
        try:
            return self._update_inner(packet, my_team)
        except Exception:
            return self._state

    # ── Internal ─────────────────────────────────────────────────────────────

    def _update_inner(self, packet: object, my_team: int) -> MatchState:
        gi = packet.game_info  # type: ignore[attr-defined]

        # ── Match over ───────────────────────────────────────────────────────
        if gi.is_match_ended:
            return self._transition(MatchState.MATCH_END)

        # ── Goal replay ──────────────────────────────────────────────────────
        if not gi.is_round_active and not gi.is_kickoff_pause:
            self._replay_frames += 1
            if self._replay_frames > 5:
                return self._transition(MatchState.REPLAY)
        else:
            self._replay_frames = 0

        # ── Score delta ──────────────────────────────────────────────────────
        scores    = packet.teams  # type: ignore[attr-defined]
        my_score  = scores[my_team].score    if len(scores) > my_team  else 0
        opp_team  = 1 - my_team
        opp_score = scores[opp_team].score   if len(scores) > opp_team else 0

        if my_score > self._prev_my_score:
            self._prev_my_score = my_score
            return self._transition(MatchState.GOAL_SCORED)
        if opp_score > self._prev_opp_score:
            self._prev_opp_score = opp_score
            return self._transition(MatchState.GOAL_CONCEDED)

        # ── Kickoff ──────────────────────────────────────────────────────────
        ball = packet.game_ball  # type: ignore[attr-defined]
        bx, by = ball.physics.location.x, ball.physics.location.y
        if math.hypot(bx, by) <= _KICKOFF_RADIUS and gi.is_kickoff_pause:
            return self._transition(MatchState.KICKOFF)

        # ── Active play ──────────────────────────────────────────────────────
        return self._transition(MatchState.ACTIVE)

    def _transition(self, new_state: MatchState) -> MatchState:
        if new_state != self._state:
            self._state = new_state
            for cb in self._callbacks:
                try:
                    cb(new_state)
                except Exception:
                    pass
        return self._state
