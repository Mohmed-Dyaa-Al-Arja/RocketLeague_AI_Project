"""
strategy/strategy_selector.py
==============================
Selects optimal strategies before each match and adapts them
dynamically during play based on score, possession, and opponent style.

Usage
-----
    selector = StrategySelector(strategy_book)
    # Before match:
    plan = selector.select_pre_match("DFHStadium")
    # In game loop (every ~60 ticks):
    plan = selector.adapt_in_match(arena, score_diff, possession_pct,
                                   opponent_style, is_kickoff, game_state, tick)
    # At match end:
    selector.record_outcome("DFHStadium", goals_scored, goals_conceded)

Author: medo dyaa
"""

from __future__ import annotations

from typing import Dict, Optional

from strategy.strategy_book import StrategyBook, STRATEGY_TYPES, STRATEGY_OPTIONS


# ---------------------------------------------------------------------------
# Trigger thresholds
# ---------------------------------------------------------------------------

_LOSING_THRESHOLD          = -2     # score diff ≤ this → escalate attack
_WINNING_THRESHOLD         = 2      # score diff ≥ this → protect lead
_HIGH_POSSESSION_THRESHOLD = 0.60   # possession fraction ≥ this + not scoring → pass more
_LOW_POSSESSION_THRESHOLD  = 0.35   # possession fraction ≤ this → focus defence
_SWITCH_COOLDOWN           = 300    # minimum ticks between strategy switches


class StrategySelector:
    """
    Pre-match strategy selector and in-match adaptation engine.

    Author: medo dyaa
    """

    def __init__(self, strategy_book: StrategyBook) -> None:
        self._book                   = strategy_book
        self._current_arena: str     = "DFHStadium"
        self._active: Dict[str, str] = {}
        self._switch_timer: int      = 0
        self._last_switch_reason: str = ""

    # ------------------------------------------------------------------
    # Pre-match
    # ------------------------------------------------------------------

    def select_pre_match(self, arena: str) -> Dict[str, str]:
        """
        Called once before each match starts.

        Promotes the best-known strategies for this arena and returns the
        full strategy selection dict.

        Author: medo dyaa
        """
        self._current_arena = arena
        # Let the book elect the best strategy per type
        self._book.promote_best_strategies(arena)
        self._active = {
            stype: self._book.get_active_strategy(arena, stype)
            for stype in STRATEGY_TYPES
        }
        self._last_switch_reason = "pre-match selection"
        return dict(self._active)

    # ------------------------------------------------------------------
    # In-match adaptation
    # ------------------------------------------------------------------

    def adapt_in_match(
        self,
        arena:          str,
        score_diff:     int,        # positive = we are ahead
        possession_pct: float,      # 0.0–1.0 our possession fraction
        opponent_style: str,        # "aggressive" | "passive" | "unknown"
        is_kickoff:     bool,
        game_state:     str,        # from game_logic.game_awareness.GameAwareness
        tick:           int,
    ) -> Dict[str, str]:
        """
        Evaluate current match conditions and switch strategies when needed.

        Returns the (possibly updated) active strategy dict.

        Author: medo dyaa
        """
        if not self._book.adaptive_enabled:
            return dict(self._active)

        self._current_arena = arena
        # Respect cooldown to prevent thrashing
        self._switch_timer -= 1
        if self._switch_timer > 0:
            return dict(self._active)

        changed = False

        # ── Losing badly: escalate attack ────────────────────────────────
        if score_diff <= _LOSING_THRESHOLD:
            new_atk = self._pick_aggressive_attack()
            if new_atk != self._active.get("attack"):
                self._apply("attack", arena, new_atk)
                self._last_switch_reason = "losing; escalating attack"
                changed = True

        # ── Winning: protect the lead ────────────────────────────────────
        elif score_diff >= _WINNING_THRESHOLD:
            new_def = self._pick_defensive_strategy()
            if new_def != self._active.get("defense"):
                self._apply("defense", arena, new_def)
                self._last_switch_reason = "leading; protecting the lead"
                changed = True

        # ── Opponent very aggressive: deepen defense ─────────────────────
        if (
            opponent_style == "aggressive"
            and game_state in ("Defense", "Goal Defense")
        ):
            new_def = "deep_rotation"
            if new_def != self._active.get("defense"):
                self._apply("defense", arena, new_def)
                self._last_switch_reason = "opponent aggressive; deep defense"
                changed = True

        # ── High possession but not converting: use passing play ─────────
        if (
            possession_pct > _HIGH_POSSESSION_THRESHOLD
            and score_diff <= 0
            and not is_kickoff
        ):
            new_pass = self._pick_passing_strategy()
            if new_pass != self._active.get("passing"):
                self._apply("passing", arena, new_pass)
                self._last_switch_reason = "high possession; switching to passing"
                changed = True

        # ── Low possession: fall back to safe/shadow defence ────────────
        if possession_pct < _LOW_POSSESSION_THRESHOLD:
            new_def = "shadow_defense"
            if new_def != self._active.get("defense"):
                self._apply("defense", arena, new_def)
                self._last_switch_reason = "low possession; shadow defense"
                changed = True

        # ── Counter-attack: direct shot is fastest ───────────────────────
        if game_state == "Counter Attack":
            if self._active.get("attack") != "direct_shot":
                self._apply("attack", arena, "direct_shot")
                self._last_switch_reason = "counter attack; direct shot"
                changed = True

        if changed:
            self._switch_timer = _SWITCH_COOLDOWN

        return dict(self._active)

    # ------------------------------------------------------------------
    # Match-end recording
    # ------------------------------------------------------------------

    def record_outcome(
        self,
        arena: str,
        goals_scored: int,
        goals_conceded: int,
    ) -> None:
        """
        Record the match result against every active strategy for this arena.

        Author: medo dyaa
        """
        success = goals_scored > goals_conceded
        for stype, name in self._active.items():
            self._book.record_outcome(arena, stype, name, success)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def active_strategies(self) -> Dict[str, str]:
        return dict(self._active)

    @property
    def current_arena(self) -> str:
        return self._current_arena

    @property
    def last_switch_reason(self) -> str:
        return self._last_switch_reason

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply(self, stype: str, arena: str, name: str) -> None:
        self._active[stype] = name
        self._book.set_active_strategy(arena, stype, name)

    def _pick_aggressive_attack(self) -> str:
        opts = STRATEGY_OPTIONS["attack"]
        for preferred in ("wall_pass_attack", "corner_attack"):
            if preferred in opts:
                return preferred
        return opts[0]

    def _pick_defensive_strategy(self) -> str:
        opts = STRATEGY_OPTIONS["defense"]
        for preferred in ("midfield_press", "shadow_defense"):
            if preferred in opts:
                return preferred
        return opts[0]

    def _pick_passing_strategy(self) -> str:
        opts = STRATEGY_OPTIONS["passing"]
        for preferred in ("quick_wall_pass", "centre_pass"):
            if preferred in opts:
                return preferred
        return opts[0]
