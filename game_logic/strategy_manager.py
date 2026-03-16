"""
game_logic/strategy_manager.py
===============================
Dynamic strategy switching manager for the Rocket League AI bot.

Strategies
----------
ATTACK        — push constantly toward opponent goal
DEFENSE       — protect our goal
COUNTER       — wait for opponent mistake, then counter-attack
GOALKEEPER    — stay near our goal and block shots
POSSESSION    — maintain ball control, slow down tempo
DEMO          — prioritise demolishing opponents
BALANCED      — mix of attack and defense

Switching conditions
--------------------
- ball position (which half is it in?)
- score difference (winning / losing / tied)
- time remaining (urgency)
- opponent behaviour (aggressive / defensive / possessive)
- boost availability (low boost → avoid risky plays)

Author: medo dyaa
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple


# ── Strategy constants ────────────────────────────────────────────────────────
class Strategy:
    ATTACK      = "attack"
    DEFENSE     = "defense"
    COUNTER     = "counter"
    GOALKEEPER  = "goalkeeper"
    POSSESSION  = "possession"
    DEMO        = "demo"
    BALANCED    = "balanced"


# Recommended algorithms per strategy
_STRATEGY_ALGOS: Dict[str, Dict[str, List[str]]] = {
    Strategy.ATTACK: {
        "search": ["Greedy", "Beam Search", "A*"],
        "rl":     ["dqn", "ppo"],
    },
    Strategy.DEFENSE: {
        "search": ["UCS", "A*"],
        "rl":     ["actor_critic", "ppo"],
    },
    Strategy.COUNTER: {
        "search": ["Beam Search", "A*"],
        "rl":     ["ppo", "a2c"],
    },
    Strategy.GOALKEEPER: {
        "search": ["UCS", "A*"],
        "rl":     ["q_learning", "actor_critic"],
    },
    Strategy.POSSESSION: {
        "search": ["A*", "BFS"],
        "rl":     ["q_learning", "online_learner"],
    },
    Strategy.DEMO: {
        "search": ["Greedy", "A*"],
        "rl":     ["dqn", "monte_carlo"],
    },
    Strategy.BALANCED: {
        "search": ["A*", "Beam Search", "Greedy"],
        "rl":     ["q_learning", "actor_critic", "dqn"],
    },
}


class StrategyManager:
    """
    Evaluates match context each tick and returns the optimal strategy.

    Usage
    -----
    ::

        manager = StrategyManager()
        strategy = manager.decide(
            ball_pos=(100, 2000),
            push_dir=1.0,
            score_diff=-1,
            time_remaining=55.0,
            our_boost=18.0,
            opp_style="aggressive",
        )
        print(strategy)  # e.g. "goalkeeper"
    """

    def __init__(self) -> None:
        self._current: str = Strategy.BALANCED
        self._prev: str = Strategy.BALANCED
        self._switch_history: List[Tuple[str, str]] = []   # (from, to)
        self._ticks_in_state: int = 0
        # Hysteresis: don't switch strategies too quickly
        self._min_hold_ticks: int = 60   # ~1 second

    # ── Main decision logic ───────────────────────────────────────────────────

    def decide(
        self,
        ball_pos: Tuple[float, float],
        push_dir: float,
        score_diff: int,
        time_remaining: float,
        our_boost: float,
        opp_style: str = "unknown",
        ball_dist_from_our_goal: float = 5000.0,
        has_teammates: bool = False,
    ) -> str:
        """
        Evaluate context and return the recommended strategy name.

        Parameters
        ----------
        ball_pos                 : (x, y) ball position.
        push_dir                 : +1 for team-0 (attack +Y), -1 for team-1.
        score_diff               : our_score - opp_score.
        time_remaining           : seconds left in match.
        our_boost                : current boost level [0, 100].
        opp_style                : string from OpponentAnalyzer.get_style().
        ball_dist_from_our_goal  : pre-computed distance ball → our goal.
        has_teammates            : True in 2v2 / 3v3.
        """
        self._ticks_in_state += 1

        candidate = self._evaluate(
            ball_pos, push_dir, score_diff, time_remaining,
            our_boost, opp_style, ball_dist_from_our_goal, has_teammates,
        )

        # Apply hysteresis — don't switch before min hold time
        if candidate != self._current and self._ticks_in_state < self._min_hold_ticks:
            return self._current

        if candidate != self._current:
            self._switch_history.append((self._current, candidate))
            self._prev = self._current
            self._current = candidate
            self._ticks_in_state = 0

        return self._current

    def _evaluate(
        self,
        ball_pos: Tuple[float, float],
        push_dir: float,
        score_diff: int,
        time_remaining: float,
        our_boost: float,
        opp_style: str,
        ball_dist_from_our_goal: float,
        has_teammates: bool,
    ) -> str:
        _, by = ball_pos

        # Ball is in our half (defensive)
        ball_in_our_half = by * push_dir < -500

        # ── Critical deficit + time running out → all-in attack ──────────
        if score_diff < -1 and time_remaining < 60:
            return Strategy.ATTACK

        # ── We're winning comfortably → sit back and hold ─────────────────
        if score_diff >= 2 and time_remaining > 60:
            return Strategy.DEFENSE

        # ── Ball very close to our goal → goalkeeper mode ─────────────────
        if ball_dist_from_our_goal < 1500 or (ball_in_our_half and ball_dist_from_our_goal < 2500):
            return Strategy.GOALKEEPER

        # ── Low boost → be conservative (possession or defense) ──────────
        if our_boost < 20:
            return Strategy.POSSESSION if not ball_in_our_half else Strategy.DEFENSE

        # ── Opponent is aggressive → play counter ─────────────────────────
        if opp_style == "aggressive":
            return Strategy.COUNTER

        # ── Opponent hides in goal → demo + long shots ────────────────────
        if opp_style == "defensive":
            return Strategy.DEMO

        # ── Opponent possesses a lot → pressure to win ball ───────────────
        if opp_style == "possessive":
            return Strategy.ATTACK

        # ── Ball in their half → attack ───────────────────────────────────
        if by * push_dir > 1000:
            return Strategy.ATTACK

        # ── Ball in our half → defend ─────────────────────────────────────
        if ball_in_our_half:
            return Strategy.DEFENSE

        return Strategy.BALANCED

    # ── Accessors ─────────────────────────────────────────────────────────────

    @property
    def current(self) -> str:
        return self._current

    def get_recommended_algorithms(self, strategy: Optional[str] = None) -> Dict[str, List[str]]:
        """Return search + RL algorithm lists for the given (or current) strategy."""
        s = strategy or self._current
        return _STRATEGY_ALGOS.get(s, _STRATEGY_ALGOS[Strategy.BALANCED])

    def force(self, strategy: str) -> None:
        """Override the strategy (bypasses hysteresis)."""
        if strategy in _STRATEGY_ALGOS:
            self._prev = self._current
            self._current = strategy
            self._ticks_in_state = 0

    def summary(self) -> Dict[str, object]:
        return {
            "current":        self._current,
            "previous":       self._prev,
            "ticks_in_state": self._ticks_in_state,
            "switch_count":   len(self._switch_history),
            "algorithms":     self.get_recommended_algorithms(),
        }
