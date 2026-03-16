"""
game_logic/strategy_layer.py  —  Dynamic AI Strategy Switching System.

The StrategyLayer continuously selects the best strategy based on:
  - Current game state (from decision_engine.GameState)
  - Score difference and time remaining
  - Available boost
  - Opponent behavior pattern

Each strategy maps to specific recommended search + RL algorithm combinations.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple


# ── Strategy constants ────────────────────────────────────────────────────────

STRATEGY_AGGRESSIVE   = "Aggressive Attack"
STRATEGY_DEFENSIVE    = "Defensive"
STRATEGY_COUNTER      = "Counter Attack"
STRATEGY_POSSESSION   = "Possession Control"
STRATEGY_DEMO         = "Demo Aggression"
STRATEGY_BOOST        = "Boost Control"
STRATEGY_BALANCED     = "Balanced"

ALL_STRATEGIES = [
    STRATEGY_AGGRESSIVE,
    STRATEGY_DEFENSIVE,
    STRATEGY_COUNTER,
    STRATEGY_POSSESSION,
    STRATEGY_DEMO,
    STRATEGY_BOOST,
    STRATEGY_BALANCED,
]

# ── Strategy → recommended algorithms ────────────────────────────────────────

_STRATEGY_ALGOS: Dict[str, Dict[str, str]] = {
    STRATEGY_AGGRESSIVE: {
        "search": "Greedy",
        "rl":     "dqn",
        "reason": "Fast path to ball + deep Q for tactical shot selection",
    },
    STRATEGY_DEFENSIVE: {
        "search": "UCS",
        "rl":     "actor_critic",
        "reason": "Least-cost safe repositioning + value-guided control",
    },
    STRATEGY_COUNTER: {
        "search": "Beam Search",
        "rl":     "ppo",
        "reason": "Long look-ahead for quick breaks + stable policy gradient",
    },
    STRATEGY_POSSESSION: {
        "search": "A*",
        "rl":     "q_learning",
        "reason": "Optimal path to ball to maintain control + fast table lookup",
    },
    STRATEGY_DEMO: {
        "search": "Greedy",
        "rl":     "model_based",
        "reason": "Sprint at opponent + imagination rollouts for demo timing",
    },
    STRATEGY_BOOST: {
        "search": "BFS",
        "rl":     "online_learner",
        "reason": "Safe pad collection route + live adaptation to pad state",
    },
    STRATEGY_BALANCED: {
        "search": "A*",
        "rl":     "actor_critic",
        "reason": "Balanced optimal path + value-guided control refinement",
    },
}

# ── Strategy descriptions for GUI display ────────────────────────────────────

STRATEGY_DESCRIPTIONS: Dict[str, str] = {
    STRATEGY_AGGRESSIVE: "Prioritises attacking. Bot pushes toward ball and shoots at every opportunity.",
    STRATEGY_DEFENSIVE:  "Bot positions to protect goal, intercepts dangerous balls, rotates safely.",
    STRATEGY_COUNTER:    "Wait for opponent to overcommit, then surge forward with full boost.",
    STRATEGY_POSSESSION: "Keep the ball close, pass when possible, avoid risky challenges.",
    STRATEGY_DEMO:       "Target opponent car for demolitions to gain free ball possession.",
    STRATEGY_BOOST:      "Prioritise boost pad collection when low. Maintain resource advantage.",
    STRATEGY_BALANCED:   "Equal weight on attack and defense. Adapts reactively each tick.",
}


class StrategyLayer:
    """Selects and maintains the current play strategy.

    Call adapt() once per decision tick with current game state and call
    get_recommended_algorithms() to retrieve the best algo pair.
    """

    def __init__(self):
        self.current_strategy: str = STRATEGY_BALANCED
        self._last_strategy_change: int = 0
        self._tick: int = 0
        self._strategy_history: list = []   # (tick, strategy) pairs

        # Hysteresis: don't switch strategy every single tick
        self._switch_cooldown: int = 30     # minimum ticks between switches

    def adapt(
        self,
        game_state: str,           # one of the 30 GameState names
        score_diff: int,           # positive = we are ahead, negative = trailing
        time_remaining: float,     # seconds; 0 = overtime
        boost: float,              # our current boost amount (0-100)
        dist_to_ball: float,       # car-to-ball distance
        ball_in_own_half: bool,
        ball_in_opp_half: bool,
        opp_behavior: str = "",    # e.g. "rush_ball", "shadow", "aerial"
    ) -> str:
        """Evaluate situation and update self.current_strategy. Returns new strategy."""
        self._tick += 1

        candidate = self._evaluate_strategy(
            game_state, score_diff, time_remaining, boost,
            dist_to_ball, ball_in_own_half, ball_in_opp_half, opp_behavior,
        )

        # Apply hysteresis
        ticks_since_change = self._tick - self._last_strategy_change
        if candidate != self.current_strategy and ticks_since_change >= self._switch_cooldown:
            self._strategy_history.append((self._tick, self.current_strategy))
            if len(self._strategy_history) > 200:
                self._strategy_history = self._strategy_history[-200:]
            self.current_strategy = candidate
            self._last_strategy_change = self._tick

        return self.current_strategy

    def get_recommended_algorithms(
        self, strategy: Optional[str] = None
    ) -> Dict[str, str]:
        """Return {'search': ..., 'rl': ..., 'reason': ...} for the given strategy."""
        s = strategy if strategy is not None else self.current_strategy
        return _STRATEGY_ALGOS.get(s, _STRATEGY_ALGOS[STRATEGY_BALANCED])

    def get_all_strategies(self) -> Dict[str, Dict[str, str]]:
        """Return all strategy → algo mappings (used by GUI)."""
        return dict(_STRATEGY_ALGOS)

    def get_strategy_description(self, strategy: Optional[str] = None) -> str:
        s = strategy if strategy is not None else self.current_strategy
        return STRATEGY_DESCRIPTIONS.get(s, "")

    # ── Internal logic ────────────────────────────────────────────────────────

    def _evaluate_strategy(
        self,
        game_state: str,
        score_diff: int,
        time_remaining: float,
        boost: float,
        dist_to_ball: float,
        ball_in_own_half: bool,
        ball_in_opp_half: bool,
        opp_behavior: str,
    ) -> str:
        # Critical low-boost: collect first
        if boost < 20 and dist_to_ball > 1500:
            return STRATEGY_BOOST

        # Ball in our half and threatening → defend
        if ball_in_own_half and game_state in (
            "GoalDefense", "EmergencyDefense", "LastDefender",
            "ShadowDefense", "Defense", "ClearBall",
        ):
            return STRATEGY_DEFENSIVE

        # Urgent trailing + little time → aggressive
        if score_diff < 0 and 0 < time_remaining < 90:
            return STRATEGY_AGGRESSIVE

        # We're comfortably ahead with lots of time → possession
        if score_diff >= 2 and time_remaining > 120:
            return STRATEGY_POSSESSION

        # Opponent has been demoed → counter attack opportunity
        if game_state in ("FastBreak", "OpenGoal", "CounterAttack"):
            return STRATEGY_COUNTER

        # Demo opportunity identified
        if game_state == "DemoOpportunity" and boost > 50:
            return STRATEGY_DEMO

        # Kickoff or ball in midfield → balanced
        if game_state in ("Kickoff", "MidfieldControl", "BallControl", "StrategicPause"):
            return STRATEGY_BALANCED

        # Ball in opp half and we're in position → attack
        if ball_in_opp_half and game_state in (
            "Attack", "ShotOpportunity", "GoalAttack", "AerialAttack",
        ):
            return STRATEGY_AGGRESSIVE

        # Opponent pressing hard → defensive
        if opp_behavior in ("rush_ball", "aerial"):
            return STRATEGY_DEFENSIVE

        return STRATEGY_BALANCED

    def state_summary(self) -> Dict[str, str]:
        """Return a compact status dict for IPC / GUI display."""
        algos = self.get_recommended_algorithms()
        return {
            "strategy":      self.current_strategy,
            "search_advice": algos["search"],
            "rl_advice":     algos["rl"],
            "reason":        algos["reason"],
        }
