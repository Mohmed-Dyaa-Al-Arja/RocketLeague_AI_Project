"""
ability_strategy.py
===================
Strategic decision engine for using discovered/known abilities at the
right moment.  Consults AbilityDiscovery knowledge and game context to
decide *when* and *how* to activate powerups.

Usage
-----
strategy = AbilityStrategy()
should_use = strategy.evaluate(
    ability_name, game_state, ball_pos, car_pos, opp_pos, score_diff, tick)
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

from game_logic.ability_discovery import AbilityDiscovery, EFFECT_CLASSES, _DEFAULT_ENTRY

# Cooldown (ticks) between strategic activations of the same ability
_ACTIVATION_COOLDOWN = 90   # ~1.5 s at 60 Hz


class AbilityStrategy:
    """
    Decides when to activate a powerup based on:
    - learned effect type (ball_pull, spike_control, …)
    - current game context (score, game state, positions)
    - per-ability cooldown tracking
    """

    def __init__(self, discovery: Optional[AbilityDiscovery] = None):
        self._discovery: AbilityDiscovery = discovery or AbilityDiscovery()
        self._last_activation_tick: Dict[str, int] = {}
        self.last_decision_reason: str = ""

    # ── Public API ────────────────────────────────────────────────────────────

    def evaluate(
        self,
        ability_name: str,
        game_state: str,
        ball_pos: Tuple[float, float],
        car_pos: Tuple[float, float],
        opp_pos: Tuple[float, float],
        score_diff: int,
        tick: int,
    ) -> bool:
        """
        Returns True if the ability should be activated this tick.

        Parameters
        ----------
        ability_name : canonical ability name (from RUMBLE_POWERUP_NAMES)
        game_state   : string from GameAwareness ("Attack", "Defense", …)
        ball_pos     : (x, y) ball world position
        car_pos      : (x, y) bot world position
        opp_pos      : (x, y) nearest opponent world position
        score_diff   : my_goals − opp_goals
        tick         : current game tick (for cooldown)
        """
        if not ability_name:
            return False

        # Respect cooldown
        last = self._last_activation_tick.get(ability_name, -9999)
        if tick - last < _ACTIVATION_COOLDOWN:
            return False

        entry = self._discovery.get_ability_info(ability_name)
        effect = entry.get("type", "unknown")
        use_case = entry.get("best_use_case", "general")

        dist_to_ball = math.hypot(ball_pos[0] - car_pos[0], ball_pos[1] - car_pos[1])
        dist_to_opp  = math.hypot(opp_pos[0] - car_pos[0], opp_pos[1] - car_pos[1])
        ball_near_opp_goal = abs(ball_pos[1]) > 3800
        ball_near_own_goal = abs(ball_pos[1]) > 3800 and (
            (car_pos[1] > 0 and ball_pos[1] > 0) or
            (car_pos[1] < 0 and ball_pos[1] < 0)
        )

        should = False
        reason = ""

        # ── Effect-specific rules ─────────────────────────────────────────
        if effect == "ball_pull":
            if ball_near_opp_goal and dist_to_ball < 1000:
                should = True
                reason = "ball_pull: ball near opp goal → pull to score"
            elif dist_to_ball < 500:
                should = True
                reason = "ball_pull: close range pull"

        elif effect == "ball_push":
            if game_state in ("Attack", "Counter Attack") and dist_to_ball < 400:
                should = True
                reason = "ball_push: attacking — launch ball toward goal"
            elif ball_near_own_goal and dist_to_ball < 600:
                should = True
                reason = "ball_push: defensive clear"

        elif effect == "spike_control":
            if dist_to_ball < 200:
                should = True
                reason = "spike_control: attach and carry ball to goal"

        elif effect == "freeze":
            if game_state in ("Defense", "Goal Defense") and dist_to_ball < 700:
                should = True
                reason = "freeze: defensive stall"
            elif score_diff < 0 and dist_to_opp < 500:
                should = True
                reason = "freeze: stall dangerous opponent"

        elif effect == "shock_push":
            if dist_to_opp < 350:
                should = True
                reason = "shock_push: displace close opponent"
            elif game_state in ("Attack",) and dist_to_opp < 600:
                should = True
                reason = "shock_push: clear defender"

        elif effect == "grapple":
            if dist_to_ball > 1500:
                should = True
                reason = "grapple: close long range gap quickly"
            elif game_state in ("Counter Attack",) and dist_to_ball > 800:
                should = True
                reason = "grapple: counter-attack intercept"

        elif effect == "boost":
            # Disruptor-style — deny opponent boost when they are boosting
            if dist_to_opp < 700 and score_diff <= 0:
                should = True
                reason = "boost_denial: disrupt nearby opponent"

        else:
            # Unknown / general: opportunistic near-ball activation
            if dist_to_ball < 400:
                should = True
                reason = "unknown ability: near-ball opportunistic use"

        # ── Success-rate gate — don't use consistently poor abilities ─────
        if should:
            sr = float(entry.get("success_rate", 0.5))
            if sr < 0.25 and entry.get("attempts", 0) > 5:
                should = False
                reason = f"low success_rate={sr:.2f}, holding off"

        if should:
            self._last_activation_tick[ability_name] = tick
            self.last_decision_reason = reason

        return should

    def record_outcome(self, ability_name: str, success: bool) -> None:
        """Feed outcome back to AbilityDiscovery for continuous learning."""
        self._discovery.record_ability_use(ability_name, success)

    def get_strategy_hint(self, ability_name: str) -> str:
        """Return a short human-readable hint for dashboard display."""
        if not ability_name:
            return "No ability"
        entry = self._discovery.get_ability_info(ability_name)
        return (
            f"{ability_name} [{entry.get('type','?')}] — "
            f"use: {entry.get('best_use_case','?')} — "
            f"SR: {entry.get('success_rate',0.5):.0%}"
        )
