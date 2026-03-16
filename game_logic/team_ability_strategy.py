"""
team_ability_strategy.py
========================
Coordinates ability usage across the team, ensuring teammates use their
powers in synergy rather than independently.

Strategies include:
  spike_carry_attack  — teammate with spikes carries ball; others support
  freeze_then_shoot   — freeze ball defensively, then push it to attack
  grapple_intercept   — grapple to stolen ball before opponent reaches it
  pass_to_spiker      — bot passes to the teammate with spike ability
  box_out_freeze      — freeze opponent near our goal, then clear

Usage
-----
tstr = TeamAbilityStrategy(awareness, discovery)
directive = tstr.get_team_directive(
    bot_index, game_state, ball_pos, opp_team_abilities, score_diff)
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from game_logic.team_ability_awareness import TeamAbilityAwareness
from game_logic.ability_discovery import AbilityDiscovery

# Named cooperative strategy identifiers
TEAM_STRATEGIES = [
    "spike_carry_attack",
    "freeze_then_shoot",
    "grapple_intercept",
    "pass_to_spiker",
    "box_out_freeze",
    "standard",          # no special coordination
]


class TeamDirective:
    """Instruction returned to the bot for this tick."""
    __slots__ = (
        "strategy_name", "should_pass", "pass_target_index",
        "should_activate_ability", "position_hint", "reason",
    )

    def __init__(self):
        self.strategy_name: str = "standard"
        self.should_pass: bool = False
        self.pass_target_index: Optional[int] = None
        self.should_activate_ability: bool = False
        self.position_hint: Optional[Tuple[float, float]] = None
        self.reason: str = ""


class TeamAbilityStrategy:
    """
    High-level team coordination for ability usage.
    Operates on top of TeamAbilityAwareness for sensing, and
    AbilityDiscovery for learned ability semantics.
    """

    def __init__(
        self,
        awareness: Optional[TeamAbilityAwareness] = None,
        discovery: Optional[AbilityDiscovery] = None,
    ):
        self._awareness = awareness or TeamAbilityAwareness()
        self._discovery = discovery or AbilityDiscovery()
        self._current_strategy: str = "standard"
        self._strategy_tick: int = 0

    # ── Public API ────────────────────────────────────────────────────────────

    def get_team_directive(
        self,
        bot_index: int,
        my_ability: Optional[str],
        game_state: str,
        ball_pos: Tuple[float, float],
        car_pos: Tuple[float, float],
        score_diff: int,
        tick: int,
    ) -> TeamDirective:
        """
        Determine the best team-coordinated action for this bot.
        Called every tick from the main bot loop.
        """
        directive = TeamDirective()
        teammate_abilities = self._awareness.get_all_teammate_abilities()
        opponent_abilities  = self._awareness.get_all_opponent_abilities()

        # ── Spike-carry attack ─────────────────────────────────────────────
        spiker = self._teammate_with_ability(teammate_abilities, {"spikes", "spike", "spike_control"})
        if spiker is not None:
            # I should pass to the spiker if I'm closer to ball
            dist_my_to_ball = math.hypot(ball_pos[0] - car_pos[0], ball_pos[1] - car_pos[1])
            spiker_pos = self._awareness._states.get(spiker)
            if spiker_pos:
                dist_sp_to_ball = math.hypot(
                    ball_pos[0] - spiker_pos.pos[0], ball_pos[1] - spiker_pos.pos[1]
                )
                if dist_my_to_ball < dist_sp_to_ball:
                    directive.strategy_name = "spike_carry_attack"
                    directive.should_pass = True
                    directive.pass_target_index = spiker
                    directive.reason = f"pass to spiker car#{spiker}"
                    return directive
                else:
                    # Spiker is closer — I support the attack
                    directive.strategy_name = "spike_carry_attack"
                    directive.reason = "support spike_carry — push forward"
                    directive.position_hint = self._support_position(ball_pos, car_pos, attack=True)
                    return directive

        # ── Freeze then shoot ──────────────────────────────────────────────
        if my_ability == "freezer":
            danger, _ = self._awareness.is_dangerous_opponent_nearby(car_pos, 700)
            if game_state in ("Defense", "Goal Defense") or danger:
                if score_diff < 0:
                    directive.strategy_name = "freeze_then_shoot"
                    directive.should_activate_ability = True
                    directive.reason = "freeze: defensive stall while behind"
                    return directive

        # ── Grapple intercept ──────────────────────────────────────────────
        if my_ability == "grappling_hook":
            dist_to_ball = math.hypot(ball_pos[0] - car_pos[0], ball_pos[1] - car_pos[1])
            if dist_to_ball > 1500 and game_state in ("Counter Attack", "Attack"):
                directive.strategy_name = "grapple_intercept"
                directive.should_activate_ability = True
                directive.reason = "grapple to close gap in counter attack"
                return directive

        # ── Box out freeze ─────────────────────────────────────────────────
        opp_freezer = self._team_with_ability(opponent_abilities, {"freezer"})
        if opp_freezer is not None:
            directive.strategy_name = "box_out_freeze"
            directive.reason = "enemy has freezer — avoid close dribble"
            directive.position_hint = self._safe_position(ball_pos, car_pos)
            return directive

        # ── Enemy punch / shock avoidance ──────────────────────────────────
        opp_puncher = self._team_with_ability(opponent_abilities, {"punch", "haymaker"})
        if opp_puncher is not None:
            directive.strategy_name = "standard"
            directive.reason = "enemy has punch — avoid direct shot line"
            directive.position_hint = self._flanking_position(ball_pos, car_pos)
            return directive

        # ── Default ────────────────────────────────────────────────────────
        directive.strategy_name = "standard"
        directive.reason = "no special team coordination needed"
        return directive

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _teammate_with_ability(
        teammate_abilities: Dict[int, Optional[str]], target: set
    ) -> Optional[int]:
        for idx, ability in teammate_abilities.items():
            if ability and ability in target:
                return idx
        return None

    @staticmethod
    def _team_with_ability(
        abilities: Dict[int, Optional[str]], target: set
    ) -> Optional[int]:
        for idx, ability in abilities.items():
            if ability and ability in target:
                return idx
        return None

    @staticmethod
    def _support_position(
        ball_pos: Tuple[float, float], car_pos: Tuple[float, float], attack: bool
    ) -> Tuple[float, float]:
        """Return a support position behind or beside the ball."""
        offset = -800 if attack else 800
        return (ball_pos[0] * 0.5, ball_pos[1] + offset)

    @staticmethod
    def _safe_position(
        ball_pos: Tuple[float, float], car_pos: Tuple[float, float]
    ) -> Tuple[float, float]:
        """Stay away from ball to avoid enemy freeze range."""
        return (car_pos[0], car_pos[1] - 800)

    @staticmethod
    def _flanking_position(
        ball_pos: Tuple[float, float], car_pos: Tuple[float, float]
    ) -> Tuple[float, float]:
        """Attack from the side to avoid a direct punch line."""
        return (ball_pos[0] + 600, ball_pos[1])
