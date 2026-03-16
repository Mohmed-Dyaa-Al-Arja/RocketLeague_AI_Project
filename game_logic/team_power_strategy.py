"""
team_power_strategy.py
======================
Comprehensive team + power (ability) strategic synthesis.

Combines:
  • TeamController       — role assignments (attacker / support / defender)
  • TeamAbilityAwareness — who has what ability (teammate + enemy)
  • AbilityDiscovery     — learned ability semantics for own powerup

Output: StrategicDecision with a full action recommendation.

Named cooperative combos
------------------------
  spike_carry_attack      — spiker carries ball; others clear the path
  freeze_then_shoot       — freeze ball defensively, then push fast
  shock_clear_defenders   — shock wave clears path before attack
  magnet_goal_pull        — magnetise ball + aim toward opponent goal
  ice_trap_defensive_save — ice-trap attacker near own goal
  teleport_intercept      — teleport-strike to steal ball when far
  spikes_rush             — spikes + full throttle ram toward goal
  enemy_avoidance         — retreat when enemy threat score is critical
  standard                — default role-based behaviour

Usage
-----
    tps = TeamPowerStrategy(team_controller, awareness, discovery)
    decision = tps.get_strategic_decision(
        bot_index, game_state, ball_pos, car_pos,
        opp_positions, score_diff, tick)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from game_logic.team_controller import TeamController, ROLE_ATTACKER, ROLE_DEFENDER
from game_logic.team_ability_awareness import TeamAbilityAwareness
from game_logic.ability_discovery import AbilityDiscovery

# ── Named combo identifiers ───────────────────────────────────────────────────
POWER_COMBOS: List[str] = [
    "spike_carry_attack",
    "freeze_then_shoot",
    "shock_clear_defenders",
    "magnet_goal_pull",
    "ice_trap_defensive_save",
    "teleport_intercept",
    "spikes_rush",
    "enemy_avoidance",
    "standard",
]

_OFFENSIVE_STATES = {
    "Attack", "Counter Attack", "Dribbling", "Shooting", "Kickoff", "Free Ball",
}
_DEFENSIVE_STATES = {
    "Defense", "Goal Defense", "Defending", "Retreat", "Saving",
}

# Threat weights — how dangerous is an enemy ability
_THREAT_WEIGHTS: Dict[str, float] = {
    "freezer":         0.90,
    "ice_attack":      0.90,
    "haymaker":        0.80,
    "punch":           0.70,
    "shock_wave":      0.85,
    "tornado":         0.60,
    "batarang":        0.65,
    "spikes":          0.70,
    "spikes_carry":    0.75,
    "spike":           0.65,
    "grappling_hook":  0.60,
    "ball_lasso":      0.55,
    "teleport_strike": 0.80,
    "disruptor":       0.50,
}

# Opportunity weights — how strong is our own / teammate ability offensively
_OPPORTUNITY_WEIGHTS: Dict[str, float] = {
    "spikes":          0.85,
    "spikes_carry":    0.90,
    "spike":           0.80,
    "ball_magnet":     0.85,
    "magnet":          0.75,
    "teleport_strike": 0.80,
    "grappling_hook":  0.65,
    "ball_lasso":      0.75,
    "plunger":         0.70,
    "haymaker":        0.60,
    "shock_wave":      0.55,
}


# ── Output dataclass ──────────────────────────────────────────────────────────

@dataclass
class StrategicDecision:
    """Full strategic recommendation for one tick."""

    action: str = "standard"             # attack / defend / pass / support / retreat
    use_ability: bool = False
    target_pos: Optional[Tuple[float, float]] = None
    pass_to: Optional[int] = None
    combo_name: str = "standard"         # which named combo this belongs to
    priority_signals: List[str] = field(default_factory=list)
    reason: str = ""
    confidence: float = 0.5


# ── Team strategic intelligence helper ───────────────────────────────────────

class TeamStrategicIntelligence:
    """
    Synthesises threat + opportunity scores from all ability signals each tick.

    Attributes
    ----------
    threat_score      : 0–1, how dangerous current enemy abilities are
    opportunity_score : 0–1, how strong friendly / own abilities are
    urgency           : 0–1, combined tactical urgency this tick
    """

    def __init__(self) -> None:
        self.threat_score: float = 0.0
        self.opportunity_score: float = 0.0
        self.urgency: float = 0.0
        self._dominant_enemy: str = ""

    def update(
        self,
        own_ability: Optional[str],
        teammate_abilities: Dict[int, str],
        opponent_abilities: Dict[int, str],
        ball_pos: Tuple[float, float],
        car_pos: Tuple[float, float],
        score_diff: int,
    ) -> None:
        """Recompute all intelligence scores for this tick."""

        # Threat: average weight of all dangerous enemy abilities present
        enemy_weights = [
            _THREAT_WEIGHTS.get(ab, 0.3)
            for ab in opponent_abilities.values() if ab
        ]
        self.threat_score = (
            min(1.0, sum(enemy_weights) / max(len(enemy_weights), 1))
            if enemy_weights else 0.0
        )
        self._dominant_enemy = max(
            (ab for ab in opponent_abilities.values() if ab),
            key=lambda a: _THREAT_WEIGHTS.get(a, 0.0),
            default="",
        )

        # Opportunity: best of own + teammate abilities
        all_friendly: List[str] = [
            ab for ab in teammate_abilities.values() if ab
        ]
        if own_ability:
            all_friendly.append(own_ability)
        opp_weights = [_OPPORTUNITY_WEIGHTS.get(ab, 0.2) for ab in all_friendly]
        self.opportunity_score = max(opp_weights) if opp_weights else 0.0

        # Urgency: blend of threat, ball danger, score pressure
        ball_danger = max(0.0, (-ball_pos[1]) / 5120.0)   # team-0 perspective
        score_pressure = min(1.0, max(0.0, -score_diff / 3.0))
        self.urgency = min(
            1.0,
            self.threat_score * 0.5 + ball_danger * 0.3 + score_pressure * 0.2,
        )

    @property
    def dominant_enemy_ability(self) -> str:
        return self._dominant_enemy


# ── Main strategy class ───────────────────────────────────────────────────────

class TeamPowerStrategy:
    """
    High-level team + power (ability) orchestration layer.

    Sits above TeamAbilityStrategy to produce a single StrategicDecision
    that combines role assignment, ability semantics, team intelligence
    scoring, and multiple synergy combos.
    """

    def __init__(
        self,
        team_controller: Optional[TeamController] = None,
        awareness: Optional[TeamAbilityAwareness] = None,
        discovery: Optional[AbilityDiscovery] = None,
    ) -> None:
        self._controller = team_controller or TeamController()
        self._awareness  = awareness  or TeamAbilityAwareness()
        self._discovery  = discovery  or AbilityDiscovery()
        self._intel      = TeamStrategicIntelligence()
        self._last_combo: str = "standard"

    # ── Public API ────────────────────────────────────────────────────────────

    def get_strategic_decision(
        self,
        bot_index: int,
        game_state: str,
        ball_pos: Tuple[float, float],
        car_pos: Tuple[float, float],
        opp_positions: Dict[int, Tuple[float, float]],
        score_diff: int,
        tick: int,
    ) -> StrategicDecision:
        """
        Return a full strategic recommendation for this bot on this tick.

        Parameters
        ----------
        bot_index     : index of this bot in the game
        game_state    : situation string (e.g. "Attack", "Defense")
        ball_pos      : (x, y) world coordinates
        car_pos       : (x, y) this bot's position
        opp_positions : {car_idx: (x, y)} for visible opponents
        score_diff    : my_score − opp_score
        tick          : current game tick counter
        """
        own_ability        = self._discovery.current_ability
        teammate_abilities = self._awareness.get_all_teammate_abilities()
        opponent_abilities = self._awareness.get_all_opponent_abilities()
        my_role            = self._controller.get_role(bot_index)

        # Update intelligence layer
        self._intel.update(
            own_ability, teammate_abilities, opponent_abilities,
            ball_pos, car_pos, score_diff,
        )

        # ── Defensive emergency combos (highest priority) ──────────────────
        dec = self._eval_ice_trap_defensive_save(
            own_ability, ball_pos, car_pos, score_diff, game_state)
        if dec:
            return dec

        # ── Offensive spike combos ─────────────────────────────────────────
        dec = self._eval_spikes_rush(
            own_ability, ball_pos, car_pos, score_diff, game_state, my_role)
        if dec:
            return dec

        # ── Teleport intercept when far from ball ──────────────────────────
        dec = self._eval_teleport_intercept(own_ability, ball_pos, car_pos, game_state)
        if dec:
            return dec

        # ── Shock / punch to clear defenders ──────────────────────────────
        dec = self._eval_shock_clear_defenders(
            own_ability, opponent_abilities, car_pos, game_state, score_diff)
        if dec:
            return dec

        # ── Magnet pull toward goal ────────────────────────────────────────
        dec = self._eval_magnet_goal_pull(own_ability, ball_pos, car_pos, game_state, score_diff)
        if dec:
            return dec

        # ── Teammate spike coordination ────────────────────────────────────
        dec = self._eval_spike_carry_attack(
            own_ability, teammate_abilities, ball_pos, car_pos, my_role)
        if dec:
            return dec

        # ── Freeze-then-shoot when losing ─────────────────────────────────
        dec = self._eval_freeze_then_shoot(
            own_ability, ball_pos, car_pos, score_diff, game_state)
        if dec:
            return dec

        # ── Retreat if enemy threat is critical ───────────────────────────
        dec = self._eval_enemy_avoidance(opponent_abilities, car_pos, ball_pos)
        if dec:
            return dec

        # ── Default: role-based standard behaviour ────────────────────────
        return self._standard_decision(my_role, ball_pos, car_pos, game_state, score_diff)

    def notify_goal_scored(self) -> None:
        """Inform the strategy layer that we just scored."""
        self._controller.trigger_rotation("attacker_scored")

    def notify_goal_conceded(self) -> None:
        """Inform the strategy layer that we just conceded."""
        self._controller.trigger_rotation("goal_conceded")

    @property
    def intelligence(self) -> TeamStrategicIntelligence:
        """Access the underlying intelligence scores."""
        return self._intel

    @property
    def last_combo(self) -> str:
        """Name of the last combo that produced a decision."""
        return self._last_combo

    # ── Combo evaluators (private) ────────────────────────────────────────────

    def _eval_ice_trap_defensive_save(
        self,
        own_ability: Optional[str],
        ball_pos: Tuple[float, float],
        car_pos: Tuple[float, float],
        score_diff: int,
        game_state: str,
    ) -> Optional[StrategicDecision]:
        """Ice-trap or freeze the ball when it's dangerously near own goal."""
        if own_ability not in ("freezer", "ice_attack"):
            return None
        if game_state not in _DEFENSIVE_STATES:
            return None
        # Ball within 2000 units of own goal (team-0: y < -3120)
        dist_to_own_goal = math.hypot(ball_pos[0], ball_pos[1] + 5120)
        if dist_to_own_goal > 2000:
            return None
        dec = StrategicDecision(
            action="defend",
            use_ability=True,
            target_pos=ball_pos,
            combo_name="ice_trap_defensive_save",
            priority_signals=["ball_near_own_goal", "freeze_available"],
            reason=(
                f"ice_trap_defensive_save: {own_ability} ball near own goal "
                f"(dist={dist_to_own_goal:.0f})"
            ),
            confidence=0.85,
        )
        self._last_combo = dec.combo_name
        return dec

    def _eval_spikes_rush(
        self,
        own_ability: Optional[str],
        ball_pos: Tuple[float, float],
        car_pos: Tuple[float, float],
        score_diff: int,
        game_state: str,
        my_role: str,
    ) -> Optional[StrategicDecision]:
        """Rush the ball with spikes when close enough to carry it."""
        if own_ability not in ("spikes", "spike", "spikes_carry"):
            return None
        if game_state not in _OFFENSIVE_STATES and my_role != ROLE_ATTACKER:
            return None
        dist_to_ball = math.hypot(ball_pos[0] - car_pos[0], ball_pos[1] - car_pos[1])
        if dist_to_ball > 600:
            return None
        opp_goal_y = 5120.0
        dec = StrategicDecision(
            action="attack",
            use_ability=True,
            target_pos=(ball_pos[0], opp_goal_y),
            combo_name="spikes_rush",
            priority_signals=["spikes_available", "ball_near"],
            reason=(
                f"spikes_rush: carry ball with {own_ability} "
                f"({dist_to_ball:.0f} units away) → opp goal"
            ),
            confidence=0.80,
        )
        self._last_combo = dec.combo_name
        return dec

    def _eval_teleport_intercept(
        self,
        own_ability: Optional[str],
        ball_pos: Tuple[float, float],
        car_pos: Tuple[float, float],
        game_state: str,
    ) -> Optional[StrategicDecision]:
        """Teleport-strike to the ball when it's far away in an offensive state."""
        if own_ability != "teleport_strike":
            return None
        dist_to_ball = math.hypot(ball_pos[0] - car_pos[0], ball_pos[1] - car_pos[1])
        if dist_to_ball < 1500:
            return None          # already close enough
        if game_state not in _OFFENSIVE_STATES:
            return None
        dec = StrategicDecision(
            action="attack",
            use_ability=True,
            target_pos=ball_pos,
            combo_name="teleport_intercept",
            priority_signals=["teleport_available", "ball_far"],
            reason=f"teleport_intercept: teleport to ball (dist={dist_to_ball:.0f})",
            confidence=0.78,
        )
        self._last_combo = dec.combo_name
        return dec

    def _eval_shock_clear_defenders(
        self,
        own_ability: Optional[str],
        opponent_abilities: Dict[int, str],
        car_pos: Tuple[float, float],
        game_state: str,
        score_diff: int,
    ) -> Optional[StrategicDecision]:
        """Use shock / punch wave to clear defenders when pushing into attack."""
        if own_ability not in ("punch", "haymaker", "shock_wave"):
            return None
        if game_state not in _OFFENSIVE_STATES:
            return None
        if score_diff > 1:
            return None   # don't waste ability when already comfortably winning
        dec = StrategicDecision(
            action="attack",
            use_ability=True,
            target_pos=None,
            combo_name="shock_clear_defenders",
            priority_signals=["shock_push_available", "pushing_attack"],
            reason=f"shock_clear_defenders: activate {own_ability} to clear path",
            confidence=0.70,
        )
        self._last_combo = dec.combo_name
        return dec

    def _eval_magnet_goal_pull(
        self,
        own_ability: Optional[str],
        ball_pos: Tuple[float, float],
        car_pos: Tuple[float, float],
        game_state: str,
        score_diff: int,
    ) -> Optional[StrategicDecision]:
        """Activate ball magnet and aim toward the opponent's goal."""
        if own_ability not in ("magnet", "ball_magnet"):
            return None
        if game_state not in _OFFENSIVE_STATES:
            return None
        dist_to_ball = math.hypot(ball_pos[0] - car_pos[0], ball_pos[1] - car_pos[1])
        if dist_to_ball > 1800:
            return None
        opp_goal_y = 5120.0
        dec = StrategicDecision(
            action="attack",
            use_ability=True,
            target_pos=(0.0, opp_goal_y),
            combo_name="magnet_goal_pull",
            priority_signals=["magnet_available", "near_opp_goal"],
            reason="magnet_goal_pull: magnetise ball toward opponent goal",
            confidence=0.75,
        )
        self._last_combo = dec.combo_name
        return dec

    def _eval_spike_carry_attack(
        self,
        own_ability: Optional[str],
        teammate_abilities: Dict[int, str],
        ball_pos: Tuple[float, float],
        car_pos: Tuple[float, float],
        my_role: str,
    ) -> Optional[StrategicDecision]:
        """Coordinate around a teammate who has spikes."""
        _spike_set = {"spikes", "spike", "spikes_carry"}
        # Must not be the spiker ourselves
        if own_ability in _spike_set:
            return None
        spiker = next(
            (idx for idx, ab in teammate_abilities.items() if ab in _spike_set),
            None,
        )
        if spiker is None:
            return None
        dist_to_ball = math.hypot(ball_pos[0] - car_pos[0], ball_pos[1] - car_pos[1])
        if dist_to_ball < 500:
            dec = StrategicDecision(
                action="pass",
                use_ability=False,
                pass_to=spiker,
                combo_name="spike_carry_attack",
                priority_signals=["teammate_has_spikes", "i_have_ball"],
                reason=f"spike_carry_attack: pass to spiker (car#{spiker})",
                confidence=0.82,
            )
        else:
            support_pos = (ball_pos[0] + 200.0, ball_pos[1] - 400.0)
            dec = StrategicDecision(
                action="support",
                use_ability=False,
                target_pos=support_pos,
                combo_name="spike_carry_attack",
                priority_signals=["teammate_has_spikes", "support_run"],
                reason=f"spike_carry_attack: position to support spiker (car#{spiker})",
                confidence=0.70,
            )
        self._last_combo = dec.combo_name
        return dec

    def _eval_freeze_then_shoot(
        self,
        own_ability: Optional[str],
        ball_pos: Tuple[float, float],
        car_pos: Tuple[float, float],
        score_diff: int,
        game_state: str,
    ) -> Optional[StrategicDecision]:
        """Freeze ball during defense when losing — buys time to re-organise."""
        if own_ability not in ("freezer", "ice_attack"):
            return None
        if game_state not in _DEFENSIVE_STATES:
            return None
        if score_diff >= 0:
            return None   # only when behind
        dec = StrategicDecision(
            action="defend",
            use_ability=True,
            target_pos=ball_pos,
            combo_name="freeze_then_shoot",
            priority_signals=["freeze_available", "losing"],
            reason=(
                f"freeze_then_shoot: freeze ball defensively "
                f"(losing by {-score_diff})"
            ),
            confidence=0.72,
        )
        self._last_combo = dec.combo_name
        return dec

    def _eval_enemy_avoidance(
        self,
        opponent_abilities: Dict[int, str],
        car_pos: Tuple[float, float],
        ball_pos: Tuple[float, float],
    ) -> Optional[StrategicDecision]:
        """Retreat when an extremely dangerous enemy ability is active."""
        if self._intel.threat_score < 0.75:
            return None
        danger_ab = self._intel.dominant_enemy_ability
        if not danger_ab:
            return None
        retreat_pos = (car_pos[0] * 0.5, car_pos[1] - 1500.0)
        dec = StrategicDecision(
            action="retreat",
            use_ability=False,
            target_pos=retreat_pos,
            combo_name="enemy_avoidance",
            priority_signals=["high_threat", f"enemy_{danger_ab}"],
            reason=(
                f"enemy_avoidance: threat={self._intel.threat_score:.2f} "
                f"from {danger_ab}"
            ),
            confidence=0.65,
        )
        self._last_combo = dec.combo_name
        return dec

    def _standard_decision(
        self,
        my_role: str,
        ball_pos: Tuple[float, float],
        car_pos: Tuple[float, float],
        game_state: str,
        score_diff: int,
    ) -> StrategicDecision:
        """Fallback: act according to role and game state."""
        if my_role == ROLE_ATTACKER or game_state in _OFFENSIVE_STATES:
            action = "attack"
            target = (ball_pos[0], min(ball_pos[1] + 200, 5120.0))
        elif my_role == ROLE_DEFENDER or game_state in _DEFENSIVE_STATES:
            action = "defend"
            target = (0.0, -4000.0)
        else:
            action = "support"
            target = (ball_pos[0] * 0.5, ball_pos[1] - 800.0)

        self._last_combo = "standard"
        return StrategicDecision(
            action=action,
            use_ability=False,
            target_pos=target,
            combo_name="standard",
            priority_signals=[],
            reason=f"standard: role={my_role} state={game_state}",
            confidence=0.5,
        )
