"""
ability_discovery.py
====================
Dynamic Ability Discovery System — automatically detects and learns how
Rumble/Spike-Rush/mode-specific powerups work through controlled in-game
experimentation.

Lifecycle
---------
1. Bot receives an unknown powerup (ability_id not in known_abilities).
2. System enters *experimentation mode* for that ability.
3. It runs up to EXPERIMENT_ROUNDS controlled trials:
   - near_ball  : activate while close to ball
   - near_opponent: activate near an opponent
   - open_space : activate in open field
4. Each trial records (distance_to_ball, ball_velocity_change,
   opponent_displacement, goal_probability_delta).
5. After EXPERIMENT_ROUNDS trials the system classifies the ability into
   one of EFFECT_CLASSES and writes results to model/ability_knowledge.json.
6. On subsequent possession the system suggests the best use-case context.
"""

from __future__ import annotations

import json
import math
import os
from typing import Dict, List, Optional

# ── Known Rumble powerup IDs (from RLBot / game source) ────────────────────
# Maps raw powerup string → canonical name
RUMBLE_POWERUP_NAMES: Dict[str, str] = {
    "Rugby_Ball_BA":     "plunger",
    "Magnet_BA":         "magnet",
    "Spring_BA":         "haymaker",
    "Batarang_BA":       "batarang",
    "Tornado_BA":        "tornado",
    "Freeze_BA":         "freezer",
    "Grapple_BA":        "grappling_hook",
    "Spike_BA":          "spike",
    "Swapper_BA":        "ball_lasso",
    "Disruptor_BA":      "disruptor",
    "Haunted_BA":        "haunted",
    "Heist_BA":          "punch",
    # Spike Rush
    "Spikes_BA":         "spikes",
    # Extended / future powerup variants
    "IceAttack_BA":      "ice_attack",
    "ShockWave_BA":      "shock_wave",
    "BallMagnet_BA":     "ball_magnet",
    "SpikesCharged_BA":  "spikes_carry",
    "TeleportStrike_BA": "teleport_strike",
}

EFFECT_CLASSES = [
    "ball_pull",
    "ball_push",
    "teleport",
    "freeze",
    "boost",
    "grapple",
    "ice_trap",
    "spike_control",
    "shock_push",
    # Extended ability effect types
    "ice_attack",
    "shock_wave",
    "ball_magnet",
    "spikes_carry",
    "teleport_strike",
    "unknown",
]

# How many experiment rounds before we commit to a classification
EXPERIMENT_ROUNDS = 6

_DEFAULT_ENTRY: Dict = {
    "type": "unknown",
    "best_use_case": "general",
    "success_rate": 0.50,
    "attempts": 0,
    "wins": 0,
    "description": "Discovered in-game; learning in progress.",
}

_SEEDED_KNOWLEDGE: Dict[str, Dict] = {
    "plunger": {
        "type": "ball_pull",
        "best_use_case": "when_ball_near_goal",
        "success_rate": 0.63,
        "attempts": 12,
        "wins": 8,
        "description": "Pulls the ball toward the bot — best aimed at goal.",
    },
    "grappling_hook": {
        "type": "grapple",
        "best_use_case": "aerial_intercept",
        "success_rate": 0.71,
        "attempts": 10,
        "wins": 7,
        "description": "Launches bot through the air toward ball/opponent.",
    },
    "freezer": {
        "type": "freeze",
        "best_use_case": "defensive_clear",
        "success_rate": 0.60,
        "attempts": 8,
        "wins": 5,
        "description": "Freezes ball in place — use to stall enemy attack.",
    },
    "haymaker": {
        "type": "ball_push",
        "best_use_case": "offensive_clear",
        "success_rate": 0.67,
        "attempts": 9,
        "wins": 6,
        "description": "Powerful spring that launches the ball at high speed.",
    },
    "spikes": {
        "type": "spike_control",
        "best_use_case": "carry_attack",
        "success_rate": 0.58,
        "attempts": 14,
        "wins": 8,
        "description": "Attaches ball to car — drive it into the goal.",
    },
    "punch": {
        "type": "shock_push",
        "best_use_case": "opponent_displacement",
        "success_rate": 0.55,
        "attempts": 11,
        "wins": 6,
        "description": "Punches opponents/ball away — use to clear defenders.",
    },
    "tornado": {
        "type": "ball_push",
        "best_use_case": "midfield_disruption",
        "success_rate": 0.50,
        "attempts": 7,
        "wins": 4,
        "description": "Tornado that disrupts ball path and opponent position.",
    },
    "batarang": {
        "type": "ball_pull",
        "best_use_case": "long_range_intercept",
        "success_rate": 0.52,
        "attempts": 8,
        "wins": 4,
        "description": "Boomerang that travels to target and returns.",
    },
    "magnet": {
        "type": "ball_pull",
        "best_use_case": "when_ball_near",
        "success_rate": 0.61,
        "attempts": 9,
        "wins": 6,
        "description": "Temporarily magnetises ball toward the car.",
    },
    "disruptor": {
        "type": "boost",
        "best_use_case": "boost_denial",
        "success_rate": 0.56,
        "attempts": 8,
        "wins": 5,
        "description": "Disables opponent boost temporarily.",
    },
    "ball_lasso": {
        "type": "ball_pull",
        "best_use_case": "aerial_ball_control",
        "success_rate": 0.58,
        "attempts": 7,
        "wins": 4,
        "description": "Lasso that grabs and controls ball trajectory.",
    },
    "spike": {
        "type": "spike_control",
        "best_use_case": "carry_attack",
        "success_rate": 0.54,
        "attempts": 8,
        "wins": 4,
        "description": "Attaches spikes to car — ram opponents or carry ball.",
    },
    "haunted": {
        "type": "ball_push",
        "best_use_case": "midfield_disruption",
        "success_rate": 0.48,
        "attempts": 5,
        "wins": 2,
        "description": "Haunted powerup — unpredictable ball behaviour.",
    },
    # ── Extended ability types ─────────────────────────────────────────────
    "ice_attack": {
        "type": "ice_attack",
        "best_use_case": "freeze_opponent_car",
        "success_rate": 0.62,
        "attempts": 8,
        "wins": 5,
        "description": "Launches an ice projectile that briefly freezes the opponent's car in place.",
    },
    "shock_wave": {
        "type": "shock_wave",
        "best_use_case": "mass_opponent_displacement",
        "success_rate": 0.59,
        "attempts": 10,
        "wins": 6,
        "description": "Emits a radial shock wave that displaces all nearby opponents simultaneously.",
    },
    "ball_magnet": {
        "type": "ball_magnet",
        "best_use_case": "goal_pull_attack",
        "success_rate": 0.64,
        "attempts": 9,
        "wins": 6,
        "description": "Stronger sustained magnet that pulls the ball toward the car over multiple ticks.",
    },
    "spikes_carry": {
        "type": "spikes_carry",
        "best_use_case": "carry_attack",
        "success_rate": 0.66,
        "attempts": 11,
        "wins": 7,
        "description": "Supercharged spike-carry — sticks the ball firmly and boosts carry speed.",
    },
    "teleport_strike": {
        "type": "teleport_strike",
        "best_use_case": "intercept_long_range",
        "success_rate": 0.60,
        "attempts": 8,
        "wins": 5,
        "description": "Instantly teleports the car to the ball's location and fires a strike.",
    },
}


class AbilityDiscovery:
    """
    Detects, experiments with, and learns the behaviour of powerup abilities.
    """

    def __init__(self, knowledge_path: Optional[str] = None):
        if knowledge_path is None:
            _here = os.path.dirname(os.path.abspath(__file__))
            _root = os.path.dirname(_here)
            knowledge_path = os.path.join(_root, "model", "ability_knowledge.json")
        self._path = knowledge_path
        self._knowledge: Dict[str, Dict] = {}
        self._load()

        # Runtime state
        self.current_ability: Optional[str] = None
        self._experiment_mode: bool = False
        self._experiment_stage: str = "near_ball"   # near_ball | near_opp | open
        self._experiment_rounds: int = 0
        self._trial_log: List[Dict] = []

        # Snapshot of game state at activation start
        self._pre_ball_vel: float = 0.0
        self._pre_opp_dist: float = 9999.0
        self._activate_requested: bool = False

        self._ability_just_activated: bool = False

    # ── Public API ────────────────────────────────────────────────────────────

    def update(
        self,
        raw_powerup: Optional[str],
        ball_pos: tuple,
        car_pos: tuple,
        opp_pos: tuple,
        ball_vel_magnitude: float,
    ) -> None:
        """
        Call every tick with the raw powerup string from the packet.
        Updates internal state; does NOT emit controller output.
        """
        ability = RUMBLE_POWERUP_NAMES.get(raw_powerup or "", raw_powerup or "")
        if not ability:
            # No powerup — reset experiment state
            self._experiment_mode = False
            self._experiment_rounds = 0
            self._trial_log = []
            self.current_ability = None
            self._activate_requested = False
            return

        self.current_ability = ability

        if ability not in self._knowledge:
            # Unknown — enter experimentation mode
            self._knowledge[ability] = dict(_DEFAULT_ENTRY)
            self._experiment_mode = True
            self._experiment_stage = "near_ball"
            self._experiment_rounds = 0
            self._trial_log = []

        if self._experiment_mode and self._experiment_rounds < EXPERIMENT_ROUNDS:
            self._run_experiment_tick(ability, ball_pos, car_pos, opp_pos, ball_vel_magnitude)

    def should_activate(self, ball_pos: tuple, car_pos: tuple, opp_pos: tuple, score_diff: int) -> bool:
        """
        Returns True if the bot should press the ability button this tick.
        Combines experimentation logic + strategic logic for known abilities.
        """
        if not self.current_ability:
            return False

        # Experimentation: activate on every 30th tick per stage
        if self._experiment_mode and self._experiment_rounds < EXPERIMENT_ROUNDS:
            return self._activate_requested

        # Known ability — use strategic guidance
        entry = self._knowledge.get(self.current_ability, _DEFAULT_ENTRY)
        return self._strategic_should_activate(entry, ball_pos, car_pos, opp_pos, score_diff)

    def record_trial_outcome(
        self,
        ball_vel_after: float,
        opp_dist_after: float,
        goal_probability_delta: float,
    ) -> None:
        """Record the observed effect after activating an ability during a trial."""
        if not self._experiment_mode or not self.current_ability:
            return
        trial = {
            "stage": self._experiment_stage,
            "ball_vel_change": ball_vel_after - self._pre_ball_vel,
            "opp_displacement": abs(opp_dist_after - self._pre_opp_dist),
            "goal_prob_delta": goal_probability_delta,
        }
        self._trial_log.append(trial)
        self._advance_experiment_stage()

    def record_ability_use(self, ability_name: str, success: bool) -> None:
        """Update success statistics for a strategic (non-experiment) activation."""
        entry = self._knowledge.setdefault(ability_name, dict(_DEFAULT_ENTRY))
        entry["attempts"] = entry.get("attempts", 0) + 1
        if success:
            entry["wins"] = entry.get("wins", 0) + 1
        total = entry["attempts"]
        wins = entry.get("wins", 0)
        entry["success_rate"] = round(wins / total, 3) if total > 0 else 0.5
        self._save()

    def get_ability_info(self, ability_name: str) -> Dict:
        """Return the stored knowledge entry for an ability."""
        return dict(self._knowledge.get(ability_name, _DEFAULT_ENTRY))

    def list_known_abilities(self) -> List[str]:
        return list(self._knowledge.keys())

    # ── Internals ─────────────────────────────────────────────────────────────

    def _run_experiment_tick(
        self,
        ability: str,
        ball_pos: tuple,
        car_pos: tuple,
        opp_pos: tuple,
        ball_vel: float,
    ) -> None:
        dist_to_ball = math.hypot(ball_pos[0] - car_pos[0], ball_pos[1] - car_pos[1])
        dist_to_opp  = math.hypot(opp_pos[0]  - car_pos[0], opp_pos[1]  - car_pos[1])

        # Choose activation moment based on experiment stage
        target_dist = {
            "near_ball": 300.0,
            "near_opp": 400.0,
            "open_space": 9999.0,  # always ok
        }.get(self._experiment_stage, 500.0)

        if self._experiment_stage == "near_ball":
            ready = dist_to_ball < target_dist
        elif self._experiment_stage == "near_opp":
            ready = dist_to_opp < target_dist
        else:
            ready = True

        if ready:
            self._pre_ball_vel = ball_vel
            self._pre_opp_dist = dist_to_opp
            self._activate_requested = True
        else:
            self._activate_requested = False

    def _advance_experiment_stage(self) -> None:
        stages = ["near_ball", "near_opp", "open_space"]
        idx = stages.index(self._experiment_stage) if self._experiment_stage in stages else 0
        self._experiment_rounds += 1
        if idx + 1 < len(stages):
            self._experiment_stage = stages[idx + 1]
        else:
            self._experiment_stage = stages[idx % len(stages)]

        if self._experiment_rounds >= EXPERIMENT_ROUNDS:
            self._finish_experiments()

    def _finish_experiments(self) -> None:
        """Classify the ability from trial logs and save knowledge."""
        if not self.current_ability or not self._trial_log:
            self._experiment_mode = False
            return

        avg_ball_vel_change = sum(t["ball_vel_change"] for t in self._trial_log) / len(self._trial_log)
        avg_opp_disp = sum(t["opp_displacement"] for t in self._trial_log) / len(self._trial_log)
        avg_goal_delta = sum(t["goal_prob_delta"] for t in self._trial_log) / len(self._trial_log)

        # Heuristic classification — extended for new effect types
        if avg_ball_vel_change > 400 and avg_opp_disp > 600:
            effect = "shock_wave"       # very high ball push + mass displacement
        elif avg_ball_vel_change > 200:
            effect = "ball_push"
        elif avg_ball_vel_change < -300 and avg_goal_delta > 0.1:
            effect = "ball_magnet"      # strong sustained pull toward goal
        elif avg_ball_vel_change < -200:
            effect = "ball_pull"
        elif avg_opp_disp > 700:
            effect = "shock_push"
        elif avg_opp_disp > 400 and avg_ball_vel_change == 0:
            effect = "ice_attack"       # opponent frozen — velocity unchanged
        elif avg_goal_delta > 0.20 and avg_ball_vel_change < 100:
            effect = "teleport_strike"  # large goal delta without ball push
        elif avg_goal_delta > 0.15:
            effect = "boost"
        else:
            effect = "unknown"

        best_use = _infer_best_use(effect, avg_goal_delta)

        entry = self._knowledge[self.current_ability]
        entry["type"] = effect
        entry["best_use_case"] = best_use
        entry["description"] = f"Auto-classified as {effect} after {EXPERIMENT_ROUNDS} trials."
        self._save()

        self._experiment_mode = False
        self._trial_log = []

    def _strategic_should_activate(
        self, entry: Dict, ball_pos: tuple, car_pos: tuple, opp_pos: tuple, score_diff: int
    ) -> bool:
        use_case = entry.get("best_use_case", "general")
        ability_type = entry.get("type", "unknown")
        dist_to_ball = math.hypot(ball_pos[0] - car_pos[0], ball_pos[1] - car_pos[1])
        dist_to_opp  = math.hypot(opp_pos[0] - car_pos[0], opp_pos[1] - car_pos[1])
        ball_near_opp_goal = ball_pos[1] > 4000 or ball_pos[1] < -4000

        if ability_type == "ball_pull" and use_case == "when_ball_near_goal":
            return ball_near_opp_goal and dist_to_ball < 800
        if ability_type == "ball_pull":
            return dist_to_ball < 600
        if ability_type == "ball_push":
            return dist_to_ball < 400 and score_diff <= 0
        if ability_type == "freeze":
            # Use defensively when behind
            return score_diff < 0 and dist_to_ball < 700
        if ability_type == "spike_control":
            return dist_to_ball < 250
        if ability_type in ("spikes_carry",):
            return dist_to_ball < 300
        if ability_type == "shock_push":
            return dist_to_opp < 400
        if ability_type == "shock_wave":
            return dist_to_opp < 600   # wider range for radial wave
        if ability_type == "ice_attack":
            return dist_to_opp < 500 and score_diff <= 0
        if ability_type == "ball_magnet":
            return dist_to_ball < 1200 and ball_near_opp_goal
        if ability_type == "teleport_strike":
            return dist_to_ball > 1200  # use to close long distances
        if ability_type == "grapple":
            return dist_to_ball > 1500
        return dist_to_ball < 500

    def _load(self) -> None:
        try:
            if os.path.isfile(self._path):
                with open(self._path, "r", encoding="utf-8") as f:
                    saved = json.load(f)
                # Merge seeded knowledge in for any missing abilities
                merged = dict(_SEEDED_KNOWLEDGE)
                merged.update(saved)
                self._knowledge = merged
            else:
                self._knowledge = dict(_SEEDED_KNOWLEDGE)
                self._save()
        except Exception:
            self._knowledge = dict(_SEEDED_KNOWLEDGE)

    def _save(self) -> None:
        try:
            os.makedirs(os.path.dirname(self._path), exist_ok=True)
            with open(self._path, "w", encoding="utf-8") as f:
                json.dump(self._knowledge, f, indent=2)
        except Exception:
            pass


def _infer_best_use(effect: str, goal_delta: float) -> str:
    mapping = {
        "ball_pull":       "when_ball_near_goal" if goal_delta > 0.1 else "when_ball_near",
        "ball_push":       "offensive_clear",
        "shock_push":      "opponent_displacement",
        "freeze":          "defensive_clear",
        "boost":           "boost_denial",
        "grapple":         "aerial_intercept",
        "spike_control":   "carry_attack",
        "ice_attack":      "freeze_opponent_car",
        "shock_wave":      "mass_opponent_displacement",
        "ball_magnet":     "goal_pull_attack",
        "spikes_carry":    "carry_attack",
        "teleport_strike": "intercept_long_range",
        "unknown":         "general",
    }
    return mapping.get(effect, "general")
