"""
team_ability_awareness.py
=========================
Tracks the powerup state of every car on the field (teammates and opponents)
so the bot can make informed decisions:

- Avoid enemies with dangerous abilities (freeze, shock_push, spike_control)
- Coordinate with teammates who have useful abilities (spikes → pass to them)
- Predict opponent activations based on proximity and game state

Usage
-----
awareness = TeamAbilityAwareness()
# Call every tick:
awareness.update(packet_game_cars, num_cars, bot_index, bot_team)
# Query:
info = awareness.get_teammate_ability(teammate_index)
danger = awareness.is_dangerous_opponent_nearby(car_pos, radius=600)
hint = awareness.best_pass_target(ball_pos, car_pos)
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from game_logic.ability_discovery import RUMBLE_POWERUP_NAMES, _DEFAULT_ENTRY


# Abilities the bot should avoid being close to when the enemy holds them
_DANGEROUS_ENEMY_ABILITIES = {
    "freezer", "punch", "haymaker", "spike", "spikes",
    "batarang", "tornado", "grappling_hook",
}

# Abilities that benefit from passing the ball to the teammate
_PASS_BENEFITING_ABILITIES = {
    "spikes", "spike", "spike_control", "ball_lasso", "magnet",
}


class CarAbilityState:
    """Snapshot of one car's ability state."""

    __slots__ = ("car_index", "team", "raw_powerup", "ability_name",
                 "pos", "is_bot")

    def __init__(self):
        self.car_index: int = 0
        self.team: int = 0
        self.raw_powerup: Optional[str] = None
        self.ability_name: Optional[str] = None
        self.pos: Tuple[float, float] = (0.0, 0.0)
        self.is_bot: bool = False


class TeamAbilityAwareness:
    """
    Monitors every car's powerup state and exposes helpers for:
    - teammate coordination
    - enemy avoidance
    """

    def __init__(self):
        self._states: Dict[int, CarAbilityState] = {}
        self._bot_index: int = 0
        self._bot_team: int = 0

    # ── Public API ────────────────────────────────────────────────────────────

    def update(self, game_cars, num_cars: int, bot_index: int, bot_team: int) -> None:
        """
        Scan every car in the packet and record their powerup state.

        game_cars  — packet.game_cars (list/array from RLBot GameTickPacket)
        num_cars   — packet.num_cars
        bot_index  — this bot's car index
        bot_team   — this bot's team (0 or 1)
        """
        self._bot_index = bot_index
        self._bot_team  = bot_team
        self._states.clear()

        for i in range(num_cars):
            try:
                car = game_cars[i]
                state = CarAbilityState()
                state.car_index = i
                state.team = getattr(car, "team", 0)
                state.is_bot = (i == bot_index)
                state.pos = (
                    car.physics.location.x,
                    car.physics.location.y,
                )
                raw = getattr(car, "powerup_pickup_timer", None)
                # In RLBot the powerup is surfaced through game_cars or game_boosts
                # depending on SDK version.  We try the common attribute names.
                raw_powerup = (
                    getattr(car, "powerup", None) or
                    getattr(car, "powerup_active", None) or
                    ""
                )
                if raw_powerup:
                    raw_str = str(raw_powerup)
                    state.raw_powerup = raw_str
                    state.ability_name = RUMBLE_POWERUP_NAMES.get(raw_str, raw_str)
                self._states[i] = state
            except Exception:
                pass  # If a car can't be read, skip silently

    def get_car_ability(self, car_index: int) -> Optional[str]:
        """Return the canonical ability name for a given car index, or None."""
        state = self._states.get(car_index)
        return state.ability_name if state else None

    def get_teammate_ability(self, teammate_index: int) -> Optional[str]:
        state = self._states.get(teammate_index)
        if state and state.team == self._bot_team and not state.is_bot:
            return state.ability_name
        return None

    def get_all_teammate_abilities(self) -> Dict[int, Optional[str]]:
        """Return {car_index: ability_name} for all teammates."""
        return {
            idx: s.ability_name
            for idx, s in self._states.items()
            if s.team == self._bot_team and not s.is_bot
        }

    def get_all_opponent_abilities(self) -> Dict[int, Optional[str]]:
        """Return {car_index: ability_name} for all opponents."""
        return {
            idx: s.ability_name
            for idx, s in self._states.items()
            if s.team != self._bot_team
        }

    def is_dangerous_opponent_nearby(
        self, car_pos: Tuple[float, float], radius: float = 600.0
    ) -> Tuple[bool, Optional[str]]:
        """
        Returns (True, ability_name) if a dangerous opponent is within radius.
        Returns (False, None) otherwise.
        """
        for idx, s in self._states.items():
            if s.team == self._bot_team:
                continue
            if not s.ability_name:
                continue
            if s.ability_name not in _DANGEROUS_ENEMY_ABILITIES:
                continue
            dist = math.hypot(s.pos[0] - car_pos[0], s.pos[1] - car_pos[1])
            if dist < radius:
                return True, s.ability_name
        return False, None

    def best_pass_target(
        self, ball_pos: Tuple[float, float], car_pos: Tuple[float, float]
    ) -> Optional[int]:
        """
        Returns car_index of the teammate most worth passing to (has a
        beneficial ability and is positioned well), or None.
        """
        best_idx: Optional[int] = None
        best_score: float = -1.0

        for idx, s in self._states.items():
            if s.team != self._bot_team or s.is_bot:
                continue
            if s.ability_name not in _PASS_BENEFITING_ABILITIES:
                continue
            # Prefer teammates closer to goal and not too far from ball
            dist_to_ball = math.hypot(
                s.pos[0] - ball_pos[0], s.pos[1] - ball_pos[1]
            )
            # Higher score = better pass target
            score = 1.0 / (1.0 + dist_to_ball / 1000.0)
            if score > best_score:
                best_score = score
                best_idx = idx

        return best_idx

    def get_position_adjustment(
        self, car_pos: Tuple[float, float], danger_radius: float = 500.0
    ) -> Tuple[float, float]:
        """
        Returns a nudge vector (dx, dy) to move away from dangerous opponents.
        Returns (0, 0) if no danger.
        """
        dx, dy = 0.0, 0.0
        for idx, s in self._states.items():
            if s.team == self._bot_team:
                continue
            if not s.ability_name or s.ability_name not in _DANGEROUS_ENEMY_ABILITIES:
                continue
            dist = math.hypot(s.pos[0] - car_pos[0], s.pos[1] - car_pos[1])
            if dist < danger_radius and dist > 0:
                scale = (danger_radius - dist) / danger_radius
                dx += (car_pos[0] - s.pos[0]) / dist * scale * 500
                dy += (car_pos[1] - s.pos[1]) / dist * scale * 500
        return dx, dy

    def summary(self) -> Dict:
        """Compact dict for IPC / dashboard display."""
        teammates = {
            str(idx): s.ability_name or "none"
            for idx, s in self._states.items()
            if s.team == self._bot_team and not s.is_bot
        }
        opponents = {
            str(idx): s.ability_name or "none"
            for idx, s in self._states.items()
            if s.team != self._bot_team
        }
        return {"teammate_abilities": teammates, "opponent_abilities": opponents}
