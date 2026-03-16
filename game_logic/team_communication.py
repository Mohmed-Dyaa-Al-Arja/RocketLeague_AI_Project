"""
game_logic/team_communication.py
=================================
Shared team-state object for coordinated 2v2 / 3v3 play.

Each bot instance writes its own state to the shared object and reads the team
snapshot to decide its role.

Roles
-----
attacker  — closest to ball, pushing toward opponent goal
defender  — closest to own goal, protecting
support   — remaining teammate; rotates depending on situation

The shared state is a plain dict so it can be serialised to IPC files when
multiple processes are involved.

Author: medo dyaa
"""

from __future__ import annotations

import math
import threading
from typing import Dict, List, Optional, Tuple


_LOCK = threading.Lock()


class TeamState:
    """
    Thread-safe shared state for the entire team.

    In a single-process multi-bot simulation each bot calls `register()` then
    `update()` every tick.  In subprocess mode, bots can share state via IPC.
    """

    def __init__(self) -> None:
        self._bots: Dict[int, dict] = {}   # bot_id → state dict
        self._ball_position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._own_goal_y:  float = -5120.0
        self._opp_goal_y:  float =  5120.0

    # ── Bot registration ──────────────────────────────────────────────────────

    def register(
        self,
        bot_id: int,
        own_goal_y: float = -5120.0,
        opp_goal_y: float = 5120.0,
    ) -> None:
        """Register a bot and set field orientation."""
        with _LOCK:
            self._own_goal_y = own_goal_y
            self._opp_goal_y = opp_goal_y
            if bot_id not in self._bots:
                self._bots[bot_id] = {
                    "bot_id":   bot_id,
                    "role":     "support",
                    "position": (0.0, 0.0, 0.0),
                    "boost":    100.0,
                    "has_ball": False,
                }

    # ── Per-tick update ───────────────────────────────────────────────────────

    def update(
        self,
        bot_id: int,
        position: Tuple[float, float, float],
        ball_position: Tuple[float, float, float],
        boost: float,
        has_ball: bool,
    ) -> str:
        """
        Update this bot's state, recalculate roles, and return this bot's role.

        Returns
        -------
        str — one of ``"attacker"``, ``"defender"``, ``"support"``.
        """
        with _LOCK:
            self._ball_position = ball_position

            if bot_id not in self._bots:
                self.register(bot_id)

            self._bots[bot_id].update({
                "position": position,
                "boost":    boost,
                "has_ball": has_ball,
            })

            self._assign_roles()
            return self._bots[bot_id]["role"]

    # ── Role assignment ───────────────────────────────────────────────────────

    def _assign_roles(self) -> None:
        """Simple distance-based role assignment."""
        ids = list(self._bots.keys())
        if not ids:
            return

        bx, by, bz = self._ball_position

        # Distances to ball
        dist_to_ball = {}
        for bid in ids:
            px, py, pz = self._bots[bid]["position"]
            dist_to_ball[bid] = math.hypot(px - bx, py - by)

        # Distances to own goal
        own_gx, own_gy = 0.0, self._own_goal_y
        dist_to_own_goal = {}
        for bid in ids:
            px, py, pz = self._bots[bid]["position"]
            dist_to_own_goal[bid] = math.hypot(px - own_gx, py - own_gy)

        sorted_by_ball = sorted(ids, key=lambda b: dist_to_ball[b])
        sorted_by_goal = sorted(ids, key=lambda b: dist_to_own_goal[b])

        n = len(ids)
        if n == 1:
            self._bots[ids[0]]["role"] = "attacker"
        elif n == 2:
            self._bots[sorted_by_ball[0]]["role"] = "attacker"
            self._bots[sorted_by_ball[1]]["role"] = "defender"
        else:
            # 3+ bots
            self._bots[sorted_by_ball[0]]["role"] = "attacker"
            # Deepest in own half = defender
            self._bots[sorted_by_goal[0]]["role"] = "defender"
            for bid in ids:
                if self._bots[bid]["role"] not in ("attacker", "defender"):
                    self._bots[bid]["role"] = "support"

    # ── Queries ───────────────────────────────────────────────────────────────

    def get_role(self, bot_id: int) -> str:
        with _LOCK:
            return self._bots.get(bot_id, {}).get("role", "support")

    def get_ball_position(self) -> Tuple[float, float, float]:
        with _LOCK:
            return self._ball_position

    def should_chase_ball(self, bot_id: int) -> bool:
        """Return True if this bot is the designated attacker."""
        return self.get_role(bot_id) == "attacker"

    def get_target(
        self,
        bot_id: int,
        push_dir: float = 1.0,
    ) -> Tuple[float, float]:
        """
        Return a 2-D target position suitable for path planning.

        - attacker  → ball
        - defender  → own goal mouth
        - support   → midfield offset
        """
        with _LOCK:
            role = self._bots.get(bot_id, {}).get("role", "support")
            bx, by, _ = self._ball_position
            if role == "attacker":
                return (bx, by)
            elif role == "defender":
                return (0.0, self._own_goal_y * 0.8)
            else:
                return (bx * 0.5, (self._own_goal_y + self._opp_goal_y) * 0.3 * push_dir)

    def snapshot(self) -> dict:
        """Return a serialisable snapshot of the full team state."""
        with _LOCK:
            return {
                "ball_position": self._ball_position,
                "bots": {
                    bid: dict(self._bots[bid]) for bid in self._bots
                },
            }


# Module-level singleton (single-process multi-agent use)
_default_team_state = TeamState()


def get_default_team_state() -> TeamState:
    return _default_team_state
