"""
team_controller.py
==================
Multi-bot role assignment, rotation management, and model-selection system.

Roles
-----
  attacker  — pushes ball toward opponent goal
  support   — second-man coverage, ready to shoot or assist
  defender  — protects own goal

Team Model Types (for custom team configuration)
--------------
  default_rl    — use Rocket League's built-in bot opponent
  custom_ai     — use this project's SearchOnlyBot AI
  specific      — user-selected model variant (attacker/support/defender)

Role Assignment Logic
---------------------
  ball near opp goal  → assign attacker
  ball midfield       → assign support
  ball near own goal  → assign defender

Professional Rotation
---------------------
After every significant game event the roles rotate:
  attacker → rotates back to support
  support  → moves forward to attack when attacker rotates
  defender → covers goal; becomes support when ball clears midfield

Usage
-----
tc = TeamController()
tc.update(ball_pos, car_positions, my_index, my_team)
my_role = tc.get_role(my_index)
tc.trigger_rotation("attacker_scored")
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

ROLE_ATTACKER = "attacker"
ROLE_SUPPORT  = "support"
ROLE_DEFENDER = "defender"

_ALL_ROLES = [ROLE_ATTACKER, ROLE_SUPPORT, ROLE_DEFENDER]

# Team composition options
TEAM_MODE_DEFAULT_RL  = "default_rl"    # Use RL Engine bots as opponents
TEAM_MODE_CUSTOM_AI   = "custom_ai"     # All bots use this project's AI
TEAM_MODE_SPECIFIC    = "specific"      # User selects individual models

# Model variants for specific mode
MODEL_ATTACKER = "attacker"
MODEL_SUPPORT  = "support"
MODEL_DEFENDER = "defender"

# Rotation cooldown in ticks to prevent thrashing
_ROTATION_COOLDOWN = 180   # ~3 s


class BotRoleState:
    __slots__ = ("bot_index", "role", "pos", "team")

    def __init__(self, bot_index: int, role: str, team: int):
        self.bot_index = bot_index
        self.role = role
        self.pos: Tuple[float, float] = (0.0, 0.0)
        self.team = team


class TeamController:
    """
    Manages role assignment and rotation for all bots on a team.
    """

    def __init__(self):
        self._roles: Dict[int, BotRoleState] = {}
        self._last_rotation_tick: int = 0
        self._team_mode: str = TEAM_MODE_DEFAULT_RL
        self._team_models: Dict[str, str] = {
            MODEL_ATTACKER: "model_attack_v1",
            MODEL_SUPPORT:  "model_support_v1",
            MODEL_DEFENDER: "model_defense_v1",
        }
        self._my_team: int = 0
        self.last_rotation_reason: str = ""

    # ── Public API ────────────────────────────────────────────────────────────

    def update(
        self,
        ball_pos: Tuple[float, float],
        car_positions: Dict[int, Tuple[float, float]],
        my_index: int,
        my_team: int,
        tick: int = 0,
    ) -> None:
        """
        Refresh role assignments based on current game state.

        car_positions: {car_index: (x, y)} for all cars on MY team
        """
        self._my_team = my_team

        # Register new cars
        for idx, pos in car_positions.items():
            if idx not in self._roles:
                # Assign roles in registration order
                existing = len(self._roles)
                role = _ALL_ROLES[existing % len(_ALL_ROLES)]
                self._roles[idx] = BotRoleState(idx, role, my_team)
            self._roles[idx].pos = pos

        # Dynamic role assignment based on ball position
        if tick - self._last_rotation_tick >= _ROTATION_COOLDOWN:
            self._assign_roles_by_position(ball_pos, my_team)
            self._last_rotation_tick = tick

    def get_role(self, bot_index: int) -> str:
        """Return the current role for a given bot index."""
        state = self._roles.get(bot_index)
        return state.role if state else ROLE_SUPPORT

    def trigger_rotation(self, event: str) -> None:
        """
        Manually trigger a rotation on a game event.
        event: "attacker_scored" | "goal_conceded" | "ball_cleared"
        """
        role_list = [
            s for s in self._roles.values() if s.team == self._my_team
        ]
        if not role_list:
            return

        if event == "attacker_scored":
            # Rotate: old attacker backs off, support steps up
            self._rotate_roles(role_list, direction=1)
            self.last_rotation_reason = "attacker scored — rotating back"
        elif event == "goal_conceded":
            # Pull everyone back
            for s in role_list:
                s.role = ROLE_DEFENDER if s.role == ROLE_SUPPORT else s.role
            self.last_rotation_reason = "goal conceded — defensive re-set"
        elif event == "ball_cleared":
            self._rotate_roles(role_list, direction=1)
            self.last_rotation_reason = "ball cleared — pushing forward"

    def get_team_summary(self) -> Dict:
        return {
            str(idx): {"role": s.role, "pos": list(s.pos)}
            for idx, s in self._roles.items()
        }

    def set_team_mode(self, mode: str) -> None:
        """Set team composition mode: default_rl | custom_ai | specific."""
        if mode in (TEAM_MODE_DEFAULT_RL, TEAM_MODE_CUSTOM_AI, TEAM_MODE_SPECIFIC):
            self._team_mode = mode

    def set_team_model(self, role: str, model_name: str) -> None:
        """Assign a model variant name to a role (for specific mode)."""
        if role in (MODEL_ATTACKER, MODEL_SUPPORT, MODEL_DEFENDER):
            self._team_models[role] = model_name

    def get_model_for_role(self, role: str) -> str:
        if self._team_mode == TEAM_MODE_SPECIFIC:
            return self._team_models.get(role, "default")
        if self._team_mode == TEAM_MODE_CUSTOM_AI:
            return "custom_ai"
        return "default_rl"

    @property
    def team_mode(self) -> str:
        return self._team_mode

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _assign_roles_by_position(
        self, ball_pos: Tuple[float, float], my_team: int
    ) -> None:
        """Assign attacker/support/defender based on car proximity to ball."""
        team_states = [
            s for s in self._roles.values() if s.team == my_team
        ]
        if not team_states:
            return

        # Sort by distance to ball
        team_states.sort(
            key=lambda s: math.hypot(s.pos[0] - ball_pos[0], s.pos[1] - ball_pos[1])
        )

        # Ball zone
        ball_y = ball_pos[1]
        # Determine which side is offensive/defensive
        # For team 0: positive Y is opponent goal; for team 1: negative Y
        if my_team == 0:
            in_attack = ball_y > 1500
            in_defense = ball_y < -1500
        else:
            in_attack = ball_y < -1500
            in_defense = ball_y > 1500

        n = len(team_states)
        if n == 1:
            team_states[0].role = ROLE_ATTACKER
        elif n == 2:
            team_states[0].role = ROLE_ATTACKER
            team_states[1].role = ROLE_DEFENDER
        else:
            # 3+ cars
            team_states[0].role = ROLE_ATTACKER
            team_states[-1].role = ROLE_DEFENDER
            for s in team_states[1:-1]:
                s.role = ROLE_SUPPORT

    @staticmethod
    def _rotate_roles(role_list: List[BotRoleState], direction: int) -> None:
        """Shift roles along the list by direction steps."""
        n = len(role_list)
        if n < 2:
            return
        roles = [s.role for s in role_list]
        roles = roles[direction:] + roles[:direction]
        for s, r in zip(role_list, roles):
            s.role = r

    # ── Multi-bot team spawning ───────────────────────────────────────────────

    def launch_team(
        self,
        project_root: str,
        venv_python: Optional[str] = None,
    ) -> Dict[str, subprocess.Popen]:
        """
        Spawn one subprocess per role (attacker, support, defender), each
        running run_match.py in headless mode with the role preset via env var.

        Returns a dict mapping role -> Popen handle.
        The caller is responsible for terminating them later.
        """
        if venv_python is None:
            venv_python = sys.executable

        run_match = os.path.join(project_root, "runtime", "run_match.py")
        processes: Dict[str, subprocess.Popen] = {}

        for role in (ROLE_ATTACKER, ROLE_SUPPORT, ROLE_DEFENDER):
            env = dict(os.environ)
            env["RL_BOT_ROLE"] = role
            env["RL_BOT_HEADLESS"] = "1"

            try:
                proc = subprocess.Popen(
                    [venv_python, run_match, "--no-gui", f"--role={role}"],
                    env=env,
                    cwd=project_root,
                    stdin=subprocess.DEVNULL,
                )
                processes[role] = proc
            except Exception:
                pass

        self._launched_processes = processes
        return processes

    def get_launched_pids(self) -> Dict[str, int]:
        """Return {role: pid} for all currently spawned team-bot processes."""
        return {
            role: proc.pid
            for role, proc in getattr(self, "_launched_processes", {}).items()
            if proc.poll() is None    # still running
        }

    def stop_team_bots(self) -> None:
        """Terminate all subprocesses launched by launch_team()."""
        for proc in getattr(self, "_launched_processes", {}).values():
            try:
                if proc.poll() is None:
                    proc.terminate()
            except Exception:
                pass
        self._launched_processes = {}
