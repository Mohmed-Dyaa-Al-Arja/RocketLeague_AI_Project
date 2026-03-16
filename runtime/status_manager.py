"""runtime/status_manager.py — Centralises all status writes for the bot subprocess."""
from __future__ import annotations

import os
import sys

_RUNTIME_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_RUNTIME_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from model.model_status import ModelStatus  # noqa: E402
import runtime.ipc as ipc                   # noqa: E402


class StatusManager:
    """Owned by the bot instance; call its methods from the match loop."""

    def __init__(self, project_root: str) -> None:
        self._project_root = project_root
        self._goals: int         = 0
        self._concedes: int      = 0
        self._episode: int       = 0
        self._win: int           = 0
        self._total_matches: int = 0

    # ── Counters ─────────────────────────────────────────────────────────────

    def on_goal_scored(self) -> None:
        self._goals += 1

    def on_goal_conceded(self) -> None:
        self._concedes += 1

    def on_episode_end(self, won: bool = False) -> None:
        self._episode += 1
        self._total_matches += 1
        if won:
            self._win += 1

    @property
    def goals(self) -> int:
        return self._goals

    @property
    def concedes(self) -> int:
        return self._concedes

    @property
    def episode(self) -> int:
        return self._episode

    @property
    def win_rate(self) -> float:
        if self._total_matches == 0:
            return 0.0
        return self._win / self._total_matches

    # ── Writers ──────────────────────────────────────────────────────────────

    def write_alive(self) -> None:
        ipc.write_bot_status({"alive": True})

    def write_dead(self) -> None:
        ipc.write_bot_status({"alive": False})

    def write_model_status(
        self,
        current_model: str = "persistent",
        training_state: str = "idle",
        learning_rate: float = 0.001,
        last_reward: float = 0.0,
        active_models: list | None = None,
        active_search: list | None = None,
        team_strategy: str = "balanced",
    ) -> None:
        ModelStatus.update(
            current_model=current_model,
            training_state=training_state,
            learning_rate=learning_rate,
            last_reward=last_reward,
            episodes=self._episode,
            goals_total=self._goals,
            concedes_total=self._concedes,
            win_rate=self.win_rate,
            active_models=active_models,
            active_search=active_search,
            team_strategy=team_strategy,
        )
