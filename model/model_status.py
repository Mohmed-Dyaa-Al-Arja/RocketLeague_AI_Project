"""model/model_status.py — Lightweight write-through model status cache.

The bot process calls ModelStatus.update() each status tick;
the GUI reads it via ModelStatus.read().
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

_MODEL_DIR   = os.path.dirname(os.path.abspath(__file__))
_RUNTIME_DIR = os.path.join(os.path.dirname(_MODEL_DIR), "runtime")
_STATUS_FILE = os.path.join(_RUNTIME_DIR, "_model_status.json")


class ModelStatus:
    """Static helper — no instances needed."""

    @staticmethod
    def update(
        current_model: str = "persistent",
        training_state: str = "idle",
        learning_rate: float = 0.001,
        last_reward: float = 0.0,
        episodes: int = 0,
        goals_total: int = 0,
        concedes_total: int = 0,
        win_rate: float = 0.0,
        active_models: Optional[List[str]] = None,
        active_search: Optional[List[str]] = None,
        team_strategy: str = "balanced",
    ) -> None:
        """Atomically write the current model status to _model_status.json."""
        data: Dict[str, Any] = {
            "current_model":  current_model,
            "training_state": training_state,
            "learning_rate":  learning_rate,
            "last_reward":    last_reward,
            "episodes":       episodes,
            "goals_total":    goals_total,
            "concedes_total": concedes_total,
            "win_rate":       win_rate,
            "active_models":  active_models or [],
            "active_search":  active_search or ["A*"],
            "team_strategy":  team_strategy,
        }
        tmp = _STATUS_FILE + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp, _STATUS_FILE)
        except Exception:
            pass

    @staticmethod
    def read() -> Dict[str, Any]:
        """Return the last written model status, or {} on any error."""
        try:
            with open(_STATUS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
