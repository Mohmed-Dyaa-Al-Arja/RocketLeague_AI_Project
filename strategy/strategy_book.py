"""
strategy/strategy_book.py
==========================
Persistent strategy database organised by arena and situation type.

Each arena stores entries for five strategy categories:
  kickoff, attack, defense, passing, boost_route

Each entry tracks:
  name, description, success_rate, attempts, wins, losses

Usage
-----
    sb = StrategyBook()
    sb.record_outcome("DFHStadium", "attack", "corner_attack", success=True)
    best = sb.best_strategy("DFHStadium", "attack")

Author: medo dyaa
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_THIS_DIR     = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
_STRATEGY_BOOK_FILE = os.path.join(_PROJECT_ROOT, "model", "strategy_book.json")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STRATEGY_TYPES: List[str] = [
    "kickoff",
    "attack",
    "defense",
    "passing",
    "boost_route",
]

STRATEGY_OPTIONS: Dict[str, List[str]] = {
    "kickoff":    ["speed_flip",        "diagonal",          "delayed",           "fake_kickoff"],
    "attack":     ["wall_pass_attack",  "corner_attack",     "direct_shot",       "fake_shot_dribble"],
    "defense":    ["deep_rotation",     "midfield_press",    "aggressive_clear",  "shadow_defense"],
    "passing":    ["quick_wall_pass",   "centre_pass",       "soft_pass",         "direct_play"],
    "boost_route":["speed_boost_route", "corner_boost_route","safe_midboost",     "diagonal_boost"],
}

KNOWN_ARENAS: List[str] = [
    "DFHStadium",
    "Mannfield",
    "ChampionsField",
    "UrbanCentral",
    "BeckwithPark",
    "NeoTokyo",
    "AquaDome",
    "Farmstead",
    "StarbaseArc",
    "WastelandHS",
    "Utopia_Retro",
    "TheSanctuary",
]

# Default descriptions keyed by (strategy_type, strategy_name)
_DESCRIPTIONS: Dict[str, str] = {
    # kickoff
    "speed_flip":         "High-speed diagonal flip toward ball",
    "diagonal":           "Diagonal approach for angle advantage",
    "delayed":            "Wait for opponent to commit, then dash",
    "fake_kickoff":       "Fake direction before attacking ball",
    # attack
    "wall_pass_attack":   "Use side walls for angled shot",
    "corner_attack":      "Attack from corner with back-pass option",
    "direct_shot":        "Fastest route — drive and shoot",
    "fake_shot_dribble":  "Fake shot, dribble into better position",
    # defense
    "deep_rotation":      "Rotate back deep; protect the goal",
    "midfield_press":     "Intercept in midfield before opponent dribbles",
    "aggressive_clear":   "Clear ball aggressively under pressure",
    "shadow_defense":     "Shadow opponent, wait for mistake",
    # passing
    "quick_wall_pass":    "One-touch pass off side wall",
    "centre_pass":        "Pass through the centre to open team-mate",
    "soft_pass":          "Soft controlled pass to maintain possession",
    "direct_play":        "Skip pass — shoot directly",
    # boost_route
    "speed_boost_route":  "Sprint to large boost pad in opponent half",
    "corner_boost_route": "Collect corner boost on the way to attack",
    "safe_midboost":      "Pick up mid-field boost pad safely",
    "diagonal_boost":     "Diagonal route collecting two small boost pads",
}


# ---------------------------------------------------------------------------
# Default entry / book builders
# ---------------------------------------------------------------------------

def _default_entry(name: str, strategy_type: str = "") -> Dict:
    desc = _DESCRIPTIONS.get(name, "")
    return {
        "name":         name,
        "description":  desc,
        "success_rate": 0.50,
        "attempts":     0,
        "wins":         0,
        "losses":       0,
    }


def _build_default_book() -> Dict:
    book: Dict = {}
    for arena in KNOWN_ARENAS:
        book[arena] = {}
        for stype, options in STRATEGY_OPTIONS.items():
            book[arena][stype] = {
                "active":  options[0],
                "entries": {name: _default_entry(name, stype) for name in options},
            }
    return book


# ---------------------------------------------------------------------------
# StrategyBook class
# ---------------------------------------------------------------------------

class StrategyBook:
    """
    Persistent strategy database organised by arena and strategy type.

    Thread-safety note: Not thread-safe; call only from the bot game-loop
    thread or the GUI thread (not both simultaneously).

    Author: medo dyaa
    """

    def __init__(self) -> None:
        self._book: Dict = _load_book()
        self.adaptive_enabled: bool = True

    # ------------------------------------------------------------------
    # Recording outcomes
    # ------------------------------------------------------------------

    def record_outcome(
        self,
        arena: str,
        strategy_type: str,
        strategy_name: str,
        success: bool,
    ) -> None:
        """
        Update success statistics for a given arena / type / strategy name.

        Author: medo dyaa
        """
        self._ensure_arena(arena)
        cat = self._book[arena][strategy_type]
        if strategy_name not in cat["entries"]:
            cat["entries"][strategy_name] = _default_entry(strategy_name, strategy_type)

        entry = cat["entries"][strategy_name]
        entry["attempts"] += 1
        if success:
            entry["wins"] += 1
        else:
            entry["losses"] += 1

        a = entry["attempts"]
        entry["success_rate"] = round(entry["wins"] / a, 4) if a > 0 else 0.50
        self.save()

    # ------------------------------------------------------------------
    # Querying strategies
    # ------------------------------------------------------------------

    def best_strategy(self, arena: str, strategy_type: str) -> str:
        """
        Return the strategy name with the highest adjusted success rate.

        Strategies with fewer than 3 attempts are penalised to avoid
        over-fitting on very small samples (cold-start bias).

        Author: medo dyaa
        """
        self._ensure_arena(arena)
        entries = self._book[arena][strategy_type]["entries"]
        if not entries:
            return STRATEGY_OPTIONS.get(strategy_type, ["default"])[0]

        def _score(item):
            name, e = item
            confidence = min(1.0, e["attempts"] / 3.0)
            return e["success_rate"] * confidence

        sorted_entries = sorted(entries.items(), key=_score, reverse=True)
        return sorted_entries[0][0]

    def get_active_strategy(self, arena: str, strategy_type: str) -> str:
        """Return the currently active strategy name for this arena/type."""
        self._ensure_arena(arena)
        return self._book[arena][strategy_type].get(
            "active",
            STRATEGY_OPTIONS.get(strategy_type, ["default"])[0],
        )

    def set_active_strategy(
        self, arena: str, strategy_type: str, name: str
    ) -> None:
        """Persist a new active strategy for this arena/type."""
        self._ensure_arena(arena)
        self._book[arena][strategy_type]["active"] = name
        self.save()

    def get_arena_summary(self, arena: str) -> Dict:
        """
        Returns a display-ready dict:
          {strategy_type: {active, best, success_rate, attempts, description}}

        Author: medo dyaa
        """
        self._ensure_arena(arena)
        summary: Dict = {}
        for stype in STRATEGY_TYPES:
            cat    = self._book[arena][stype]
            active = cat.get("active", "—")
            best   = self.best_strategy(arena, stype)
            entry  = cat["entries"].get(active, {})
            summary[stype] = {
                "active":       active,
                "best":         best,
                "success_rate": round(entry.get("success_rate", 0.50), 3),
                "attempts":     entry.get("attempts", 0),
                "description":  entry.get("description", ""),
            }
        return summary

    def get_all_arenas(self) -> List[str]:
        return list(self._book.keys())

    def promote_best_strategies(self, arena: str) -> None:
        """
        Set the active strategy to the best-known one for every category.
        Call this before each match.

        Author: medo dyaa
        """
        if not self.adaptive_enabled:
            return
        for stype in STRATEGY_TYPES:
            best = self.best_strategy(arena, stype)
            self.set_active_strategy(arena, stype, best)

    def get_pre_match_selection(self, arena: str) -> Dict[str, str]:
        """Return {strategy_type: active_strategy_name} for all types."""
        self._ensure_arena(arena)
        return {
            stype: self.get_active_strategy(arena, stype)
            for stype in STRATEGY_TYPES
        }

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Wipe all learned data and restore factory defaults."""
        self._book = _build_default_book()
        self.save()

    def save(self) -> None:
        _save_book(self._book)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_arena(self, arena: str) -> None:
        """Make sure the given arena exists in the book with all categories."""
        if arena not in self._book:
            self._book[arena] = {}
        for stype, options in STRATEGY_OPTIONS.items():
            if stype not in self._book[arena]:
                self._book[arena][stype] = {
                    "active":  options[0],
                    "entries": {n: _default_entry(n, stype) for n in options},
                }


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def _load_book() -> Dict:
    if not os.path.exists(_STRATEGY_BOOK_FILE):
        return _build_default_book()
    try:
        with open(_STRATEGY_BOOK_FILE, "r", encoding="utf-8") as fh:
            data: Dict = json.load(fh)
        # Forward-fill: ensure any newly-added arenas/types exist
        default = _build_default_book()
        for arena, atypes in default.items():
            if arena not in data:
                data[arena] = atypes
            else:
                for stype, cat in atypes.items():
                    if stype not in data[arena]:
                        data[arena][stype] = cat
        return data
    except (json.JSONDecodeError, OSError):
        return _build_default_book()


def _save_book(data: Dict) -> None:
    os.makedirs(os.path.dirname(_STRATEGY_BOOK_FILE), exist_ok=True)
    try:
        with open(_STRATEGY_BOOK_FILE, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
    except OSError:
        pass
