"""
learning/match_learning.py
===========================
Records per-match metrics to ``model/match_history.json`` and derives
simple strategy recommendations from match history trends.

Recommendations produced
------------------------
* kickoff_strategy  — which KickoffStrategy string to favour
* passing_threshold — float multiplier (>1 = pass more / <1 = pass less)
* rotation_aggression — float multiplier for how aggressively to rotate

All maths uses only the standard library.

Author: medo dyaa
"""

from __future__ import annotations

import json
import math
import os
from typing import Dict, List

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

_THIS_DIR    = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
_HISTORY_FILE = os.path.join(_PROJECT_ROOT, "model", "match_history.json")

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULT_HISTORY: Dict = {
    "matches": [],
    "totals": {
        "goals_scored":      0,
        "goals_conceded":    0,
        "kickoff_successes": 0,
        "shot_successes":    0,
        "matches_played":    0,
    },
}

# ---------------------------------------------------------------------------
# Module
# ---------------------------------------------------------------------------


class MatchLearning:
    """
    Persists match history and exposes simple learning recommendations.

    Author: medo dyaa
    """

    def __init__(self) -> None:
        self._data: Dict = _load_history()

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_match(
        self,
        goals_scored:      int,
        goals_conceded:    int,
        kickoff_successes: int,
        shot_successes:    int,
        possession_pct:    float,
    ) -> None:
        """
        Append one match record and update running totals.

        Parameters
        ----------
        goals_scored      : goals we scored
        goals_conceded    : goals against
        kickoff_successes : number of kickoffs where we got first touch
        shot_successes    : shots on target that resulted in a goal
        possession_pct    : 0.0–1.0 fraction of ticks we "had the ball"

        Author: medo dyaa
        """
        entry = {
            "goals_scored":      goals_scored,
            "goals_conceded":    goals_conceded,
            "kickoff_successes": kickoff_successes,
            "shot_successes":    shot_successes,
            "possession_pct":    round(possession_pct, 4),
        }
        self._data["matches"].append(entry)

        t = self._data["totals"]
        t["goals_scored"]      += goals_scored
        t["goals_conceded"]    += goals_conceded
        t["kickoff_successes"] += kickoff_successes
        t["shot_successes"]    += shot_successes
        t["matches_played"]    += 1

        _save_history(self._data)

    # ------------------------------------------------------------------
    # Recommendations
    # ------------------------------------------------------------------

    def get_kickoff_strategy_recommendation(self) -> str:
        """
        Returns a KickoffStrategy constant string based on win-rate patterns.

        * If win-rate > 60% → stick with Speed Flip (aggressive)
        * If kickoff success rate < 40% → try Diagonal
        * Otherwise → Speed Flip (default)

        Author: medo dyaa
        """
        n = self._data["totals"]["matches_played"]
        if n == 0:
            return "speed_flip"

        ks_rate = (
            self._data["totals"]["kickoff_successes"] / max(n, 1)
        )

        scored    = self._data["totals"]["goals_scored"]
        conceded  = self._data["totals"]["goals_conceded"]
        win_rate  = scored / max(scored + conceded, 1)

        if win_rate > 0.60:
            return "speed_flip"
        if ks_rate < 0.40:
            return "diagonal"
        return "speed_flip"

    def get_passing_threshold_adjustment(self) -> float:
        """
        Returns a multiplier (0.5–2.0) for how readily the bot should pass.

        * High possession but low goals → increase passing (>1.0)
        * Low possession but ok goals  → decrease passing (<1.0)

        Author: medo dyaa
        """
        recent = self._recent_matches(10)
        if not recent:
            return 1.0

        avg_poss  = sum(m["possession_pct"] for m in recent) / len(recent)
        avg_goals = sum(m["goals_scored"]   for m in recent) / len(recent)

        # High possession + low scoring → pass more to break down defence
        if avg_poss > 0.55 and avg_goals < 1.5:
            return 1.4
        # Low possession → hold ball more, pass less
        if avg_poss < 0.35:
            return 0.7
        return 1.0

    def get_rotation_aggression(self) -> float:
        """
        Returns a rotation aggression multiplier (0.6–1.4).

        * Conceding a lot relative to goals → more defensive (< 1)
        * Scoring a lot → more aggressive rotation (> 1)

        Author: medo dyaa
        """
        recent = self._recent_matches(10)
        if not recent:
            return 1.0

        avg_scored   = sum(m["goals_scored"]   for m in recent) / len(recent)
        avg_conceded = sum(m["goals_conceded"]  for m in recent) / len(recent)

        ratio = avg_scored / max(avg_conceded, 0.5)
        # Clamp between 0.6 and 1.4
        return max(0.6, min(1.4, ratio))

    def summary(self) -> dict:
        """Returns a copy of the totals dict."""
        return dict(self._data["totals"])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _recent_matches(self, n: int) -> List[Dict]:
        return self._data["matches"][-n:]


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def _load_history() -> Dict:
    if not os.path.exists(_HISTORY_FILE):
        return dict(
            matches=[],
            totals=dict(
                goals_scored=0,
                goals_conceded=0,
                kickoff_successes=0,
                shot_successes=0,
                matches_played=0,
            ),
        )
    try:
        with open(_HISTORY_FILE, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return dict(
            matches=[],
            totals=dict(
                goals_scored=0,
                goals_conceded=0,
                kickoff_successes=0,
                shot_successes=0,
                matches_played=0,
            ),
        )


def _save_history(data: Dict) -> None:
    os.makedirs(os.path.dirname(_HISTORY_FILE), exist_ok=True)
    try:
        with open(_HISTORY_FILE, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
    except OSError:
        pass  # Non-critical — learning data is optional
