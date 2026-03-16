"""
game_logic/team_rotation.py
============================
Team rotation system for 2v2 and 3v3 team play.

Roles
-----
FIRST_MAN   — attacker; currently pressing or shooting
SECOND_MAN  — support; follows up shots, ready for 50/50s
THIRD_MAN   — last defender; stays behind midfield

Rotation logic
--------------
- If this bot attacked last → rotate back to second/third position.
- If a teammate is attacking → move to support rotation.
- If last defender on team → stay between ball and our goal.
- In 1v1 the system is a no-op (always FIRST_MAN).

Author: medo dyaa
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple


class TeamRole:
    FIRST_MAN  = "first_man"   # attacker – on the ball
    SECOND_MAN = "second_man"  # support – ready to challenge
    THIRD_MAN  = "third_man"   # last defender – do not commit


# ── Helper ────────────────────────────────────────────────────────────────────

def _dist2d(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


class TeamRotationManager:
    """
    Determines the appropriate role for THIS bot within a team.

    Usage
    -----
    ::

        rot = TeamRotationManager()
        role, target = rot.update(
            my_pos=(100, -1000),
            ball_pos=(0, 0),
            teammate_positions=[(500, 500)],
            push_dir=1.0,
            own_goal_y=-5120,
        )
    """

    def __init__(self) -> None:
        self._my_role: str = TeamRole.FIRST_MAN
        self._last_attacker_dist: float = float("inf")

    # ── Main update ───────────────────────────────────────────────────────────

    def update(
        self,
        my_pos: Tuple[float, float],
        ball_pos: Tuple[float, float],
        teammate_positions: List[Tuple[float, float]],
        push_dir: float,
        own_goal_y: float,
    ) -> Tuple[str, Tuple[float, float]]:
        """
        Decide role and return a recommended target position.

        Parameters
        ----------
        my_pos              : (x, y) this bot's position.
        ball_pos            : (x, y) ball position.
        teammate_positions  : list of (x, y) for all team-mates (excluding self).
        push_dir            : +1 for team-0, -1 for team-1.
        own_goal_y          : y-coordinate of our goal.

        Returns
        -------
        (role, target_pos)
        """
        if not teammate_positions:
            # 1v1 — always first man
            self._my_role = TeamRole.FIRST_MAN
            return TeamRole.FIRST_MAN, ball_pos

        my_dist   = _dist2d(my_pos, ball_pos)
        tm_dists  = [(_dist2d(t, ball_pos), t) for t in teammate_positions]
        tm_dists.sort(key=lambda x: x[0])

        closest_tm_dist, closest_tm_pos = tm_dists[0]

        # ── Assign roles based on distance ordering ───────────────────────
        # The player nearest the ball is the attacker.
        if my_dist <= closest_tm_dist:
            # We are closest → FIRST MAN
            role = TeamRole.FIRST_MAN
        elif len(teammate_positions) >= 2:
            # Multiple teammates: determine whether we are second or third
            second_tm_dist = tm_dists[1][0]
            role = TeamRole.SECOND_MAN if my_dist < second_tm_dist else TeamRole.THIRD_MAN
        else:
            role = TeamRole.SECOND_MAN

        self._my_role = role
        target = self._recommend_target(
            role, my_pos, ball_pos, closest_tm_pos, push_dir, own_goal_y,
        )
        return role, target

    def _recommend_target(
        self,
        role: str,
        my_pos: Tuple[float, float],
        ball_pos: Tuple[float, float],
        attacker_pos: Tuple[float, float],
        push_dir: float,
        own_goal_y: float,
    ) -> Tuple[float, float]:
        bx, by = ball_pos

        if role == TeamRole.FIRST_MAN:
            # Attacker: go straight for the ball
            return ball_pos

        if role == TeamRole.SECOND_MAN:
            # Support: position slightly behind and to the side of the ball
            # so we can follow up or challenge on a 50/50
            side = 1.0 if my_pos[0] < bx else -1.0
            tx = bx + side * 600.0
            ty = by - push_dir * 800.0   # slightly behind the ball
            return (tx, ty)

        if role == TeamRole.THIRD_MAN:
            # Last defender: stay between ball and our goal
            # Position at ~30% of the way from our goal to the ball
            gx = 0.0
            gy = own_goal_y
            tx = gx + (bx - gx) * 0.3
            ty = gy + (by - gy) * 0.3
            return (tx, ty)

        return ball_pos   # fallback

    # ── Accessors ─────────────────────────────────────────────────────────────

    @property
    def role(self) -> str:
        return self._my_role

    def should_attack(self) -> bool:
        return self._my_role == TeamRole.FIRST_MAN

    def should_support(self) -> bool:
        return self._my_role == TeamRole.SECOND_MAN

    def should_defend(self) -> bool:
        return self._my_role == TeamRole.THIRD_MAN

    def summary(self) -> Dict[str, object]:
        return {"role": self._my_role}
