"""
game_logic/passing_ai.py
=========================
Evaluates whether the bot should shoot, pass, or dribble and computes
an appropriate pass target when a pass is preferred.

Decision hierarchy
------------------
1. If the bot has a clear shooting lane → SHOOT_CENTER
2. If a teammate is better positioned and open → PASS to teammate
3. If the ball is on the wall / high → PASS_WALL chip
4. If the bot needs to keep possession safely → DRIBBLE
5. Fall-back → SHOOT_CENTER anyway

Author: medo dyaa
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple


class PassDecision:
    SHOOT_CENTER = "shoot_center"
    PASS_CENTER  = "pass_center"
    PASS_WALL    = "pass_wall"
    PASS_SOFT    = "pass_soft"
    DRIBBLE      = "dribble"


def _dist3d(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)


def _dist2d(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _has_clear_shot(
    ball_pos: Tuple[float, float, float],
    opp_goal_y: float,
    defender_positions: List[Tuple[float, float]],
    goal_half_width: float = 893.0,
) -> bool:
    """
    Very cheap ray-vs-circle check: returns True if no defender blocks the
    straight-line path from ball to goal centre.

    Author: medo dyaa
    """
    gx, gy = 0.0, opp_goal_y
    bx, by = ball_pos[0], ball_pos[1]
    dx, dy = gx - bx, gy - by
    length = math.hypot(dx, dy)
    if length < 1.0:
        return True
    ux, uy = dx / length, dy / length

    block_radius = 200.0  # uu, rough car blocking radius

    for dpos in defender_positions:
        # project defender onto the ray
        ex, ey = dpos[0] - bx, dpos[1] - by
        proj = ex * ux + ey * uy
        if proj < 0 or proj > length:
            continue
        perp = abs(ex * uy - ey * ux)
        if perp < block_radius:
            return False

    return True


def _best_teammate(
    ball_pos: Tuple[float, float, float],
    car_pos: Tuple[float, float, float],
    teammate_positions: List[Tuple[float, float]],
    opp_goal_y: float,
) -> Optional[int]:
    """
    Returns index of the teammate that is:
      a) closer to the opponent goal than we are, AND
      b) open (more than 400 uu from any passing obstruction heuristic)

    Author: medo dyaa
    """
    my_goal_dist = abs(car_pos[1] - opp_goal_y)
    best_idx: Optional[int] = None
    best_dist = my_goal_dist - 50.0  # must be meaningfully better

    for idx, tpos in enumerate(teammate_positions):
        t_goal_dist = abs(tpos[1] - opp_goal_y)
        if t_goal_dist < best_dist:
            best_dist = t_goal_dist
            best_idx = idx

    return best_idx


class PassingAI:
    """
    Stateless evaluator — call ``evaluate()`` every tick (or when ball is
    close and decision is needed).

    Author: medo dyaa
    """

    _WALL_HEIGHT_THRESH = 100.0   # uu — ball is "on the wall" above this Z
    _OPP_PRESSURE_DIST  = 500.0   # uu — opponent is "close" if within this

    def evaluate(
        self,
        ball_pos: Tuple[float, float, float],
        car_pos: Tuple[float, float, float],
        car_vel: Tuple[float, float, float],
        teammate_positions: List[Tuple[float, float]],
        opp_positions: List[Tuple[float, float]],
        opp_goal_y: float,
    ) -> str:
        """
        Returns a PassDecision constant.

        Parameters
        ----------
        ball_pos           : (x, y, z) ball world position
        car_pos            : (x, y, z) own car world position
        car_vel            : (x, y, z) own car velocity
        teammate_positions : list of (x, y) for each team-mate (can be empty)
        opp_positions      : list of (x, y) for each opponent
        opp_goal_y         : Y coordinate of the opponent's goal line (+/-5120)

        Author: medo dyaa
        """
        # Wall pass if ball is elevated
        if ball_pos[2] > self._WALL_HEIGHT_THRESH:
            return PassDecision.PASS_WALL

        # Under heavy pressure — soft pass / dribble
        under_pressure = any(
            _dist2d(op, (car_pos[0], car_pos[1])) < self._OPP_PRESSURE_DIST
            for op in opp_positions
        )

        if under_pressure and teammate_positions:
            tm_idx = _best_teammate(ball_pos, car_pos, teammate_positions, opp_goal_y)
            if tm_idx is not None:
                return PassDecision.PASS_CENTER
            return PassDecision.PASS_SOFT

        # Clear shot available
        if _has_clear_shot(ball_pos, opp_goal_y, opp_positions):
            return PassDecision.SHOOT_CENTER

        # Teammate is better placed
        if teammate_positions:
            tm_idx = _best_teammate(ball_pos, car_pos, teammate_positions, opp_goal_y)
            if tm_idx is not None:
                return PassDecision.PASS_CENTER

        # Default — try to dribble
        return PassDecision.DRIBBLE

    def get_pass_target(
        self,
        decision: str,
        teammate_positions: List[Tuple[float, float]],
        opp_goal_y: float,
    ) -> Tuple[float, float]:
        """
        Returns world (x, y) pass target for PASS_* decisions.

        Author: medo dyaa
        """
        goal_sign = 1.0 if opp_goal_y > 0 else -1.0

        if decision == PassDecision.PASS_WALL:
            # Aim for the far wall corner near the goal
            return (900.0 * goal_sign, opp_goal_y * 0.7)

        if decision in (PassDecision.PASS_CENTER, PassDecision.PASS_SOFT):
            if teammate_positions:
                best_tm = min(
                    teammate_positions,
                    key=lambda tp: abs(tp[1] - opp_goal_y),
                )
                return best_tm
            # No team-mates — aim centre
            return (0.0, opp_goal_y * 0.5)

        # Shoot / dribble — aim at goal centre
        return (0.0, opp_goal_y)
