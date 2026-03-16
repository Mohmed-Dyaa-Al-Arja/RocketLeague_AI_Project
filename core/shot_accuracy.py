"""
core/shot_accuracy.py
======================
Evaluates the quality of a potential shot and recommends the best aim
point, accounting for defenders blocking the goal.

Constants
---------
_GOAL_HALF_WIDTH  — half-width of the Rocket League goal mouth (uu)
_GOAL_HEIGHT      — height of the goal mouth (uu)

Decisions
---------
SHOOT_CENTER  — clear central shot available
AIM_LEFT_POST — aim for the left post (from attacker's perspective)
AIM_RIGHT_POST — aim for the right post
CHIP          — elevated shot to clear or chip over a low block
WAIT          — no good angle; hold possession

Author: medo dyaa
"""

from __future__ import annotations

import math
from typing import List, Tuple


class ShotDecision:
    SHOOT_CENTER   = "shoot_center"
    AIM_LEFT_POST  = "aim_left_post"
    AIM_RIGHT_POST = "aim_right_post"
    CHIP           = "chip"
    WAIT           = "wait"


_GOAL_HALF_WIDTH = 893.0    # uu — official Rocket League goal half-width
_GOAL_HEIGHT     = 642.8    # uu — goal mouth height
_MIN_SHOT_ANGLE  = 5.0      # degrees — if open angle < this, don't shoot
_DEFENDER_RADIUS = 250.0    # uu — sphere around a defending car


def _dist2d(
    a: Tuple[float, float],
    b: Tuple[float, float],
) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _ray_blocked(
    origin: Tuple[float, float],
    target: Tuple[float, float],
    blockers: List[Tuple[float, float]],
    block_radius: float,
) -> bool:
    """
    Returns True if any blocker circle intersects the ray from origin to target.

    Author: medo dyaa
    """
    ox, oy = origin
    tx, ty = target
    dx, dy = tx - ox, ty - oy
    length = math.hypot(dx, dy)
    if length < 1.0:
        return False
    ux, uy = dx / length, dy / length

    for bx, by in blockers:
        ex, ey = bx - ox, by - oy
        proj = ex * ux + ey * uy
        if proj < 0.0 or proj > length:
            continue
        perp = abs(ex * uy - ey * ux)
        if perp < block_radius:
            return True
    return False


class ShotAccuracyEvaluator:
    """
    Stateless evaluator — call ``evaluate()`` when a shot decision is needed.

    Author: medo dyaa
    """

    def evaluate(
        self,
        ball_pos: Tuple[float, float, float],
        car_pos:  Tuple[float, float, float],
        opp_goal_y: float,
        defender_positions: List[Tuple[float, float]],
    ) -> str:
        """
        Parameters
        ----------
        ball_pos           : (x, y, z)
        car_pos            : (x, y, z)
        opp_goal_y         : Y coordinate of the opponent's goal line
        defender_positions : list of (x, y) defending car positions

        Returns
        -------
        ShotDecision constant string.

        Author: medo dyaa
        """
        angle = self.open_goal_angle(ball_pos, opp_goal_y, defender_positions)

        if angle < _MIN_SHOT_ANGLE:
            return ShotDecision.WAIT

        # Try centre first
        origin  = (ball_pos[0], ball_pos[1])
        centre  = (0.0, opp_goal_y)
        if not _ray_blocked(origin, centre, defender_positions, _DEFENDER_RADIUS):
            return ShotDecision.SHOOT_CENTER

        # Try posts
        left_post  = (-_GOAL_HALF_WIDTH, opp_goal_y)
        right_post = ( _GOAL_HALF_WIDTH, opp_goal_y)

        left_clear  = not _ray_blocked(origin, left_post,  defender_positions, _DEFENDER_RADIUS)
        right_clear = not _ray_blocked(origin, right_post, defender_positions, _DEFENDER_RADIUS)

        if left_clear and right_clear:
            # Pick the post with the wider open angle
            angle_left  = _angle_to_target(origin, left_post)
            angle_right = _angle_to_target(origin, right_post)
            return ShotDecision.AIM_LEFT_POST if angle_left > angle_right else ShotDecision.AIM_RIGHT_POST
        if left_clear:
            return ShotDecision.AIM_LEFT_POST
        if right_clear:
            return ShotDecision.AIM_RIGHT_POST

        # Defenders are covering all ground lanes — try a chip
        if ball_pos[2] < 100.0:   # only chip if ball is on/near the ground
            return ShotDecision.CHIP

        return ShotDecision.WAIT

    def open_goal_angle(
        self,
        ball_pos: Tuple[float, float, float],
        opp_goal_y: float,
        defender_positions: List[Tuple[float, float]],
    ) -> float:
        """
        Compute the visible angular width of the goal mouth from the ball
        position (degrees), taking defenders as blockers into account.

        Uses a simple triangle area method:
          angle = 2 * arctan(half_width / distance)

        Author: medo dyaa
        """
        bx, by = ball_pos[0], ball_pos[1]
        dist_to_goal = math.hypot(bx, by - opp_goal_y)

        if dist_to_goal < 1.0:
            return 180.0

        raw_angle = 2.0 * math.degrees(
            math.atan(_GOAL_HALF_WIDTH / dist_to_goal)
        )

        # Subtract angle blocked by each defender (rough estimate)
        blocked_angle = 0.0
        origin = (bx, by)
        for dpos in defender_positions:
            d_dist = _dist2d(origin, dpos)
            if d_dist < 1.0:
                continue
            block = math.degrees(math.atan2(_DEFENDER_RADIUS, d_dist)) * 2.0
            blocked_angle += block

        return max(0.0, raw_angle - blocked_angle)

    def best_aim_point(
        self,
        ball_pos: Tuple[float, float, float],
        opp_goal_y: float,
        defender_positions: List[Tuple[float, float]],
    ) -> Tuple[float, float]:
        """
        Returns the (x, y) world coordinate of the best aim point in the
        goal mouth, considering defenders.

        Author: medo dyaa
        """
        decision = self.evaluate(ball_pos, (0.0, 0.0, 0.0), opp_goal_y, defender_positions)

        if decision == ShotDecision.AIM_LEFT_POST:
            return (-_GOAL_HALF_WIDTH * 0.8, opp_goal_y)
        if decision == ShotDecision.AIM_RIGHT_POST:
            return ( _GOAL_HALF_WIDTH * 0.8, opp_goal_y)

        # Centre / chip / default
        return (0.0, opp_goal_y)


def _angle_to_target(
    origin: Tuple[float, float],
    target: Tuple[float, float],
) -> float:
    """Angle in degrees from positive-X axis to target (unused in ranking but available)."""
    return math.degrees(math.atan2(target[1] - origin[1], target[0] - origin[0]))
