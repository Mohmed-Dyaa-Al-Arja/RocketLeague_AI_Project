"""
core/ball_prediction.py
=======================
Physics-based ball trajectory prediction for the Rocket League AI.

Features
--------
- Simulates ball movement for 2-3 seconds into the future (default 60 steps @ ~1/60 s)
- Accounts for gravity, ball velocity, and wall/ceiling/floor bounces
- Returns a list of (x, y, z) predicted positions
- `find_best_intercept()` returns the earliest reachable point along the path

Author: medo dyaa
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

# ── Field dimensions (standard Rocket League) ────────────────────────────────
_FIELD_X   = 4096.0    # half-width  (walls)
_FIELD_Y   = 5120.0    # half-length (back walls / goals)
_FIELD_Z   = 2044.0    # ceiling height
_BALL_R    = 93.0      # ball radius
_GRAVITY   = -650.0    # uu/s²  (game uses ≈ 650 uu/s²)
_DAMPING   = 0.60      # velocity damping on bounce
_FRICTION  = 0.35      # speed reduction along bounced surface (tangential)
_DT        = 1.0 / 60.0  # physics timestep (~60 Hz)

# How fast the bot's car travels (used for intercept feasibility check)
_CAR_MAX_SPEED = 2300.0   # uu/s (ground max speed)


def predict_ball_path(
    ball_pos: Tuple[float, float, float],
    ball_vel: Tuple[float, float, float],
    steps: int = 60,
    dt: float = _DT,
) -> List[Tuple[float, float, float]]:
    """
    Simulate the ball trajectory for `steps` physics ticks.

    Parameters
    ----------
    ball_pos : (x, y, z) current ball position in Rocket League units.
    ball_vel : (vx, vy, vz) current ball velocity.
    steps    : number of simulation steps (default 60 ≈ 1 second at 60 Hz).
    dt       : timestep per step in seconds.

    Returns
    -------
    List of (x, y, z) predicted positions, length == steps.
    """
    x, y, z = float(ball_pos[0]), float(ball_pos[1]), float(ball_pos[2])
    vx, vy, vz = float(ball_vel[0]), float(ball_vel[1]), float(ball_vel[2])

    path: List[Tuple[float, float, float]] = []

    for _ in range(steps):
        # Apply gravity
        vz += _GRAVITY * dt

        # Move ball
        x += vx * dt
        y += vy * dt
        z += vz * dt

        # ── Bounce off floor ──
        if z <= _BALL_R:
            z = _BALL_R
            if vz < 0:
                vz = -vz * _DAMPING
                vx *= (1.0 - _FRICTION)
                vy *= (1.0 - _FRICTION)

        # ── Bounce off ceiling ──
        if z >= _FIELD_Z - _BALL_R:
            z = _FIELD_Z - _BALL_R
            if vz > 0:
                vz = -vz * _DAMPING

        # ── Bounce off side walls (X) ──
        if x > _FIELD_X - _BALL_R:
            x = _FIELD_X - _BALL_R
            if vx > 0:
                vx = -vx * _DAMPING
        elif x < -(_FIELD_X - _BALL_R):
            x = -(_FIELD_X - _BALL_R)
            if vx < 0:
                vx = -vx * _DAMPING

        # ── Bounce off back walls (Y) - ball treated as crossing goal if |y|>5120 ──
        if y > _FIELD_Y - _BALL_R:
            y = _FIELD_Y - _BALL_R
            if vy > 0:
                vy = -vy * _DAMPING
        elif y < -(_FIELD_Y - _BALL_R):
            y = -(_FIELD_Y - _BALL_R)
            if vy < 0:
                vy = -vy * _DAMPING

        path.append((x, y, z))

    return path


def find_best_intercept(
    predicted_path: List[Tuple[float, float, float]],
    car_pos: Tuple[float, float],
    car_speed: float = 1400.0,
    dt: float = _DT,
) -> Optional[Tuple[float, float, float]]:
    """
    Find the earliest point in `predicted_path` that the car can reach
    before (or at the same time as) the ball.

    Parameters
    ----------
    predicted_path : output of predict_ball_path().
    car_pos        : (x, y) current car ground position.
    car_speed      : current car speed (used to estimate travel time).
    dt             : timestep used to generate the path.

    Returns
    -------
    The best (x, y, z) intercept point, or None if no reachable point found.
    """
    if not predicted_path:
        return None

    # Use an average speed estimate: current speed blended toward max
    eff_speed = max(500.0, min(car_speed * 1.1, _CAR_MAX_SPEED))

    for step_idx, (bx, by, bz) in enumerate(predicted_path):
        ball_time = (step_idx + 1) * dt
        car_dist = math.hypot(bx - car_pos[0], by - car_pos[1])
        car_travel_time = car_dist / eff_speed

        if car_travel_time <= ball_time * 1.15:   # 15% margin
            return (bx, by, bz)

    # If nothing is reachable in time, return the closest ground point
    return min(
        predicted_path,
        key=lambda p: math.hypot(p[0] - car_pos[0], p[1] - car_pos[1]),
    )


def detect_open_goal(
    ball_pos: Tuple[float, float],
    opponent_positions: List[Tuple[float, float]],
    opp_goal_y: float,
    goal_half_width: float = 893.0,
    corridor_margin: float = 300.0,
) -> bool:
    """
    Return True when there is a clear shooting corridor to the opponent goal.

    The corridor check: for each opponent, test whether they project into the
    shooting lane (ball_x ± corridor_margin towards goal).

    Parameters
    ----------
    ball_pos          : (x, y) ball position.
    opponent_positions: list of (x, y) for all opponent cars.
    opp_goal_y        : y-coordinate of the opponent goal.
    goal_half_width   : half the width of the goal mouth.
    corridor_margin   : lateral half-width of the shooting corridor.
    """
    bx, by = ball_pos
    goal_x_min = -goal_half_width
    goal_x_max =  goal_half_width
    # The corridor spans from ball_x ± margin toward the goal
    lane_x_min = bx - corridor_margin
    lane_x_max = bx + corridor_margin

    for (ox, oy) in opponent_positions:
        # Is opponent between ball and goal?
        if opp_goal_y > 0:
            between_y = by < oy < opp_goal_y
        else:
            between_y = opp_goal_y < oy < by
        if not between_y:
            continue
        # Is opponent within the shooting corridor?
        if lane_x_min <= ox <= lane_x_max:
            return False   # blocked

    return True


def aim_for_goal_shot(
    ball_pos: Tuple[float, float],
    opp_goal_y: float,
    keeper_x: Optional[float] = None,
    goal_half_width: float = 893.0,
) -> Tuple[float, float]:
    """
    Compute an approach target that places the car behind the ball aimed at
    the centre of the opponent goal (or the far post if keeper is central).

    Returns the push/contact point (x, y) behind the ball.
    """
    bx, by = ball_pos
    goal_aim_x = 0.0

    if keeper_x is not None:
        # Aim at the far post from the keeper
        if keeper_x > 0:
            goal_aim_x = -goal_half_width * 0.7
        else:
            goal_aim_x = goal_half_width * 0.7

    gx, gy = goal_aim_x, opp_goal_y

    # Back-of-ball offset: come from the side opposite the goal-aim angle
    dx = gx - bx
    dy = gy - by
    d  = max(1.0, math.hypot(dx, dy))
    # Stand 200 units behind the ball relative to goal direction
    contact_x = bx - dx / d * 200.0
    contact_y = by - dy / d * 200.0
    return contact_x, contact_y
