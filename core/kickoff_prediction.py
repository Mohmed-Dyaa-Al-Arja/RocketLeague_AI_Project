"""
core/kickoff_prediction.py
===========================
Physics-based ball trajectory prediction during kickoffs and helper
functions to find intercept points.

All calculations are pure Python / math — no external dependencies.
Units: Unreal Engine Units (uu), seconds.

API
---
predict_kickoff_ball_path(ball_pos, ball_vel, steps, dt)
    → List of (x, y, z) positions

estimate_opponent_arrival(opp_pos, ball_pos, opp_speed)
    → float (seconds)

find_kickoff_intercept(ball_pos, ball_vel, car_pos, car_speed, dt)
    → (x, y, z) intercept world position

Author: medo dyaa
"""

from __future__ import annotations

import math
from typing import List, Tuple

# Rocket League physics constants (approximate)
_GRAVITY      = -650.0    # uu/s²
_BOUNCE_DAMP  = 0.60      # fraction of Z velocity retained on ground bounce
_GROUND_Z     = 93.0      # uu — ball rests at this height when on ground
_BALL_RADIUS  = 92.75     # uu

# Field boundaries (uu)
_FIELD_HALF_X = 4096.0
_FIELD_HALF_Y = 5120.0


def predict_kickoff_ball_path(
    ball_pos: Tuple[float, float, float],
    ball_vel: Tuple[float, float, float],
    steps: int = 30,
    dt: float = 1 / 60,
) -> List[Tuple[float, float, float]]:
    """
    Predict ball positions over the next ``steps`` ticks using simple
    projectile physics with ground bounce and wall reflection.

    Parameters
    ----------
    ball_pos : current (x, y, z)
    ball_vel : current velocity (vx, vy, vz)
    steps    : number of ticks to simulate
    dt       : seconds per tick (default 1/60 s)

    Returns
    -------
    List of (x, y, z) predicted positions (length == steps).

    Author: medo dyaa
    """
    x, y, z   = ball_pos
    vx, vy, vz = ball_vel
    path: List[Tuple[float, float, float]] = []

    for _ in range(steps):
        # Apply gravity
        vz += _GRAVITY * dt

        # Integrate position
        x += vx * dt
        y += vy * dt
        z += vz * dt

        # Ground bounce
        if z <= _GROUND_Z:
            z = _GROUND_Z
            if vz < 0:
                vz = -vz * _BOUNCE_DAMP

        # Wall bounces (X)
        if x > _FIELD_HALF_X - _BALL_RADIUS:
            x = _FIELD_HALF_X - _BALL_RADIUS
            vx = -vx
        elif x < -(_FIELD_HALF_X - _BALL_RADIUS):
            x = -(_FIELD_HALF_X - _BALL_RADIUS)
            vx = -vx

        # Wall bounces (Y)
        if y > _FIELD_HALF_Y - _BALL_RADIUS:
            y = _FIELD_HALF_Y - _BALL_RADIUS
            vy = -vy
        elif y < -(_FIELD_HALF_Y - _BALL_RADIUS):
            y = -(_FIELD_HALF_Y - _BALL_RADIUS)
            vy = -vy

        path.append((x, y, z))

    return path


def estimate_opponent_arrival(
    opp_pos: Tuple[float, float, float],
    ball_pos: Tuple[float, float, float],
    opp_speed: float = 1400.0,
) -> float:
    """
    Estimate seconds until an opponent arrives at the ball using a simple
    straight-line distance / speed model.

    Author: medo dyaa
    """
    dist = math.sqrt(
        (opp_pos[0] - ball_pos[0]) ** 2
        + (opp_pos[1] - ball_pos[1]) ** 2
        + (opp_pos[2] - ball_pos[2]) ** 2
    )
    if opp_speed <= 0.0:
        return float("inf")
    return dist / opp_speed


def find_kickoff_intercept(
    ball_pos: Tuple[float, float, float],
    ball_vel: Tuple[float, float, float],
    car_pos:  Tuple[float, float, float],
    car_speed: float = 1400.0,
    dt: float = 1 / 60,
) -> Tuple[float, float, float]:
    """
    Find the earliest predicted ball position that a car travelling at
    ``car_speed`` can reach before the ball moves past that point.

    Parameters
    ----------
    ball_pos  : current ball (x, y, z)
    ball_vel  : current ball velocity
    car_pos   : current car (x, y, z)
    car_speed : car travel speed in uu/s (default ~supersonic kick-off)
    dt        : tick duration in seconds

    Returns
    -------
    (x, y, z) intercept point — falls back to current ball position if no
    intercept found within the simulated window.

    Author: medo dyaa
    """
    predicted = predict_kickoff_ball_path(ball_pos, ball_vel, steps=60, dt=dt)

    for tick, pos in enumerate(predicted):
        time_to_reach_ball = (tick + 1) * dt          # time for ball to get there
        car_travel_dist    = car_speed * time_to_reach_ball
        dist_car_to_point  = math.sqrt(
            (car_pos[0] - pos[0]) ** 2
            + (car_pos[1] - pos[1]) ** 2
            + (car_pos[2] - pos[2]) ** 2
        )
        if dist_car_to_point <= car_travel_dist:
            return pos

    # No intercept found — return earliest predicted position
    if predicted:
        return predicted[0]
    return ball_pos
