"""
core/reward_calculator.py — Structured reward system inspired by RLBot Championship bots.

Event rewards (one-shot, via signal_* methods):
  goal_scored             = +120    assist                  = +60
  shot_on_target          = +25     pass_success / passing  = +15
  save                    = +40     clear_ball              = +20
  own_goal                = -150    goal_conceded           = -80
  missed_open_shot        = -20     double_commit           = -8
  collision_with_teammate = -6      passing_play            = +10
  assist_chain            = +15     team_rotation_correct   = +6
  demo_basic              = +15     demo_plus_ball          = +25
  demo_defend             = +30

Per-tick continuous rewards (computed in compute_reward()):
  ball_touch (dist < 150)           ≈ +0.033 / tick  (~+2/s)
  ball_near_enemy_goal              ≈ +0.083 / tick  (~+5/s)
  dribble_control (streak ≥ 10)    ≈ +0.10  / tick  (~+6/s)
  ball_intercept (closing fast)     = +10 one-shot per closure
  good_offensive_position          ≈ +0.067 / tick  (~+4/s)
  good_defensive_position          ≈ +0.067 / tick  (~+4/s)
  goal_side_positioning            ≈ +0.10  / tick  (~+6/s)
  velocity_toward_enemy_goal        = up to +0.5 / tick at max speed
  backward_velocity_penalty         = -0.5 / tick
  idle_penalty                      = -0.03 / tick  (constant, prevents passivity)
  boost_wasted                     ≈ -0.083 / tick  when hoarding boost far from ball
  boost_no_pressure                ≈ -0.050 / tick  when burning boost with no ball pressure
  bad_positioning (over-commit)    ≈ -0.167 / tick

Final reward is clipped to [-200, +200] per tick.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple


# ── Event reward constants ────────────────────────────────────────────────────

_R_GOAL_SCORED        = 120.0
_R_ASSIST             = 60.0
_R_SHOT_ON_TARGET     = 25.0
_R_PASS_SUCCESS       = 15.0
_R_SAVE               = 40.0
_R_CLEAR              = 20.0
_R_DEMO_BASIC         = 15.0
_R_DEMO_BALL          = 25.0
_R_DEMO_DEFEND        = 30.0
_R_PASSING_PLAY       = 10.0
_R_ASSIST_CHAIN       = 15.0
_R_TEAM_ROTATION      = 6.0

_P_OWN_GOAL           = -150.0
_P_GOAL_CONCEDED      = -80.0
_P_MISSED_SHOT        = -20.0
_P_DOUBLE_COMMIT      = -8.0
_P_COLLISION          = -6.0

# ── Per-tick continuous reward rates ─────────────────────────────────────────
# Values are per-tick at 60 Hz; comments show approximate per-second equivalent.

_R_BALL_TOUCH_TICK      = 0.033    # ~+2 / s when in contact radius
_R_NEAR_ENEMY_GOAL_TICK = 0.083    # ~+5 / s when controlling ball near opponent goal
_R_DRIBBLE_TICK         = 0.10     # ~+6 / s for sustained dribble (≥ 10 consecutive ticks)
_R_INTERCEPT_ONESHOT    = 10.0     # once per closure event (was far, now <150 u)

_R_OFFENSE_POS_TICK     = 0.067    # ~+4 / s when in good attacking position
_R_DEFENSE_POS_TICK     = 0.067    # ~+4 / s when in good defensive position
_R_GOAL_SIDE_TICK       = 0.10     # ~+6 / s when goal-side of ball while defending

_R_MOMENTUM_FWD_MAX     = 0.5      # max per tick for full forward velocity toward enemy goal
_P_MOMENTUM_BWD         = -0.5     # per tick when driving backward significantly

_P_IDLE_TICK            = -0.03    # constant small penalty every tick (prevents idling)
_P_BOOST_WASTED_TICK    = -0.083   # ~-5 / s when hoarding boost far from ball
_P_BOOST_NO_PRESSURE    = -0.050   # ~-3 / s when burning boost without ball pressure
_P_BAD_POSITIONING_TICK = -0.167   # ~-10 / s for over-committing (leaving goal exposed)

# Normalization bounds
_REWARD_MIN = -200.0
_REWARD_MAX = +200.0


class RewardCalculator:
    """Calculates reward signals from game state changes.

    All one-off events are queued via signal_*() methods and flushed inside
    compute_reward() on the next frame, so every RL model sees them correctly
    regardless of tick timing.
    """

    def __init__(self):
        self._prev_dist_to_ball: float = 9999.0
        self._prev_ball_y_rel: float = 0.0
        self._prev_speed: float = 0.0
        self._prev_boost: float = 0.0
        self._prev_car_pos: Tuple[float, float] = (0.0, 0.0)

        # Pending one-shot event rewards
        self._pending: float = 0.0

        # Episode stats accumulator
        self._stats: Dict[str, float] = {
            "goal":        0.0,
            "assist":      0.0,
            "shot_target": 0.0,
            "pass":        0.0,
            "save":        0.0,
            "clear":       0.0,
            "demo_basic":  0.0,
            "demo_ball":   0.0,
            "demo_defend": 0.0,
            "idle":        0.0,
            "camping":     0.0,
            "boost_waste": 0.0,
            "continuous":  0.0,
            "ball_control":0.0,
            "positioning": 0.0,
            "momentum":    0.0,
            "penalty":     0.0,
            "team_play":   0.0,
        }
        self._episode_total: float = 0.0

        # Idle / camping detection state
        self._idle_timer: int = 0
        self._idle_threshold: int = 120     # ~2 s at 60 Hz
        self._camping_timer: int = 0

        # Ball touch / dribble tracking
        self._ball_touch_frames: int = 0    # consecutive ticks within touch radius
        self._intercept_fired: bool = False # prevent repeated intercept rewards

    # ── One-shot event signals ──────────────────────────────────────────────

    def signal_goal_scored(self) -> None:
        """Bot scored a goal: +120."""
        self._queue(_R_GOAL_SCORED, "goal")

    def signal_assist(self) -> None:
        """Bot assisted a goal: +60."""
        self._queue(_R_ASSIST, "assist")

    def signal_shot_on_target(self) -> None:
        """Bot fired a shot on target: +25."""
        self._queue(_R_SHOT_ON_TARGET, "shot_target")

    def signal_pass_success(self) -> None:
        """Successful pass to teammate: +15."""
        self._queue(_R_PASS_SUCCESS, "pass")

    def signal_save(self) -> None:
        """Bot made a save: +40."""
        self._queue(_R_SAVE, "save")

    def signal_clear_ball(self) -> None:
        """Bot cleared the ball from danger: +20."""
        self._queue(_R_CLEAR, "clear")

    def signal_demo_basic(self) -> None:
        """Destroyed opponent: +15."""
        self._queue(_R_DEMO_BASIC, "demo_basic")

    def signal_demo_plus_ball(self) -> None:
        """Destroyed opponent and gained ball: +25."""
        self._queue(_R_DEMO_BALL, "demo_ball")

    def signal_demo_defender_near_goal(self) -> None:
        """Destroyed a defender near their goal: +30."""
        self._queue(_R_DEMO_DEFEND, "demo_defend")

    def signal_passing_play(self) -> None:
        """Coordinated passing play: +10."""
        self._queue(_R_PASSING_PLAY, "team_play")

    def signal_assist_chain(self) -> None:
        """Assist chain combination: +15."""
        self._queue(_R_ASSIST_CHAIN, "team_play")

    def signal_team_rotation_correct(self) -> None:
        """Correct team rotation: +6."""
        self._queue(_R_TEAM_ROTATION, "team_play")

    def signal_own_goal(self) -> None:
        """Bot scored in own goal: -150."""
        self._queue(_P_OWN_GOAL, "penalty")

    def signal_missed_open_shot(self) -> None:
        """Missed a clear scoring opportunity: -20."""
        self._queue(_P_MISSED_SHOT, "penalty")

    def signal_double_commit_with_teammate(self) -> None:
        """Both bot and teammate chasing the same ball: -8."""
        self._queue(_P_DOUBLE_COMMIT, "penalty")

    def signal_collision_with_teammate(self) -> None:
        """Collision with a teammate: -6."""
        self._queue(_P_COLLISION, "penalty")

    # ── Backward-compatible aliases ─────────────────────────────────────────

    def signal_goal_conceded(self) -> None:
        """Opponent scored: -80.  Stats bucket kept as 'idle' for compatibility."""
        self._queue(_P_GOAL_CONCEDED, "idle")

    def signal_save_made(self) -> None:
        """Alias for signal_save()."""
        self.signal_save()

    def signal_demo(self) -> None:
        """Alias for signal_demo_basic()."""
        self.signal_demo_basic()

    # ── Per-frame reward ────────────────────────────────────────────────────

    def compute_reward(
        self,
        car_pos: Tuple[float, float],
        ball_pos: Tuple[float, float],
        ball_vel: Tuple[float, float],
        opp_pos: Tuple[float, float],
        car_speed: float,
        car_boost: float,
        push_dir: float,          # +1.0 if team==0 (attacking +Y), -1.0 otherwise
        situation: str,
        car_vel: Tuple[float, float] = (0.0, 0.0),
    ) -> float:
        """Compute a single-step reward from game observables.

        ``car_vel`` is optional; if omitted the velocity is inferred from the
        position delta since the previous frame (accurate enough for shaping).
        """
        reward: float = 0.0

        # ── Spatial helpers ──────────────────────────────────────────────────
        dist_to_ball = math.hypot(ball_pos[0] - car_pos[0],
                                  ball_pos[1] - car_pos[1])
        ball_y_rel = ball_pos[1] * push_dir   # positive = near enemy goal
        car_y_rel  = car_pos[1] * push_dir

        # Infer car velocity from position delta if caller didn't supply it
        if car_vel == (0.0, 0.0):
            inferred_vx = (car_pos[0] - self._prev_car_pos[0]) * 60.0
            inferred_vy = (car_pos[1] - self._prev_car_pos[1]) * 60.0
            car_vel = (inferred_vx, inferred_vy)

        # ── Constant idle penalty (prevents passive play) ────────────────────
        reward += _P_IDLE_TICK
        self._stats["idle"] += _P_IDLE_TICK

        # ── Ball approach reward / retreat penalty ───────────────────────────
        dist_delta = self._prev_dist_to_ball - dist_to_ball
        if dist_delta > 0:
            proximity_bonus = 1.0 + 4.0 * max(0.0, 1.0 - dist_to_ball / 5000.0)
            approach_r = dist_delta * 0.002 * proximity_bonus
            reward += approach_r
            self._stats["continuous"] += approach_r
        else:
            retreat_p = dist_delta * 0.002 * 3.0   # negative value
            reward += retreat_p
            self._stats["continuous"] += retreat_p

        # ── Ball progress toward enemy goal ──────────────────────────────────
        ball_progress = ball_y_rel - self._prev_ball_y_rel
        influence = (1.0 if dist_to_ball < 1400
                     else 0.4 if dist_to_ball < 2500
                     else 0.1)
        ball_goal_r = ball_progress * 0.001 * 3.0 * influence
        reward += ball_goal_r
        self._stats["continuous"] += ball_goal_r

        # ── Ball threatening own goal ─────────────────────────────────────────
        if ball_progress < 0 and ball_y_rel < -2000 and ball_vel[1] * push_dir < -300:
            ball_danger_p = -3.0 * 0.002
            reward += ball_danger_p
            self._stats["continuous"] += ball_danger_p

        # ── Ball touch / dribble / intercept ────────────────────────────────
        _TOUCH_RADIUS = 150.0
        if dist_to_ball < _TOUCH_RADIUS:
            # Basic ball touch
            reward += _R_BALL_TOUCH_TICK
            self._stats["ball_control"] += _R_BALL_TOUCH_TICK

            # Ball control near enemy goal
            if ball_y_rel > 2500:
                reward += _R_NEAR_ENEMY_GOAL_TICK
                self._stats["ball_control"] += _R_NEAR_ENEMY_GOAL_TICK

            # Dribble: sustained ball contact
            self._ball_touch_frames += 1
            if self._ball_touch_frames >= 10:
                reward += _R_DRIBBLE_TICK
                self._stats["ball_control"] += _R_DRIBBLE_TICK

            # Ball intercept: closing fast from distance
            if self._prev_dist_to_ball > 500 and not self._intercept_fired:
                reward += _R_INTERCEPT_ONESHOT
                self._stats["ball_control"] += _R_INTERCEPT_ONESHOT
                self._intercept_fired = True
        else:
            self._ball_touch_frames = 0
            self._intercept_fired = False

        # ── Positioning rewards ──────────────────────────────────────────────
        if situation == "defending" or ball_y_rel < -1000:
            # Defensive positioning: stay between ball and own goal
            if car_y_rel < ball_y_rel:
                reward += _R_DEFENSE_POS_TICK
                self._stats["positioning"] += _R_DEFENSE_POS_TICK
                # Extra goal-side bonus when well back
                if car_y_rel < -1000:
                    reward += _R_GOAL_SIDE_TICK
                    self._stats["positioning"] += _R_GOAL_SIDE_TICK
        elif ball_y_rel > 500:
            # Offensive positioning: stay ahead of ball toward enemy goal
            if car_y_rel > ball_y_rel:
                reward += _R_OFFENSE_POS_TICK
                self._stats["positioning"] += _R_OFFENSE_POS_TICK

        # Bad positioning: over-committing (car deep in enemy half, ball in ours)
        if car_y_rel > 3000 and ball_y_rel < -2000:
            reward += _P_BAD_POSITIONING_TICK
            self._stats["positioning"] += _P_BAD_POSITIONING_TICK

        # ── Possession advantage ─────────────────────────────────────────────
        opp_dist = math.hypot(ball_pos[0] - opp_pos[0], ball_pos[1] - opp_pos[1])
        if dist_to_ball < opp_dist:
            reward += 0.002
            self._stats["continuous"] += 0.002

        # ── Momentum rewards ────────────────────────────────────────────────
        car_vel_y_rel = car_vel[1] * push_dir
        if car_vel_y_rel > 0:
            # Forward momentum toward enemy goal
            fwd_r = min(_R_MOMENTUM_FWD_MAX,
                        car_vel_y_rel / 2300.0 * _R_MOMENTUM_FWD_MAX)
            reward += fwd_r
            self._stats["momentum"] += fwd_r
        elif car_vel_y_rel < -200:
            # Driving backward — discourage backing away from play
            reward += _P_MOMENTUM_BWD
            self._stats["momentum"] += _P_MOMENTUM_BWD

        # ── Boost management penalties ────────────────────────────────────────
        boost_delta = car_boost - self._prev_boost
        # Hoarding boost while far from play
        if car_boost > 60 and dist_to_ball > 2000 and car_speed < 800:
            reward += _P_BOOST_WASTED_TICK
            self._stats["boost_waste"] += _P_BOOST_WASTED_TICK
        # Using boost when no ball pressure
        if boost_delta < -5 and dist_to_ball > 1500:
            reward += _P_BOOST_NO_PRESSURE
            self._stats["boost_waste"] += _P_BOOST_NO_PRESSURE

        # ── Speed bonus ──────────────────────────────────────────────────────
        if car_speed > 900:
            reward += 0.001
            self._stats["continuous"] += 0.001

        # ── Stationary idle penalty (extra on top of constant idle) ─────────
        if car_speed < 100:
            self._idle_timer += 1
        else:
            self._idle_timer = 0

        if self._idle_timer >= self._idle_threshold:
            stationary_p = -5.0 * 0.002
            reward += stationary_p
            self._stats["idle"] += stationary_p
            self._idle_timer = self._idle_threshold  # cap

        # ── Camping own goal ─────────────────────────────────────────────────
        if car_y_rel < -2500 and dist_to_ball > 2000 and situation != "defending":
            self._camping_timer += 1
        else:
            self._camping_timer = 0

        if self._camping_timer >= 90:
            camping_p = -10.0 * 0.002
            reward += camping_p
            self._stats["camping"] += camping_p

        # ── Flush pending one-shot events ────────────────────────────────────
        if self._pending != 0.0:
            reward += self._pending
            self._pending = 0.0

        # ── Normalize to stable training range ───────────────────────────────
        reward = max(_REWARD_MIN, min(_REWARD_MAX, reward))

        # ── Update tracking state ─────────────────────────────────────────────
        self._prev_dist_to_ball = dist_to_ball
        self._prev_ball_y_rel   = ball_y_rel
        self._prev_speed        = car_speed
        self._prev_boost        = car_boost
        self._prev_car_pos      = car_pos
        self._episode_total    += reward

        return reward

    # ── Episode stats ───────────────────────────────────────────────────────

    @property
    def episode_reward(self) -> float:
        """Total accumulated reward this episode."""
        return self._episode_total

    def get_episode_stats(self) -> Dict[str, float]:
        """Return a snapshot of accumulated reward by category."""
        return dict(self._stats, total=self._episode_total)

    def reset_episode(self) -> None:
        """Call at the start of a new episode to clear accumulators."""
        for k in self._stats:
            self._stats[k] = 0.0
        self._episode_total = 0.0
        self._idle_timer = 0
        self._camping_timer = 0
        self._ball_touch_frames = 0
        self._intercept_fired = False
        self._prev_dist_to_ball = 9999.0
        self._prev_ball_y_rel = 0.0
        self._prev_car_pos = (0.0, 0.0)

    def reset(self) -> None:
        """Alias for reset_episode() — called by match restart commands."""
        self.reset_episode()

    # ── Internal helper ─────────────────────────────────────────────────────

    def _queue(self, amount: float, category: str) -> None:
        self._pending += amount
        self._stats[category] = self._stats.get(category, 0.0) + amount

