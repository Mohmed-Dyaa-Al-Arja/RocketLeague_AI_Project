from __future__ import annotations

import datetime as _dt
import json
import math
import os
import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from navigation.path_planner import PathPlanner, PathResult
from game_logic.mode_manager import MODE_ATTACK, MODE_BALANCED, MODE_DEFENSE
from core.all_algorithms import build_decision_tree, evaluate_tree, predict_trajectory, find_intercept, ball_landing_pos
from core.adaptive_learner import AdaptiveLearner

# ── Game Modes (identifiers, auto-detected from game) ──
GM_SOCCER = "soccer"
GM_HOOPS = "hoops"
GM_SNOW = "snow"
GM_DROPSHOT = "dropshot"
GM_RUMBLE = "rumble"
GM_HEATSEEKER = "heatseeker"
GM_GRIDIRON = "gridiron"
GM_KNOCKOUT = "knockout"
GM_SPIKE_RUSH = "spike_rush"
GM_BOOMER = "boomer"
GM_BEACH_BALL = "beach_ball"
GM_GHOST_HUNT = "ghost_hunt"
GM_SPRING_LOADED = "spring_loaded"
GM_SUPER_CUBE = "super_cube"
GM_ROCKET_LABS = "rocket_labs"
GM_CUSTOM_MUTATOR = "custom_mutator"
GM_TRAINING = "training"
GM_TOURNAMENT = "tournament"
GM_DROPSHOT_RUMBLE = "dropshot_rumble"
GM_SPEED_DEMON = "speed_demon"
GM_WINTER_BREAKAWAY = "winter_breakaway"
GM_SPLIT_SHOT = "split_shot"
GM_TACTICAL_RUMBLE = "tactical_rumble"
GM_GOTHAM_RUMBLE = "gotham_rumble"

# Sets for easy category checks
_SOCCER_VARIANTS = {GM_SOCCER, GM_BOOMER, GM_BEACH_BALL, GM_SPEED_DEMON,
                    GM_ROCKET_LABS, GM_CUSTOM_MUTATOR, GM_TRAINING, GM_TOURNAMENT}
_RUMBLE_VARIANTS = {GM_RUMBLE, GM_DROPSHOT_RUMBLE, GM_TACTICAL_RUMBLE, GM_GOTHAM_RUMBLE}
_SPIKE_VARIANTS = {GM_SPIKE_RUSH, GM_GRIDIRON}
_SPLIT_VARIANTS = {GM_SPLIT_SHOT}
_CUBE_VARIANTS = {GM_SUPER_CUBE}

_SEARCH_ALGORITHMS = [
    "A*",
    "BFS",
    "UCS",
    "DFS",
    "Greedy",
    "Decision Tree",
    "Beam Search",
    "IDA*",
]


# ══════════════════════════════════════════════════════════════════
#  30 Professional Game States
# ══════════════════════════════════════════════════════════════════

class GameState:
    """30 professional Rocket League game states."""
    KICKOFF           = "Kickoff"
    ATTACK            = "Attack"
    DEFENSE           = "Defense"
    COUNTER_ATTACK    = "CounterAttack"
    POSSESSION        = "Possession"
    BALL_CHASE        = "BallChase"
    RECOVERY          = "Recovery"
    GOAL_DEFENSE      = "GoalDefense"
    GOAL_ATTACK       = "GoalAttack"
    BOOST_COLLECTION  = "BoostCollection"
    AERIAL_ATTACK     = "AerialAttack"
    AERIAL_DEFENSE    = "AerialDefense"
    WALL_PLAY         = "WallPlay"
    CORNER_PLAY       = "CornerPlay"
    MIDFIELD_CONTROL  = "MidfieldControl"
    LAST_DEFENDER     = "LastDefender"
    TEAM_ROTATION     = "TeamRotation"
    PASSING_OPPORTUNITY = "PassingOpportunity"
    SHOT_OPPORTUNITY  = "ShotOpportunity"
    DEMO_OPPORTUNITY  = "DemoOpportunity"
    OPPONENT_PRESSURE = "OpponentPressure"
    BALL_PREDICTION   = "BallPrediction"
    EMERGENCY_DEFENSE = "EmergencyDefense"
    OPEN_GOAL         = "OpenGoal"
    CLEAR_BALL        = "ClearBall"
    SHADOW_DEFENSE    = "ShadowDefense"
    BOOST_STARVATION  = "BoostStarvation"
    FAST_BREAK        = "FastBreak"
    BALL_CONTROL      = "BallControl"
    STRATEGIC_PAUSE   = "StrategicPause"


# State → recommended (search_algo, rl_model, strategy_tag)
_STATE_STRATEGY_MAP: dict = {
    GameState.KICKOFF:            ("Greedy",        "q_learning",    "attack"),
    GameState.ATTACK:             ("A*",            "dqn",           "attack"),
    GameState.DEFENSE:            ("UCS",           "actor_critic",  "defense"),
    GameState.COUNTER_ATTACK:     ("Beam Search",   "ppo",           "counter"),
    GameState.POSSESSION:         ("A*",            "q_learning",    "possession"),
    GameState.BALL_CHASE:         ("Greedy",        "online_learner","attack"),
    GameState.RECOVERY:           ("BFS",           "actor_critic",  "balanced"),
    GameState.GOAL_DEFENSE:       ("UCS",           "actor_critic",  "defense"),
    GameState.GOAL_ATTACK:        ("A*",            "dqn",           "attack"),
    GameState.BOOST_COLLECTION:   ("BFS",           "online_learner","boost"),
    GameState.AERIAL_ATTACK:      ("IDA*",          "dqn",           "attack"),
    GameState.AERIAL_DEFENSE:     ("IDA*",          "actor_critic",  "defense"),
    GameState.WALL_PLAY:          ("A*",            "model_based",   "balanced"),
    GameState.CORNER_PLAY:        ("A*",            "dqn",           "attack"),
    GameState.MIDFIELD_CONTROL:   ("A*",            "q_learning",    "balanced"),
    GameState.LAST_DEFENDER:      ("UCS",           "actor_critic",  "defense"),
    GameState.TEAM_ROTATION:      ("BFS",           "q_learning",    "balanced"),
    GameState.PASSING_OPPORTUNITY:("A*",            "ppo",           "possession"),
    GameState.SHOT_OPPORTUNITY:   ("Greedy",        "dqn",           "attack"),
    GameState.DEMO_OPPORTUNITY:   ("Greedy",        "model_based",   "demo"),
    GameState.OPPONENT_PRESSURE:  ("UCS",           "actor_critic",  "defense"),
    GameState.BALL_PREDICTION:    ("IDA*",          "model_based",   "balanced"),
    GameState.EMERGENCY_DEFENSE:  ("UCS",           "actor_critic",  "defense"),
    GameState.OPEN_GOAL:          ("Greedy",        "dqn",           "attack"),
    GameState.CLEAR_BALL:         ("Greedy",        "actor_critic",  "defense"),
    GameState.SHADOW_DEFENSE:     ("UCS",           "q_learning",    "defense"),
    GameState.BOOST_STARVATION:   ("BFS",           "online_learner","boost"),
    GameState.FAST_BREAK:         ("Beam Search",   "ppo",           "attack"),
    GameState.BALL_CONTROL:       ("A*",            "q_learning",    "possession"),
    GameState.STRATEGIC_PAUSE:    ("A*",            "actor_critic",  "balanced"),
}


def detect_game_state(
    car_pos: tuple,
    ball_pos: tuple,
    opp_pos: tuple,
    boost: float,
    score_diff: int,
    time_remaining: float,
    ball_vel: tuple = (0.0, 0.0),
    car_vel: tuple = (0.0, 0.0),
    ball_z: float = 0.0,
    push_dir: float = 1.0,
    car_on_ground: bool = True,
    opp_demolished: bool = False,
    has_teammates: bool = False,
    my_team: int = 0,
) -> str:
    """Classify the current match situation into one of 30 GameState values.

    Parameters
    ----------
    push_dir : +1.0 if team==0 (attacks toward +Y), -1.0 otherwise
    """
    import math as _math

    car_x, car_y = car_pos[0], car_pos[1]
    ball_x, ball_y = ball_pos[0], ball_pos[1]
    opp_x, opp_y  = opp_pos[0], opp_pos[1]
    bvx, bvy      = (ball_vel[0], ball_vel[1]) if len(ball_vel) >= 2 else (0.0, 0.0)
    cvx, cvy      = (car_vel[0], car_vel[1]) if len(car_vel) >= 2 else (0.0, 0.0)

    dist_car_ball = _math.hypot(ball_x - car_x, ball_y - car_y)
    dist_opp_ball = _math.hypot(ball_x - opp_x, ball_y - opp_y)

    ball_y_rel  = ball_y * push_dir    # positive = opp side
    car_y_rel   = car_y * push_dir
    bvy_rel     = bvy * push_dir       # positive = toward opp goal
    opp_y_rel   = opp_y * push_dir

    car_speed = _math.hypot(cvx, cvy)
    ball_speed = _math.hypot(bvx, bvy)

    # ── Kickoff ──────────────────────────────────────────────────────────────
    if abs(ball_x) < 150 and abs(ball_y) < 150 and ball_speed < 50:
        return GameState.KICKOFF

    # ── Emergency defense: ball flying toward our goal fast ──────────────────
    if ball_y_rel < -3000 and bvy_rel < -600:
        return GameState.EMERGENCY_DEFENSE

    # ── Open goal: opp goalkeeper is far, ball in opp half ───────────────────
    if ball_y_rel > 2000 and opp_y_rel > 0 and dist_opp_ball > 3500 and boost > 30:
        return GameState.OPEN_GOAL

    # ── Demo opportunity: fast car, close to opponent, enough boost ───────────
    if dist_car_ball < 2500 and dist_opp_ball < 1200 and car_speed > 1500 and boost > 40:
        return GameState.DEMO_OPPORTUNITY

    # ── Aerial states ─────────────────────────────────────────────────────────
    if ball_z > 500:
        if ball_y_rel < 0:
            return GameState.AERIAL_DEFENSE
        return GameState.AERIAL_ATTACK

    # ── Wall play: ball near wall ─────────────────────────────────────────────
    if abs(ball_x) > 3600 or abs(ball_y) > 4600:
        return GameState.WALL_PLAY

    # ── Corner play ───────────────────────────────────────────────────────────
    if abs(ball_x) > 2800 and abs(ball_y_rel) > 3500:
        return GameState.CORNER_PLAY

    # ── Boost starvation ──────────────────────────────────────────────────────
    if boost < 15:
        return GameState.BOOST_STARVATION

    # ── Boost collection: not in danger but low on boost ─────────────────────
    if boost < 35 and ball_y_rel > -1000:
        return GameState.BOOST_COLLECTION

    # ── Recovery: car in air and not close to ball ────────────────────────────
    if not car_on_ground and dist_car_ball > 2000:
        return GameState.RECOVERY

    # ── Goal defense: ball near our goal ─────────────────────────────────────
    if ball_y_rel < -3000 and dist_car_ball < 2000:
        return GameState.GOAL_DEFENSE

    # ── Last defender: we are the only car between ball and own goal ──────────
    if car_y_rel < ball_y_rel < opp_y_rel and ball_y_rel < -1000:
        return GameState.LAST_DEFENDER

    # ── Clear ball: ball in our half, we're close ────────────────────────────
    if ball_y_rel < -500 and dist_car_ball < 1200:
        return GameState.CLEAR_BALL

    # ── Shadow defense: opponent has ball in their half, we track ────────────
    if opp_y_rel > 1000 and dist_opp_ball < dist_car_ball and ball_y_rel > 0:
        return GameState.SHADOW_DEFENSE

    # ── Opponent pressure: opp close to ball in our half ─────────────────────
    if ball_y_rel < 0 and dist_opp_ball < 800:
        return GameState.OPPONENT_PRESSURE

    # ── Defense: ball in our half and opp close ───────────────────────────────
    if ball_y_rel < -500:
        return GameState.DEFENSE

    # ── Fast break: opp is behind ball and we have boost ─────────────────────
    if opp_y_rel < ball_y_rel and boost > 50 and car_speed > 1200:
        return GameState.FAST_BREAK

    # ── Counter attack transition ─────────────────────────────────────────────
    if opp_y_rel < ball_y_rel + 1000 and bvy_rel > 200:
        return GameState.COUNTER_ATTACK

    # ── Shot opportunity: close to ball in opp half ───────────────────────────
    if ball_y_rel > 2000 and dist_car_ball < 1500:
        return GameState.SHOT_OPPORTUNITY

    # ── Goal attack: car close to opp goal ────────────────────────────────────
    if car_y_rel > 3500 and dist_car_ball < 2000:
        return GameState.GOAL_ATTACK

    # ── Passing opportunity: teammate present ─────────────────────────────────
    if has_teammates and dist_car_ball < 1500 and ball_y_rel > 0:
        return GameState.PASSING_OPPORTUNITY

    # ── Possession: we're closest to ball ─────────────────────────────────────
    if dist_car_ball < dist_opp_ball and dist_car_ball < 1200:
        return GameState.POSSESSION

    # ── Ball control: we're very close to ball ────────────────────────────────
    if dist_car_ball < 500:
        return GameState.BALL_CONTROL

    # ── Ball prediction: ball fast, we're planning intercept ─────────────────
    if ball_speed > 800 and dist_car_ball > 1500:
        return GameState.BALL_PREDICTION

    # ── Ball chase: ball in opp half, we're chasing ──────────────────────────
    if ball_y_rel > 0 and dist_car_ball > 1000:
        return GameState.BALL_CHASE

    # ── Attack: default offensive state ──────────────────────────────────────
    if ball_y_rel > 0:
        return GameState.ATTACK

    # ── Team rotation ─────────────────────────────────────────────────────────
    if has_teammates:
        return GameState.TEAM_ROTATION

    # ── Midfield control ──────────────────────────────────────────────────────
    if abs(ball_y_rel) < 1500:
        return GameState.MIDFIELD_CONTROL

    # ── Strategic pause: nothing urgent ──────────────────────────────────────
    return GameState.STRATEGIC_PAUSE


# ══════════════════════════════════════════════════════════════════
#  Mechanics Skill System — reward-based learning for each mechanic
# ══════════════════════════════════════════════════════════════════

_DEFAULT_MECHANICS: Dict[str, float] = {
    "car_control": 1.0,
    "single_jump": 1.0,
    "double_jump": 0.5,
    "front_flip": 0.8,
    "side_flip": 0.3,
    "half_flip": 0.1,
    "speed_flip": 0.1,
    "aerial": 0.1,
    "dribble": 0.1,
    "power_shot": 0.3,
    "goal_save": 0.5,
    "fast_recovery": 0.2,
    "air_dribble": 0.0,
    "flip_reset": 0.0,
    "wave_dash": 0.1,
    "boost_management": 0.5,
    "demo_play": 0.3,
}

# Skill threshold: bot only attempts a mechanic if its confidence >= threshold
_SKILL_THRESHOLDS: Dict[str, float] = {
    "double_jump": 0.4,
    "front_flip": 0.3,
    "side_flip": 0.5,
    "half_flip": 0.6,
    "speed_flip": 0.5,
    "aerial": 0.4,
    "dribble": 0.5,
    "power_shot": 0.3,
    "air_dribble": 0.7,
    "flip_reset": 0.9,
    "wave_dash": 0.6,
    "demo_play": 0.3,
}


@dataclass
class DecisionOutput:
    algorithm: str
    target: Tuple[float, float]
    path_result: PathResult
    situation: str = "free_ball"


#  Game Settings — detected from packet or defaulted

@dataclass
class GameSettings:
    game_mode: str = GM_SOCCER
    gravity: float = -650.0
    ball_weight: float = 1.0
    ball_speed: float = 1.0
    ball_size: float = 1.0
    bounce: float = 1.0
    ground_friction: float = 1.0
    steer_mult: float = 1.0
    throttle_mult: float = 1.0
    boost_aggression: float = 1.0
    # Field dimensions (populated from FieldInfo if available)
    field_w: float = 4096.0              # half-width of field
    field_l: float = 5120.0              # half-length of field
    goal_depth: float = 880.0
    ball_radius: float = 92.75
    # Goal positions (set from FieldInfo at init; defaults for Soccer)
    opp_goal_y_blue: float = 5120.0    # where blue team scores
    opp_goal_y_orange: float = -5120.0 # where orange team scores
    goal_height: float = 643.0
    goal_width: float = 893.0
    ceiling_z: float = 2044.0
    # Mode-specific tuning
    aerial_bias: float = 0.0            # extra willingness to fly (hoops +)
    ground_hit_bias: float = 0.0        # dropshot: hit ball on their floor
    max_aerial_z: float = 400.0         # default intercept max Z
    puck_mode: bool = False             # snow day: ball never flies high
    # Kickoff spawn (detected dynamically from car position)
    kickoff_spawn: str = "unknown"

    def adapt(self):
        if self.ground_friction < 0.7:
            self.steer_mult = 0.6
            self.throttle_mult = 0.7
            self.boost_aggression = 0.7
        elif self.ground_friction < 0.9:
            self.steer_mult = 0.8
            self.throttle_mult = 0.85
            self.boost_aggression = 0.85
        else:
            self.steer_mult = 1.0
            self.throttle_mult = 1.0
            self.boost_aggression = 1.0
        if self.gravity > -450:
            self.boost_aggression *= 0.8

    # Score / overtime awareness
    we_are_trailing: bool = True
    is_overtime: bool = True
    score_diff: int = 1  # positive = we lead, negative = trailing

    # Team play
    num_teammates: int = 0       # excluding self
    teammate_positions: list = None  # list of (x,y)

    # Dropshot tile info
    damaged_tiles: list = None   # list of (index, tile_state) for opponent half
    car_surface: str = "ground"
    ball_surface: str = "ground"
    car_surface_time: float = 0.0
    ball_surface_time: float = 0.0
    car_surface_max_speed: float = 0.0
    ball_surface_max_speed: float = 0.0
    _last_car_surface: str = field(default="ground", repr=False)
    _last_ball_surface: str = field(default="ground", repr=False)
    _car_surface_start: float = field(default=0.0, repr=False)
    _ball_surface_start: float = field(default=0.0, repr=False)

    def __post_init__(self):
        if self.teammate_positions is None:
            self.teammate_positions = []
        if self.damaged_tiles is None:
            self.damaged_tiles = []

    def apply_mode(self):
        """Tune physics biases per game mode."""
        m = self.game_mode
        if m == GM_HOOPS:
            self.aerial_bias = 0.5
            self.max_aerial_z = 900.0
        elif m == GM_DROPSHOT:
            self.ground_hit_bias = 0.6
            self.max_aerial_z = 600.0
        elif m in (GM_SNOW, GM_WINTER_BREAKAWAY):
            self.puck_mode = True
            self.max_aerial_z = 200.0
            self.bounce = 0.4
        elif m == GM_HEATSEEKER:
            self.max_aerial_z = 600.0
        elif m in _RUMBLE_VARIANTS:
            self.boost_aggression = 1.2
        elif m in (GM_GRIDIRON, GM_SPIKE_RUSH):
            self.max_aerial_z = 300.0
            self.boost_aggression = 1.3
        elif m == GM_KNOCKOUT:
            self.boost_aggression = 1.5
            self.max_aerial_z = 600.0
        elif m == GM_BOOMER:
            self.boost_aggression = 1.0
            self.max_aerial_z = 500.0
        elif m == GM_BEACH_BALL:
            self.aerial_bias = 0.3
            self.max_aerial_z = 700.0
        elif m == GM_SPEED_DEMON:
            self.boost_aggression = 1.4
        elif m == GM_SPRING_LOADED:
            self.aerial_bias = 0.4
            self.max_aerial_z = 800.0
        elif m in _CUBE_VARIANTS:
            self.bounce = 0.6
            self.max_aerial_z = 400.0
        elif m == GM_DROPSHOT_RUMBLE:
            self.ground_hit_bias = 0.6
            self.max_aerial_z = 600.0
            self.boost_aggression = 1.2
        elif m in _SPLIT_VARIANTS:
            self.boost_aggression = 0.9

    def get_opp_goal_y(self, my_team: int) -> float:
        return self.opp_goal_y_blue if my_team == 0 else self.opp_goal_y_orange

    def get_own_goal_y(self, my_team: int) -> float:
        return self.opp_goal_y_orange if my_team == 0 else self.opp_goal_y_blue

    def populate_from_field_info(self, field_info):
        """Auto-fill settings from FieldInfo provided by the game."""
        # Goal positions and dimensions
        for i in range(field_info.num_goals):
            g = field_info.goals[i]
            team = g.team_num
            y = g.location.y
            h = g.height
            w = g.width
            if team == 0:  # blue
                self.opp_goal_y_orange = y
            else:  # orange
                self.opp_goal_y_blue = y
            self.goal_height = max(self.goal_height, h)
            self.goal_width = max(self.goal_width, w)
        # Deduce field length from goal positions
        if abs(self.opp_goal_y_blue) > 100:
            self.field_l = abs(self.opp_goal_y_blue)

    def detect_kickoff_spawn(self, car_pos, my_team: int) -> str:
        """Detect kickoff spawn position from actual car coordinates."""
        flip = -1.0 if my_team == 0 else 1.0
        x = car_pos[0]
        y = car_pos[1] * flip
        abs_x = abs(x)

        if abs_x > 1500 and 2000 < y < 3200:
            self.kickoff_spawn = "diagonal_right" if x > 0 else "diagonal_left"
        elif abs_x < 500 and y > 4000:
            self.kickoff_spawn = "far_back"
        elif y > 3000:
            self.kickoff_spawn = "back_right" if x > 0 else "back_left"
        else:
            self.kickoff_spawn = "unknown"
        return self.kickoff_spawn

    def detect_car_surface(self, car_pos: Tuple[float, float, float], car_on_ground: bool) -> str:
        x, y, z = car_pos
        if car_on_ground:
            if z > self.ceiling_z - 180.0:
                return "ceiling"
            if abs(x) > self.field_w - 180.0 or abs(y) > self.field_l - 180.0:
                return "wall"
            return "ground"
        return "air"

    def detect_ball_surface(self, ball_pos: Tuple[float, float, float]) -> str:
        x, y, z = ball_pos
        if z <= self.ball_radius + 15.0:
            return "ground"
        if z >= self.ceiling_z - self.ball_radius - 20.0:
            return "ceiling"
        if abs(x) >= self.field_w - self.ball_radius - 25.0 or abs(y) >= self.field_l - self.ball_radius - 25.0:
            return "wall"
        return "air"

    def update_surface_state(self,
                             now_seconds: float,
                             car_pos: Tuple[float, float, float],
                             car_speed: float,
                             car_on_ground: bool,
                             ball_pos: Tuple[float, float, float],
                             ball_speed: float):
        car_surface = self.detect_car_surface(car_pos, car_on_ground)
        ball_surface = self.detect_ball_surface(ball_pos)

        if car_surface != self._last_car_surface:
            self._car_surface_start = now_seconds
            self._last_car_surface = car_surface
            self.car_surface_max_speed = 0.0
        if ball_surface != self._last_ball_surface:
            self._ball_surface_start = now_seconds
            self._last_ball_surface = ball_surface
            self.ball_surface_max_speed = 0.0

        self.car_surface = car_surface
        self.ball_surface = ball_surface
        self.car_surface_time = max(0.0, now_seconds - self._car_surface_start)
        self.ball_surface_time = max(0.0, now_seconds - self._ball_surface_start)
        self.car_surface_max_speed = max(self.car_surface_max_speed, car_speed)
        self.ball_surface_max_speed = max(self.ball_surface_max_speed, ball_speed)


#  Knowledge Store

class KnowledgeStore:
    def __init__(self, knowledge_path: Optional[str], persist: bool = True):
        self.knowledge_path = knowledge_path
        self.persist = bool(persist and knowledge_path)
        self.data: Dict = {
            "navigation_patterns": {},
            "successful_strategies": {"attack": 0, "defense": 0},
            "goal_paths": [],
            "algorithm_success": {
                "A*": 0, "BFS": 0, "UCS": 0, "Greedy": 0,
                "DFS": 0, "Decision Tree": 0, "Beam Search": 0, "IDA*": 0,
            },
            "human_play": [],
            "total_ticks": 0,
            "goals_conceded": 0,
            "demos_given": 0,
            "saves_made": 0,
            "possession_lost": 0,
            "mechanics_skill": dict(_DEFAULT_MECHANICS),
            "opponent_patterns": [],
        }
        self.load()

    def load(self):
        if not self.persist or not self.knowledge_path:
            return
        # Also load adaptive RL data
        rl_path = self._rl_data_path()
        if rl_path and os.path.exists(rl_path):
            try:
                with open(rl_path, "r", encoding="utf-8") as f:
                    self._rl_data_cache = json.load(f)
            except Exception:
                self._rl_data_cache = {}
        else:
            self._rl_data_cache = {}
        if os.path.exists(self.knowledge_path):
            try:
                with open(self.knowledge_path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict):
                    self.data.update(loaded)
                    for key in ("human_play", "goal_paths", "opponent_patterns"):
                        if key not in self.data:
                            self.data[key] = []
                    algs = self.data.setdefault("algorithm_success", {})
                    for a in ("A*", "BFS", "UCS", "Greedy", "DFS",
                              "Decision Tree", "Beam Search", "IDA*"):
                        algs.setdefault(a, 0)
                    for k in ("total_ticks", "goals_conceded", "demos_given",
                              "saves_made", "possession_lost"):
                        self.data.setdefault(k, 0)
                    ms = self.data.setdefault("mechanics_skill", {})
                    for mk, mv in _DEFAULT_MECHANICS.items():
                        ms.setdefault(mk, mv)
            except Exception:
                pass

    def _rl_data_path(self) -> Optional[str]:
        if not self.knowledge_path:
            return None
        base = os.path.dirname(self.knowledge_path)
        return os.path.join(base, "adaptive_rl_data.json")

    def save_rl_data(self, rl_data: Dict):
        """Save adaptive RL learning data."""
        if not self.persist or not self.knowledge_path:
            return
        rl_path = self._rl_data_path()
        if not rl_path:
            return
        os.makedirs(os.path.dirname(rl_path), exist_ok=True)
        tmp_path = rl_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(rl_data, f, indent=2)
        os.replace(tmp_path, rl_path)

    def load_rl_data(self) -> Dict:
        """Load adaptive RL learning data."""
        return getattr(self, '_rl_data_cache', {})

    def save(self):
        if not self.persist or not self.knowledge_path:
            return
        os.makedirs(os.path.dirname(self.knowledge_path), exist_ok=True)
        tmp_path = self.knowledge_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2)
        os.replace(tmp_path, self.knowledge_path)

    def save_checkpoint(self, tag: str = "auto", keep_last: int = 25):
        if not self.persist or not self.knowledge_path:
            return
        base_dir = os.path.dirname(self.knowledge_path)
        ckpt_dir = os.path.join(base_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)

        safe_tag = "".join(c for c in tag.strip().lower() if c.isalnum() or c in ("-", "_"))
        if not safe_tag:
            safe_tag = "auto"

        ts = _dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        ckpt_path = os.path.join(ckpt_dir, f"search_knowledge_{ts}_{safe_tag}.json")
        with open(ckpt_path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2)

        try:
            files = [
                os.path.join(ckpt_dir, name)
                for name in os.listdir(ckpt_dir)
                if name.startswith("search_knowledge_") and name.endswith(".json")
            ]
            files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            for p in files[keep_last:]:
                try:
                    os.remove(p)
                except Exception:
                    pass
        except Exception:
            pass

    def tick(self):
        self.data["total_ticks"] = self.data.get("total_ticks", 0) + 1

    def record_node(self, node: Tuple[int, int]):
        key = f"{node[0]}:{node[1]}"
        nav = self.data["navigation_patterns"]
        nav[key] = int(nav.get(key, 0)) + 1

    def record_goal_path(self, path: List[Tuple[float, float]], mode: str, algorithm: str):
        entries = self.data["goal_paths"]
        entries.append({"mode": mode, "algorithm": algorithm, "path": path[-20:]})
        self.data["goal_paths"] = entries[-200:]
        strats = self.data["successful_strategies"]
        strats[mode] = strats.get(mode, 0) + 1
        algs = self.data["algorithm_success"]
        if algorithm in algs:
            algs[algorithm] += 2

    def punish_conceded_goal(self, algorithm: str):
        self.data["goals_conceded"] = self.data.get("goals_conceded", 0) + 1
        algs = self.data.get("algorithm_success", {})
        if algorithm in algs:
            algs[algorithm] = max(0, algs[algorithm] - 3)
        strats = self.data.get("successful_strategies", {})
        strats["defense"] = max(0, strats.get("defense", 0) - 2)
        self._adjust_skill("goal_save", -0.03)

    def reward_demo(self, algorithm: str):
        self.data["demos_given"] = self.data.get("demos_given", 0) + 1
        algs = self.data.get("algorithm_success", {})
        if algorithm in algs:
            algs[algorithm] += 4
        strats = self.data.get("successful_strategies", {})
        strats["attack"] = strats.get("attack", 0) + 2
        self._adjust_skill("demo_play", 0.05)

    def reward_save(self, algorithm: str):
        self.data["saves_made"] = self.data.get("saves_made", 0) + 1
        algs = self.data.get("algorithm_success", {})
        if algorithm in algs:
            algs[algorithm] += 3
        strats = self.data.get("successful_strategies", {})
        strats["defense"] = strats.get("defense", 0) + 2
        self._adjust_skill("goal_save", 0.04)

    def punish_possession_lost(self, algorithm: str):
        self.data["possession_lost"] = self.data.get("possession_lost", 0) + 1
        algs = self.data.get("algorithm_success", {})
        if algorithm in algs:
            algs[algorithm] = max(0, algs[algorithm] - 1)

    def reward_mechanic(self, mechanic: str, amount: float = 0.02):
        self._adjust_skill(mechanic, amount)

    def punish_mechanic(self, mechanic: str, amount: float = 0.015):
        self._adjust_skill(mechanic, -amount)

    def _adjust_skill(self, mechanic: str, delta: float):
        ms = self.data.setdefault("mechanics_skill", {})
        old = ms.get(mechanic, _DEFAULT_MECHANICS.get(mechanic, 0.0))
        ms[mechanic] = max(0.0, min(1.0, old + delta))

    def skill(self, mechanic: str) -> float:
        return self.data.get("mechanics_skill", {}).get(
            mechanic, _DEFAULT_MECHANICS.get(mechanic, 0.0))

    def can_do(self, mechanic: str) -> bool:
        threshold = _SKILL_THRESHOLDS.get(mechanic, 0.0)
        return self.skill(mechanic) >= threshold

    def record_opponent_state(self, opp_pos: Tuple[float, float],
                              opp_vel: Tuple[float, float],
                              ball_pos: Tuple[float, float]):
        entries = self.data.setdefault("opponent_patterns", [])
        entries.append({
            "op": [round(opp_pos[0], 0), round(opp_pos[1], 0)],
            "ov": [round(opp_vel[0], 0), round(opp_vel[1], 0)],
            "b": [round(ball_pos[0], 0), round(ball_pos[1], 0)],
        })
        self.data["opponent_patterns"] = entries[-600:]

    def predict_opponent_target(self, opp_pos: Tuple[float, float],
                                opp_vel: Tuple[float, float],
                                ball_pos: Tuple[float, float]) -> Optional[Tuple[float, float]]:
        patterns = self.data.get("opponent_patterns", [])
        if len(patterns) < 30:
            return None
        best_dist = float("inf")
        best_idx = -1
        for idx, p in enumerate(patterns[:-1]):
            d = (math.hypot(opp_pos[0] - p["op"][0], opp_pos[1] - p["op"][1])
                 + math.hypot(ball_pos[0] - p["b"][0], ball_pos[1] - p["b"][1]) * 0.5)
            if d < best_dist:
                best_dist = d
                best_idx = idx
        if best_dist > 2500 or best_idx < 0:
            return None
        nxt = patterns[min(best_idx + 5, len(patterns) - 1)]
        return (nxt["op"][0], nxt["op"][1])

    def record_human_frame(self, car_pos: Tuple[float, float],
                           ball_pos: Tuple[float, float], car_speed: float,
                           car_boost: float, my_team: int,
                           throttle: float = 0.0, steer: float = 0.0,
                           boost_input: float = 0.0, jump_input: float = 0.0):
        entries = self.data["human_play"]
        entries.append({
            "car": [round(car_pos[0], 1), round(car_pos[1], 1)],
            "ball": [round(ball_pos[0], 1), round(ball_pos[1], 1)],
            "speed": round(car_speed, 1),
            "boost": round(car_boost, 1),
            "team": my_team,
            "throttle": round(throttle, 2),
            "steer": round(steer, 2),
            "boost_in": round(boost_input, 2),
            "jump": round(jump_input, 2),
        })
        self.data["human_play"] = entries[-1000:]

    def human_play_count(self) -> int:
        return len(self.data.get("human_play", []))

    def lookup_human_hint(self, car_pos: Tuple[float, float],
                          ball_pos: Tuple[float, float], car_speed: float,
                          my_team: int) -> Optional[Dict]:
        frames = self.data.get("human_play", [])
        if len(frames) < 15:
            return None
        best_dist = float("inf")
        best: Optional[Dict] = None
        for f in frames:
            if f.get("team") != my_team:
                continue
            fc, fb = f["car"], f["ball"]
            d = (math.hypot(car_pos[0] - fc[0], car_pos[1] - fc[1])
                 + math.hypot(ball_pos[0] - fb[0], ball_pos[1] - fb[1]) * 0.5
                 + abs(car_speed - f.get("speed", 0)) * 0.3)
            if d < best_dist:
                best_dist = d
                best = f
        if best is None or best_dist > 1500.0:
            return None
        return {"throttle": best.get("throttle", 0.0),
                "steer": best.get("steer", 0.0),
                "boost_in": best.get("boost_in", 0.0),
                "jump": best.get("jump", 0.0)}


#  Helper Math

def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _predict_ball(ball_pos: Tuple[float, float], ball_vel: Tuple[float, float],
                  dt: float) -> Tuple[float, float]:
    decay = max(0.0, 1.0 - 0.03 * dt)
    px = ball_pos[0] + ball_vel[0] * dt * decay
    py = ball_pos[1] + ball_vel[1] * dt * decay
    return (_clamp(px, -4096.0, 4096.0), _clamp(py, -5120.0, 5120.0))


def _time_to_reach(pos: Tuple[float, float], target: Tuple[float, float],
                   speed: float) -> float:
    d = _dist(pos, target)
    avg_speed = max(500.0, (speed + 1400.0) / 2.0)
    return d / avg_speed


def _angle_to(car_yaw: float, car_pos: Tuple[float, float],
              target: Tuple[float, float]) -> float:
    dx = target[0] - car_pos[0]
    dy = target[1] - car_pos[1]
    target_yaw = math.atan2(dy, dx)
    return (target_yaw - car_yaw + math.pi) % (2.0 * math.pi) - math.pi


# ══════════════════════════════════════════════════════════════════
#  Goal opportunity detection
# ══════════════════════════════════════════════════════════════════

def detect_open_goal(
    ball_pos: Tuple[float, float],
    opponent_positions: List[Tuple[float, float]],
    opp_goal_y: float,
    goal_half_width: float = 893.0,
    corridor_margin: float = 350.0,
) -> bool:
    """
    Return True when there is a clear shooting corridor to the opponent goal.

    Checks whether any opponent car projects into the lane between the ball
    and the goal mouth (ball_x ± corridor_margin).
    """
    bx, by = ball_pos
    lane_x_min = bx - corridor_margin
    lane_x_max = bx + corridor_margin

    for (ox, oy) in opponent_positions:
        # Is opponent between ball and goal on the Y axis?
        if opp_goal_y > 0:
            between_y = by < oy < opp_goal_y
        else:
            between_y = opp_goal_y < oy < by
        if not between_y:
            continue
        # Is opponent within the shooting corridor?
        if lane_x_min <= ox <= lane_x_max:
            return False
    return True


# ══════════════════════════════════════════════════════════════════
#  Smart Aiming / Kickoff / Boost Collection helpers
# ══════════════════════════════════════════════════════════════════

def aim_at_goal(ball_pos: Tuple[float, float], opp_goal_y: float,
                keeper_pos: Optional[Tuple[float, float]] = None,
                goal_w: float = 893.0) -> Tuple[float, float]:
    """Pick the best target point on the goal face (away from keeper)."""
    aim_w = goal_w * 0.7
    if keeper_pos is not None and abs(keeper_pos[1] - opp_goal_y) < 2500:
        if keeper_pos[0] > 100:
            return (-aim_w, opp_goal_y)
        elif keeper_pos[0] < -100:
            return (aim_w, opp_goal_y)
    if ball_pos[0] > 0:
        return (-aim_w, opp_goal_y)
    return (aim_w, opp_goal_y)


def mode_attack_anchor(game_mode: str, opp_goal_y: float,
                       field_l: float) -> Tuple[float, float]:
    """Return the scoring anchor appropriate for the current mode."""
    if game_mode == GM_HOOPS:
        return (0.0, opp_goal_y * 0.78)
    if game_mode == GM_DROPSHOT:
        return (0.0, opp_goal_y * 0.45)
    return (0.0, _clamp(opp_goal_y, -field_l, field_l))


def is_kickoff_state(ball_pos: Tuple[float, float],
                     ball_vel: Tuple[float, float]) -> bool:
    """Detect kickoff: ball at center, essentially still."""
    return (abs(ball_pos[0]) < 80 and abs(ball_pos[1]) < 80
            and math.hypot(ball_vel[0], ball_vel[1]) < 50)


def find_best_boost_pad(car_pos: Tuple[float, float],
                        target: Tuple[float, float],
                        pads: List[Dict],
                        max_detour: float = 1000.0) -> Optional[Dict]:
    """Find a boost pad roughly on the way to *target*.

    *pads*: list of dicts with keys ``pos`` (x, y), ``active``, ``is_large``.
    Returns the best pad dict or *None*.
    """
    if not pads:
        return None
    direct_dist = _dist(car_pos, target)
    best = None
    best_score = float('inf')
    for pad in pads:
        if not pad.get('active', False):
            continue
        pad_pos = pad['pos']
        detour = _dist(car_pos, pad_pos) + _dist(pad_pos, target) - direct_dist
        if detour > max_detour:
            continue
        value = 100.0 if pad.get('is_large', False) else 12.0
        score = detour / max(1.0, value)
        if score < best_score:
            best_score = score
            best = pad
    return best


# ══════════════════════════════════════════════════════════════════
#  Situation Classifier
# ══════════════════════════════════════════════════════════════════

def classify_situation(car_pos: Tuple[float, float], ball_pos: Tuple[float, float],
                       ball_vel: Tuple[float, float], opp_pos: Tuple[float, float],
                       my_team: int) -> str:
    push_dir = 1.0 if my_team == 0 else -1.0
    my_dist = _dist(car_pos, ball_pos)
    opp_dist = _dist(opp_pos, ball_pos)
    ball_on_our_side = (ball_pos[1] * push_dir) < -200
    ball_heading_to_us = (ball_vel[1] * push_dir) < -200
    ball_speed = math.hypot(ball_vel[0], ball_vel[1])

    # DEFEND: ball on our side AND moving toward our goal
    if ball_on_our_side and ball_heading_to_us and ball_speed > 300:
        return "defending"
    # DEFEND: ball very deep in our half even if slow
    if (ball_pos[1] * push_dir) < -3500 and my_dist > 1500:
        return "defending"
    # WE HAVE BALL: we are closer
    if my_dist < opp_dist and my_dist < 1800:
        return "we_have_ball"
    # OPP HAS BALL: they are clearly closer
    if opp_dist < my_dist - 100 and opp_dist < 1800:
        return "opp_has_ball"
    # FREE BALL: nobody clearly has it
    return "free_ball"


# ══════════════════════════════════════════════════════════════════
#  Decision Engine
# ══════════════════════════════════════════════════════════════════

class DecisionEngine:
    def __init__(self, root_dir: str):
        self.planner = PathPlanner(grid_step=200)
        self.knowledge = KnowledgeStore(
            os.path.join(root_dir, "model", "search_knowledge.json"),
            persist=True,
        )
        self.settings = GameSettings()
        self._last_algorithm = "A*"
        self._last_path: Optional[PathResult] = None
        self._ball_trajectory: List = []
        self._boost_pads: List[Dict] = []
        self._keeper_pos: Optional[Tuple[float, float]] = None
        self._kickoff_spawn: str = "unknown"  # detected spawn type

        # Adaptive RL system
        self.adaptive = AdaptiveLearner()
        rl_data = self.knowledge.load_rl_data()
        if rl_data:
            self.adaptive.from_dict(rl_data)
        self._rl_save_counter = 0

        # User-chosen algorithm (empty string = let adaptive selector decide)
        self.forced_algorithm: str = ""
        # Algorithms enabled by the user in the Settings GUI (empty list = all)
        self.active_search_algorithms: List[str] = []

    def set_knowledge_store(self, knowledge: KnowledgeStore):
        self.knowledge = knowledge
        # If switching to a fresh session knowledge, reset adaptive learner too
        rl_data = knowledge.load_rl_data()
        if rl_data:
            self.adaptive.from_dict(rl_data)
        else:
            self.adaptive = AdaptiveLearner()

    def set_ball_trajectory(self, trajectory: List):
        self._ball_trajectory = trajectory

    def set_boost_pads(self, pads: List[Dict]):
        self._boost_pads = pads

    def set_keeper_pos(self, pos: Optional[Tuple[float, float]]):
        self._keeper_pos = pos

    def set_teammates(self, positions: List[Tuple[float, float]]):
        """Set teammate positions (excluding self)."""
        self.settings.teammate_positions = positions
        self.settings.num_teammates = len(positions)

    def set_score_state(self, my_score: int, opp_score: int, is_overtime: bool):
        """Update score difference and overtime state for aggression tuning."""
        self.settings.score_diff = my_score - opp_score
        self.settings.we_are_trailing = my_score < opp_score
        self.settings.is_overtime = is_overtime

    def set_dropshot_tiles(self, tiles: list):
        """tiles: list of (index, tile_state) for opponent-half tiles."""
        self.settings.damaged_tiles = tiles

    def save_rl_data(self):
        """Persist adaptive RL learning data."""
        self.knowledge.save_rl_data(self.adaptive.to_dict())

    def _adaptive_context_tag(self) -> str:
        gravity_bucket = int(round(abs(self.settings.gravity) / 100.0))
        width_bucket = int(round(self.settings.field_w / 256.0))
        length_bucket = int(round(self.settings.field_l / 256.0))
        teammates_bucket = min(3, self.settings.num_teammates)
        return (
            f"gm={self.settings.game_mode}|g={gravity_bucket}|"
            f"fw={width_bucket}|fl={length_bucket}|tm={teammates_bucket}|"
            f"puck={int(self.settings.puck_mode)}"
        )

    def adaptive_decide_role(self, situation: str, car_pos: Tuple[float, float],
                             ball_pos: Tuple[float, float],
                             ball_vel: Tuple[float, float],
                             opp_pos: Tuple[float, float],
                             opp_vel: Tuple[float, float],
                             car_speed: float, car_boost: float,
                             my_team: int, user_mode: str) -> str:
        """Use adaptive RL system to decide the effective mode/role."""
        push_dir = 1.0 if my_team == 0 else -1.0
        return self.adaptive.decide_role(
            situation=situation,
            car_pos=car_pos,
            ball_pos=ball_pos,
            ball_vel=ball_vel,
            opp_pos=opp_pos,
            opp_vel=opp_vel,
            car_speed=car_speed,
            car_boost=car_boost,
            score_diff=self.settings.score_diff,
            push_dir=push_dir,
            user_mode=user_mode,
            context_tag=self._adaptive_context_tag(),
        )

    def adaptive_blend_controls(self, controls: Dict[str, float],
                                car_pos: Tuple[float, float],
                                ball_pos: Tuple[float, float],
                                car_speed: float, car_boost: float,
                                opp_pos: Tuple[float, float],
                                my_team: int) -> Dict[str, float]:
        """Blend controls with policy gradient (human-learned) and actor-critic."""
        push_dir = 1.0 if my_team == 0 else -1.0

        # Get policy gradient suggestion from human demonstrations
        pg_controls = self.adaptive.get_human_policy_controls(
            car_pos, ball_pos, car_speed, car_boost, opp_pos, push_dir,
            self._adaptive_context_tag())

        if pg_controls is not None:
            # Blend with higher weight since human policy is learned
            w = 0.35 if self.adaptive.policy_human.has_learned() else 0.15
            controls["throttle"] = _clamp(
                controls["throttle"] * (1 - w) + pg_controls["throttle"] * w, -1, 1)
            controls["steer"] = _clamp(
                controls["steer"] * (1 - w) + pg_controls["steer"] * w, -1, 1)
            controls["yaw"] = controls["steer"]
            if pg_controls["boost"] > 0.5:
                controls["boost"] = max(controls["boost"], w)
            if pg_controls["jump"] > 0.5:
                controls["jump"] = max(controls["jump"], 0.3)

        # Apply actor-critic aggression adjustment
        state_key = self.adaptive.get_current_state_key()
        if state_key:
            adj = self.adaptive.get_actor_critic_adjustments(state_key)
            aggression = adj.get("aggression", 0.0)
            # Positive aggression → more throttle/boost, negative → more cautious
            if aggression > 0.05:
                controls["throttle"] = _clamp(
                    controls["throttle"] + aggression * 0.3, -1, 1)
                if controls["boost"] > 0 and aggression > 0.2:
                    controls["boost"] = 1.0
            elif aggression < -0.05:
                controls["throttle"] = _clamp(
                    controls["throttle"] + aggression * 0.2, -1, 1)

        return controls

    def record_human_demo(self, car_pos: Tuple[float, float],
                          ball_pos: Tuple[float, float],
                          car_speed: float, car_boost: float,
                          opp_pos: Tuple[float, float],
                          my_team: int,
                          throttle: float, steer: float,
                          boost_in: float, jump_in: float):
        """Record human demonstration for RL policy learning (called in M mode)."""
        push_dir = 1.0 if my_team == 0 else -1.0
        self.adaptive.record_human_frame(
            car_pos, ball_pos, car_speed, car_boost,
            opp_pos, push_dir, self._adaptive_context_tag(),
            throttle, steer, boost_in, jump_in)

    def detect_kickoff_spawn(self, car_pos: Tuple[float, float], my_team: int):
        """Detect kickoff spawn dynamically from actual car coordinates."""
        spawn = self.settings.detect_kickoff_spawn(car_pos, my_team)
        self._kickoff_spawn = spawn
        return spawn

    def detect_game_mode(self, packet_game_info) -> str:
        try:
            world_gravity = getattr(packet_game_info, 'world_gravity_z', -650.0)
            if world_gravity != 0:
                self.settings.gravity = world_gravity
        except Exception:
            pass
        return self.settings.game_mode

    def detect_game_mode_from_settings(self, match_settings) -> str:
        """Read game mode from rlbot match_settings (flatbuffer)."""
        _MODE_MAP = {
            0: GM_SOCCER,
            1: GM_HOOPS,
            2: GM_DROPSHOT,
            3: GM_SNOW,
            4: GM_RUMBLE,
            5: GM_HEATSEEKER,
            6: GM_GRIDIRON,
        }
        try:
            gm_int = match_settings.GameMode()
            detected = _MODE_MAP.get(gm_int, GM_SOCCER)
            self.settings.game_mode = detected
            self.settings.apply_mode()
            print(f"[MODE] Game mode from settings: {detected} (id={gm_int})")
            return detected
        except Exception:
            return self.settings.game_mode

    def detect_game_mode_from_packet(self, num_tiles: int, world_gravity: float,
                                     ball_z: float) -> str:
        """Heuristic game-mode detection from per-frame data.

        RLBot's flatbuffer GameMode only covers 7 modes.  Many LTMs / mutator
        modes report as Soccer (0).  We refine by checking packet properties
        that vary between modes.
        """
        prev = self.settings.game_mode
        detected = prev

        # Dropshot has tiles
        if num_tiles > 0:
            if prev in _RUMBLE_VARIANTS:
                detected = GM_DROPSHOT_RUMBLE
            else:
                detected = GM_DROPSHOT

        # Track ball speed across frames for heuristic detection
        if not hasattr(self, '_frame_ball_speeds'):
            self._frame_ball_speeds = []
        self._frame_ball_speeds.append(ball_z)
        if len(self._frame_ball_speeds) > 120:
            self._frame_ball_speeds = self._frame_ball_speeds[-120:]

        # Mutator gravity differs significantly from default -650
        if world_gravity != 0:
            grav = world_gravity
            self.settings.gravity = grav
            # Very low gravity might indicate a mutator mode
            if prev == GM_SOCCER and abs(grav) < 400:
                detected = GM_CUSTOM_MUTATOR
                self.settings.game_mode = detected
                self.settings.apply_mode()

        if detected != prev:
            self.settings.game_mode = detected
            self.settings.apply_mode()
            print(f"[MODE] Heuristic detection changed mode: {prev} -> {detected}")
        return detected

    def refine_mode_from_mutators(self, match_settings) -> str:
        """Detect LTM / mutator-based modes beyond the 7 official GameMode values.

        Called once after settings are available.  Inspects mutators,
        match name, map name, etc.
        """
        try:
            map_name = ""
            try:
                map_name = match_settings.GameMap().decode() if hasattr(match_settings, 'GameMap') else ""
            except Exception:
                pass
            map_lower = map_name.lower() if map_name else ""

            # Check for known maps
            if "labs" in map_lower or "pillars" in map_lower or "cosmic" in map_lower or "double_goal" in map_lower:
                self.settings.game_mode = GM_ROCKET_LABS
                self.settings.apply_mode()
                return GM_ROCKET_LABS

            # Detect mutator settings
            try:
                # Ball type mutator: 1=Cube, 2=Puck, 3=Basketball
                ball_type = match_settings.MutatorSettings().BallType() if hasattr(match_settings.MutatorSettings(), 'BallType') else 0
                if ball_type == 1:
                    self.settings.game_mode = GM_SUPER_CUBE
                    self.settings.apply_mode()
                    return GM_SUPER_CUBE
            except Exception:
                pass

            try:
                ball_speed_opt = match_settings.MutatorSettings().BallMaxSpeedOption() if hasattr(match_settings.MutatorSettings(), 'BallMaxSpeedOption') else 0
                boost_opt = match_settings.MutatorSettings().BoostOption() if hasattr(match_settings.MutatorSettings(), 'BoostOption') else 0
                # Very high ball speed → could be Boomer Ball
                if ball_speed_opt >= 3:
                    self.settings.game_mode = GM_BOOMER
                    self.settings.apply_mode()
                    return GM_BOOMER
            except Exception:
                pass

        except Exception:
            pass

        return self.settings.game_mode

    def set_goal_positions(self, field_info):
        """Populate settings from FieldInfo (preferred) or legacy goals list."""
        self.settings.populate_from_field_info(field_info)

    def detect_ground_type(self, car_on_ground: bool, car_speed: float,
                           throttle: float, actual_accel: float):
        if car_on_ground and abs(throttle) > 0.8 and car_speed > 200:
            expected = 1600.0 * abs(throttle)
            if actual_accel < expected * 0.5:
                self.settings.ground_friction = max(0.3, self.settings.ground_friction - 0.02)
            else:
                self.settings.ground_friction = min(1.0, self.settings.ground_friction + 0.01)
            self.settings.adapt()

    def _target_weights(self, mode: str, reset_mode: bool) -> Dict[str, float]:
        if reset_mode:
            return {"A*": 0.13, "BFS": 0.13, "UCS": 0.13, "Greedy": 0.13,
                    "DFS": 0.10, "Decision Tree": 0.13, "Beam Search": 0.12, "IDA*": 0.13}
        if mode == MODE_ATTACK:
            return {"A*": 0.22, "Greedy": 0.16, "Beam Search": 0.15, "IDA*": 0.15,
                    "Decision Tree": 0.14, "UCS": 0.08, "BFS": 0.06, "DFS": 0.04}
        if mode == MODE_DEFENSE:
            return {"UCS": 0.19, "A*": 0.19, "IDA*": 0.16, "Beam Search": 0.14,
                    "Decision Tree": 0.14, "BFS": 0.10, "Greedy": 0.05, "DFS": 0.03}
        return {"A*": 0.18, "IDA*": 0.15, "Beam Search": 0.14, "Decision Tree": 0.15,
                "UCS": 0.14, "Greedy": 0.11, "BFS": 0.08, "DFS": 0.05}

    def choose_algorithm(self, usage_pct: Dict[str, float], mode: str, reset_mode: bool) -> str:
        # If the user pinned a specific algorithm, always use it
        if self.forced_algorithm:
            self._last_algorithm = self.forced_algorithm
            return self.forced_algorithm
        target = self._target_weights(mode, reset_mode)
        scores: Dict[str, float] = {}
        for alg, desired in target.items():
            deficit = desired * 100.0 - usage_pct.get(alg, 0.0)
            success_bonus = 0.0
            if not reset_mode:
                success = self.knowledge.data.get("algorithm_success", {}).get(alg, 0)
                success_bonus = min(15.0, float(success) * 0.25)
            scores[alg] = deficit + success_bonus
        chosen = max(scores, key=scores.get)
        self._last_algorithm = chosen
        return chosen

    def _plan_candidate(self, algorithm: str, mode: str, my_team: int,
                        car_pos: Tuple[float, float], ball_pos: Tuple[float, float],
                        ball_vel: Tuple[float, float], car_speed: float,
                        opp_pos: Tuple[float, float], reset_mode: bool,
                        situation: str = "free_ball",
                        car_boost: float = 50.0,
                        ball_z: float = 93.0) -> DecisionOutput:
        gs = self.settings
        push_dir = 1.0 if my_team == 0 else -1.0
        fw = gs.field_w * 0.95
        fl = gs.field_l
        opp_goal_y = gs.get_opp_goal_y(my_team)
        own_goal_y = gs.get_own_goal_y(my_team)

        # Default target from main strategy
        base_target = self.select_target(mode, my_team, car_pos, ball_pos,
                                         ball_vel, car_speed, opp_pos,
                                         reset_mode, situation,
                                         car_boost=car_boost,
                                         ball_z=ball_z)

        if reset_mode:
            target = base_target
        elif algorithm == "Decision Tree":
            tree = build_decision_tree(car_pos, ball_pos, car_speed, my_team,
                                       self.knowledge.data)
            dt_result = evaluate_tree(tree)
            target = dt_result.target

        elif algorithm == "Greedy":
            # Greedy: short look-ahead, aggressive direct chase
            short_pred = _predict_ball(ball_pos, ball_vel, 0.15)
            target = (_clamp(short_pred[0], -fw, fw),
                      _clamp(short_pred[1], -fl, fl))

        elif algorithm == "UCS":
            # UCS (safest cost): defensive bias — position between ball and own goal
            bx, by = base_target
            w = 0.7 if situation in ("defending", "opp_has_ball") else 0.85
            target = (_clamp(bx * w, -fw, fw),
                      _clamp(by * w + own_goal_y * (1 - w), -fl, fl))

        elif algorithm == "Beam Search":
            # Beam Search: longer prediction window — plan further ahead
            my_time = _time_to_reach(car_pos, ball_pos, car_speed)
            far_pred = _predict_ball(ball_pos, ball_vel,
                                     _clamp(my_time * 1.2, 0.2, 2.0))
            if situation in ("we_have_ball", "free_ball"):
                goal_aim = aim_at_goal(far_pred, opp_goal_y,
                                       self._keeper_pos, goal_w=gs.goal_width)
                target = self._push_target(far_pred, goal_aim, 450.0)
            else:
                target = (_clamp(far_pred[0], -fw, fw),
                          _clamp(far_pred[1], -fl, fl))

        elif algorithm == "DFS":
            # DFS: aggressive push — target a spot behind ball toward opponent goal
            bx, by = base_target
            push_y = by + push_dir * 350.0
            target = (_clamp(bx, -fw, fw), _clamp(push_y, -fl, fl))

        elif algorithm == "BFS":
            # BFS: wide exploration — target a supporting rotation position
            bx, by = base_target
            if situation in ("we_have_ball", "free_ball"):
                # Offset to the side for a wider angle of approach
                side = 1.0 if car_pos[0] < bx else -1.0
                target = (_clamp(bx + side * 500.0, -fw, fw),
                          _clamp(by, -fl, fl))
            else:
                target = base_target

        elif algorithm == "IDA*":
            # IDA*: intercept-focused — use ball trajectory if available
            if self._ball_trajectory and not is_kickoff_state(ball_pos, ball_vel):
                intercept = find_intercept(
                    self._ball_trajectory,
                    car_pos[0], car_pos[1],
                    car_speed, car_boost,
                    max_z=400.0,
                )
                if intercept is not None:
                    ix, iy = intercept[0], intercept[1]
                    target = (_clamp(ix, -fw, fw), _clamp(iy, -fl, fl))
                else:
                    target = base_target
            else:
                target = base_target
        else:
            # A* and any other: use the standard optimal target
            target = base_target

        # ── Exploration noise (scales with DQN epsilon, fades as model matures) ──
        explore_amp = getattr(self, "_explore_amp", 0.0)
        if explore_amp > 60.0 and not reset_mode:
            target = (
                _clamp(target[0] + random.gauss(0, explore_amp * 0.40), -fw, fw),
                _clamp(target[1] + random.gauss(0, explore_amp * 0.60), -fl, fl),
            )

        path_algo = algorithm
        if algorithm in ("Decision Tree", "IDA*"):
            path_algo = "A*"
        _allowed = set(self.active_search_algorithms) if self.active_search_algorithms else None
        path_result = self.planner.find_path(car_pos, target, path_algo, allowed_algorithms=_allowed)
        return DecisionOutput(algorithm=algorithm, target=target,
                              path_result=path_result, situation=situation)

    def _score_candidate(self, decision: DecisionOutput, mode: str,
                         reset_mode: bool, situation: str) -> float:
        path_cost = decision.path_result.cost
        if math.isinf(path_cost):
            return float("-inf")

        weights = self._target_weights(mode, reset_mode)
        mode_bias = weights.get(decision.algorithm, 0.0) * 100.0
        success_bonus = float(self.knowledge.data.get("algorithm_success", {})
                              .get(decision.algorithm, 0)) * 0.35
        node_count_penalty = len(decision.path_result.node_path) * 0.05
        path_cost_penalty = path_cost * 1.5
        tactical_bonus = 0.0

        # Each algorithm has a unique situational strength
        alg = decision.algorithm
        if alg == "Decision Tree":
            # Good at contextual decisions, always a reasonable bonus
            tactical_bonus += 3.0
        if alg == "A*" and situation in ("defending", "opp_has_ball"):
            # A* finds optimal short-path — great for urgent defense
            tactical_bonus += 4.0
        if alg == "Greedy" and situation in ("we_have_ball", "free_ball"):
            # Greedy is fast and aggressive — good when we have opportunity
            tactical_bonus += 3.5
        if alg == "Beam Search" and situation == "free_ball":
            # Beam search explores multiple promising paths
            tactical_bonus += 3.0
        if alg == "UCS" and situation == "defending":
            # UCS finds minimum-cost path — best for safe defensive clears
            tactical_bonus += 3.5
        if alg == "IDA*" and situation in ("defending", "opp_has_ball"):
            # IDA* memory-efficient optimal — good under pressure
            tactical_bonus += 3.0
        if alg == "BFS" and situation == "we_have_ball":
            # BFS explores widely — finds alternative routes
            tactical_bonus += 2.0
        if alg == "DFS" and situation == "free_ball":
            # DFS explores deep — sometimes finds creative paths
            tactical_bonus += 1.5

        # Friction-aware: on slippery ground, prefer algorithms with shorter paths
        gf = self.settings.ground_friction
        if gf < 0.8:
            # Penalize long paths more on slippery ground
            path_cost_penalty *= 1.5
            if alg in ("A*", "UCS"):
                tactical_bonus += 2.0  # shortest-path algos better on ice

        # Anomaly-threat aware: under high threat prefer safe/optimal algorithms
        threat = getattr(self.adaptive, '_last_threat_level', 0.0)
        if threat > 0.6:
            if alg in ("A*", "UCS"):
                tactical_bonus += 3.5  # safe, complete — best under threat
            elif alg in ("DFS", "Greedy"):
                tactical_bonus -= 2.0  # risky/suboptimal — penalise under threat

        return mode_bias + success_bonus + tactical_bonus - path_cost_penalty - node_count_penalty

    def plan_parallel_ensemble(self, mode: str, my_team: int,
                               car_pos: Tuple[float, float], ball_pos: Tuple[float, float],
                               ball_vel: Tuple[float, float], car_speed: float,
                               opp_pos: Tuple[float, float], reset_mode: bool,
                               situation: str = "free_ball",
                               car_boost: float = 50.0,
                               ball_z: float = 93.0) -> DecisionOutput:
        # Exploration amplitude: high when DQN is still learning, fades to 0
        dqn_eps = getattr(self.adaptive.dqn, "epsilon", 0.1)
        self._explore_amp = max(0.0, dqn_eps - 0.08) * 2600.0

        self.knowledge.tick()

        # Respect user-selected algorithm whitelist (empty list = all enabled)
        algo_whitelist = self.active_search_algorithms
        algorithms_to_run = (
            [a for a in _SEARCH_ALGORITHMS if a in algo_whitelist]
            if algo_whitelist else _SEARCH_ALGORITHMS
        )
        # Always keep at least one algorithm to avoid empty ensemble
        if not algorithms_to_run:
            algorithms_to_run = _SEARCH_ALGORITHMS

        with ThreadPoolExecutor(max_workers=len(algorithms_to_run)) as pool:
            futures = [
                pool.submit(
                    self._plan_candidate,
                    algorithm,
                    mode,
                    my_team,
                    car_pos,
                    ball_pos,
                    ball_vel,
                    car_speed,
                    opp_pos,
                    reset_mode,
                    situation,
                    car_boost,
                    ball_z,
                )
                for algorithm in algorithms_to_run
            ]
            candidates = [future.result() for future in futures]

        best = max(
            candidates,
            key=lambda decision: self._score_candidate(decision, mode, reset_mode, situation),
        )

        if not reset_mode:
            for node in best.path_result.node_path[:8]:
                self.knowledge.record_node(node)
        self._last_path = best.path_result
        self._last_algorithm = best.algorithm
        return best

    @staticmethod
    def smart_effective_mode(my_team: int, car_pos: Tuple[float, float],
                             car_speed: float, ball_pos: Tuple[float, float],
                             ball_vel: Tuple[float, float],
                             opp_pos: Tuple[float, float]) -> str:
        push_dir = 1.0 if my_team == 0 else -1.0
        ball_on_our_side = (ball_pos[1] * push_dir) < -500
        ball_heading_to_us = (ball_vel[1] * push_dir) < -500
        ball_speed = math.hypot(ball_vel[0], ball_vel[1])
        if ball_on_our_side and ball_heading_to_us and ball_speed > 400:
            return MODE_DEFENSE
        return MODE_ATTACK

    def _push_target(self, ball_pos: Tuple[float, float], toward: Tuple[float, float],
                     push_dist: float = 400.0) -> Tuple[float, float]:
        dx = toward[0] - ball_pos[0]
        dy = toward[1] - ball_pos[1]
        d = max(1.0, math.hypot(dx, dy))
        tx = ball_pos[0] + dx / d * push_dist
        ty = ball_pos[1] + dy / d * push_dist
        fw = self.settings.field_w * 0.95
        fl = self.settings.field_l
        return (_clamp(tx, -fw, fw), _clamp(ty, -fl, fl))

    def _approach_ball(self, ball_pos: Tuple[float, float], opp_goal: Tuple[float, float],
                       offset: float = 250.0) -> Tuple[float, float]:
        dx = opp_goal[0] - ball_pos[0]
        dy = opp_goal[1] - ball_pos[1]
        d = max(1.0, math.hypot(dx, dy))
        ax = ball_pos[0] - dx / d * offset
        ay = ball_pos[1] - dy / d * offset
        fw = self.settings.field_w * 0.95
        fl = self.settings.field_l
        return (_clamp(ax, -fw, fw), _clamp(ay, -fl, fl))

    def select_target(self, mode: str, my_team: int, car_pos: Tuple[float, float],
                      ball_pos: Tuple[float, float], ball_vel: Tuple[float, float],
                      car_speed: float, opp_pos: Tuple[float, float],
                      reset_mode: bool, situation: str = "free_ball",
                      car_boost: float = 50.0,
                      ball_z: float = 93.0) -> Tuple[float, float]:
        push_dir = 1.0 if my_team == 0 else -1.0
        gs = self.settings
        fw = gs.field_w * 0.95  # clamp margin inside walls
        fl = gs.field_l
        gw = gs.goal_width
        opp_goal_y = gs.get_opp_goal_y(my_team)
        own_goal_y = gs.get_own_goal_y(my_team)
        opp_goal = (0.0, opp_goal_y)

        my_time = _time_to_reach(car_pos, ball_pos, car_speed)
        look_ahead = _clamp(my_time * 0.8, 0.1, 1.5)
        pred_ball = _predict_ball(ball_pos, ball_vel, look_ahead)
        bx, by = pred_ball
        dist_ball = _dist(car_pos, pred_ball)
        car_behind_ball = (car_pos[1] * push_dir) < (by * push_dir)
        ball_speed = math.hypot(ball_vel[0], ball_vel[1])
        opp_dist_ball = _dist(opp_pos, pred_ball)

        # Use opponent prediction if available
        opp_pred = self.knowledge.predict_opponent_target(opp_pos, (0, 0), ball_pos)

        # ── Kickoff: smart route based on spawn position ──
        if is_kickoff_state(ball_pos, ball_vel):
            spawn = self._kickoff_spawn
            # Diagonal spawns: slight offset to hit ball with momentum
            if spawn == "diagonal_right":
                return (300.0 * push_dir, 0.0)
            elif spawn == "diagonal_left":
                return (-300.0 * push_dir, 0.0)
            elif spawn == "far_back":
                # Far back: let teammate go first if we have one
                if gs.num_teammates > 0:
                    return (0.0, own_goal_y * 0.3)
                return (0.0, 0.0)
            elif spawn in ("back_right", "back_left"):
                # Grab corner boost then approach
                if car_boost < 40:
                    bx_off = 3584.0 if spawn == "back_right" else -3584.0
                    return (bx_off, 0.0)
                return (0.0, 0.0)
            return (0.0, 0.0)

        # Mode-specific strategy overrides
        gm = gs.game_mode

        # ── HOOPS: basket is above, need to push ball up & over the rim ──
        if gm == GM_HOOPS:
            rim_center = mode_attack_anchor(gm, opp_goal_y, fl)
            # Hoops goal height ~300-ish; we need to pop the ball up.
            # Get under the ball, then flip into it to launch upward.
            if situation in ("we_have_ball", "free_ball"):
                if ball_z > 250:
                    # Ball already airborne: go for an aerial hit
                    if self._ball_trajectory:
                        intercept = find_intercept(
                            self._ball_trajectory,
                            car_pos[0], car_pos[1],
                            car_speed, car_boost,
                            max_z=gs.max_aerial_z,
                        )
                        if intercept:
                            return (_clamp(intercept[0], -fw, fw),
                                    _clamp(intercept[1], -fl, fl))
                # Stay ball-centric: approach behind the ball toward the rim,
                # not inside the basket itself.
                if car_behind_ball:
                    return self._push_target(pred_ball, rim_center, 280.0)
                return self._approach_ball(pred_ball, rim_center, 260.0)
            # Defending / opp has ball: just challenge the ball fast
            return (_clamp(bx, -fw, fw), _clamp(by, -fl, fl))

        # ── DROPSHOT: hit the ball onto opponent's floor to damage tiles ──
        if gm == GM_DROPSHOT:
            # Primary goal: make the ball bounce hard on the opponent side.
            # Our half is y < 0 (blue) or y > 0 (orange).
            # Push ball into the air on THEIR side.
            opp_center = (0.0, opp_goal_y * 0.5)  # center of opponent half
            if situation in ("we_have_ball", "free_ball"):
                # Push ball aggressively toward opponent half
                push_d = 600.0 if ball_speed < 500 else 350.0
                return self._push_target(pred_ball, opp_center, push_d)
            if situation == "defending":
                # Clear ball to their side
                return self._push_target(pred_ball, opp_center, 700.0)
            # opp_has_ball: challenge
            return (_clamp(bx, -fw, fw), _clamp(by, -fl, fl))

        # ── SNOW DAY (Hockey): puck slides on ground, rarely airborne ──
        if gm == GM_SNOW:
            # Puck doesn't fly — ignore aerials, focus on ground play.
            # Steer more carefully (ice/low friction).
            goal_aim = aim_at_goal(pred_ball, opp_goal_y, self._keeper_pos, goal_w=gw)
            if situation in ("we_have_ball", "free_ball"):
                push_d = 500.0 if ball_speed < 400 else 300.0
                return self._push_target(pred_ball, goal_aim, push_d)
            if situation == "defending":
                w = 0.65
                return (_clamp(bx * w, -fw, fw),
                        _clamp(by * w + own_goal_y * (1 - w), -fl, fl))
            return (_clamp(bx, -fw, fw), _clamp(by, -fl, fl))

        # ── HEATSEEKER: ball auto-targets opponent goal after a hit ──
        if gm == GM_HEATSEEKER:
            # Just touch the ball — it seeks their goal automatically.
            # On defense: touch it to redirect to their goal.
            if dist_ball < 2000 or situation in ("defending", "we_have_ball"):
                return (_clamp(bx, -fw, fw), _clamp(by, -fl, fl))
            # Far away: go toward the ball
            if self._ball_trajectory:
                intercept = find_intercept(
                    self._ball_trajectory,
                    car_pos[0], car_pos[1],
                    car_speed, car_boost,
                    max_z=gs.max_aerial_z,
                )
                if intercept:
                    return (_clamp(intercept[0], -fw, fw),
                            _clamp(intercept[1], -fl, fl))
            return (_clamp(bx, -fw, fw), _clamp(by, -fl, fl))

        # ── RUMBLE: powerups make aggression pay off, be extra aggressive ──
        if gm in _RUMBLE_VARIANTS:
            # Challenge more aggressively, go for ball/demos more
            if situation in ("opp_has_ball", "free_ball"):
                if _dist(car_pos, opp_pos) < 1300 and _dist(car_pos, ball_pos) > 1500:
                    return (_clamp(opp_pos[0], -fw, fw),
                            _clamp(opp_pos[1], -fl, fl))
            # For the rest, fall through to standard soccer logic

        # ── GRIDIRON / SPIKE RUSH: ball attaches to car ──
        if gm in _SPIKE_VARIANTS:
            if dist_ball < 300:
                goal_aim = aim_at_goal(car_pos, opp_goal_y, self._keeper_pos, goal_w=gw)
                return goal_aim
            if situation == "defending":
                w = 0.6
                return (_clamp(bx * w, -fw, fw),
                        _clamp(by * w + own_goal_y * (1 - w), -fl, fl))
            return (_clamp(bx, -fw, fw), _clamp(by, -fl, fl))

        # ── KNOCKOUT: ram opponents off the arena, no ball ──
        if gm == GM_KNOCKOUT:
            # Objective: hit other players, not a ball. Target nearest opponent.
            return (_clamp(opp_pos[0], -fw, fw),
                    _clamp(opp_pos[1], -fl, fl))

        # ── BOOMER BALL: extremely fast ball, react quickly ──
        if gm == GM_BOOMER:
            # Ball moves very fast; minimize approach distance, just hit it
            goal_aim = aim_at_goal(pred_ball, opp_goal_y, self._keeper_pos, goal_w=gw)
            if situation == "defending":
                # Get between ball and goal immediately
                w = 0.75
                return (_clamp(bx * w, -fw, fw),
                        _clamp(by * w + own_goal_y * (1 - w), -fl, fl))
            if dist_ball < 2500:
                return self._push_target(pred_ball, goal_aim, 300.0)
            return (_clamp(bx, -fw, fw), _clamp(by, -fl, fl))

        # ── BEACH BALL: floaty ball, more aerial play ──
        if gm == GM_BEACH_BALL:
            goal_aim = aim_at_goal(pred_ball, opp_goal_y, self._keeper_pos, goal_w=gw)
            if ball_z > 200 and self._ball_trajectory:
                intercept = find_intercept(
                    self._ball_trajectory, car_pos[0], car_pos[1],
                    car_speed, car_boost, max_z=gs.max_aerial_z)
                if intercept:
                    return (_clamp(intercept[0], -fw, fw),
                            _clamp(intercept[1], -fl, fl))
            if situation in ("we_have_ball", "free_ball"):
                return self._push_target(pred_ball, goal_aim, 450.0)
            return (_clamp(bx, -fw, fw), _clamp(by, -fl, fl))

        # ── SPEED DEMON: very fast gameplay, minimal detours ──
        if gm == GM_SPEED_DEMON:
            goal_aim = aim_at_goal(pred_ball, opp_goal_y, self._keeper_pos, goal_w=gw)
            if situation in ("we_have_ball", "free_ball"):
                return self._push_target(pred_ball, goal_aim, 350.0)
            if situation == "defending":
                w = 0.7
                return (_clamp(bx * w, -fw, fw),
                        _clamp(by * w + own_goal_y * (1 - w), -fl, fl))
            return (_clamp(bx, -fw, fw), _clamp(by, -fl, fl))

        # ── SPRING LOADED: extra jump power, more aerials ──
        if gm == GM_SPRING_LOADED:
            goal_aim = aim_at_goal(pred_ball, opp_goal_y, self._keeper_pos, goal_w=gw)
            if ball_z > 300 and self._ball_trajectory:
                intercept = find_intercept(
                    self._ball_trajectory, car_pos[0], car_pos[1],
                    car_speed, car_boost, max_z=gs.max_aerial_z)
                if intercept:
                    return (_clamp(intercept[0], -fw, fw),
                            _clamp(intercept[1], -fl, fl))
            if situation in ("we_have_ball", "free_ball"):
                return self._push_target(pred_ball, goal_aim, 500.0)
            return (_clamp(bx, -fw, fw), _clamp(by, -fl, fl))

        # ── SUPER CUBE / SPOOKY CUBE: cube bounces unpredictably ──
        if gm in _CUBE_VARIANTS:
            # Cube bounces weird; get close and hit it toward goal
            goal_aim = aim_at_goal(pred_ball, opp_goal_y, self._keeper_pos, goal_w=gw)
            if situation in ("we_have_ball", "free_ball"):
                return self._push_target(pred_ball, goal_aim, 400.0)
            if situation == "defending":
                return (_clamp(bx, -fw, fw), _clamp(by, -fl, fl))
            return (_clamp(bx, -fw, fw), _clamp(by, -fl, fl))

        # ── SPLIT SHOT variants: stay in your zone ──
        if gm in _SPLIT_VARIANTS:
            # Restrict targets to our half of the field
            zone_limit_y = 0.0  # midfield
            goal_aim = aim_at_goal(pred_ball, opp_goal_y, self._keeper_pos, goal_w=gw)
            clamped_by = _clamp(by, -fl, fl)
            # In split shot, stay on our side (blue = y<0, orange = y>0)
            if push_dir > 0:
                clamped_by = min(clamped_by, 200.0)
            else:
                clamped_by = max(clamped_by, -200.0)
            if situation == "defending":
                w = 0.7
                return (_clamp(bx * w, -fw, fw),
                        _clamp(clamped_by * w + own_goal_y * (1 - w), -fl, fl))
            return (_clamp(bx, -fw, fw), clamped_by)

        # ── WINTER BREAKAWAY: hockey mode (same as Snow Day) ──
        if gm == GM_WINTER_BREAKAWAY:
            goal_aim = aim_at_goal(pred_ball, opp_goal_y, self._keeper_pos, goal_w=gw)
            if situation in ("we_have_ball", "free_ball"):
                push_d = 500.0 if ball_speed < 400 else 300.0
                return self._push_target(pred_ball, goal_aim, push_d)
            if situation == "defending":
                w = 0.65
                return (_clamp(bx * w, -fw, fw),
                        _clamp(by * w + own_goal_y * (1 - w), -fl, fl))
            return (_clamp(bx, -fw, fw), _clamp(by, -fl, fl))

        # ── GHOST HUNT / other event modes: general ball-chase ──
        if gm == GM_GHOST_HUNT:
            return (_clamp(bx, -fw, fw), _clamp(by, -fl, fl))

        # ── DROPSHOT with tile awareness ──
        if gm == GM_DROPSHOT and gs.damaged_tiles:
            # Prefer targeting areas where tiles are already damaged/open
            opp_center = (0.0, opp_goal_y * 0.5)
            if situation in ("we_have_ball", "free_ball"):
                push_d = 600.0 if ball_speed < 500 else 350.0
                return self._push_target(pred_ball, opp_center, push_d)

        # ── Score-aware aggression: be more aggressive when trailing / overtime ──
        trailing_aggression = 1.0
        if gs.is_overtime:
            trailing_aggression = 1.4  # overtime: hyper-aggressive
        elif gs.we_are_trailing:
            trailing_aggression = 1.0 + min(0.4, abs(gs.score_diff) * 0.15)
        elif gs.score_diff >= 3:
            trailing_aggression = 0.7  # comfortably ahead: play safe

        # ── Team play: avoid double-committing with teammates ──
        if gs.num_teammates > 0 and situation in ("free_ball", "we_have_ball"):
            for tm_pos in gs.teammate_positions:
                tm_dist_ball = _dist(tm_pos, pred_ball)
                # Teammate is closer to ball — rotate back instead of both chasing
                if tm_dist_ball < dist_ball - 400:
                    # Hold a midfield / supporting position
                    support_y = by * 0.3 + own_goal_y * 0.2
                    support_x = car_pos[0] * 0.5  # drift toward center
                    return (_clamp(support_x, -fw, fw),
                            _clamp(support_y, -fl, fl))
            # If two teammates are close to each other, we spread out
            if len(gs.teammate_positions) >= 2:
                tm1 = gs.teammate_positions[0]
                tm2 = gs.teammate_positions[1]
                if _dist(tm1, tm2) < 800:
                    # They're stacked — we go opposite side
                    avg_x = (tm1[0] + tm2[0]) / 2
                    spread_x = -avg_x * 0.5
                    return (_clamp(spread_x, -fw, fw),
                            _clamp(by * 0.5, -fl, fl))

        # ── Wall play detection: if ball is near wall, approach from angle ──
        ball_near_wall = abs(bx) > 3500
        if ball_near_wall and situation in ("we_have_ball", "free_ball"):
            # Ball is on the wall — drive toward a spot that centers it
            center_offset = -600.0 if bx > 0 else 600.0
            wall_target_x = _clamp(bx + center_offset, -fw, fw)
            wall_target_y = _clamp(by + push_dir * 200.0, -fl, fl)
            if dist_ball < 1500:
                return (wall_target_x, wall_target_y)

        # ── Boost collection when safe and we're running low ──
        if (car_boost < 15 and situation == "free_ball"
                and dist_ball > 2000 and self._boost_pads):
            pad = find_best_boost_pad(car_pos, pred_ball, self._boost_pads,
                                      max_detour=1200.0)
            if pad is not None:
                return pad['pos']

        # ── Defense-only behavior: prioritize positioning when ball is free ──
        if mode == MODE_DEFENSE and situation == "free_ball":
            # If ball is very safe (deep in opponent half), hold a back-post-ish position.
            if (by * push_dir) > 1200:
                return (0.0, _clamp(own_goal_y * 0.55, -fl, fl))

            # If opponent is about to play the ball near our half, consider a bump/demo.
            if (
                self.knowledge.can_do("demo_play")
                and (opp_pos[1] * push_dir) < -300
                and _dist(car_pos, opp_pos) < 1200
                and opp_dist_ball < 800
            ):
                return (_clamp(opp_pos[0], -fw, fw),
                        _clamp(opp_pos[1], -fl, fl))

            # Otherwise shadow between ball and our goal.
            w = 0.55
            sx = bx * w + 0.0 * (1 - w)
            sy = by * w + own_goal_y * (1 - w)
            return (_clamp(sx, -fw, fw),
                    _clamp(sy, -fl, fl))

        # ── Attack / Balanced: disruption as a secondary objective ──
        if mode in (MODE_ATTACK, MODE_BALANCED) and situation in ("opp_has_ball", "defending"):
            if (
                self.knowledge.can_do("demo_play")
                and _dist(car_pos, opp_pos) < 900
                and opp_dist_ball < 650
            ):
                return (_clamp(opp_pos[0], -fw, fw),
                        _clamp(opp_pos[1], -fl, fl))

        # ═══ DEFENDING — use trajectory to intercept ═══
        if situation == "defending":
            # Try physics-based intercept from trajectory
            if self._ball_trajectory:
                intercept = find_intercept(
                    self._ball_trajectory,
                    car_pos[0], car_pos[1],
                    car_speed, car_boost,
                    max_z=400.0,
                )
                if intercept is not None:
                    ix, iy, iz, it = intercept
                    if abs(iy - own_goal_y) < 1500:
                        return (_clamp(ix, -fw, fw),
                                _clamp(iy, -fl, fl))
                    w = 0.65
                    return (_clamp(ix * w, -fw, fw),
                            _clamp(iy * w + own_goal_y * (1 - w), -fl, fl))
            # Fallback: linear prediction toward goal
            weight_ball = 0.7
            intercept_x = bx * weight_ball + 0.0 * (1 - weight_ball)
            intercept_y = by * weight_ball + own_goal_y * (1 - weight_ball)
            if abs(by - own_goal_y) < 1500:
                return (_clamp(bx, -fw, fw),
                        _clamp(by, -fl, fl))
            return (_clamp(intercept_x, -fw, fw),
                    _clamp(intercept_y, -fl, fl))

        # ═══ OPP HAS BALL — challenge or cut off ═══
        if situation == "opp_has_ball":
            my_t = _time_to_reach(car_pos, pred_ball, car_speed)
            opp_t = _time_to_reach(opp_pos, pred_ball, 1200.0)
            if mode == MODE_DEFENSE:
                # Stay between opponent/ball and our goal unless it's an immediate threat.
                if abs(by - own_goal_y) < 2600 or my_t < opp_t + 0.3:
                    return (_clamp(bx, -fw, fw),
                            _clamp(by, -fl, fl))
                w = 0.6
                sx = bx * w + 0.0 * (1 - w)
                sy = by * w + own_goal_y * (1 - w)
                return (_clamp(sx, -fw, fw),
                        _clamp(sy, -fl, fl))

            # Always challenge aggressively — go for the ball
            if my_t < opp_t + 0.5:
                return (_clamp(bx, -fw, fw),
                        _clamp(by, -fl, fl))
            # Can we demo the opponent?
            if self.knowledge.can_do("demo_play") and opp_dist_ball < 400:
                return (_clamp(opp_pos[0], -fw, fw),
                        _clamp(opp_pos[1], -fl, fl))
            # Cut between opp and our goal
            cut_x = opp_pos[0] * 0.4
            cut_y = (opp_pos[1] + own_goal_y) * 0.3
            return (_clamp(cut_x, -fw, fw),
                    _clamp(cut_y, -fl, fl))

        # ═══ WE HAVE BALL — push to goal ASAP ═══
        if situation == "we_have_ball":
            attack_anchor = mode_attack_anchor(gm, opp_goal_y, fl)
            goal_aim = aim_at_goal(pred_ball, attack_anchor[1], self._keeper_pos, goal_w=gw)

            # ── GOAL SCORING PRIORITY LOGIC ──────────────────────────────
            dist_ball_to_opp_goal = math.hypot(pred_ball[0], pred_ball[1] - opp_goal_y)

            # Collect opponent positions for open-goal check
            opp_positions_list = [opp_pos] if opp_pos != (0.0, 0.0) else []
            open_goal = detect_open_goal(pred_ball, opp_positions_list, opp_goal_y, goal_half_width=gw)

            # Ball is very close to opponent goal → commit to shot immediately
            if dist_ball_to_opp_goal < 1500:
                # Aim at far post away from keeper
                shot_aim = aim_at_goal(pred_ball, opp_goal_y, self._keeper_pos, goal_w=gw)
                return self._push_target(pred_ball, shot_aim, 150.0)

            # Open goal AND ball in shooting range → fast direct shot
            if open_goal and dist_ball_to_opp_goal < 3000:
                shot_aim = aim_at_goal(pred_ball, opp_goal_y, self._keeper_pos, goal_w=gw)
                push_d = 200.0 if car_behind_ball else 350.0
                return self._push_target(pred_ball, shot_aim, push_d)

            # Ball is in opponent half and car has clear shot angle → shoot
            if (pred_ball[1] * push_dir) > 0 and dist_ball_to_opp_goal < 3500:
                # Check if we're behind the ball (can shoot with momentum)
                if car_behind_ball:
                    shot_aim = aim_at_goal(pred_ball, opp_goal_y, self._keeper_pos, goal_w=gw)
                    return self._push_target(pred_ball, shot_aim, 300.0)
            # ── END GOAL SCORING PRIORITY ─────────────────────────────────

            if mode == MODE_DEFENSE:
                # Safe clear toward opponent half / corners.
                clear_x = 3200.0 if bx < 0 else -3200.0
                clear_target = (clear_x, opp_goal_y * 0.75)
                return self._push_target(pred_ball, clear_target, 650.0)
            # Backboard read: if ball is heading toward opponent backwall at height, chase the rebound
            if (self._ball_trajectory and ball_z > 200
                    and abs(ball_vel[1]) > 500
                    and (ball_vel[1] * push_dir) > 0):
                landing = ball_landing_pos(self._ball_trajectory)
                if landing and abs(landing[1]) > 4000:
                    # Ball will hit backboard — position for rebound
                    rebound_x = _clamp(landing[0] * 0.3, -2000, 2000)
                    rebound_y = _clamp(opp_goal_y * 0.85, -fl, fl)
                    return (rebound_x, rebound_y)
            # Trailing aggression: push harder
            push_mult = trailing_aggression
            if car_behind_ball:
                # Behind ball — push toward goal corner (away from keeper)
                push_d = (700.0 if ball_speed < 600 else 450.0) * push_mult
                return self._push_target(pred_ball, goal_aim, push_d)
            else:
                # In front of ball, go around it to get behind
                return self._approach_ball(pred_ball, goal_aim, 350.0)

        # ═══ FREE BALL — use intercept prediction ═══
        attack_anchor = mode_attack_anchor(gm, opp_goal_y, fl)
        goal_aim = aim_at_goal(pred_ball, attack_anchor[1], self._keeper_pos, goal_w=gw)
        if self._ball_trajectory:
            intercept = find_intercept(
                self._ball_trajectory,
                car_pos[0], car_pos[1],
                car_speed, car_boost,
                max_z=350.0,
            )
            if intercept is not None:
                ix, iy, iz, it = intercept
                push_d = (500.0 if ball_speed < 500 else 300.0) * trailing_aggression
                return self._push_target((ix, iy), goal_aim, push_d)
        # Fallback: chase predicted ball
        if dist_ball < 2500:
            push_d = (550.0 if ball_speed < 500 else 350.0) * trailing_aggression
            return self._push_target(pred_ball, goal_aim, push_d)
        return (_clamp(bx, -fw, fw),
                _clamp(by, -fl, fl))

    def plan(self, algorithm: str, mode: str, my_team: int,
             car_pos: Tuple[float, float], ball_pos: Tuple[float, float],
             ball_vel: Tuple[float, float], car_speed: float,
             opp_pos: Tuple[float, float], reset_mode: bool,
             situation: str = "free_ball",
             car_boost: float = 50.0,
             ball_z: float = 93.0) -> DecisionOutput:
        self.knowledge.tick()
        decision = self._plan_candidate(
            algorithm=algorithm,
            mode=mode,
            my_team=my_team,
            car_pos=car_pos,
            ball_pos=ball_pos,
            ball_vel=ball_vel,
            car_speed=car_speed,
            opp_pos=opp_pos,
            reset_mode=reset_mode,
            situation=situation,
            car_boost=car_boost,
            ball_z=ball_z,
        )
        path_result = decision.path_result
        if not reset_mode:
            for node in path_result.node_path[:8]:
                self.knowledge.record_node(node)
        self._last_path = path_result
        self._last_algorithm = decision.algorithm
        return decision

    def build_controls(self, car_yaw: float, car_pos: Tuple[float, float],
                       target: Tuple[float, float], speed: float,
                       dist_to_ball: float = 9999.0, car_on_ground: bool = True,
                       ball_z: float = 0.0, car_boost_amount: float = 0.0,
                       situation: str = "free_ball") -> Dict[str, float]:
        dx = target[0] - car_pos[0]
        dy = target[1] - car_pos[1]
        distance = math.hypot(dx, dy)
        target_yaw = math.atan2(dy, dx)
        diff = (target_yaw - car_yaw + math.pi) % (2.0 * math.pi) - math.pi
        abs_diff = abs(diff)

        sm = self.settings.steer_mult
        tm = self.settings.throttle_mult
        ba = self.settings.boost_aggression
        gf = self.settings.ground_friction

        # On slippery ground: reduce steering to avoid spinouts, keep throttle up
        if gf < 0.8:
            sm *= 0.7
            # On ice, handbrake turns are fatal — never handbrake
            slip_mode = True
        else:
            slip_mode = False

        # Only reverse when target is almost directly behind and very close
        reverse = False
        if abs_diff > 2.5 and distance < 500:
            reverse = True
            diff = -((target_yaw - car_yaw + math.pi) % (2.0 * math.pi) - math.pi)
            abs_diff = abs(diff)

        # Proportional steering: softer gain for smooth arcs
        steer = _clamp(diff * 4.0 * sm, -1.0, 1.0)

        if reverse:
            throttle = -0.8 * tm
            boost_ctrl = False
        else:
            # Always full throttle unless very sharp turn
            if abs_diff < 1.2:
                throttle = 1.0 * tm
            elif abs_diff < 1.8:
                throttle = 0.85 * tm
            else:
                throttle = 0.65 * tm

            boost_ctrl = (abs_diff < 0.5 and speed < 2280
                          and distance > 200 and car_boost_amount > 5
                          and ba > 0.4)
            # More aggressive boost when chasing the ball
            if situation in ("opp_has_ball", "free_ball", "defending") and abs_diff < 0.7:
                boost_ctrl = car_boost_amount > 3 and speed < 2300

        handbrake = (not reverse and not slip_mode
                     and abs_diff > 1.2 and car_on_ground and speed > 500)

        jump = False
        pitch = 0.0
        roll_val = 0.0

        if car_on_ground and not reverse:
            # Front flip into ball when really close & well-aligned
            if (dist_to_ball < 350 and abs_diff < 0.3 and speed > 400
                    and ball_z < 180
                    and self.knowledge.can_do("front_flip")):
                jump = True
                pitch = -1.0
                roll_val = _clamp(math.sin(diff) * 0.3, -0.4, 0.4)
                self.knowledge.reward_mechanic("front_flip", 0.01)
                if speed > 800:
                    self.knowledge.reward_mechanic("power_shot", 0.01)

        # ── Aerial attempts when ball is high ──
        if (not car_on_ground or ball_z > 300) and ball_z > 250:
            mode_max_z = self.settings.max_aerial_z
            if (dist_to_ball < 1500 and abs_diff < 0.6
                    and self.knowledge.can_do("aerial")
                    and ball_z < mode_max_z):
                boost_ctrl = True
                jump = not car_on_ground
                pitch = _clamp(-0.5 if ball_z > 500 else -0.2, -1.0, 0.0)
                self.knowledge.reward_mechanic("aerial", 0.005)

        if speed > 2200 and car_on_ground:
            boost_ctrl = False
            self.knowledge.reward_mechanic("boost_management", 0.002)

        # ── Mode-specific control tweaks ──
        gm = self.settings.game_mode

        if gm == GM_KNOCKOUT:
            boost_ctrl = abs_diff < 0.8 and car_boost_amount > 5
            jump = False

        if gm in (GM_SNOW, GM_WINTER_BREAKAWAY) and self.settings.puck_mode:
            if jump and ball_z < 100:
                jump = False
                pitch = 0.0

        if gm in (GM_BOOMER, GM_SPEED_DEMON):
            if abs_diff < 0.6 and car_boost_amount > 5:
                boost_ctrl = True

        if gm in (GM_HOOPS, GM_BEACH_BALL, GM_SPRING_LOADED):
            if ball_z > 200 and dist_to_ball < 1500 and abs_diff < 1.0:
                boost_ctrl = True
                if not car_on_ground:
                    pitch = _clamp(-0.4, -1.0, 0.0)

        return {
            "throttle": _clamp(throttle, -1.0, 1.0),
            "steer": steer, "yaw": steer, "pitch": pitch,
            "roll": roll_val,
            "boost": 1.0 if boost_ctrl else 0.0,
            "handbrake": 1.0 if handbrake else 0.0,
            "jump": 1.0 if jump else 0.0,
        }

    def blend_human_experience(self, controls: Dict[str, float],
                               car_pos: Tuple[float, float],
                               ball_pos: Tuple[float, float],
                               speed: float, my_team: int,
                               weight: float = 0.3) -> Dict[str, float]:
        hint = self.knowledge.lookup_human_hint(car_pos, ball_pos, speed, my_team)
        if hint is None:
            return controls
        w = weight
        controls["throttle"] = _clamp(
            controls["throttle"] * (1 - w) + hint["throttle"] * w, -1, 1)
        controls["steer"] = _clamp(
            controls["steer"] * (1 - w) + hint["steer"] * w, -1, 1)
        controls["yaw"] = controls["steer"]
        if hint["boost_in"] > 0.5:
            controls["boost"] = max(controls["boost"], w)
        if hint["jump"] > 0.5:
            controls["jump"] = max(controls["jump"], 0.2)
        return controls
