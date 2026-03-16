"""
Master Adaptive Learning System for Rocket League AI Bot.

Orchestrates all individual RL algorithms:
- Q-Learning (core/q_learning.py)      - Role/mode selection
- SARSA (core/sarsa_opponent.py)        - Opponent behavior prediction
- Policy Gradient (core/policy_gradient.py) - Human demo learning
- Actor-Critic (core/actor_critic.py)   - Control refinement
- Ensemble Voting (core/ensemble_voter.py)  - Strategy blending
- Online Learner (core/online_learner.py)   - Real-time adaptation
- Reward Calculator (core/reward_calculator.py) - Frame reward signals

The bot learns:
1. From the USER when M is pressed (expert demonstrations)
2. From the OPPONENT by tracking their patterns and adapting
3. From its OWN experience through reward signals (goals, saves, possession)
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from core.rl_state import (
    _zone, _speed_zone, _boost_zone, _score_zone, build_state_key,
    ROLE_ACTIONS, OPP_ACTIONS,
)
from core.rl_algorithms import (
    QLearningRoleSelector, SARSAOpponentModel, PolicyGradientHumanLearner,
    ActorCritic, EnsembleVoter, OnlinePatternLearner,
    DQNRoleSelector, PPORoleSelector, A2CRoleSelector,
    MonteCarloRoleSelector, ModelBasedRLSelector,
)
from core.reward_calculator import RewardCalculator
from core.advanced_ml import AdvancedMLSystem, ALL_ADVANCED_COMPONENTS

# Full set of toggleable RL models
ALL_MODELS = frozenset({
    "q_learning", "actor_critic", "online_learner",
    "dqn", "ppo", "a2c", "monte_carlo", "model_based",
})


class AdaptiveLearner:
    """
    Master system that coordinates all learning algorithms.
    This is the main interface used by DecisionEngine.
    """

    def __init__(self):
        self.q_role = QLearningRoleSelector()
        self.sarsa_opp = SARSAOpponentModel()
        self.policy_human = PolicyGradientHumanLearner()
        self.actor_critic = ActorCritic()
        self.ensemble = EnsembleVoter()
        self.online = OnlinePatternLearner()
        self.reward_calc = RewardCalculator()
        # Advanced RL algorithms
        self.dqn         = DQNRoleSelector()
        self.ppo         = PPORoleSelector()
        self.a2c         = A2CRoleSelector()
        self.monte_carlo = MonteCarloRoleSelector()
        self.model_based = ModelBasedRLSelector()
        self.advanced    = AdvancedMLSystem()

        self._current_role = "balanced"
        self._current_state_key = ""
        self._opp_state_key = ""
        self._opp_action = "rush_ball"
        self._frame_count = 0
        self._is_manual_mode = False
        # Which models participate in ensemble voting (all active by default)
        self.active_models: set = set(ALL_MODELS)
        self.active_advanced_components: set = set(ALL_ADVANCED_COMPONENTS)

        # Opponent scoring pattern tracking
        self._opp_pre_goal_history: List[Dict] = []
        self._opp_frame_buffer: List[Dict] = []
        self._opp_scoring_patterns: List[List[Dict]] = []
        self._context_tag = "soccer|std"

        # Advanced-ML state
        self._prev_state_key: str = ""
        self._last_threat_level: float = 0.0
        self._last_cf_regret: float = 0.0

    def set_active_models(self, models: set) -> None:
        """Set which RL models participate in ensemble voting.
        Pass an empty set to restore all models."""
        self.active_models = set(models) if models else set(ALL_MODELS)

    def set_active_advanced_components(self, components: set) -> None:
        """Set which advanced-ML components are active.
        Pass an empty set to restore all advanced systems."""
        self.active_advanced_components = set(components) if components else set(ALL_ADVANCED_COMPONENTS)
        self.advanced.set_active_components(self.active_advanced_components)

    def _compose_state_key(self, base_state: str, context_tag: str) -> str:
        return f"{context_tag}::{base_state}" if context_tag else base_state

    def build_state(self, situation: str, car_pos: Tuple[float, float],
                    ball_pos: Tuple[float, float], car_speed: float,
                    car_boost: float, score_diff: int,
                    push_dir: float, possession: str) -> str:
        """Build the current state key from game observables."""
        ball_zone = _zone(ball_pos[0], ball_pos[1], push_dir)
        car_zone = _zone(car_pos[0], car_pos[1], push_dir)
        speed_z = _speed_zone(car_speed)
        boost_z = _boost_zone(car_boost)
        score_z = _score_zone(score_diff)
        return build_state_key(situation, ball_zone, car_zone,
                               speed_z, boost_z, score_z, possession)

    def decide_role(self, situation: str, car_pos: Tuple[float, float],
                    ball_pos: Tuple[float, float], ball_vel: Tuple[float, float],
                    opp_pos: Tuple[float, float], opp_vel: Tuple[float, float],
                    car_speed: float, car_boost: float,
                    score_diff: int, push_dir: float,
                    user_mode: str,
                    context_tag: str = "") -> str:
        """
        Decide the best role to play right now.
        If user set a specific mode (attack/defense), respect it but
        still let the RL system learn. If balanced, let RL choose.
        """
        self._frame_count += 1
        self._is_manual_mode = (user_mode == "manual")

        opp_dist = math.hypot(ball_pos[0] - opp_pos[0],
                              ball_pos[1] - opp_pos[1])
        my_dist = math.hypot(ball_pos[0] - car_pos[0],
                             ball_pos[1] - car_pos[1])
        possession = "us" if my_dist < opp_dist else "them"

        base_state = self.build_state(
            situation, car_pos, ball_pos, car_speed, car_boost,
            score_diff, push_dir, possession)
        self._context_tag = context_tag or self._context_tag
        state_key = self._compose_state_key(base_state, self._context_tag)
        self._current_state_key = state_key

        reward = self.reward_calc.compute_reward(
            car_pos, ball_pos, ball_vel, opp_pos,
            car_speed, car_boost, push_dir, situation)

        self.q_role.update(state_key, reward)

        opp_zone = _zone(opp_pos[0], opp_pos[1], -push_dir)
        opp_state = self._compose_state_key(
            f"{opp_zone}|{_zone(ball_pos[0], ball_pos[1], push_dir)}",
            self._context_tag,
        )
        new_opp_action = self.sarsa_opp.classify_opp_action(
            opp_pos, opp_vel, ball_pos, push_dir)
        if self._opp_state_key:
            opp_reward = -reward
            self.sarsa_opp.update(self._opp_state_key, self._opp_action,
                                  opp_reward, opp_state, new_opp_action)
        self._opp_state_key = opp_state
        self._opp_action = new_opp_action

        # Track opponent behavior for scoring pattern analysis
        self._opp_frame_buffer.append({
            "state": opp_state,
            "action": new_opp_action,
            "opp_pos": opp_pos,
            "ball_pos": ball_pos,
        })
        if len(self._opp_frame_buffer) > 180:
            self._opp_frame_buffer.pop(0)

        self.actor_critic.update(state_key, reward)
        self.online.record(state_key, self._current_role, reward)
        # Advanced RL updates
        self.dqn.update(state_key, reward)
        self.ppo.update(state_key, reward)
        self.a2c.update(state_key, reward)
        self.monte_carlo.record(state_key, self._current_role, reward)
        self.model_based.record(state_key, self._current_role, reward)
        # Dyna-Q: inject model-generated samples into DQN replay
        imaginary = self.model_based.generate_sample()
        if imaginary is not None:
            self.dqn.add_imaginary(*imaginary)

        q_role = self.q_role.choose_role(state_key)

        opp_pred = self.sarsa_opp.predict(opp_state)
        counter_role = self._counter_strategy(opp_pred)

        # Check if opponent is in a known scoring pattern
        if self._is_opp_scoring_pattern():
            counter_role = "defense"

        value = self.actor_critic.evaluate_state(state_key)
        ac_role = "attack" if value > 0.3 else "defense" if value < -0.3 else "balanced"

        online_role = self.online.best_action(state_key) or "balanced"
        dqn_role    = self.dqn.choose_role(state_key)
        ppo_role    = self.ppo.choose_role(state_key)
        a2c_role    = self.a2c.choose_role(state_key)
        mc_role     = self.monte_carlo.choose_role(state_key)
        mb_role     = self.model_based.choose_role(state_key)

        # Advanced ML tick (Causal, Anomaly, MAML, MTL, PA, Deep RL)
        from core.rl_state import ROLE_ACTIONS as _RA
        role_idx = _RA.index(q_role) if q_role in _RA else 0

        # Mechanic index: infer from current role/situation for DeepRLNet heads
        # 0=drive 1=dodge 2=aerial 3=boost_pickup 4=challenge 5=rotate 6=dribble 7=save
        _mech_idx = (
            4 if self._current_role == "challenge" else
            5 if self._current_role == "rotate_back" else
            7 if situation == "defending" else
            6 if situation == "we_have_ball" else
            0
        )

        _dist_to_ball   = math.hypot(ball_pos[0] - car_pos[0], ball_pos[1] - car_pos[1])
        _opp_dist_ball  = math.hypot(ball_pos[0] - opp_pos[0], ball_pos[1] - opp_pos[1])
        _ball_speed     = math.hypot(ball_vel[0], ball_vel[1])
        _opp_spd        = math.hypot(opp_vel[0], opp_vel[1])
        adv_metrics = {
            "opp_speed":        _opp_spd,
            "car_speed":        car_speed,
            "dist_to_ball":     _dist_to_ball,
            "ball_vel_y":       ball_vel[1],
            "ball_vel_x":       ball_vel[0],
            "ball_speed":       _ball_speed,
            "opp_dist_to_ball": _opp_dist_ball,
            "car_boost":        car_boost,
            "score_diff":       float(score_diff),
            "ball_y_abs":       abs(ball_pos[1]),
            "car_y_rel":        car_pos[1] * push_dir,
            "pressure":         float(
                situation in ("defending", "opp_has_ball")
                and ball_pos[1] * push_dir < -2000),
        }
        _prev_key = self._prev_state_key if self._prev_state_key else state_key
        adv_result = self.advanced.tick(
            _prev_key, role_idx, _mech_idx, reward,
            state_key, False, adv_metrics,
        )
        self._prev_state_key    = state_key
        self._last_threat_level = adv_result.get("threat_level", 0.0)
        self._last_cf_regret    = adv_result.get("cf_regret", 0.0)
        adv_override = adv_result.get("role_override")

        # Counterfactual regret feedback: when regret is high the current model made
        # a suboptimal choice — gently penalise q_learning and dqn in ensemble weights.
        if self._last_cf_regret > 1.0:
            _regret_penalty = -0.05 * min(self._last_cf_regret, 4.0)
            self.ensemble.reward_source("q_learning", _regret_penalty)
            self.ensemble.reward_source("dqn",        _regret_penalty * 0.5)

        am = self.active_models
        recommendations = {
            "sarsa_counter": counter_role,   # always active — opponent tracking
            "heuristic":     self._heuristic_role(situation, score_diff),  # always
        }
        if "q_learning"    in am: recommendations["q_learning"]    = q_role
        if "actor_critic"  in am: recommendations["actor_critic"]  = ac_role
        if "online_learner"in am: recommendations["online_learner"]= online_role
        if "dqn"           in am: recommendations["dqn"]           = dqn_role
        if "ppo"           in am: recommendations["ppo"]           = ppo_role
        if "a2c"           in am: recommendations["a2c"]           = a2c_role
        if "monte_carlo"   in am: recommendations["monte_carlo"]   = mc_role
        if "model_based"   in am: recommendations["model_based"]   = mb_role
        if adv_override:
            recommendations["advanced_ml"] = adv_override
            self.ensemble.source_weights["advanced_ml"] = 1.2
        voted_role = self.ensemble.vote(recommendations)

        if user_mode == "attack":
            voted_role = "attack" if voted_role not in ("challenge", "rotate_back") else voted_role
        elif user_mode == "defense":
            voted_role = "defense" if voted_role not in ("shadow", "rotate_back") else voted_role
        elif user_mode == "balanced":
            pass

        self._current_role = self._map_to_engine_mode(voted_role)
        return self._current_role

    def update_rewards_only(self, car_pos: Tuple[float, float],
                            ball_pos: Tuple[float, float],
                            ball_vel: Tuple[float, float],
                            opp_pos: Tuple[float, float],
                            opp_vel: Tuple[float, float],
                            car_speed: float, car_boost: float,
                            score_diff: int, push_dir: float,
                            situation: str,
                            context_tag: str = ""):
        """
        Update reward tracking and opponent model during manual mode.
        Does NOT decide a role -- the user is in control.
        """
        self._frame_count += 1
        self._is_manual_mode = True

        opp_dist = math.hypot(ball_pos[0] - opp_pos[0],
                              ball_pos[1] - opp_pos[1])
        my_dist = math.hypot(ball_pos[0] - car_pos[0],
                             ball_pos[1] - car_pos[1])
        possession = "us" if my_dist < opp_dist else "them"

        base_state = self.build_state(
            situation, car_pos, ball_pos, car_speed, car_boost,
            score_diff, push_dir, possession)
        self._context_tag = context_tag or self._context_tag
        state_key = self._compose_state_key(base_state, self._context_tag)
        self._current_state_key = state_key

        reward = self.reward_calc.compute_reward(
            car_pos, ball_pos, ball_vel, opp_pos,
            car_speed, car_boost, push_dir, situation)

        # Keep tracking opponent during manual mode
        opp_zone = _zone(opp_pos[0], opp_pos[1], -push_dir)
        opp_state = self._compose_state_key(
            f"{opp_zone}|{_zone(ball_pos[0], ball_pos[1], push_dir)}",
            self._context_tag,
        )
        new_opp_action = self.sarsa_opp.classify_opp_action(
            opp_pos, opp_vel, ball_pos, push_dir)
        if self._opp_state_key:
            opp_reward = -reward
            self.sarsa_opp.update(self._opp_state_key, self._opp_action,
                                  opp_reward, opp_state, new_opp_action)
        self._opp_state_key = opp_state
        self._opp_action = new_opp_action

        self._opp_frame_buffer.append({
            "state": opp_state,
            "action": new_opp_action,
            "opp_pos": opp_pos,
            "ball_pos": ball_pos,
        })
        if len(self._opp_frame_buffer) > 180:
            self._opp_frame_buffer.pop(0)

        self.actor_critic.update(state_key, reward)
        self.online.record(state_key, "manual", reward)

        # Advanced ML update during manual mode (anomaly, MAML, causal, etc.)
        from core.rl_state import ROLE_ACTIONS as _RA_m
        _role_i_m = _RA_m.index("balanced")  # manual mode → balanced signal
        _adv_metrics_m = {
            "opp_speed":        math.hypot(opp_vel[0], opp_vel[1]),
            "car_speed":        car_speed,
            "dist_to_ball":     math.hypot(ball_pos[0] - car_pos[0], ball_pos[1] - car_pos[1]),
            "ball_vel_y":       ball_vel[1],
            "ball_vel_x":       ball_vel[0],
            "ball_speed":       math.hypot(ball_vel[0], ball_vel[1]),
            "opp_dist_to_ball": math.hypot(ball_pos[0] - opp_pos[0], ball_pos[1] - opp_pos[1]),
            "car_boost":        car_boost,
            "score_diff":       float(score_diff),
            "pressure":         float(situation in ("defending", "opp_has_ball")),
        }
        _prev_key_m = self._prev_state_key if self._prev_state_key else state_key
        self.advanced.tick(_prev_key_m, _role_i_m, 0, reward, state_key, False, _adv_metrics_m)
        self._prev_state_key = state_key

    def record_human_frame(self, car_pos: Tuple[float, float],
                             ball_pos: Tuple[float, float],
                             car_speed: float, car_boost: float,
                             opp_pos: Tuple[float, float],
                             push_dir: float,
                             context_tag: str,
                             throttle: float, steer: float,
                             boost_in: float, jump_in: float):
        """Record human demonstration for policy gradient learning."""
        self.policy_human.record_human_frame(
            car_pos, ball_pos, car_speed, car_boost,
            opp_pos, push_dir, context_tag, throttle, steer, boost_in, jump_in)

    def get_human_policy_controls(self, car_pos: Tuple[float, float],
                                  ball_pos: Tuple[float, float],
                                  car_speed: float, car_boost: float,
                                  opp_pos: Tuple[float, float],
                                  push_dir: float,
                                  context_tag: str = "") -> Optional[Dict[str, float]]:
        """Get control suggestions from the learned human policy."""
        return self.policy_human.suggest_controls(
            car_pos, ball_pos, car_speed, car_boost, opp_pos, push_dir, context_tag)

    def get_actor_critic_adjustments(self, state_key: str) -> Dict[str, float]:
        """Get Actor-Critic's adjustments for control blending."""
        return self.actor_critic.get_adjustments(state_key)

    def signal_goal_scored(self):
        self.reward_calc.signal_goal_scored()
        self.advanced.on_goal_scored()
        self.ensemble.reward_source("q_learning",    1.0)
        self.ensemble.reward_source("sarsa_counter", 0.5)
        self.ensemble.reward_source("actor_critic",  0.5)
        self.ensemble.reward_source("online_learner",0.4)
        self.ensemble.reward_source("dqn",           0.6)
        self.ensemble.reward_source("ppo",           0.6)
        self.ensemble.reward_source("a2c",           0.5)
        self.ensemble.reward_source("monte_carlo",   0.5)
        self.ensemble.reward_source("model_based",   0.4)
        # Monte Carlo episode boundary
        self.monte_carlo.end_episode(terminal_reward=10.0)

    def signal_goal_conceded(self):
        self.reward_calc.signal_goal_conceded()
        self.advanced.on_goal_conceded()
        self.ensemble.reward_source("q_learning",    -0.8)
        self.ensemble.reward_source("sarsa_counter", -0.3)
        self.ensemble.reward_source("online_learner",-0.3)
        self.ensemble.reward_source("dqn",           -0.5)
        self.ensemble.reward_source("ppo",           -0.5)
        self.ensemble.reward_source("a2c",           -0.4)
        self.ensemble.reward_source("monte_carlo",   -0.4)
        self.ensemble.reward_source("model_based",   -0.3)
        # Monte Carlo episode boundary
        self.monte_carlo.end_episode(terminal_reward=-8.0)
        # Store opponent's pre-goal behavior for pattern learning
        if self._opp_frame_buffer:
            self._opp_scoring_patterns.append(list(self._opp_frame_buffer))
            self._opp_scoring_patterns = self._opp_scoring_patterns[-20:]
            for frame in self._opp_frame_buffer[-30:]:
                state = frame["state"]
                action = frame["action"]
                q = self.sarsa_opp._get_q(state)
                q[action] = q.get(action, 0.0) + 0.5

    def signal_save_made(self):
        self.reward_calc.signal_save_made()
        self.ensemble.reward_source("sarsa_counter", 0.5)

    def signal_demo(self):
        self.reward_calc.signal_demo()

    def get_threat_level(self) -> float:
        """Return the current anomaly-detection threat level (0.0–1.0)."""
        return self._last_threat_level

    def get_opponent_prediction(self) -> str:
        """Return what we think the opponent is doing."""
        return self._opp_action

    def get_current_state_key(self) -> str:
        return self._current_state_key

    def get_trend(self) -> float:
        """Return the recent performance trend."""
        return self.online.recent_trend()

    def _is_opp_scoring_pattern(self) -> bool:
        """Check if opponent's current behavior matches a known scoring pattern."""
        if not self._opp_scoring_patterns or len(self._opp_frame_buffer) < 10:
            return False
        recent = [f["action"] for f in self._opp_frame_buffer[-10:]]
        for pattern in self._opp_scoring_patterns:
            if len(pattern) < 10:
                continue
            stored = [f["action"] for f in pattern[-10:]]
            matches = sum(1 for a, b in zip(recent, stored) if a == b)
            if matches >= 7:
                return True
        return False

    @staticmethod
    def _counter_strategy(opp_action: str) -> str:
        counter = {
            "rush_ball": "challenge",
            "rotate_back": "attack",
            "boost_grab": "attack",
            "shadow": "attack",
            "demolish": "shadow",
            "aerial": "rotate_back",
        }
        return counter.get(opp_action, "balanced")

    @staticmethod
    def _heuristic_role(situation: str, score_diff: int) -> str:
        if situation == "defending":
            return "defense"
        if situation == "we_have_ball":
            return "attack"
        if score_diff < 0:
            return "attack"
        if score_diff > 2:
            return "defense"
        return "balanced"

    @staticmethod
    def _map_to_engine_mode(rl_action: str) -> str:
        mapping = {
            "attack": "attack",
            "defense": "defense",
            "balanced": "balanced",
            "rotate_back": "defense",
            "challenge": "attack",
            "shadow": "defense",
        }
        return mapping.get(rl_action, "balanced")

    def to_dict(self) -> Dict:
        return {
            "q_role":       self.q_role.to_dict(),
            "sarsa_opp":    self.sarsa_opp.to_dict(),
            "policy_human": self.policy_human.to_dict(),
            "actor_critic": self.actor_critic.to_dict(),
            "ensemble":     self.ensemble.to_dict(),
            "online":       self.online.to_dict(),
            "dqn":          self.dqn.to_dict(),
            "ppo":          self.ppo.to_dict(),
            "a2c":          self.a2c.to_dict(),
            "monte_carlo":  self.monte_carlo.to_dict(),
            "model_based":  self.model_based.to_dict(),
            "context_tag":  self._context_tag,
            "opp_scoring_patterns": [
                [{"state": f["state"], "action": f["action"]} for f in pat[-30:]]
                for pat in self._opp_scoring_patterns[-10:]
            ],
            "advanced_ml":  self.advanced.to_dict(),
        }

    def from_dict(self, data: Dict):
        if "q_role" in data:
            self.q_role.from_dict(data["q_role"])
        if "sarsa_opp" in data:
            self.sarsa_opp.from_dict(data["sarsa_opp"])
        if "policy_human" in data:
            self.policy_human.from_dict(data["policy_human"])
        if "actor_critic" in data:
            self.actor_critic.from_dict(data["actor_critic"])
        if "ensemble" in data:
            self.ensemble.from_dict(data["ensemble"])
        if "online" in data:
            self.online.from_dict(data["online"])
        if "dqn" in data:
            self.dqn.from_dict(data["dqn"])
        if "ppo" in data:
            self.ppo.from_dict(data["ppo"])
        if "a2c" in data:
            self.a2c.from_dict(data["a2c"])
        if "monte_carlo" in data:
            self.monte_carlo.from_dict(data["monte_carlo"])
        if "model_based" in data:
            self.model_based.from_dict(data["model_based"])
        self._context_tag = data.get("context_tag", self._context_tag)
        for pat in data.get("opp_scoring_patterns", []):
            self._opp_scoring_patterns.append(pat)
        if "advanced_ml" in data:
            self.advanced.from_dict(data["advanced_ml"])